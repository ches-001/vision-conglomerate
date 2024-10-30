import os
import sys
import logging
import random
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, Sampler, DistributedSampler
from dataset.segmentation_dataset import SegmentationDataset
from modules.segmentation import SegmentationNetwork
from modules.segmentation_loss import SegmentationLoss
from pipeline.segmentation_trainer import TrainSegmentationPipeline
from utils.utils import load_yaml
from utils.make_anchors import generate_anchors_and_class_weights
from utils.ddp_utils import ddp_setup, ddp_destroy, ddp_broadcast
from typing import *

logger = logging.getLogger(__name__)

def make_dataset(data_dir: str, overlap_masks: bool, **kwargs) -> SegmentationDataset:
    return SegmentationDataset(data_dir=data_dir, overlap_masks=overlap_masks, **kwargs)

def make_dataloader(dataset: SegmentationDataset, batch_size: int, sampler: Optional[Sampler]=None, **kwargs) -> DataLoader:
    kwargs = dict(batch_size=batch_size, **kwargs)
    if "num_workers" not in kwargs:
        kwargs["num_workers"] = os.cpu_count()
    if sampler:
        kwargs["shuffle"] = False
    return DataLoader(dataset, collate_fn=SegmentationDataset.collate_fn, sampler=sampler, **kwargs)

def make_model(in_channels: int, num_classes: int, config: Dict[str, Any], anchors: Dict[str, Any]) -> SegmentationNetwork:
    model = SegmentationNetwork(in_channels=in_channels, num_classes=num_classes, config=config, anchors=anchors)
    model.train()
    return model

def make_loss_fn(model: SegmentationNetwork, class_weights: torch.Tensor, overlap_masks: bool, **kwargs) -> SegmentationLoss:
    return SegmentationLoss(model=model, class_weights=class_weights, overlap_masks=overlap_masks, **kwargs)

def make_optimizer(model: SegmentationNetwork, **kwargs) -> torch.optim.Optimizer:
    optimizer_name = kwargs.pop("name")
    kwargs["lr"] = kwargs["lr"] * torch.cuda.device_count()
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **kwargs)
    return optimizer

def make_lr_scheduler(optimizer: torch.optim.Optimizer, **kwargs) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_name = kwargs.pop("name")
    lr_scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **kwargs)
    return lr_scheduler

def run(args: argparse.Namespace, config: Dict[str, Any]):
    if args.use_ddp:
        # setup DDP process group
        ddp_setup()
    data_path = config["train_config"]["data_path"]
    train_path = os.path.join(data_path, "train")
    eval_path = os.path.join(data_path, "valid")
    img_config = config["train_config"]["img_config"]
    dataloader_config = config["train_config"]["dataloader_config"]
    model_config = config["model_config"]
    loss_config = config["train_config"]["loss_config"]
    optimizer_config = config["train_config"]["optimizer_config"]
    lr_scheduler_config = config["train_config"]["lr_scheduler_config"]
    auto_anchors_config = config["auto_anchors_config"]
    overlap_masks = config["train_config"]["overlap_masks"]
    anchors = load_yaml(ANCHORS_PATH)["anchors"]
    device_or_rank = config["train_config"]["device"] if torch.cuda.is_available() else "cpu"

    train_dataset = make_dataset(train_path, overlap_masks=overlap_masks, **img_config)
    eval_dataset = make_dataset(eval_path, overlap_masks=overlap_masks, **img_config)
    train_sampler = None
    eval_sampler = None
    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
        try:
            device_or_rank = int(os.environ["LOCAL_RANK"])
        except KeyError as e:
            logger.error(
                f"{e}. This LOCAL_RANK key not existing in the environment variable is a clear "
                "indication that you need to execute this script with torchrun if you wish to "
                "use the DDP mode (ddp=True)"
            )
            sys.exit(0)

    def print_logs(log: str, rank_to_log: Union[str, int]=-1):
        if not args.no_verbose:
            if args.use_ddp:
                if rank_to_log != -1:
                    if device_or_rank == rank_to_log:
                        print(log)
                else:
                    print(log)
                return
            print(log)

    train_dataloader = make_dataloader(train_dataset, args.batch_size, train_sampler, **dataloader_config)
    eval_dataloader = make_dataloader(eval_dataset, args.batch_size, eval_sampler, **dataloader_config)

    if not args.use_ddp or (args.use_ddp and device_or_rank in [0, "cuda:0"]):
        new_anchors, class_weights = generate_anchors_and_class_weights(
            train_path, 
            predefined_anchors=anchors,
            from_polygons=True, 
            verbose=(not args.no_verbose), 
            anchors_path=ANCHORS_PATH,
            device=device_or_rank,
            **auto_anchors_config, 
        )
        num_classes = torch.tensor([class_weights.shape[0]], dtype=torch.int64, device=device_or_rank)
    else:
        num_classes = torch.tensor([0], dtype=torch.int64, device=device_or_rank)
        new_anchors = torch.zeros(
            len(anchors), 
            len(anchors["sm"]), 
            len(anchors["sm"][0]), 
            dtype=torch.float32, device=device_or_rank
        )

    if args.use_ddp:
        ddp_broadcast(new_anchors, src_rank=0)
        ddp_broadcast(num_classes, src_rank=0)
        if device_or_rank not in [0, "cuda:0"]:
            class_weights = torch.zeros(num_classes.item(), dtype=torch.int64, device=device_or_rank)
        ddp_broadcast(class_weights, src_rank=0)

    new_anchors = {"sm": new_anchors[0], "md": new_anchors[1], "lg": new_anchors[2]}
    in_channels = train_dataset[0][0].shape[0]
    model = make_model(in_channels, num_classes.item(), model_config, new_anchors)
    loss_fn = make_loss_fn(model, class_weights, overlap_masks=overlap_masks, **loss_config)
    optimizer = make_optimizer(model, **optimizer_config)
    lr_scheduler = make_lr_scheduler(optimizer, **lr_scheduler_config) if args.lr_schedule else None
    pipeline = TrainSegmentationPipeline(
        model, 
        loss_fn, 
        optimizer, 
        lr_scheduler=lr_scheduler,
        lr_schedule_interval=args.lr_schedule_interval,
        device_or_rank=device_or_rank,
        ddp_mode=args.use_ddp
    )
    best_loss = np.inf
    best_model_epoch = None
    last_epoch = pipeline.last_epoch
    for epoch in range(last_epoch, args.epochs):
        print_logs(f"train step @ epoch: {epoch} on device: {device_or_rank}", -1)
        _ = pipeline.train(train_dataloader, verbose=(not args.no_verbose))
        if epoch % args.eval_interval == 0:
            print_logs(f"evaluation step @ epoch: {epoch} on device: {device_or_rank}", -1)
            eval_metrics = pipeline.evaluate(eval_dataloader, verbose=(not args.no_verbose))
            if eval_metrics["aggregate_loss"] < best_loss:
                best_loss = eval_metrics["aggregate_loss"]
                best_model_epoch = epoch
                pipeline.save_best_model()
                print_logs(f"best model saved at epoch: {best_model_epoch}", 0)
        if (args.checkpoint_interval > 0) and (epoch % args.checkpoint_interval == 0):
            print_logs(f"checkpoint saved at epoch: {best_model_epoch}", 0)
            pipeline.save_checkpoint()
    pipeline.metrics_to_csv()
    pipeline.save_metrics_plots()
    print_logs(f"\nBest model saved at epoch {best_model_epoch} with loss value of {best_loss :.4f}", 0)
    if args.use_ddp:
        # Destroy process group
        ddp_destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segmentation Network")
    parser.add_argument("--batch_size", type=int, default=32, metavar="", help="Training batch size")
    parser.add_argument("--epochs", type=int, default=200, metavar="", help="Number of training epochs")
    parser.add_argument("--checkpoint_interval", type=int, default=10, metavar="", help="Number of epochs before persisting checkpoint to disk")
    parser.add_argument("--eval_interval", type=int, default=1, metavar="", help="Number of training steps before each evaluation")
    parser.add_argument("--no_verbose", action="store_true", help="Reduce training output verbosity")
    parser.add_argument("--lr_schedule", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--use_ddp", action="store_true", help="Use DDP (Distributed Data Parallelization)")
    parser.add_argument("--lr_schedule_interval", type=int, default=1, metavar="", help="Number of training steps before lr scheduling")
    args = parser.parse_args()

    SEED = 42
    CONFIG_PATH = "config/segmentation/config.yaml"
    ANCHORS_PATH = "config/segmentation/anchors.yaml"
    LOG_FORMAT = "%(asctime)s %(levelname)s %(filename)s: %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    config = load_yaml(CONFIG_PATH)

    # For single GPU / CPU training:: train_seg.py --use_ddp --lr_schedule --batch_size=32
    # For multiple GPU training:: torchrun --standalone --nproc_per_node=gpu train_seg.py --use_ddp --lr_schedule --batch_size=32
    run(args, config)
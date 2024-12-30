import os
import sys
import random
import logging
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, DistributedSampler
from dataset.tracknet_dataset import TrackNetDataset
from modules.tracknet import TrackNet
from pipeline.tracknet_trainer import TrainTrackNetPipeline
from utils.utils import load_yaml
from utils.ddp_utils import ddp_setup, ddp_destroy
from typing import *

logger = logging.getLogger(__name__)


def make_datasets(data_dir: str, **kwargs) -> Tuple[TrackNetDataset, TrackNetDataset]:
    train_dataset = TrackNetDataset(data_dir, split_percentage=0.7, **kwargs)
    eval_dataset = TrackNetDataset(labels_df=train_dataset.unused_labels_df, **kwargs)
    return train_dataset, eval_dataset

def make_dataloader(dataset: TrackNetDataset, batch_size: int, sampler: Optional[Sampler]=None, **kwargs) -> DataLoader:
    kwargs = dict(batch_size=batch_size, **kwargs)
    if "num_workers" not in kwargs:
        kwargs["num_workers"] = os.cpu_count()
    if sampler:
        kwargs["shuffle"] = False
    return DataLoader(dataset, sampler=sampler, **kwargs)

def make_model(in_channels: int, config: Dict[str, Any]) -> TrackNet:
    model = TrackNet(in_channels=in_channels, config=config)
    model.train()
    return model

def make_optimizer(model: TrackNet, **kwargs) -> torch.optim.Optimizer:
    optimizer_name = kwargs.pop("name")
    kwargs["lr"] *= torch.cuda.device_count()
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
    img_config = config["train_config"]["img_config"]
    dataloader_config = config["train_config"]["dataloader_config"]
    tp_dist_tol = config["train_config"]["tp_dist_tol"]
    heatmap_threshold = config["train_config"]["heatmap_threshold"]
    hough_grad_config = config["train_config"]["hough_grad_config"]
    model_config = config["model_config"]
    optimizer_config = config["train_config"]["optimizer_config"]
    lr_scheduler_config = config["train_config"]["lr_scheduler_config"]
    device_or_rank = config["train_config"]["device"] if torch.cuda.is_available() else "cpu"

    train_dataset, eval_dataset = make_datasets(data_path, **img_config)
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

    in_channels = train_dataset[0][0].shape[0]
    model = make_model(in_channels, model_config)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, **optimizer_config)
    lr_scheduler = make_lr_scheduler(optimizer, **lr_scheduler_config) if args.lr_schedule else None
    pipeline = TrainTrackNetPipeline(
        model, 
        loss_fn, 
        optimizer, 
        lr_scheduler=lr_scheduler,
        lr_schedule_interval=args.lr_schedule_interval,
        device_or_rank=device_or_rank,
        ddp_mode=args.use_ddp,
        config_path=CONFIG_PATH,
        tp_dist_tol=tp_dist_tol,
        hough_grad_kwargs=hough_grad_config,
        heatmap_threshold=heatmap_threshold,
    )
    best_loss = np.inf
    best_model_epoch = None
    last_epoch = pipeline.last_epoch
    for epoch in range(last_epoch, args.epochs):
        print_logs(f"train step @ epoch: {epoch} on device: {device_or_rank}", -1)
        _ = pipeline.train(train_dataloader, verbose=(not args.no_verbose), steps_per_epoch=args.steps_per_epoch)
        if epoch % args.eval_interval == 0:
            print_logs(f"evaluation step @ epoch: {epoch} on device: {device_or_rank}", -1)
            eval_metrics = pipeline.evaluate(eval_dataloader, verbose=(not args.no_verbose))
            if eval_metrics["loss"] < best_loss:
                best_loss = eval_metrics["loss"]
                best_model_epoch = epoch
                pipeline.save_best_model()
                print_logs(f"best model saved at epoch {best_model_epoch}", 0)
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
    parser = argparse.ArgumentParser(description="Train Detection Network")
    parser.add_argument("--batch_size", type=int, default=2, metavar="", help="Training batch size")
    parser.add_argument("--epochs", type=int, default=500, metavar="", help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=400, metavar="", help="Number of batch steps per epoch")
    parser.add_argument("--checkpoint_interval", type=int, default=10, metavar="", help="Number of epochs before persisting checkpoint to disk")
    parser.add_argument("--eval_interval", type=int, default=5, metavar="", help="Number of training steps before each evaluation")
    parser.add_argument("--no_verbose", action="store_true", help="Reduce training output verbosity")
    parser.add_argument("--lr_schedule", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--use_ddp", action="store_true", help="Use DDP (Distributed Data Parallelization)")
    parser.add_argument("--lr_schedule_interval", type=int, default=1, metavar="", help="Number of training steps before lr scheduling")
    args = parser.parse_args()

    SEED = 42
    CONFIG_PATH = "config/tracknet/config.yaml"
    ANCHORS_PATH = "config/tracknet/anchors.yaml"
    LOG_FORMAT = "%(asctime)s %(levelname)s %(filename)s: %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    config = load_yaml(CONFIG_PATH)

    # For single GPU / CPU training:: train_tracknet.py --use_ddp --lr_schedule --batch_size=32
    # For multiple GPU training:: torchrun --standalone --nproc_per_node=gpu train_tracknet.py --use_ddp --lr_schedule --batch_size=32
    run(args, config)
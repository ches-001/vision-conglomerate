import os
import tqdm
import math
import logging
import torch
from .detection_trainer import TrainDetectionPipeline
from torch.utils.data import DataLoader
from utils.ddp_utils import ddp_sync_metrics
from typing import Dict

logger = logging.getLogger(__name__)

class TrainSegmentationPipeline(TrainDetectionPipeline):
    metrics_dir = "metrics/segmentation"
    checkpoints_dir = "saved_model/segmentation/checkpoints"
    best_model_dir = "saved_model/segmentation/best_model"

    def __init__(self, *args, **kwargs):
        super(TrainSegmentationPipeline, self).__init__(*args, **kwargs)

    def step(self, dataloader: DataLoader, mode: str, verbose: bool=False) -> Dict[str, float]:
        if mode not in self._valid_modes:
            raise ValueError(f"Invalid mode {mode} expected either one of {self._valid_modes}")
        getattr(self.model, mode)()
        metrics = {}
        if self.ddp_mode:
            # invert progress bar position such that the last (rank n-1) is at
            # the top and the first (rank 0) at the bottom. This is because the
            # first rank will be the one logging all the metrics
            world_size = int(os.environ["WORLD_SIZE"])
            position = (
                self.device_or_rank 
                if not isinstance(self.device_or_rank, str) else 
                int(self.device_or_rank.replace("cuda:", ""))
            )
            position = abs(position - (world_size - 1))
            pbar = tqdm.tqdm(enumerate(dataloader), position=position)
        else:
            total = math.ceil(len(dataloader.dataset) / dataloader.batch_size)
            pbar = tqdm.tqdm(enumerate(dataloader), total=total)

        for count, (images, labels, target_masks) in pbar:
            images: torch.Tensor = images.to(self.device_or_rank)
            labels: torch.Tensor = labels.to(self.device_or_rank)       
            target_masks: torch.Tensor = target_masks.to(self.device_or_rank)        
            (sm_preds, md_preds, lg_preds), protos = self.model(images)
            batch_loss: torch.Tensor
            batch_loss, batch_metrics = self.loss_fn((sm_preds, md_preds, lg_preds), labels, protos, target_masks)
            if mode == "train":
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            # sum metrics across all batches
            for key in batch_metrics.keys(): 
                if key not in metrics.keys(): metrics[key] = batch_metrics[key]
                else: metrics[key] += batch_metrics[key]

        # average metrics from all batches
        for key in metrics.keys(): 
            metrics[key] = (metrics[key] / (count + 1))
        
        # sync metrics across all devices if ddp_mode=True
        if self.ddp_mode:
            metrics = ddp_sync_metrics(self.device_or_rank, metrics)

        if not self.ddp_mode or (self.ddp_mode and self.device_or_rank in [0, f"cuda:0"]):
            getattr(self, f"_{mode}_metrics").append(metrics)
            if verbose:
                log = "[" + mode.title() + "]: " + "\t".join([f"{k.replace('_', ' ')}: {v :.4f}" for k, v in metrics.items()])
                print(log)
        return metrics
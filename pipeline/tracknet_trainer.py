import os
import cv2
import yaml
import tqdm
import math
import logging
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from .detection_trainer import TrainDetectionPipeline
from torch.utils.data import DataLoader
from utils.ddp_utils import ddp_sync_metrics, ddp_sync_vals
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TrainTrackNetPipeline(TrainDetectionPipeline):
    metrics_dir = "metrics/tracknet"
    checkpoints_dir = "saved_model/tracknet/checkpoints"
    best_model_dir = "saved_model/tracknet/best_model"

    def __init__(
            self,
            *args,
            tp_dist_tol: int=4,
            hough_grad_kwargs:Optional[Dict[str, Any]]=None,
            heatmap_threshold: int=128, 
            **kwargs
        ):
        super(TrainTrackNetPipeline, self).__init__(*args, **kwargs)
        self.tp_dist_tol = tp_dist_tol
        self.hough_grad_kwargs = hough_grad_kwargs
        self.heatmap_threshold = heatmap_threshold

    def _save(self, path: str, snapshot_mode: bool=True):
        network_params = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        if snapshot_mode:
            state_dicts = {
                "LAST_EPOCH": self.last_epoch,
                "NETWORK_PARAMS": network_params,
                "OPTIMIZER_PARAMS": self.optimizer.state_dict(),
                "METRICS": {
                    "TRAIN": self._train_metrics,
                    "EVAL": self._eval_metrics
                }
            }
            if self.lr_scheduler:
                state_dicts["LR_SCHEDULER_PARAMS"] = self.lr_scheduler.state_dict()
        else:
            state_dicts = {
                "LAST_EPOCH": self.last_epoch,
                "NETWORK_PARAMS": network_params,
            }
        return torch.save(state_dicts, path)

    def _save_config_copy(self, config_path: str, to_checkpoint_dir: bool):
        if not (self.ddp_mode) or (self.ddp_mode and self.device_or_rank in [0, "cuda:0"]):
            dest_path = os.path.join(
                (self.checkpoints_dir if to_checkpoint_dir else self.best_model_dir), "config"
            )
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            f.close()
            os.makedirs(dest_path, exist_ok=True)
            with open(os.path.join(dest_path, "config.yaml"), "w") as f:
                yaml.safe_dump(config, f, sort_keys=False, default_flow_style=True)
            f.close()

    def train(
            self, 
            dataloader: DataLoader, 
            verbose: bool=False, 
            steps_per_epoch: Optional[int]=None
        ) -> float:
        self.model.train()
        loss = self.step(dataloader, verbose, steps_per_epoch)
        if self.lr_scheduler and (self.last_epoch % self.lr_schedule_interval == 0):
            self.lr_scheduler.step()
        self.last_epoch += 1
        return loss
    
    def evaluate(self, dataloader: DataLoader, verbose: bool=False) -> Dict[str, float]:
        self.model.eval()
        loss = 0
        tp = np.zeros(4)
        fp = np.zeros(4)
        tn = np.zeros(4)
        fn = np.zeros(4)
        hough_grad_kwargs = dict(
            method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7
        )
        hough_grad_kwargs.update(self.hough_grad_kwargs)
        if isinstance(hough_grad_kwargs.get("method"), str):
            hough_grad_kwargs["method"] = getattr(cv2, hough_grad_kwargs.get("method"))

        if self.ddp_mode:
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

        for count, (stacked_frames, gt_heatmaps, others) in pbar:
            stacked_frames: torch.Tensor = stacked_frames.to(self.device_or_rank)
            gt_heatmaps: torch.Tensor = gt_heatmaps.to(dtype=torch.int64, device=self.device_or_rank)  
            others: torch.Tensor = others.to(self.device_or_rank)
            with torch.no_grad():
                pred_heatmaps: torch.Tensor = self.model(stacked_frames)
            batch_loss = self._compute_loss(pred_heatmaps, gt_heatmaps)
            loss += batch_loss.item()
            pred_heatmaps = pred_heatmaps.argmax(dim=-1).cpu().numpy()
            for i in range(0, pred_heatmaps.shape[0]):
                pred_heatmap = pred_heatmaps[i].astype(np.uint8)
                pred_heatmap[pred_heatmap < self.heatmap_threshold] = 0
                pred_heatmap[pred_heatmap >= self.heatmap_threshold] = 255
                visibility, x_gt, y_gt, _ = others[i]
                visibility = int(visibility.item())
                circles = cv2.HoughCircles(pred_heatmap, **hough_grad_kwargs)
                x_pred, y_pred = None, None
                if circles is not None and len(circles) == 1:
                    x_pred = circles[0][0][0]
                    y_pred = circles[0][0][1]
                if x_pred is not None:
                    if visibility != 0:
                        dist = np.sqrt((x_pred - x_gt.item())**2 + (y_pred - y_gt.item())**2)
                        tp[visibility] += dist <= self.tp_dist_tol
                        fp[visibility] += dist > self.tp_dist_tol
                    else:
                        fp[visibility] += 1
                else:
                    if visibility != 0:
                        fn[visibility] += 1
                    else:
                        tn[visibility] += 1
        eps = 1e-8
        loss /= count     
        precision = tp.sum() / (tp.sum() + fp.sum() + eps)
        recall = tp.sum() / (tp[1:].sum() + tn[1:].sum() + fp[1:].sum() + fn[1:].sum() + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        metrics = dict(
            loss=loss, 
            tp=tp.sum(),
            tn=tn.sum(),
            fp=fp.sum(),
            fn=fn.sum(),
            precision=precision, 
            recall=recall, 
            f1=f1
        )

        # sync metrics across all devices if ddp_mode=True
        if self.ddp_mode:
            metrics = ddp_sync_metrics(self.device_or_rank, metrics)

        if not self.ddp_mode or (self.ddp_mode and self.device_or_rank in [0, f"cuda:0"]):
            self._eval_metrics.append(metrics)
            if verbose:
                log = "[eval]: " + "\n".join([f"{k.replace('_', ' ')}: {v :.4f}" for k, v in metrics.items()])
                print(log)
                print(f"tp(vc0, vc1, vc2, vc3): {tp.astype(int)}")
                print(f"tn(vc0, vc1, vc2, vc3): {tn.astype(int)}")
                print(f"fp(vc0, vc1, vc2, vc3): {fp.astype(int)}")
                print(f"fn(vc0, vc1, vc2, vc3): {fn.astype(int)}")
        return metrics
                    
    def step(
            self, 
            dataloader: DataLoader, 
            verbose: bool=False, 
            steps_per_epoch: Optional[int]=None
        ) -> float:
        loss = 0
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
            if steps_per_epoch is not None:
                total = min(total, steps_per_epoch)
            pbar = tqdm.tqdm(enumerate(dataloader), total=total)

        for count, (stacked_frames, gt_heatmaps, others) in pbar:
            stacked_frames: torch.Tensor = stacked_frames.to(self.device_or_rank)
            gt_heatmaps: torch.Tensor = gt_heatmaps.to(dtype=torch.int64, device=self.device_or_rank)  
            others: torch.Tensor = others.to(self.device_or_rank)
            pred_heatmaps_pdf: torch.Tensor = self.model(stacked_frames)
            batch_loss = self._compute_loss(pred_heatmaps_pdf, gt_heatmaps)
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            loss += batch_loss.item()
            if steps_per_epoch is not None and count == (steps_per_epoch-1):
                break
        
        loss /= (count + 1)
        if self.ddp_mode:
            loss = ddp_sync_vals(self.device_or_rank, loss)
        if not self.ddp_mode or (self.ddp_mode and self.device_or_rank in [0, f"cuda:0"]):
            self._train_metrics.append({"loss": loss})
            if verbose:
                print(f"[train]: CE Loss: {loss :.5f}")
        return loss


    def _compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(pred.flatten(0, -2), gt.flatten(0, -1))
        return loss
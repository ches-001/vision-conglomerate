import torch
import warnings
import torch.nn as nn
import pandas as pd
from dataset.detection_dataset import DetectionDataset
from modules.detection import DetectionNetwork
warnings.filterwarnings(action="ignore")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Optional, Dict, List, Union


class FocalLoss(nn.Module):
    def __init__(
            self, 
            reduction: str="mean", 
            gamma: float=1.5, 
            alpha: float=0.25, 
            with_logits: bool=False, 
            **kwargs):
        
        super().__init__()
        self.with_logits = with_logits
        self.loss_fn = getattr(
            nn, ("BCEWithLogitsLoss" if with_logits else "BCELoss")
        )(reduction="none", **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss: torch.Tensor = self.loss_fn(pred, targets)
        if self.with_logits:
            pred = pred.sigmoid()
        pt = torch.exp(-bce_loss)
        flloss = (self.alpha * (1 - pt) ** self.gamma) * bce_loss

        if self.reduction == "none":
            return flloss
        return getattr(flloss, self.reduction)()
 

class DetectionLoss(nn.Module):
    def __init__(
        self,
        model: DetectionNetwork,
        anchor_t: float=4.0,
        edge_t: float=0.5,
        box_w: float=1.0,
        conf_w: float=1.0,
        class_w: float=1.0,
        class_weights: Optional[torch.Tensor]=None,
        label_smoothing: float=0,
        batch_scale_loss: bool=False,
        alpha: Optional[float]=None,
        gamma: Optional[float]=None,
        scale_w: Optional[List]=None
    ):
        super(DetectionLoss, self).__init__()
        self.anchor_t = anchor_t
        self.edge_t = edge_t
        self.box_w = box_w
        self.conf_w = conf_w
        self.class_w = class_w
        self.label_smoothing = label_smoothing
        self.batch_scale_loss = batch_scale_loss
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.model = model
        self.scale_w = scale_w or [4.0, 2.0, 1.0]

        if alpha and gamma:
            self.conf_lossfn = FocalLoss(alpha=alpha, gamma=gamma, with_logits=True)
            self.cls_lossfn = FocalLoss(alpha=alpha, gamma=gamma, with_logits=True)
        else:
            self.conf_lossfn = nn.BCEWithLogitsLoss()
            self.cls_lossfn = nn.BCEWithLogitsLoss()


    def forward(
        self, 
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:        
        metrics_dict = {}
        sm_preds, md_preds, lg_preds = preds
        sm_anchors = self.model.sm_anchors.data.clone().detach()
        md_anchors = self.model.md_anchors.data.clone().detach()
        lg_anchors = self.model.lg_anchors.data.clone().detach()
        (sm_lbox, sm_lconf, sm_lcls), sm_metrics_dict = self.loss_fn(sm_preds, targets, anchors=sm_anchors)
        (md_lbox, md_lconf, md_lcls), md_metrics_dict = self.loss_fn(md_preds, targets, anchors=md_anchors)
        (lg_lbox, lg_lconf, lg_lcls), lg_metrics_dict = self.loss_fn(lg_preds, targets, anchors=lg_anchors)

        lbox = (self.scale_w[0] * sm_lbox) + (self.scale_w[1] * md_lbox) + (self.scale_w[2] * lg_lbox)
        lconf = (self.scale_w[0] * sm_lconf) + (self.scale_w[1] * md_lconf) + (self.scale_w[2] * lg_lconf)
        lcls = (self.scale_w[0] * sm_lcls) + (self.scale_w[1] * md_lcls) + (self.scale_w[2] * lg_lcls)

        _b = preds[-1].shape[0] if self.batch_scale_loss else 1.0
        loss = ((self.box_w * lbox) + (self.conf_w * lconf) + (self.class_w * lcls)) * _b
        metrics_df = pd.DataFrame([sm_metrics_dict, md_metrics_dict, lg_metrics_dict])

        metrics_dict["aggregate_loss"] = loss.item()
        for key in metrics_df.columns:
            metrics_dict[key] = metrics_df[key].mean()
        return loss, metrics_dict


    def loss_fn(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor, 
            anchors: Union[List[float], torch.Tensor],
        ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]]:
        
        _device = preds.device
        indices, t_classes, anchors, t_xywh, _ = DetectionDataset.build_target_by_scale(
            targets, 
            preds.shape[1:3],
            anchors,
            anchor_threshold=self.anchor_t, 
            edge_threshold=self.edge_t
        )
        batch_idx, grid_j, grid_i, anchor_idx = indices
        match_preds = preds[batch_idx, grid_j, grid_i, anchor_idx]
        p_cls_proba = match_preds[:, 1:1+self.model.num_classes]
        p_xywh = match_preds[:, 1+self.model.num_classes:]
        pxy, pwh = p_xywh.chunk(2, dim=-1)
        p_xywh = torch.cat([pxy, pwh*anchors], dim=-1)

        # bbox loss
        ciou = DetectionLoss.compute_ciou(p_xywh, t_xywh)
        ciou_loss = (1.0 - ciou).mean()

        # conf loss
        ciou = ciou.detach()
        t_conf = torch.zeros(preds.shape[:-1], device=_device, dtype=preds.dtype)
        t_conf[batch_idx, grid_j, grid_i, anchor_idx] = ciou
        p_conf = preds[..., 0]
        conf_loss = self.conf_lossfn(p_conf, t_conf)
        pos_conf = p_conf[batch_idx, grid_j, grid_i, anchor_idx].sigmoid()
        neg_conf = p_conf[t_conf == 0].sigmoid()
        avg_pos_conf = pos_conf.mean()
        avg_neg_conf = neg_conf.mean()

        # class loss
        cn = 0.5 * self.label_smoothing
        cp = 1.0 - cn
        t_cls_proba = torch.full_like(p_cls_proba, cn)
        t_cls_proba[range(batch_idx.shape[0]), t_classes] = cp
        class_loss = self.cls_lossfn(p_cls_proba, t_cls_proba)

        # accuracy, precision, recall
        if t_classes.shape[0]:
            pred_labels = p_cls_proba.detach().argmax(dim=-1).cpu().numpy()
            target_labels = t_classes.cpu().numpy()
            accuracy = accuracy_score(target_labels, pred_labels)
            f1 = f1_score(target_labels, pred_labels, average="macro")
            precision = precision_score(target_labels, pred_labels, average="macro")    
            recall = recall_score(target_labels, pred_labels, average="macro")
        else:
             accuracy, f1, precision, recall = [torch.nan] * 4

        # aggregate losses
        handle_nan = lambda val : val if val == val else torch.tensor(0.0, device=_device)
        losses = (handle_nan(ciou_loss), conf_loss, handle_nan(class_loss))
        metrics_dict = {}
        metrics_dict["mean_ciou"] = ciou.mean().item()
        metrics_dict["conf_loss"] = conf_loss.item()
        metrics_dict["avg_pos_conf"] = avg_pos_conf.item()
        metrics_dict["avg_neg_conf"] = avg_neg_conf.item()
        metrics_dict["class_loss"] = class_loss.item()
        metrics_dict["accuracy"] = accuracy
        metrics_dict["f1"] = f1
        metrics_dict["precision"] = precision
        metrics_dict["recall"] = recall
        return losses, metrics_dict


    @staticmethod
    def compute_ciou(preds_xywh: torch.Tensor, targets_xywh: torch.Tensor, e: float=1e-7) -> torch.Tensor:
        assert (preds_xywh.ndim == targets_xywh.ndim + 1) or (preds_xywh.ndim == targets_xywh.ndim)
        if targets_xywh.ndim != preds_xywh.ndim:
                targets_xywh = targets_xywh.unsqueeze(dim=-2)

        preds_w = preds_xywh[..., 2:3]
        preds_h = preds_xywh[..., 3:]
        preds_x1 = preds_xywh[..., 0:1] - (preds_w / 2)
        preds_y1 = preds_xywh[..., 1:2] - (preds_h / 2)
        preds_x2 = preds_x1 + preds_w
        preds_y2 = preds_y1 + preds_h

        targets_w = targets_xywh[..., 2:3]
        targets_h = targets_xywh[..., 3:]
        targets_x1 = targets_xywh[..., 0:1] - (targets_w / 2)
        targets_y1 = targets_xywh[..., 1:2] - (targets_h / 2)
        targets_x2 = targets_x1 + targets_w
        targets_y2 = targets_y1 + targets_h

        intersection_w = (torch.min(preds_x2, targets_x2) - torch.max(preds_x1, targets_x1)).clip(min=0)
        intersection_h = (torch.min(preds_y2, targets_y2) - torch.max(preds_y1, targets_y1)).clip(min=0)
        intersection = intersection_w * intersection_h
        union = (preds_w * preds_h) + (targets_w * targets_h) - intersection
        iou = intersection / (union + e)

        cw = (torch.max(preds_x2, targets_x2) - torch.min(preds_x1, targets_x1))
        ch = (torch.max(preds_y2, targets_y2) - torch.min(preds_y1, targets_y1))
        c2 = cw.pow(2) + ch.pow(2) + e
        v = (4 / (torch.pi**2)) * (torch.arctan(targets_w / targets_h) - torch.arctan(preds_w / preds_h)).pow(2)
        rho2 = (preds_xywh[..., :1] - targets_xywh[..., :1]).pow(2) + (preds_xywh[..., 1:2] - targets_xywh[..., 1:2]).pow(2)
        with torch.no_grad():
            a = v / (v - iou + (1 + e))
        ciou = iou - ((rho2/c2) + (a * v))
        return ciou.squeeze(-1)
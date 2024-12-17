import torch
import warnings
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from dataset.detection_dataset import DetectionDataset
warnings.filterwarnings(action="ignore")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Dict, List, Union
from .detection_loss import DetectionLoss, FocalLoss
from utils.utils import crop_section, compute_dice_score


class SegmentationLoss(DetectionLoss):
    def __init__(self, *args, seg_w: float=1.0, overlap_masks: bool=True, **kwargs):
        super(SegmentationLoss, self).__init__(*args, **kwargs)
        self.seg_w = seg_w
        self.overlap_masks = overlap_masks

        if self.alpha and self.gamma:
            self.seg_lossfn = FocalLoss(alpha=self.alpha, gamma=self.gamma, with_logits=True, reduction="none")
        else:
            self.seg_lossfn = nn.BCEWithLogitsLoss(reduction="none")
     

    def forward(
        self, 
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        targets: torch.Tensor, 
        protos: torch.Tensor, 
        target_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:        
        metrics_dict = {}
        sm_preds, md_preds, lg_preds = preds
        sm_anchors = self.model.sm_anchors.data.clone().detach()
        md_anchors = self.model.md_anchors.data.clone().detach()
        lg_anchors = self.model.lg_anchors.data.clone().detach()
        sm_losses, sm_metrics_dict = self.loss_fn(
            sm_preds, targets, protos, target_masks, anchors=sm_anchors
        )
        md_losses, md_metrics_dict = self.loss_fn(
            md_preds, targets, protos, target_masks, anchors=md_anchors
        )
        lg_losses, lg_metrics_dict = self.loss_fn(
            lg_preds, targets, protos, target_masks, anchors=lg_anchors
        )
    
        if len(sm_losses) == 4:
            (sm_lbox, sm_lconf, sm_lcls, sm_lseg) = sm_losses
            (md_lbox, md_lconf, md_lcls, md_lseg) = md_losses
            (lg_lbox, lg_lconf, lg_lcls, lg_lseg) = lg_losses
        else:
            (sm_lbox, sm_lconf, sm_lcls, sm_lseg, sm_lkp) = sm_losses
            (md_lbox, md_lconf, md_lcls, md_lseg, md_lkp) = md_losses
            (lg_lbox, lg_lconf, lg_lcls, lg_lseg, lg_lkp) = lg_losses

        lbox = (self.scale_w[0] * sm_lbox) + (self.scale_w[1] * md_lbox) + (self.scale_w[2] * lg_lbox)
        lconf = (self.scale_w[0] * sm_lconf) + (self.scale_w[1] * md_lconf) + (self.scale_w[2] * lg_lconf)
        lcls = (self.scale_w[0] * sm_lcls) + (self.scale_w[1] * md_lcls) + (self.scale_w[2] * lg_lcls)
        lseg = (self.scale_w[0] * sm_lseg) + (self.scale_w[1] * md_lseg) + (self.scale_w[2] * lg_lseg)
        loss = (self.box_w * lbox) + (self.conf_w * lconf) + (self.class_w * lcls) + (self.seg_w * lseg)

        if len(sm_losses) > 4:
            lkp = (self.scale_w[0] * sm_lkp) + (self.scale_w[1] * md_lkp) + (self.scale_w[2] * lg_lkp)
            loss = loss + (self.keypoints_w * lkp)

        loss = loss * (preds[-1].shape[0] if self.batch_scale_loss else 1.0)
        metrics_df = pd.DataFrame([sm_metrics_dict, md_metrics_dict, lg_metrics_dict])
        
        metrics_dict["aggregate_loss"] = loss.item()
        for key in metrics_df.columns:
            metrics_dict[key] = metrics_df[key].mean()
        return loss, metrics_dict


    def loss_fn(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor, 
            protos: torch.Tensor,
            target_masks: torch.Tensor,
            anchors: Union[List[float], torch.Tensor],
        ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        
        _device = preds.device
        indices, t_classes, anchors, t_xywh, tmask_idx, t_keypoints = DetectionDataset.build_target_by_scale(
            targets, 
            preds.shape[1:3],
            anchors,
            anchor_threshold=self.anchor_t, 
            edge_threshold=self.edge_t,
            overlap_masks=self.overlap_masks,
            batch_size=preds.shape[0]
        )
        batch_idx, grid_j, grid_i, anchor_idx = indices
        match_preds = preds[batch_idx, grid_j, grid_i, anchor_idx]
        p_cls_proba = match_preds[:, 1:1+self.model.num_classes]
        p_xywh = match_preds[:, 1+self.model.num_classes:5+self.model.num_classes]
        k_i = 5+self.model.num_classes # begining slice index of mask_coefs
        k_j = k_i + self.model.proto_seg_module.out_channels # end slice index of mask_coefs
        pmask_coefs = match_preds[:, k_i:k_j]
        p_keypoints = match_preds[:, k_j:]
        pxy, pwh = p_xywh.chunk(2, dim=-1)
        p_xywh = torch.cat([pxy, pwh*anchors], dim=-1)

        kpv_loss = None
        kpc_loss = None
        kp_loss =  None
        # t_keypoints: [x, y, v(visibility_class_index)]
        # p_keypoints: [x, y, p_visible, p_occluded, p_deleted]
        if torch.is_tensor(t_keypoints):
            num_keypoints = self.model.num_keypoints
            assert p_keypoints.shape[1] == num_keypoints*5 and t_keypoints.shape[1] == num_keypoints*3

            # keypoints loss
            if t_keypoints.shape[1] != 0:
                p_keypoints = p_keypoints.reshape(*p_keypoints.shape[:-1], -1, 5)
                t_keypoints = t_keypoints.reshape(*t_keypoints.shape[:-1], -1, 3)

                # kpv = keypoint visibility
                # kpc = keypoint coordinates
                kpv_loss = self.kpv_lossfn(
                    p_keypoints[..., 2:].flatten(start_dim=0, end_dim=-2), 
                    t_keypoints[..., 2].flatten(start_dim=0, end_dim=-1).to(dtype=torch.int64, device=_device)
                )
                kpc_loss = nn.functional.mse_loss(p_keypoints[..., :2], t_keypoints[..., :2], reduction="none")
                # remove invalid keypoints (keypoints with infinity or nan loss)
                # samples with unequal number of keypoints or no keypoints at all 
                # are padded with torch.inf or -torch.inf, so their losses will 
                # either be NaN or (+/-)infinity
                kpc_loss = kpc_loss[~(torch.isnan(kpc_loss) | torch.isinf(kpc_loss))].mean()
                kp_loss = (1 + kpv_loss) * kpc_loss

        # bbox loss
        ciou = SegmentationLoss.compute_ciou(p_xywh, t_xywh)
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

        # segmentation loss
        if tuple(target_masks.shape[1:]) != tuple(protos.shape[2:]):
            target_masks = F.interpolate(target_masks.unsqueeze(dim=0), size=protos.shape[2:], mode="nearest")[0]
        seg_loss = torch.tensor(0.0, device=protos.device)
        dice_score = torch.tensor(0.0, device=protos.device)
        for i in batch_idx.unique():
            # read the comments on the build_target_by_scale static method in the DetectionDataset class in datection
            # dataset.py for information on why this code is the way it is, I tried my best there (Good luck)
            m = (batch_idx == i)
            if self.overlap_masks:
                tmask = torch.where(target_masks[i].unsqueeze(dim=0) == tmask_idx[m].reshape(-1, 1, 1), 1.0, 0.0)
            else:
                tmask = target_masks[tmask_idx][m].to(dtype=protos.dtype, device=protos.device)
            sl, ds = self.segmentation_metrics(pmask_coefs[m], protos[i], tmask, t_xywh[m])
            seg_loss += sl
            dice_score += ds
        seg_loss /= preds.shape[0]
        dice_score /= preds.shape[0]

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
        losses = [handle_nan(ciou_loss), conf_loss, handle_nan(class_loss), seg_loss]
        metrics_dict = {}
        metrics_dict["mean_ciou"] = ciou.mean().item()
        metrics_dict["conf_loss"] = conf_loss.item()
        metrics_dict["seg_loss"] = seg_loss.item()
        metrics_dict["dice_score"] = dice_score.item()
        metrics_dict["avg_pos_conf"] = avg_pos_conf.item()
        metrics_dict["avg_neg_conf"] = avg_neg_conf.item()
        metrics_dict["class_loss"] = class_loss.item()
        metrics_dict["accuracy"] = accuracy
        metrics_dict["f1"] = f1
        metrics_dict["precision"] = precision
        metrics_dict["recall"] = recall
        if torch.is_tensor(kp_loss):
            losses.append(handle_nan(kp_loss))
            metrics_dict["kpv_loss"] = kpv_loss.item()
            metrics_dict["kpc_loss"] = kpc_loss.item()
            metrics_dict["kp_loss"] = kp_loss.item()
        return losses, metrics_dict


    def segmentation_metrics(
            self, 
            pmask_coefs: torch.Tensor, 
            protos: torch.Tensor, 
            target_mask: torch.Tensor, 
            t_xywh: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pmask_coefs: (n, k)
        # protos: (k, H, W) 
        # target_mask: (n, H, W)
        # t_xywh: (n, 4)
        mask_area = t_xywh[:, 2:].prod(dim=-1)
        pred_mask = (pmask_coefs @ protos.reshape(protos.shape[0], -1)).reshape(-1, *protos.shape[1:])
        sigmoid_pred_mask = torch.sigmoid(pred_mask)
        losses = self.seg_lossfn(pred_mask, target_mask)
        dice_loss = 1.0 - compute_dice_score(sigmoid_pred_mask, target_mask, round_tensor=False)
        dice_score = compute_dice_score(sigmoid_pred_mask, target_mask, round_tensor=True).detach()
        losses = crop_section(losses, t_xywh).mean(dim=(1, 2)) / mask_area
        losses = (1 - losses) * dice_loss
        return losses.mean(), dice_score
import os
import tqdm
import glob
import json
import logging
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import *
from utils.utils import load_bbox_labels, load_and_process_img, xywh2x1y1x2y2

logger = logging.getLogger(__name__)

class DetectionDataset(Dataset):
    def __init__(self, data_dir: str, img_ext: str="png", img_wh: Union[int, Tuple[int, int]]=(640,  640)):
        if isinstance(img_wh, int):
            img_wh = (img_wh, img_wh)
        self.img_wh = img_wh

        self.img_files = glob.glob(os.path.join(data_dir, "**", f"*.{img_ext}"), recursive=True)
        self.annotation_files = glob.glob(os.path.join(data_dir, "**", f"*.txt"), recursive=True)
        self.img_files = sorted(self.img_files)
        self.annotation_files = sorted(self.annotation_files)
        if len(self.img_files) == 0:
            raise Exception(f"{data_dir} does not contain any .{img_ext} files in its base and sub directories")
        if len(self.annotation_files) == 0:
            raise Exception(f"{data_dir} does not contain any .txt files in its base and sub directories")
        assert len(self.img_files) == len(self.annotation_files)
        logger.info(f"Number of image samples: {self.__len__()}")

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_file = self.img_files[idx]
        annotation_file = self.annotation_files[idx]
        # bbox_only set to False, incase keypoint annotations exists
        raw_labels = load_bbox_labels(annotation_file, bbox_only=False)

        if raw_labels.shape[1] > 5:
            # scale keypoint coordinates to be relative to corresponding bbox dimensions rather image dimensions
            bbox = raw_labels[:, :5]
            keypoints = raw_labels[:, 5:].reshape(raw_labels.shape[0], -1, 3)
            bbox_xyxy = xywh2x1y1x2y2(bbox[:, 1:])
            keypoints[..., :2] = (
                (keypoints[..., :2] - bbox_xyxy[:, None, :2]) / (bbox_xyxy[:, None, 2:] - bbox_xyxy[:, None, :2])
            ).clip(min=0.0, max=1.0)
            keypoints = keypoints.reshape(keypoints.shape[0], -1)
            raw_labels = np.concatenate([bbox, keypoints], axis=1)
            raw_labels = np.ascontiguousarray(raw_labels)

        img = load_and_process_img(img_file, img_wh=self.img_wh[::-1], permute=True, scale=True, convert_to="RGB")
        labels = torch.zeros((raw_labels.shape[0], raw_labels.shape[1]+1), dtype=torch.float32)
        if labels.shape[0] > 0:
            labels[:, 1:] = torch.from_numpy(raw_labels).to(dtype=torch.float32)
        return img, labels
    
    def get_class_weights(self, device: Optional[str]=None) -> torch.Tensor:
        if device is None: device = "cpu"
        classes = []
        logger.info("getting class weights...")
        for annotation_file in tqdm.tqdm(self.annotation_files):
            boxes = load_bbox_labels(annotation_file)
            classes.extend(boxes[:, 0].tolist())
        classes = sorted(classes)
        class_counts = np.bincount(classes)
        class_counts = torch.from_numpy(class_counts).to(dtype=torch.float32, device=device)
        label_weights = class_counts.sum() / (class_counts.shape[0] * class_counts)
        return label_weights

    @staticmethod
    def save_label_map(class2idx_map: Dict[str, int], _dir: str):
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
        idx2class_map = {v:k for k, v in class2idx_map.items()}
        with open(os.path.join(_dir, "class_map.json"), "w") as f:
            json.dump(idx2class_map, f)
        f.close()

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        img, labels = zip(*batch)
        for i, label in enumerate(labels):
            label[:, 0] = i
        img = torch.stack(img, dim=0)
        labels = torch.cat(labels, dim=0)
        return img, labels
    
    @staticmethod
    def build_target_by_scale(
            targets: torch.Tensor, 
            fmap_shape: Union[int, List[int], torch.Size, torch.Tensor],
            anchors: Union[List[List[float]], torch.Tensor],
            anchor_threshold: float=4.0,
            edge_threshold: float=0.5,
            overlap_masks: Optional[bool]=None,
            batch_size: Optional[int]=None,
        ) -> Tuple[
            List[torch.Tensor], 
            torch.Tensor, 
            torch.Tensor, 
            torch.Tensor, 
            Optional[torch.Tensor], 
            Optional[torch.Tensor]
        ]:
        # NOTE: I apologize if the comments here are excessive, I basically reimplemented the target builder of the
        # official YOLOv5 implementation with extra stuff (like taking keypoints and segment mask indexes into account). 
        # I made these comments to help me follow through after I had understood what was going on and also incase anyone
        # view this and wonder whatever the hell is actually going on here.

        # targets contains (batch_idx, cls, x, y, w, h) or (batch_idx, cls, x, y, w, h, x1, y1, v1, x2, y2, v2,...)
        # where x1, y1, v1, x2, y2, v2,... are keypoint coordinates and visibility (visible, occluded or deleted)
        # targets shape: (num_boxes across the batch, 6) or (num_boxes across the batch, 6 + (3*num_keypoints))
        _device = targets.device
        _dtype = targets.dtype
        if not isinstance(fmap_shape, torch.Tensor):
            # fmap_shape: [h,  w]
            fmap_shape = torch.tensor(fmap_shape, device=_device)

        if isinstance(anchors, list):
            anchors = torch.tensor([anchors], dtype=_dtype, device=_device)

        num_anchors = anchors.shape[0]
        num_targets = targets.shape[0]
        anchor_idx = torch.arange(num_anchors, device=_device).unsqueeze(dim=-1).tile(1, num_targets)
        _t = targets.unsqueeze(dim=0).tile(num_anchors, 1, 1)
        if overlap_masks is None:
            # normalized to gridspace gain
            gain = torch.ones(7, device=_device)
            targets = torch.cat([_t[..., :6], anchor_idx[..., None], _t[..., 6:]], dim=-1)
        else:
            gain = torch.ones(8, device=_device)
            # NOTE: The collate_fn discussed in this else block is implemented in the SegmentationDataset class
            # (in segmentation_dataset.py) file
            if overlap_masks:
                # overlap implies that masks (m, H, W) for a given image sample has been compressed to (1, H, W) with 
                # the objects of smaller area having more pixel intensity than objects of larger area
                # (see polygons_2_overlapped_mask function in utils/utils.py line). Suppose that on average each sample
                # has 4 polygons (practically, the number of polygons per sample will vary), we can extract 4 bboxes 
                # (of shape (4, 6), 6 coz: [batch_idx, cls, x, y, w, h]) from these polygons. Typically we also expect the
                # mask to to be of shape (4, H, W), however, since overlap_masks=True, the mask will be of shape (1, H, W).
                # Suppose we have a batch size of 10, the collate_fn is designed in such a way that the bboxes will be
                # concatinated along the first axis, making it a tensor with shape (40, 6), since the mask tensors are of
                # shape (1, H, W) per image sample, concatinating along the first dimension will make the batch of masks
                # have a shape (10, H, W).
                # In the for-loop below, we determine the number of objects per mask in the batch of masks since the target
                # bbox tensor and the target mask tensor do not have a one-to-one correspondence along their first axis after 
                # the collate_fn.
                tmask_idx = []
                if not batch_size:
                    raise ValueError("batch_size is required when overlap_mask is set to True")
                for i in range(0, batch_size):
                    num_t = (targets[:, 0] == i).sum().item()
                    tmask_idx.append(torch.arange(num_t, device=_device).unsqueeze(dim=0).tile(num_anchors, 1) + 1)
                tmask_idx = torch.cat(tmask_idx, dim=-1)
            else:
                # Due to the fact that in the collate function, masks (of shape (n, H, W)) are concatinated along
                # the first axis, if there are no mask overlaps, the number of target masks will be n and equal to
                # the size of the first dimension of the target bbox tensor after concatination in the collate_fn, 
                # as such the tmask_idx will be actually just be the same as the index of each bbox in the targets
                # tensor (in order) prior to the final preprocessing.
                # Following the previous example, if a given image sample has 4 bboxes (shape: (4, 6)), then the
                # shape of the mask will be (4, H, W) because overlap_mask=False. Hence in the collate_fn, where
                # the bbox targets and the masks are of shame size along their first dimensions, there is a 
                # one-to-one correspondence, hence no need for any fancy loops to extract the number of target bboxes
                # per mask, because we know that for 4 boxes, there are 4 masks.
                tmask_idx = torch.arange(targets.shape[0], device=_device).unsqueeze(dim=0).tile(num_anchors, 1)

            targets = torch.cat(
                [_t[..., :6], anchor_idx[..., None], tmask_idx[..., None], _t[..., 6:]], dim=-1
            )
        
        # pad gain tensor if need be (if keypoint annotations are available anyways)
        gain_pad_size = _t[..., 6:].shape[-1]
        if gain_pad_size > 0:
            gain = torch.cat([gain, torch.ones(gain_pad_size, device=_device)], dim=0)

        gain[2:6] = fmap_shape[[1, 0, 1, 0]]
        # anchors are originally normalized relative to image size, so they range from 0 to 1
        # here we scale these anchors to the dimensions of the feature map to compare with the
        # targets that have been normalized to grid space
        anchors = anchors * fmap_shape[[1, 0]]
        targets = targets * gain
        
        if num_targets > 0:
            # compute the ratio between the boxes and the anchors, boxes that are
            # (anchor_threshold times) more or (anchor_threshold times) less than the corresponding achors
            # are filtered, while the rest are discarded
            r = targets[..., 4:6] / anchors[:, None]
            targets = targets[torch.max(r, 1/r).max(dim=2)[0] < anchor_threshold]

            grid_xy = targets[:, 2:4]

            grid_xy_i = gain[[2, 3]] - grid_xy

            # next we look for boxes, whose corresponding centers are close to the edge of 
            # its grid cell (either leftwards or rightwards) while simultanously not being at
            # the edge of the image / spectrogram (in the first grid or in the last grid)
            j_mask, k_mask = ((grid_xy % 1 < edge_threshold) & (grid_xy > 1)).T
            l_mask, m_mask = ((grid_xy_i % 1 < edge_threshold) & (grid_xy_i > 1)).T

            # we formulate a mask that includes all the selected targets as well as all the targets
            # that satisfy the conditions of the c_mask and i_mask (Note that there will mostly always
            # be repititions of boxes in the final targets)
            mask = torch.stack([torch.ones_like(j_mask), j_mask, k_mask, l_mask, m_mask], dim=0)
            targets = targets.repeat(mask.shape[0], 1, 1)[mask]

            # based on the masks generated above, we formulate offsets. The idea here is that for a certain
            # box whose center falls close to the left edge of a grid cell, there is also a chance that 
            # the preceding grid cell can be capable of predicting that box. Similarly, if the center
            # falls close to the right edge of a grid cell, there is also a good chance that the superseding
            # grid cell can also be capable of predicting that box. Hence the offsets are made such that
            # when such boxes are found, we match them against the predictions at its grid cell, as well as
            # the preceding and superseding grid cells if applicable.
            # The offsets are defined such that 0 corresponds to the offset of the original grids, indicating no changes
            # to them, -1 indicates a leftward shift and downwards shift to the preceding grid cell if box's center 
            # is close to the left edge or bottom edge of its grid, and 1 indicates a rightward or upward shift to next 
            # grid cell if box's center is close to the right or top edge of its grid.
            offset = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=_device).float() * edge_threshold
            offset = (torch.zeros_like(grid_xy)[None] + offset[:, None])[mask]
        else:
            targets = targets[0]
            offset = 0
        kp_i = 7 # starting index of keypoint coordinates
        batch_idx = targets[:, 0].long()
        classes = targets[:, 1].long()
        grid_xy = targets[:, 2:4]
        grid_wh = targets[:, 4:6]
        anchor_idx = targets[:, 6].long()
        grid_ij = (grid_xy - offset).long()
        grid_i, grid_j = grid_ij.T
        grid_j, grid_i = grid_j.clamp_(0, fmap_shape[0] - 1), grid_i.clamp_(0, fmap_shape[1] - 1)
        
        anchors = anchors[anchor_idx]
        indices = [batch_idx, grid_j, grid_i, anchor_idx]
        boxes = torch.cat((grid_xy - grid_ij, grid_wh), dim=1)
        tmask_idx = None
        if overlap_masks is not None:
            kp_i += 1
            tmask_idx = targets[:, 7].long()
        
        keypoints = targets[:, kp_i:]
        if keypoints.shape[1] == 0:
            keypoints = None
        return indices, classes, anchors, boxes, tmask_idx, keypoints
import tqdm
import logging
import torch
import numpy as np
from PIL import Image
from typing import *
from .detection_dataset import DetectionDataset
from utils.utils import (
    load_and_process_img,
    load_polygon_labels, 
    interpolate_polygons, 
    polygons_2_xywh, 
    polygons_2_masks,
    polygons_2_overlapped_mask, 
)

logger = logging.getLogger(__name__)

class SegmentationDataset(DetectionDataset):
    def __init__(self, *args, overlap_masks: bool=True, mask_scale_factor: float=1.0, **kwargs):
        super(SegmentationDataset, self).__init__(*args, **kwargs)
        self.overlap_masks = overlap_masks
        self.mask_scale_factor = mask_scale_factor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_file = self.img_files[idx]
        annotation_file = self.annotation_files[idx]
        labels = load_polygon_labels(annotation_file)
        classes = np.asarray([p[0] for p in labels])
        polygons = interpolate_polygons([p[1:] for p in labels])
        bboxes = np.asarray(polygons_2_xywh(polygons))
        img = load_and_process_img(img_file, img_wh=self.img_wh[::-1], permute=True, scale=True, convert_to="RGB")
        labels = np.zeros((len(polygons), 6), dtype=np.float32)
        if len(polygons) > 0:
            labels[:, 1] = classes
            labels[:, 2:] = bboxes
            if not self.overlap_masks:
                masks = polygons_2_masks(
                    polygons, img.shape[2], img.shape[1], scale_factor=self.mask_scale_factor
                )
            else:
                masks, sorted_indices = polygons_2_overlapped_mask(
                    polygons, img.shape[2], img.shape[1], scale_factor=self.mask_scale_factor
                )
                labels = labels[sorted_indices]
        else:
            mask_shape = np.asarray(tuple(img.shape[1:])) * self.mask_scale_factor
            mask_shape = mask_shape.round().astype(int)
            masks = np.zeros(((1 if self.overlap_masks else 0), *mask_shape), dtype=np.uint8)
        labels = torch.from_numpy(labels)
        masks = torch.from_numpy(masks)
        return img, labels, masks
    
    def get_class_weights(self, device: Optional[str]=None) -> torch.Tensor:
        if device is None: device = "cpu"
        classes = []
        logger.info("getting class weights...")
        for annotation_file in tqdm.tqdm(self.annotation_files):
            polygons = load_polygon_labels(annotation_file)
            classes.extend([p[0] for p in polygons])
        classes = sorted(classes)
        class_counts = np.bincount(classes)
        class_counts = torch.from_numpy(class_counts).to(dtype=torch.float32, device=device)
        label_weights = class_counts.sum() / (class_counts.shape[0] * class_counts)
        return label_weights

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, labels, masks = zip(*batch)
        for i, label in enumerate(labels):
            label[:, 0] = i
        img = torch.stack(img, dim=0)
        labels = torch.cat(labels, dim=0)
        masks = torch.cat(masks, dim=0)
        return img, labels, masks
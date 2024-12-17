import os
import glob
import tqdm
import cv2
import yaml
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from PIL import Image
from supervision import Detections
from typing import *

def load_yaml(config_path: str) -> Dict[str, Dict[str, List[List[float]]]]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    f.close()
    return config


def load_model(model: nn.Module, path: str, device: str="cpu", eval: bool=True):
    model.load_state_dict(torch.load(path, map_location=device)["NETWORK_PARAMS"])
    model.eval() if eval else model.train()

def load_and_process_img(
        img_path: str, 
        img_wh: Optional[Tuple[int, int]]=None, 
        permute: bool=True, 
        scale: bool=True,
        convert_to: str="RGB"
    ) -> torch.Tensor:
    img = Image.open(img_path).convert(convert_to)
    if img_wh is not None:
        img = img.resize(img_wh)
    img = np.asarray(img).copy()
    img = torch.from_numpy(img)
    if permute:
        img = img.permute(2, 0, 1)
    if scale:
        img = (img / 255).to(dtype=torch.float32)
    return img


def load_bbox_labels(annotation_file: str, bbox_only: bool=True) -> np.ndarray:
    with open(annotation_file, "r") as f:
        text = f.read()
        lines = text.split("\n")
        boxes = np.asarray([line.split() for line in lines if len(line.split()) > 0]).astype(np.float32)
        if bbox_only:
            boxes = boxes[:, :5]
    f.close()
    return boxes


def load_polygon_labels(annotation_file: str) -> List[np.ndarray]:
    with open(annotation_file, "r") as f:
        text = f.read()
        lines = text.split("\n")
        polygons = [np.asarray(line.split()).astype(np.float32) for line in lines if len(line.split()) > 5]
    f.close()
    return polygons


def interpolate_polygons(polygons: List[np.ndarray], n: int=500) -> List[np.ndarray]:
    for i, polygon in enumerate(polygons):
        if polygon.ndim == 1:
            assert polygon.shape[0] % 2 == 0
            # if the polygon points is a flattened ndarray, reshape it to a 2d ndarray of points (x, y)
            polygon = np.stack([polygon[slice(0, None, 2)], polygon[slice(1, None, 2)]], axis=1)

        if not np.all(polygon[0] == polygon[-1]):
            # if first point is not equal to last point, make it so
            polygon = np.concatenate([polygon, polygon[:1, :]], axis=0)

        x = np.linspace(0, polygon.shape[0]-1, num=n)
        xp = np.arange(polygon.shape[0])
        polygons[i] = np.stack(
            [np.interp(x, xp, polygon[:, i]) for i in range(0, polygon.shape[1])],
            axis=1
        )
    return polygons


def polygons_2_xywh(polygons: List[np.ndarray]) -> List[np.ndarray]:
    bboxes = []
    for polygon in polygons:
        assert polygon.ndim == 2
        x1, y1, x2, y2 = polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()
        w, h = x2 - x1, y2 - y1
        x, y = x1 + (w/2), y1 + (h/2)
        bboxes.append(np.asarray([x, y, w, h]))
    return bboxes


def polygons_2_masks(
        polygons: List[np.ndarray], 
        img_width: int,
        img_height: int, 
        scale_factor: float=1.0,
        color: int=1,
    ) -> np.ndarray:
    masks = []
    for polygon in polygons:
        assert polygon.ndim == 2
        mask = np.zeros((round(img_height * scale_factor), round(img_width * scale_factor)), dtype=np.uint8)
        polygon = polygon * np.asarray([img_width, img_height])
        polygon = polygon.astype(int)
        masks.append(cv2.fillPoly(mask, pts=polygon[None], color=color))
    masks = np.stack(masks, axis=0)
    return masks


def overlap_masks(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert masks.ndim == 3
    areas = masks.sum((1, 2))
    sorted_indices = np.argsort(-areas)
    final_mask = np.zeros(masks.shape[1:], dtype=(np.uint8 if masks.shape[0] <= 255 else np.uint32))
    for i, sorted_idx in enumerate(sorted_indices):
        final_mask += (masks[sorted_idx] * (i + 1)).astype(final_mask.dtype)
        final_mask = np.clip(final_mask, a_min=0, a_max=i+1)
    final_mask = final_mask.reshape(-1, *final_mask.shape)
    return final_mask, sorted_indices


def polygons_2_overlapped_mask(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    masks = polygons_2_masks(*args, **kwargs)
    return overlap_masks(masks)


def crop_section(
        image: Union[np.ndarray, torch.Tensor], 
        bboxes: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
    # image: 
    # bboxes: (n, 4)
    is_numpy_img = isinstance(image, np.ndarray)
    if is_numpy_img:
        image = torch.from_numpy(image)
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes)
    _, h, w = image.shape
    bboxes = torch.cat([bboxes[:, :2]-(bboxes[:, 2:]/2), bboxes[:, :2]+(bboxes[:, 2:]/2)], dim=-1)
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)
    r = torch.arange(w, device=image.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=image.device, dtype=x1.dtype)[None, :, None]
    section = image * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    if is_numpy_img:
        section = section.numpy()
    return section


def compute_dice_score(
        mask1: torch.Tensor, 
        mask2: torch.Tensor, 
        round_tensor: bool=False, 
        e: float=1e-5
    ) -> torch.Tensor:
    #mask1 shape: N, C, H, W or (N, H, W)
    #mask2 shape: N, C, H, W or (N, H, W)
    assert tuple(mask1.shape) == tuple(mask2.shape) and mask1.ndim in [3, 4]
    if mask1.ndim == 3:
        mask1 = mask1.unsqueeze(dim=1)
        mask2 = mask2.unsqueeze(dim=1)
    mask1 = mask1.clip(0.0, 1.0)
    mask2 = mask2.clip(0.0, 1.0)
    if round_tensor:
        mask1 = mask1.round()
        mask2 = mask2.round()
    intersection = torch.abs(mask1 * mask2).sum(dim=(2, 3))
    denominator = mask1.sum(dim=(2, 3)) + mask2.sum(dim=(2, 3))
    dice_coeff = ((2 * intersection + e) / (denominator + e)).mean(dim=(0, 1))
    return dice_coeff


def get_class_weights(classes) -> np.ndarray:
    classes = sorted(classes)
    class_counts = np.bincount(classes)
    label_weights = class_counts.sum() / (class_counts.shape[0] * class_counts)
    return label_weights


def get_box_sizes_and_class_weights(path: str) -> Tuple[np.ndarray, np.ndarray]:
    annotation_files = glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)
    box_sizes = []
    classes = []
    for file in tqdm.tqdm(annotation_files):
        bbox = load_bbox_labels(file)
        if len(bbox) == 0: 
            continue
        classes.append(bbox[:, 0])
        box_sizes.append(bbox[:, -2:])
    box_sizes = np.concatenate(box_sizes, axis=0)
    classes = np.concatenate(classes, axis=0)
    return box_sizes, get_class_weights(classes)


def get_box_sizes_and_class_weights_from_polygons(path: str) -> Tuple[np.ndarray, np.ndarray]:
    annotation_files = glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)
    box_sizes = []
    classes = []
    for file in tqdm.tqdm(annotation_files):
        polygons = load_polygon_labels(file)
        if len(polygons) == 0:
            continue
        classes.append([p[0] for p in polygons])
        polygons = [p[1:] for p in polygons]
        polygons = interpolate_polygons(polygons)
        bboxes = np.asarray(polygons_2_xywh(polygons))
        box_sizes.append(bboxes[:, -2:])
    box_sizes = np.concatenate(box_sizes, axis=0)
    classes = np.concatenate(classes, axis=0)
    return box_sizes, get_class_weights(classes)


def xywh2x1y1x2y2(bboxes: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    # convert xywh -> xyxy
    # this implementation could be done inplace, and made easier, but I need this to
    # always return a new tensor at a new memory address
    assert isinstance(bboxes, (np.ndarray, torch.Tensor))
    x1y1 = bboxes[..., :2] - (bboxes[..., 2:] / 2)
    x2y2 = x1y1 + bboxes[..., 2:]
    if isinstance(bboxes, torch.Tensor):
        bboxes = torch.cat([x1y1, x2y2], dim=-1)
    else:
        bboxes = np.concatenate([x1y1, x2y2], axis=-1)
    return bboxes

def x1y1x2y22xywh(bboxes: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    # convert xyxy -> xywh
    # this implementation could be done inplace, and made easier, but I need this to
    # always return a new tensor at a new memory address
    assert isinstance(bboxes, (np.ndarray, torch.Tensor))
    wh = bboxes[..., 2:] - bboxes[..., :2]
    xy = bboxes[..., :2] + (wh / 2)
    if isinstance(bboxes, torch.Tensor):
        bboxes = torch.cat([xy, wh], dim=-1)
    else:
        bboxes = np.concatenate([xy, wh], axis=-1)
    return bboxes

def apply_segments(
        img: np.ndarray, 
        masks: np.ndarray, 
        alpha: float=0.5, 
        colormap: Optional[np.ndarray]=None
    ) -> np.ndarray:
    # img shape: (C, H, W)
    # masks: (1 or m, H, W)
    assert img.ndim == 3 and masks.ndim == 3
    assert img.dtype in [np.uint8, np.float32, float]
    if img.shape[0] == 3:
        img = np.ascontiguousarray(img.transpose(1, 2, 0), dtype=img.dtype)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    masks = masks.astype(np.uint8)
    colored_masks = np.zeros_like(img)
    if masks.shape[0] > 1:
        masks, _ = overlap_masks(masks)
    masks = masks.squeeze(axis=0)
    if colormap is None:
        num_objects = masks.max() + 1
        colormap = np.random.randint(0, 255, size=(num_objects, 3))
    for obj_id in range(0, colormap.shape[0]):
        colored_masks[masks == obj_id+1] = colormap[obj_id]
    final_img = cv2.addWeighted(src1=img, alpha=alpha, src2=colored_masks, beta=1-alpha, gamma=0)
    return final_img


def apply_bboxes(
        img: np.ndarray, 
        bboxes: np.ndarray, 
        box_thickness: int=2,
        text_thickness: int=2, 
        font: int=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float=0.4,
        colormap: Optional[np.ndarray]=None,
        classmap: Optional[List[Dict[str, Any]]]=None
    ) -> np.ndarray:
    # img shape: (C, H, W)
    # bboxes: (n, 6): format -> (score, class_idx, x1, y1, x2, y2)
    assert img.ndim == 3 and bboxes.ndim == 2 and bboxes.shape[1] == 6
    assert img.dtype in [np.uint8, np.float32, float]
    if img.shape[0] == 3:
        img = np.ascontiguousarray(img.transpose(1, 2, 0), dtype=img.dtype)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    if colormap is None:
        colormap = np.random.randint(0, 255, size=(int(bboxes[:, 1].max()), 3))
    for box in bboxes:
        score, class_idx, x1, y1, x2, y2 = box
        class_idx = int(class_idx)
        x1, y1, x2, y2 = list(map(lambda x : round(x), [x1, y1, x2, y2]))
        color = tuple(colormap[int(class_idx)].tolist())
        img = cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=box_thickness)
        class_ = classmap[class_idx]["name"] if classmap else class_idx
        text = f"({class_} {score :.2f})"
        text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        img = cv2.rectangle(img, (x1, y1-text_size[1]-4), (x1+text_size[0]+2, y1), color, cv2.FILLED)
        img = cv2.putText(
            img,
            text=text, 
            org=(x1, y1-2), 
            color=(0, 0, 0), 
            fontFace=font,
            fontScale=font_scale, 
            thickness=text_thickness
        )
    return img

def apply_keypoints(img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    assert img.ndim == 3
    assert img.dtype in [np.uint8, np.float32, float]
    if img.shape[0] == 3:
        img = np.ascontiguousarray(img.transpose(1, 2, 0), dtype=img.dtype)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    keypoints = keypoints.astype(int)

    for i in range(keypoints.shape[0]):
        if keypoints[i][2] == 0:
            # visible keypoint
            color = (255, 255, 255)
        elif keypoints[i][2] == 1:
            # occluded keypoint 
            color = (255, 255, 100)
        else:
            continue
        img = cv2.circle(img, keypoints[i][:2], 3, color=color, thickness=-1)
    return img

def apply_bboxes_from_tracks(
        img: np.ndarray, 
        detections: Detections, 
        box_thickness: int=2,
        text_thickness: int=2, 
        font: int=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float=0.4,
        colormap: Optional[np.ndarray]=None,
        classmap: Optional[List[Dict[str, Any]]]=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # img shape: (C, H, W)
    assert img.ndim == 3 and img.dtype in [np.uint8, np.float32, float]
    if img.shape[0] == 3:
        img = np.ascontiguousarray(img.transpose(1, 2, 0), dtype=img.dtype)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # if colormap is None: TODO
    #   colormap = np.random.randint(0, 255, size=(int(bboxes[:, 1].max()), 3))
    boxes = []
    for i in range(0, len(detections)):
        track_id = detections.tracker_id[i]
        class_idx = int(detections.class_id[i])
        score = detections.confidence[i]
        if score is None:
            continue
        x1, y1, x2, y2 = detections.xyxy[i]
        boxes.append([int(track_id), score, class_idx, x1, y1, x2, y2])
        x1, y1, x2, y2 = list(map(lambda x : round(x), [x1, y1, x2, y2]))
        color = tuple(colormap[int(class_idx)].tolist())
        img = cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=box_thickness)
        class_ = classmap[class_idx]["name"] if classmap else class_idx

        text = f"id:{track_id} ({class_} {score :.2f})"
        text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        img = cv2.rectangle(img, (x1, y1-text_size[1]-4), (x1+text_size[0]+2, y1), color, cv2.FILLED)
        img = cv2.putText(
            img,
            text=text, 
            org=(x1, y1-2), 
            color=(0, 0, 0), 
            fontFace=font,
            fontScale=font_scale, 
            thickness=text_thickness
        )
    return img, np.asarray(boxes)

def detection_summary_df(
        bboxes: np.ndarray, 
        classmap: Optional[List[Dict[str, Any]]]=None,
        box_coord_label: Optional[List[str]]=None
    ) -> Optional[pd.DataFrame]:
    data = []
    box_coord_label = box_coord_label or ["X", "Y", "W", "H"]
    bboxes = bboxes.copy()
    for box in bboxes:
        row = {}
        if len(box) == 6:
            score, class_idx, x_or_x1, y_or_y1, x2_or_w, y2_or_h = box
        else:
            track_id, score, class_idx, x_or_x1, y_or_y1, x2_or_w, y2_or_h = box
            row["track_id"] = track_id
        class_idx =  int(class_idx)
        class_ = classmap[class_idx]["name"] if classmap else class_idx
        row.update({"confidence": score, "class": class_})
        row.update({k:int(v) for k, v in zip(box_coord_label, [x_or_x1, y_or_y1, x2_or_w, y2_or_h])})
        data.append(row)
    if len(data) > 0:
        return pd.DataFrame(data)

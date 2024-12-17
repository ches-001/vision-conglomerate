import os
import math
import tqdm
import cv2
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision
import supervision as sv
from pathlib import Path
from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset, IterableDataset, DataLoader
from dataset.inference_dataset import SingleImgSample, InferenceImgDataset, InferenceVideoDataset
from modules.detection import DetectionNetwork
from utils.utils import(
    load_yaml, 
    xywh2x1y1x2y2,
    x1y1x2y22xywh,
    apply_bboxes, 
    apply_bboxes_from_tracks, 
    apply_keypoints, 
    detection_summary_df
)
from typing import *

STORAGE_PATH = os.path.join("outputs", "detection", str(datetime.now()).replace(":", "_"))
CLASS_MAP_PATH = os.path.join("classmap", "detection", "classmap.json")
ANCHORS_PATH = os.path.join("config", "detection", "anchors.yaml")
LOG_FORMAT="%(asctime)s %(levelname)s %(filename)s: %(message)s"
LOG_DATE_FORMAT="%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)

def post_process_preds(
        imgs: torch.Tensor,
        preds: torch.Tensor, 
        num_classes: int, 
        colormap: Optional[np.ndarray]=None,
        iou_threshold: float=0.5,
        score_threshold: float=0.1,
        vwriter: Optional[cv2.VideoWriter]=None,
        tracker: Optional[sv.ByteTrack]=None,
        classmap: Optional[List[Dict[str, Any]]]=None,
        with_summary: bool=False,
        tracked_classes: Optional[List[int]]=None,
        start_idx: int=0,
        box_allowance: Optional[int]=None,
    ) -> Optional[pd.DataFrame]:
    # img shape: (N, C, H, W)
    # preds shape: (N, m, 5+ncls+(kp or 0))
    if colormap is None:
        colormap = np.random.randint(0, 255, size=(num_classes, 3))
    batch_size, bboxes_per_batch = preds.shape[0], preds.shape[1]
    confidneces = torch.sigmoid(preds[..., :1])
    classes = torch.sigmoid(preds[..., 1:1+num_classes])
    scores = torch.max(classes, dim=-1)[0].unsqueeze(dim=-1) * confidneces
    xywh = preds[..., 1+num_classes:5+num_classes]
    keypoints = preds[...,  5+num_classes:]
    
    assert classes.shape[-1] == num_classes
    sample_idxs = torch.zeros(batch_size * bboxes_per_batch, dtype=torch.int64, device=preds.device)
    idx = 0
    for i in range(0, sample_idxs.shape[0], bboxes_per_batch):
        sample_idxs[i:i+bboxes_per_batch] = idx
        idx += 1

    scores = scores.reshape(-1)
    classes = classes.reshape(-1, num_classes)
    xywh = xywh.reshape(-1, 4)
    if box_allowance:
        xywh[:, 2:] += box_allowance
    keypoints = keypoints.reshape(math.prod(keypoints.shape[:-1]), keypoints.shape[-1])
    x1y1x2y2 = xywh2x1y1x2y2(xywh)
    keep_idxs = torchvision.ops.batched_nms(
        boxes=x1y1x2y2,
        scores=scores,
        idxs=sample_idxs,
        iou_threshold=iou_threshold,
    )
    scores = scores[keep_idxs]
    m = (scores > score_threshold)
    scores = scores[m]
    classes = classes[keep_idxs][m]
    sample_idxs = sample_idxs[keep_idxs][m]
    x1y1x2y2 = x1y1x2y2[keep_idxs][m]
    keypoints = keypoints[keep_idxs][m]
    if keypoints.shape[0] > 0 and keypoints.shape[-1] > 0:
        keypoints = keypoints.reshape(*keypoints.shape[:-1], -1, 5)
        keypoints = torch.cat([keypoints[..., :2], keypoints.argmax(-1, keepdim=True)], dim=-1)
    pred_boxes = torch.cat([
        scores.unsqueeze(dim=-1), 
        classes.argmax(dim=-1, keepdim=True), 
        x1y1x2y2
    ], dim=-1)
    summary = []

    for idx, i in enumerate(sample_idxs.unique()):
        m = (sample_idxs == i)
        img = imgs[i]
        # boxes format (confidence, class_idx, x1, y1, x2, y2)
        boxes = pred_boxes[m]
        kp = keypoints[m]
        # classes (class indexes) to track
        if tracked_classes:
            tracked_obj_mask = torch.isin(boxes[:, 1], torch.tensor(tracked_classes, device=device))
            boxes = boxes[tracked_obj_mask]
        if boxes.shape[0] == 0:
            logger.info(f"frame {start_idx + idx} has no detected boxes")
            continue
        # img type is already in uint8, so no need to convert from float32 to uint8
        img = img.permute(1, 2, 0).contiguous().detach().cpu().numpy()
        apply_bboxes_kwargs = {
            "colormap": colormap,
            "box_thickness": 2,
            "text_thickness": 1,
            "classmap": classmap
        }
        if not tracker:
            boxes = boxes.detach().cpu().numpy()
            img = apply_bboxes(img, boxes, **apply_bboxes_kwargs)
            if kp.shape[-1] > 0:
                kp = kp.reshape(-1, kp.shape[-1]).cpu().numpy()
                img = apply_keypoints(img, kp)
        else:
            # boxes = torch.concat([boxes[:, 2:], boxes[:, :2]], dim=1)
            boxes = boxes.cpu().numpy()
            detections = sv.Detections(
                xyxy=boxes[:, 2:], 
                confidence=boxes[:, 0], 
                class_id=boxes[:, 1],
                data={"keypoints": kp.cpu().numpy()} if kp.shape[-1] > 0 else {}
            )
            detections = tracker.update_with_detections(detections)
            if detections.xyxy.shape[0] == 0:
                logger.info(f"frame {start_idx + idx} has no tracked detections")
                continue
            img, boxes = apply_bboxes_from_tracks(img, detections, **apply_bboxes_kwargs)
            if detections.data.get("keypoints") is not None:
                kp = detections.data["keypoints"].reshape(-1, kp.shape[-1])
                img = apply_keypoints(img, kp)

        if with_summary:
            boxes[:, -4:] = x1y1x2y22xywh(boxes[:, -4:])
            box_coord_label = ["X", "Y", "W", "H"]
            summary_df = detection_summary_df(
                boxes, 
                classmap=classmap, 
                box_coord_label=box_coord_label
            )
            if summary_df is not None:
                summary_df.insert(0, "frame", np.zeros(summary_df.shape[0], dtype=int) + (idx + start_idx))
                summary.append(summary_df)
        if vwriter is None:
            with open(os.path.join(STORAGE_PATH, f"img_{idx + start_idx}.png"), "wb") as f:
                Image.fromarray(img).save(f)
            f.close()
        else:
            vwriter.write(img)
            
    if len(summary) > 0:
        summary = pd.concat(summary, axis=0)
        return summary


def evaluate_frames(
        dataset: Union[SingleImgSample, Dataset, IterableDataset],
        model: DetectionNetwork, 
        batch_size: int=32,
        num_workers: int=0,
        device: Union[int, str]="cpu",
        is_video: bool=False,
        fps: int=30,
        **kwargs
    ):
    if not isinstance(dataset, (SingleImgSample, Dataset, IterableDataset)):
        raise ValueError((
            f"imgs is expected to be of type {SingleImgSample.__name__} or"
            f" {Dataset.__name__}, got {type(dataset)}"
        ))
    model.to(device)
    num_classes = model.num_classes
    colormap = np.random.randint(0, 255, size=(num_classes, 3))
    os.makedirs(STORAGE_PATH, exist_ok=True)
    
    classmap = None
    if os.path.isfile(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH, "r") as f:
            classmap = json.load(f)[1:]
        f.close()

    with torch.no_grad():
        if isinstance(dataset, SingleImgSample):
            touched_img, og_img = dataset[0]
            touched_img = touched_img.to(device).unsqueeze(0)
            og_img = og_img.to(device).unsqueeze(0)
            preds = model(
                touched_img, 
                combine_scales=True, 
                to_img_scale=True, 
                og_size=og_img.shape[1:]
            )
            summary = post_process_preds(
                og_img,
                preds,
                num_classes=num_classes,
                classmap=classmap,
                colormap=colormap,
                **kwargs
            )
        else:
            summary = None
            if isinstance(dataset, IterableDataset):
                if num_workers > 0:
                    logger.warning("num workers will be set to 0 when processing video")
                    num_workers = 0                
            dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            vwriter = None
            start_idx = 0
            for i, (touched_img_batch, og_img_batch) in tqdm.tqdm(enumerate(dataset)):
                if is_video and (vwriter is None):
                    w, h = og_img_batch.shape[-1], og_img_batch.shape[-2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vwriter = cv2.VideoWriter(
                        os.path.join(STORAGE_PATH, "video.mp4"), 
                        fourcc=fourcc, 
                        fps=fps, 
                        frameSize=(w, h)
                    )
                touched_img_batch = touched_img_batch.to(device)
                og_img_batch = og_img_batch.to(device)
                preds = model(
                    touched_img_batch, 
                    combine_scales=True, 
                    to_img_scale=True, 
                    og_size=og_img_batch.shape[2:]
                )
                summary_df = post_process_preds(
                    og_img_batch, 
                    preds, 
                    num_classes=num_classes, 
                    colormap=colormap, 
                    vwriter=vwriter,
                    classmap=classmap,
                    start_idx=start_idx,
                    **kwargs
                )
                if summary_df is not None:
                    if summary is None:
                        summary = []
                    summary.append(summary_df)
                start_idx += touched_img_batch.shape[0]
            if vwriter is not None:
                vwriter.release()

            if summary is not None:
                summary = pd.concat(summary, axis=0)
        
        if summary is not None:
            summary.to_csv(os.path.join(STORAGE_PATH, "output.csv"), index=False)


def run(args: argparse.Namespace, config_path: str):
    is_video = False
    if os.path.isdir(args.path):
        dataset = InferenceImgDataset(
            img_dir=args.path, 
            img_exts=["png", "jpg", "jpeg"], 
            img_wh=args.img_size
        )
    elif os.path.isfile(args.path):
        if args.path.endswith(("avi", "mkv", "mp4")):
            is_video = True
            dataset = InferenceVideoDataset(
                video_path=args.path, 
                img_wh=args.img_size, 
                frame_skips=args.frame_skips
            )
        elif args.path.endswith(("png", "jpg", "jpeg")):
            dataset = SingleImgSample(args.path, args.img_size)
    else:
        raise OSError(f"{args.path} not found")
    
    config = load_yaml(config_path)["model_config"]
    anchors = load_yaml(ANCHORS_PATH)["anchors"]

    state_dict = torch.load(args.weights_path, map_location=args.device)
    model = DetectionNetwork(
        in_channels=3,
        num_classes=state_dict["NUM_CLASSES"], 
        config=config, 
        anchors=anchors,
        num_keypoints=config["num_keypoints"]
    )
    model.load_state_dict(state_dict["NETWORK_PARAMS"])
    model.eval()

    tracker = sv.ByteTrack(
        track_activation_threshold=0.35,
        lost_track_buffer=30,
        minimum_matching_threshold=.85,
        frame_rate=30,
        minimum_consecutive_frames=1
    ) if is_video else None
    logger.info("Commencing inference on input data")

    evaluate_frames(
        dataset, 
        model, 
        batch_size=args.batch_size, 
        num_workers=args.dl_workers, 
        device=args.device,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        is_video=is_video,
        tracker=tracker,
        fps=args.fps,
        with_summary=args.with_summary,
        tracked_classes=[int(i) for i in args.tracked_classes.split(",") if i != ''] or None,
        box_allowance=args.box_allowance,
    )
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    best_model_path = f"saved_model/detection/best_model/{DetectionNetwork.__name__}.pth.tar"
    config_path = os.path.join(Path(best_model_path).parent.resolve(), "config", "config.yaml")
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    parser = argparse.ArgumentParser(description="Segmentation Inference")
    parser.add_argument("--path", type=str, metavar="", help="input path (image, folder of images or single video)")
    parser.add_argument("--batch_size", type=int, default=32, metavar="", help="Training batch size")
    parser.add_argument("--img_size", type=int, default=640, metavar="", help="Height and width of images")
    parser.add_argument("--weights_path", type=str, default=best_model_path, metavar="", help="saved model path")
    parser.add_argument("--dl_workers", type=int, default=0, metavar="", help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default=device, metavar="", help="device to run inference on")
    parser.add_argument("--fps", type=int, default=30, metavar="", help="Number of frames per second for video")
    parser.add_argument("--iou_threshold", type=float, default=0.35, metavar="", help="IOU threshold for NMS")
    parser.add_argument("--score_threshold", type=float, default=0.3, metavar="", help="Confidence score threshold")
    parser.add_argument("--with_summary", action="store_true", help="Store output with csv summary of detection")
    parser.add_argument("--tracked_classes", type=str, default="", metavar="", help="class indexes to track")
    parser.add_argument("--frame_skips", type=int, default=0, metavar="", help="Number of frames to skip (only applicable to video stream)")
    parser.add_argument("--box_allowance", type=int, default=4, metavar="", help="Bounding box width and height allowance")
    args = parser.parse_args()
    # python inference_det.py --path="test_vid/20241003T122001.mkv" --with_summary
    run(args, config_path)
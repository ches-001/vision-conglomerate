import os
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
from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset, IterableDataset, DataLoader
from dataset.inference_dataset import SingleImgSample, InferenceImgDataset, InferenceVideoDataset
from modules.segmentation import SegmentationNetwork
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.utils import(
    load_yaml, 
    xywh2x1y1x2y2, 
    apply_bboxes, 
    apply_bboxes_from_tracks, 
    apply_segments, 
    detection_summary_df
)
from typing import *

STORAGE_PATH = os.path.join("outputs", "segmentation", str(datetime.now()).replace(":", "_"))
CONFIG_PATH = os.path.join("config", "segmentation", "config.yaml")
CLASS_MAP_PATH = os.path.join("classmap", "segmentation", "classmap.json")
ANCHORS_PATH = os.path.join("config", "segmentation", "anchors.yaml")
LOG_FORMAT="%(asctime)s %(levelname)s %(filename)s: %(message)s"
LOG_DATE_FORMAT="%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)

def post_process_preds(
        imgs: torch.Tensor,
        preds: torch.Tensor, 
        protos: torch.Tensor, 
        num_classes: int, 
        colormap: Optional[np.ndarray]=None,
        iou_threshold: float=0.5,
        score_threshold: float=0.1,
        vwriter: Optional[cv2.VideoWriter]=None,
        deepsort_tracker: Optional[DeepSort]=None,
        classmap: Optional[List[Dict[str, Any]]]=None,
        with_summary: bool=False,
        tracked_classes: Optional[List[int]]=None,
        start_idx: int=0,
        og_img_whs: Optional[torch.Tensor]=None,
        frame_skips: int=0,
        box_allowance: Optional[int]=None,
    ) -> Optional[pd.DataFrame]:
    # k = num_masks; na=num_anchors; m = (ny *nx * na)_sm + (ny *nx * na)_md + (ny *nx * na)_lg
    # img shape: (N, C, H, W)
    # preds shape: (N, m, 5+ncls+k)
    # protos shape: (N, k, W, H)
    if torch.is_tensor(og_img_whs):
        assert og_img_whs.shape[0] == imgs.shape[0]
    if colormap is None:
        colormap = np.random.randint(0, 255, size=(num_classes, 3))
    img_size = imgs.shape[2:]
    batch_size, bboxes_per_batch, num_masks = preds.shape[0], preds.shape[1], protos.shape[1]
    confidneces = torch.sigmoid(preds[..., :1])
    classes = torch.sigmoid(preds[..., 1:1+num_classes])
    scores = torch.max(classes, dim=-1)[0].unsqueeze(dim=-1) * confidneces
    xywh = preds[..., 1+num_classes:5+num_classes]
    mask_coefs = preds[..., 5+num_classes:]
    
    assert classes.shape[-1] == num_classes and mask_coefs.shape[-1] == (preds.shape[-1] - (5 + num_classes))
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
    mask_coefs = mask_coefs.reshape(-1, mask_coefs.shape[-1])
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
    mask_coefs = mask_coefs[keep_idxs][m]
    pred_boxes = torch.cat([scores.unsqueeze(dim=-1), classes.argmax(dim=-1, keepdim=True), x1y1x2y2], dim=-1)
    summary = []
    for idx, i in enumerate(sample_idxs.unique()):
        if (idx + start_idx) % (frame_skips + 1) != 0: 
            continue
        m = (sample_idxs == i)
        img = imgs[i]
        # boxes format (confidence, class_idx, x1, y1, x2, y2)
        boxes = pred_boxes[m]
        coefs = mask_coefs[m]
        # classes (class indexes) to track
        if tracked_classes:
            tracked_obj_mask = torch.isin(boxes[:, 1], torch.tensor(tracked_classes, device=device))
            boxes = boxes[tracked_obj_mask]
            coefs = coefs[tracked_obj_mask]
        masks = (coefs @ protos[i].reshape(num_masks, -1)).reshape(-1, *protos.shape[2:]).sigmoid()
        masks = F.interpolate(masks.unsqueeze(dim=0), size=img.shape[1:], mode="bilinear", align_corners=False)
        masks = torch.gt(masks, other=0.5).squeeze(dim=0).detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        masks_colormap = colormap[boxes[:, 1].squeeze().astype(int)]
        img = img.permute(1, 2, 0).contiguous().detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        if not deepsort_tracker:
            img = apply_bboxes(
                img, 
                boxes, 
                colormap=colormap, 
                box_thickness=2, 
                text_thickness=1, 
                classmap=classmap,
            )
            img = apply_segments(img, masks, alpha=0.5, colormap=masks_colormap)
        else:
            # (x1, y1, x2, y2) -> (left (x1), top(y1), w, h)
            boxes[:, 4:] = boxes[:, 4:] - boxes[:, 2:4]
            boxes = [(box[2:].tolist(), box[0], int(box[1])) for box in boxes]
            # tracker format: ([left,top,w,h], confidence, class)
            tracks = deepsort_tracker.update_tracks(raw_detections=boxes, frame=img)
            img, boxes = apply_bboxes_from_tracks(
                img,
                tracks,
                colormap=colormap,
                box_thickness=1,
                text_thickness=1,
                classmap=classmap,
            )
            img = apply_segments(img, masks, alpha=0.5, colormap=masks_colormap)
        og_img_wh = tuple(og_img_whs[i].cpu().numpy()) if torch.is_tensor(og_img_whs) else None

        if og_img_wh:
            img = cv2.resize(img, og_img_wh)
        if with_summary:
            summary_df = detection_summary_df(
                boxes,
                img_wh=(img_size[1], img_size[0]),
                og_img_wh=og_img_wh,
                classmap=classmap,
            )
            if summary_df is not None:
                summary_df.insert(0, "frame", np.zeros(summary_df.shape[0], dtype=int) + (idx + start_idx))
                summary.append(summary_df)
        if vwriter is None:
            with open(os.path.join(STORAGE_PATH, f"img_{i}.png"), "wb") as f:
                Image.fromarray(img).save(f)
            f.close()
        else:
            vwriter.write(img)
            
    if len(summary) > 0:
        summary = pd.concat(summary, axis=0)
        return summary


def evaluate_frames(
        imgs: Union[SingleImgSample, Dataset],
        model: SegmentationNetwork, 
        batch_size: int=32,
        num_workers: int=0,
        device: Union[int, str]="cpu",
        is_video: bool=False,
        fps: int=30,
        save_og_size: bool=False,
        **kwargs
    ):
    if not isinstance(imgs, (SingleImgSample, Dataset, IterableDataset)):
        raise ValueError(
            f"imgs is expected to be of type {SingleImgSample.__name__} or {Dataset.__name__}, got {type(imgs)}"
        )
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
        if isinstance(imgs, SingleImgSample):
            imgs.img = imgs.img.to(device)
            preds, protos = model(imgs.img, combine_scales=True, to_img_scale=True)
            summary = post_process_preds(
                imgs.img, 
                preds, 
                protos, 
                num_classes=num_classes, 
                classmap=classmap, 
                colormap=colormap,
                og_img_whs=imgs.og_img_wh if save_og_size else None, 
                **kwargs
            )
            imgs.img = imgs.img.cpu()
            imgs.og_img_wh = imgs.og_img_wh.cpu()

        else:
            summary = None
            dataloader = DataLoader(imgs, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            vwriter = None
            start_idx = 0
            for i, (img_batch, og_img_wh_batch) in tqdm.tqdm(enumerate(dataloader)):
                if is_video and (vwriter is None):
                    if save_og_size:
                        w, h = og_img_wh_batch[0][0].item(), og_img_wh_batch[0][1].item()
                    else:
                        w, h = img_batch.shape[3], img_batch.shape[2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vwriter = cv2.VideoWriter(
                        os.path.join(STORAGE_PATH, "video.mp4"), fourcc=fourcc, fps=fps, frameSize=(w, h)
                    )
                img_batch = img_batch.to(device)
                preds, protos = model(img_batch, combine_scales=True, to_img_scale=True)
                summary_df = post_process_preds(
                    img_batch, 
                    preds, 
                    protos, 
                    num_classes=num_classes, 
                    colormap=colormap, 
                    vwriter=vwriter,
                    classmap=classmap,
                    start_idx=start_idx,
                    og_img_whs=og_img_wh_batch if save_og_size else None,
                    **kwargs
                )
                if summary_df is not None:
                    if summary is None:
                        summary = []
                    summary.append(summary_df)
                start_idx += len(img_batch)
            if vwriter is not None:
                vwriter.release()

            if summary is not None:
                summary = pd.concat(summary, axis=0)
        
        if summary is not None:
            summary.to_csv(os.path.join(STORAGE_PATH, "output.csv"), index=False)


def run(args: argparse.Namespace):
    is_video = False
    if os.path.isdir(args.path):
        imgs = InferenceImgDataset(img_dir=args.path, img_exts=["png", "jpg", "jpeg"], img_wh=args.img_size)
    elif os.path.isfile(args.path):
        if args.path.endswith(("avi", "mkv", "mp4")):
            is_video = True
            imgs = InferenceVideoDataset(video_path=args.path, img_wh=args.img_size)
        elif args.path.endswith(("png", "jpg", "jpeg")):
            img = Image.open(args.path)
            og_img_wh = img.size
            img = img.resize(args.img_size if isinstance(args.img_size, tuple) else (args.img_size, args.img_size))
            img = (torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1) / 255).to(dtype=torch.float32)
            imgs = SingleImgSample(img, og_img_wh)
    else:
        raise OSError(f"{args.path} not found")
    
    config = load_yaml(CONFIG_PATH)["model_config"]
    anchors = load_yaml(ANCHORS_PATH)["anchors"]

    state_dict = torch.load(args.weights_path, map_location=args.device)
    model = SegmentationNetwork(in_channels=3, num_classes=state_dict["NUM_CLASSES"], config=config, anchors=anchors)
    model.load_state_dict(state_dict["NETWORK_PARAMS"])
    model.eval()

    deepsort_tracker = DeepSort(
        max_age=30, 
        n_init=2, 
        embedder="mobilenet",
        embedder_gpu=args.device=="cuda",
        max_cosine_distance=0.2,
    ) if is_video else None
    logger.info("Commencing inference on input data")
    frame_skips = args.frame_skips
    if (not is_video) and frame_skips > 0:
        logger.warning("frame_skips cannot be greater than 0 when input is a video stream")
        frame_skips = 0
    evaluate_frames(
        imgs, 
        model, 
        batch_size=args.batch_size, 
        num_workers=args.dl_workers, 
        device=args.device,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        is_video=is_video,
        deepsort_tracker=deepsort_tracker,
        fps=args.fps,
        with_summary=args.with_summary,
        save_og_size=args.save_og_size,
        tracked_classes=[int(i) for i in args.tracked_classes.split(",") if i != ''] or None,
        frame_skips=frame_skips,
        box_allowance=args.box_allowance,
    )
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    best_model_path = f"saved_model/segmentation/best_model/{SegmentationNetwork.__name__}.pth.tar"
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
    parser.add_argument("--tracked_classes", type=str, default="1,4,7,13,16,17", metavar="", help="class indexes to track")
    parser.add_argument("--save_og_size", action="store_true", help="Save detection outputs with original size")
    parser.add_argument("--frame_skips", type=int, default=0, metavar="", help="Number of frames to skip")
    parser.add_argument("--box_allowance", type=int, default=4, metavar="", help="Bounding box width and height allowance")
    args = parser.parse_args()
    # python inference_seg.py --path="test_vid/20241003T122001.mkv" --with_summary --iou_threshold=0.35 --score_threshold=0.3 --save_og_size
    run(args)
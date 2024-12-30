import os
import tqdm
import cv2
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from dataset.inference_dataset import TrackNetInferenceImgDataset, TrackNetInferenceVideoDataset
from modules.tracknet import TrackNet
from utils.utils import load_yaml
from typing import *

logger = logging.getLogger(__name__)

STORAGE_PATH = os.path.join("outputs", "tracknet", str(datetime.now()).replace(":", "_"))
LOG_FORMAT="%(asctime)s %(levelname)s %(filename)s: %(message)s"
LOG_DATE_FORMAT="%Y-%m-%d %H:%M:%S"

def post_process_preds(
        imgs: torch.Tensor, 
        preds: torch.Tensor, 
        hough_grad_kwargs: Dict[str, Any],
        threshold: int=128,
        vwriter: Optional[cv2.VideoWriter]=None,
        with_summary: bool=False,
        start_idx: int=0,
        max_num_trace: int=5,
        max_circle_thickness: int=10,
    ) -> pd.DataFrame:
    # imgs shape: (N, C*num_stacks, H, W)
    # preds shape: (N, H, W)

    num_stacks = imgs.shape[1] // 3
    imgs = imgs.permute(0, 2, 3, 1).contiguous()

    if start_idx != 0:
        imgs = imgs[..., :3].contiguous()
        tracks = np.empty((imgs.shape[0], 3))
        start_iter = 0
    else:
        pre_imgs = imgs[0, ..., 3:]
        pre_imgs = pre_imgs.reshape(*pre_imgs.shape[:-1], 3, num_stacks-1)
        pre_imgs = pre_imgs.permute(3, 0, 1, 2)
        imgs = torch.concat([pre_imgs, imgs[..., :3]], dim=0)
        tracks = np.empty((imgs.shape[0], 3))
        tracks[:pre_imgs.shape[0]] = [torch.nan]*tracks.shape[1]
        start_iter = pre_imgs.shape[0]

    summary = None
    if with_summary:
        summary = dict(x=[], y=[], r=[])

    imgs = imgs.to(dtype=torch.uint8, device=imgs.device)
    preds[preds < threshold] = 0
    preds[preds >= threshold] = 255

    for i in range(start_iter, imgs.shape[0]):
        pred = preds[i-start_iter].cpu().numpy()
        x, y, r = [np.nan] * 3
        circles = cv2.HoughCircles(pred, **hough_grad_kwargs)
        if circles is not None and len(circles) == 1:
            x = circles[0][0][0]
            y = circles[0][0][1]
            r = circles[0][0][2]
        tracks[i, :] = [x, y, r]

    not_nan_mask = ~np.isnan(tracks[:, 0])
    indexes = np.linspace(0, tracks.shape[0]-1, num=tracks.shape[0])
    if np.any(not_nan_mask) and not_nan_mask.sum() >= not_nan_mask.shape[0]//2:
        for i in range(0, 3):
            tracks[:, i] = np.interp(indexes, indexes[not_nan_mask], tracks[:, i][not_nan_mask])

    for i in range(0, imgs.shape[0]):
        img = imgs[i].cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if summary is not None:
            summary["x"].append(tracks[i][0])
            summary["y"].append(tracks[i][1])
            summary["r"].append(tracks[i][2])
        for j in range(0, max_num_trace):
            if (i - j) <= 0:
                break
            if ~np.isnan(tracks[i-j, 0]):
                x, y, _ = tracks[i-j].astype(int)
                img = cv2.circle(img, (x, y), radius=0, color=(100, 100, 255), thickness=max_circle_thickness-j)
        vwriter.write(img)
    
    if summary is not None:
        summary = pd.DataFrame.from_dict(summary)
    return summary


def evaluate_frames(
        dataset: Union[TrackNetInferenceImgDataset, TrackNetInferenceVideoDataset],
        model: TrackNet, 
        batch_size: int=32,
        num_workers: int=0,
        device: Union[int, str]="cpu",
        fps: int=30,
        **kwargs
    ):
    model.to(device)
    os.makedirs(STORAGE_PATH, exist_ok=True)
    if not isinstance(dataset, (TrackNetInferenceImgDataset, TrackNetInferenceVideoDataset)):
        raise ValueError((
            f"input dataset is expected to be of type {TrackNetInferenceImgDataset.__name__},"
            f" or {TrackNetInferenceVideoDataset.__name__}, got {type(dataset)}"
        ))

    summary = None
    with torch.no_grad():
        if issubclass(dataset.__class__, IterableDataset):
            if num_workers > 0:
                logger.warning("num workers will be set to 0 when processing video")
                num_workers = 0 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        vwriter = None
        start_idx = 0
        for touched_img_batch, og_img_batch in tqdm.tqdm(dataloader):
            if (vwriter is None):
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
            preds = model(touched_img_batch, inference=True, og_size=og_img_batch.shape[2:])
            batch_summary = post_process_preds(
                og_img_batch, preds, vwriter=vwriter, start_idx=start_idx, **kwargs
            )
            if batch_summary is not None:
                if summary is None:
                    summary = []
                summary.append(batch_summary)
            start_idx += touched_img_batch.shape[0]
        if vwriter is not None:
            vwriter.release()

        if summary is not None:
            summary = pd.concat(summary, axis=0)
            summary["frame"] = range(1, summary.shape[0]+1)
            summary = summary[["frame", "x", "y", "r"]]
            summary.dropna(axis=0, inplace=True)
            summary.to_csv(os.path.join(STORAGE_PATH, "output.csv"), index=False)


def run(args: argparse.Namespace, config_path: str):
    config = load_yaml(config_path)
    num_stacks = config["train_config"]["img_config"]["num_stacks"]
    img_wh = config["train_config"]["img_config"]["img_wh"]
    video_formats = ("avi", "mkv", "mp4")
    if os.path.isdir(args.path):
        dataset = TrackNetInferenceImgDataset(
            data_path=args.path, 
            img_wh=img_wh, 
            img_ext=args.img_ext, 
            num_stacks=num_stacks
        )
    elif os.path.isfile(args.path):
        if args.path.endswith(("avi", "mkv", "mp4")):
            dataset = TrackNetInferenceVideoDataset(
                video_path=args.path, 
                img_wh=img_wh,
                frame_skips=args.frame_skips,
                num_stacks=num_stacks
            )
        else:
            Exception(f"Unsupported video file format, only supports {video_formats}")
    else:
        raise OSError(f"{args.path} not found")

    state_dict = torch.load(args.weights_path, map_location=args.device)
    model = TrackNet(in_channels=3*num_stacks, config=config["model_config"])
    model.load_state_dict(state_dict["NETWORK_PARAMS"])
    model.eval()

    logger.info("Commencing inference on input data")
    hough_grad_kwargs = config["train_config"]["hough_grad_config"]
    if isinstance(hough_grad_kwargs.get("method"), str):
        hough_grad_kwargs["method"] = getattr(cv2, hough_grad_kwargs.get("method"))
    evaluate_frames(
        dataset, 
        model, 
        batch_size=args.batch_size, 
        num_workers=args.dl_workers,
        device=args.device,
        fps=args.fps,
        with_summary=args.with_summary,
        threshold=config["train_config"]["heatmap_threshold"],
        hough_grad_kwargs=hough_grad_kwargs,
        max_num_trace=args.max_num_trace,
        max_circle_thickness=args.max_circle_thickness
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    best_model_path = f"saved_model/tracknet/best_model/{TrackNet.__name__}.pth.tar"
    config_path = os.path.join(Path(best_model_path).parent.resolve(), "config", "config.yaml")
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    parser = argparse.ArgumentParser(description="Segmentation Inference")
    parser.add_argument("--path", type=str, metavar="", help="input path (image, folder of images or single video)")
    parser.add_argument("--img_ext", type=str, default="jpg", metavar="", help="Image extensions if path is a folder of sequence of frames")
    parser.add_argument("--batch_size", type=int, default=16, metavar="", help="Training batch size")
    parser.add_argument("--weights_path", type=str, default=best_model_path, metavar="", help="saved model path")
    parser.add_argument("--dl_workers", type=int, default=0, metavar="", help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default=device, metavar="", help="device to run inference on")
    parser.add_argument("--fps", type=int, default=30, metavar="", help="Number of frames per second for video")
    parser.add_argument("--with_summary", action="store_true", help="Store output with csv summary of detection")
    parser.add_argument("--frame_skips", type=int, default=0, metavar="", help="Number of frames to skip (only applicable to video stream)")
    parser.add_argument("--max_num_trace", type=int, default=5, metavar="", help="Number of ball tracks per frame")
    parser.add_argument("--max_circle_thickness", type=int, default=10, metavar="", help="Max thickness of circle for each track drawn")

    args = parser.parse_args()
    # python inference_det.py --path="test_vid/20241003T122001.mkv" --with_summary
    run(args, config_path)
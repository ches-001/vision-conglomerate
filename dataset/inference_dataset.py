import os
import cv2
import glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from utils.utils import load_and_process_img
from typing import *

class SingleImgSample:
    def __init__(self, img_file: str, img_wh: Tuple[int, int]):
        self.img_file = img_file
        if isinstance(img_wh, int):
            img_wh = (img_wh, img_wh)
        self.img_wh = img_wh
        
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx != 0:
            raise IndexError(f"{idx} is out of range for size {len(self)}")
        og_img = load_and_process_img(self.img_file, permute=True, scale=False, convert_to="RGB")
        touched_img = (og_img / 255).to(dtype=torch.float32)
        touched_img = F.interpolate(
            touched_img.unsqueeze(0), size=self.img_wh[::-1], mode="bilinear"
        ).squeeze(0)
        return touched_img, og_img


class InferenceImgDataset(Dataset):
    def __init__(
            self, 
            img_dir: Optional[str]=None, 
            img_exts: List[str]=["png", "jpg", "jpeg"], 
            img_wh: Union[int, Tuple[int, int]]=(640,  640)):
        
        if isinstance(img_wh, int):
            img_wh = (img_wh, img_wh)
        self.img_dir = img_dir
        self.img_wh = img_wh

        self.img_files = []
        for img_ext in img_exts:
            self.img_files.extend(glob.glob(os.path.join(img_dir, "**", f"*.{img_ext}"), recursive=True))
        assert len(self.img_files) > 0

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_file = self.img_files[idx]
        og_img = load_and_process_img(img_file, permute=True, scale=False, convert_to="RGB")
        touched_img = (og_img / 255).to(dtype=torch.float32)
        touched_img = F.interpolate(
            touched_img.unsqueeze(0), size=self.img_wh[::-1], mode="bilinear"
        ).squeeze(0)
        return touched_img, og_img


class InferenceVideoDataset(IterableDataset):
    def __init__(
            self, 
            video_path: Optional[str]=None, 
            img_wh: Union[int, Tuple[int, int]]=(640,  640), 
            frame_skips: int=0
        ):
        if isinstance(img_wh, int):
            img_wh = (img_wh, img_wh)
        self.video_path = video_path
        self.img_wh = img_wh
        self.frame_skips = frame_skips
        self.frame_idx = 0

        if not os.path.isfile(video_path):
            raise FileNotFoundError(F"{video_path} not found")
        self.video_cap = cv2.VideoCapture(video_path)
    
    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], Any, None]:
        while self.video_cap.isOpened():
            ret, og_frame = self.video_cap.read()
            if ret:
                if self.frame_idx % (self.frame_skips + 1) != 0:
                    self.frame_idx += 1
                    continue
                og_frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
                og_frame = torch.from_numpy(og_frame).permute(2, 0, 1)
                touched_frame = (og_frame / 255).to(dtype=torch.float32)
                touched_frame = F.interpolate(
                    touched_frame.unsqueeze(0), size=self.img_wh[::-1], mode="bilinear"
                ).squeeze(0)
                yield touched_frame, og_frame
                self.frame_idx += 1
            else:
                break
        self.video_cap.release()
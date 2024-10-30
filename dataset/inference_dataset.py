import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from typing import *

class SingleImgSample:
    def __init__(self, img: torch.Tensor, og_img_wh: Tuple[int, int]):
        self.img = img.unsqueeze(dim=0)
        self.og_img_wh = torch.tensor(og_img_wh, dtype=torch.int16).unsqueeze(dim=0)


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

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_file = self.img_files[idx]
        img = Image.open(img_file).convert("RGB")
        og_img_wh = torch.tensor(img.size, dtype=torch.int16)
        img = img.resize(self.img_wh)
        img = np.asarray(img).copy()
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = (img / 255).to(dtype=torch.float32)
        return img, og_img_wh


class InferenceVideoDataset(IterableDataset):
    def __init__(self, video_path: Optional[str]=None, img_wh: Union[int, Tuple[int, int]]=(640,  640)):
        if isinstance(img_wh, int):
            img_wh = (img_wh, img_wh)
        self.video_path = video_path
        self.img_wh = img_wh

        if not os.path.isfile(video_path):
            raise FileNotFoundError(F"{video_path} not found")
        self.video_cap = cv2.VideoCapture(video_path)
    
    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], Any, None]:
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                og_img_wh = torch.tensor((frame.shape[1], frame.shape[0]), dtype=torch.int16)
                frame = cv2.resize(frame, dsize=self.img_wh)
                frame = torch.from_numpy(frame).permute(2, 0, 1)
                frame = (frame / 255).to(dtype=torch.float32)
                yield frame, og_img_wh
            else:
                break
        self.video_cap.release()
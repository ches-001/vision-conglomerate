import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.utils import load_and_process_img
from typing import *

class TrackNetDataset(Dataset):
    def __init__(
            self, 
            data_path: Optional[str]=None, 
            labels_df: Optional[pd.DataFrame]=None,
            *,
            num_stacks: int=3,
            img_wh: Union[int, Tuple[int, int]]=(640,  352), 
            avg_diameter: int=5,
            split_percentage: Optional[float]=None
        ):
        if (labels_df is not None and data_path is not None) or (labels_df is None and data_path is None):
            raise ValueError("You either pass in label_df or data_path, not both and both cannot be NoneType")
        # for avg_diameter the paper used 10px (default), but since we resize from 
        # (1280, 720) to (640, 352) we will use 5px (we cannot use 360 directly because input sizes are required
        # by the model architecture to be divisible by 32)
        self.data_path = data_path
        self.img_wh = img_wh
        self.num_stacks = num_stacks
        self.avg_diameter = avg_diameter
        self.split_percentage = split_percentage or 1.0
        if data_path is not None:
            df = self._aggregate_labels_dfs()
        else:
            df = labels_df
        df = df.sample(frac=1)
        split_size = int(self.split_percentage * df.shape[0])
        self.labels_df = df.iloc[:split_size, :]
        self.labels_df.index = range(0, self.labels_df.shape[0])
        self.unused_labels_df = df.iloc[split_size:, :] # stored incase of an evaluation / validation set
        self.unused_labels_df.index = range(0, self.unused_labels_df.shape[0])

    def __len__(self) -> int:
        return self.labels_df.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        *frames_paths, visibility, x, y, status = self.labels_df.iloc[idx, :]
        frames = [load_and_process_img(path, None) for path in frames_paths][::-1]
        stacked_frames = torch.concat(frames, dim=0)
        if visibility == 0:
            x, y = -1, -1
        else:
            x *= (self.img_wh[0] / stacked_frames.shape[2])
            y *= (self.img_wh[1] / stacked_frames.shape[1])
        stacked_frames = torch.nn.functional.interpolate(
            stacked_frames.unsqueeze(0), 
            size=(self.img_wh[1], self.img_wh[0]), 
            mode="bilinear"
        ).squeeze(0)
        gt_heatmap = self._make_gt_heatmap(int(x), int(y), int(visibility))
        others = torch.tensor([visibility, x, y, status], dtype=torch.float32)
        return stacked_frames, gt_heatmap, others

    def _make_gt_heatmap(self, x: int, y: int, visibility: int) -> torch.Tensor:
        w, h = self.img_wh
        if visibility != 0:
            x_grid, y_grid = np.mgrid[0-y:h-y, 0-x:w-x]
            variance = self.avg_diameter
            heatmap = (np.exp(-(x_grid**2 + y_grid**2) / (2 * variance)) * 255).astype(np.uint8)
            heatmap = cv2.resize(heatmap, self.img_wh)
            heatmap = torch.from_numpy(heatmap)
            return heatmap
        return torch.zeros((self.img_wh[1], self.img_wh[0]), dtype=torch.uint8)
    
    def _aggregate_labels_dfs(self) -> pd.DataFrame:
        label_dfs = []
        clip_dirs = (glob.glob(os.path.join(self.data_path, "*/Clip*"), recursive=True))
        for clip_dir in clip_dirs:
            label_df = pd.read_csv(os.path.join(clip_dir, "Label.csv"))
            label_df = self._finalize_label_df(label_df, clip_dir)
            label_dfs.append(label_df)
        df = pd.concat(label_dfs, axis=0)
        df.index = range(0, df.shape[0])
        return df

    def _finalize_label_df(self, label_df: pd.DataFrame, dir: str) -> pd.DataFrame:
        label_df["paths"] = os.path.join(dir, "") + label_df["file name"]
        final_df = pd.DataFrame()
        for i in range(0, self.num_stacks):
            final_df[f"frame{i+1}"] = label_df["paths"].iloc[i : label_df.shape[0]-(self.num_stacks-i)+1].to_list()
        final_df = final_df[[f"frame{i+1}" for i in range(0, self.num_stacks)]]
        extra_labels = label_df.iloc[self.num_stacks-1:][["visibility", "x-coordinate", "y-coordinate", "status"]]
        extra_labels.index = range(0, extra_labels.shape[0])
        final_df = pd.concat([final_df, extra_labels], axis=1)
        return final_df
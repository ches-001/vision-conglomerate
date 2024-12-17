import os
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
import logging
import argparse
import shutil
import kaggle
from dotenv import load_dotenv
from roboflow import Roboflow
from typing import Optional

logger = logging.getLogger(__name__)

class KagglePadelBallDataDownloader:
    def __init__(self, dtype: Optional[str]=None):
        os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
        self.dataset_url = "https://www.kaggle.com/datasets/ichimarugin/padel-ball-dataset"
        __default_dtype = "detection"
        self.dataset_dir = f"data/{dtype or __default_dtype}"

    def download(self):
        dataset_url = self.dataset_url
        dataset_dir = self.dataset_dir
        temp_dir = os.path.join(dataset_dir, "temp")
        train_dir = os.path.join(dataset_dir, "train")
        validation_dir = os.path.join(dataset_dir, "valid")

        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(validation_dir, exist_ok=True)
        kaggle.api.dataset_download_cli(dataset_url.split("datasets/")[-1], path=temp_dir, unzip=True)
        
        _padel_path = os.path.join(temp_dir, "padel", "padel")
        _new_annotations_path = os.path.join(temp_dir, "new_annotations", "new_annotations")
        shutil.move(os.path.join(_padel_path, "train", "images"), train_dir)
        shutil.move(os.path.join(_padel_path, "valid", "images"), validation_dir)
        shutil.move(os.path.join(_new_annotations_path, "train", "labels"), train_dir)
        shutil.move(os.path.join(_new_annotations_path, "valid", "labels"), validation_dir)
        shutil.rmtree(temp_dir)


class RoboFlowDataDownloader:
    def __init__(self, api_key: str, workspace: str, project_id: str, version: int, dtype: Optional[str]=None):
        self.version = version
        self.dformat = "yolov5"
        self.rf = Roboflow(api_key=api_key)
        self.workspace = self.rf.workspace(the_workspace=workspace)
        self.project = self.workspace.project(project_id=project_id)
        self.version = self.project.version(version_number=version)
        __default_dtype = "segmentation"
        self.dataset_dir = f"data/{dtype or __default_dtype}"

    def download(self):
        self.version.download(self.dformat, location=self.dataset_dir, overwrite=True)


if __name__ == "__main__":
    load_dotenv()
    download_source = "kaggle"
    rf_api_key = os.getenv("ROBOFLOW_API_KEY", "")
    rf_workspace = os.getenv("ROBOFLOW_WORKSPACE", "")
    rf_project = os.getenv("ROBOFLOW_PROJECT", "")
    rf_version = os.getenv("VERSION", "1")

    parser = argparse.ArgumentParser(description=f"Dataset Downloader")
    parser.add_argument(
        "--source", type=str, choices=["kaggle", "roboflow"], 
        default=download_source, metavar="", help=f"Dataset download source"
    )
    parser.add_argument(
        "--dtype", type=str, default="", choices=["detection", "segmentation"],
        metavar="", help=f"Dataset type (detection or segmentation)"
    )
    parser.add_argument(
        "--rf_api_key", type=str, default=rf_api_key, metavar="", help=f"Roboflow API key"
    )
    parser.add_argument(
        "--rf_workplace", type=str, default=rf_workspace, metavar="", help=f"Workplace"
    )
    parser.add_argument(
        "--rf_project_id", type=str, default=rf_project, metavar="", help=f"project_id"
    )
    parser.add_argument(
        "--rf_version", type=int, default=int(rf_version), metavar="", help=f"version"
    )
    args = parser.parse_args()

    if args.source == "kaggle":
        dataset_downloader = KagglePadelBallDataDownloader(dtype=args.dtype)
    elif args.source == "roboflow":
        dataset_downloader = RoboFlowDataDownloader(
            args.rf_api_key, args.rf_workplace, args.rf_project_id, args.rf_version, dtype=args.dtype
        )
    dataset_downloader.download()
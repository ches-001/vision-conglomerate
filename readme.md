# Custom Object Detection Repository

# Setup and Training

1. Create a virtual environment (Optional)

2. run the command `pip install -r requirementst.txt`

3. visit [the official pytorch website](https://pytorch.org/) and install the specific version of torch suitable for your hardware

4. create a `.env` file and arrange your RoboFlow details like so:

    ```
    ROBOFLOW_API_KEY = "xxxxxxxxxxxx"

    ROBOFLOW_WORKSPACE = "workspace_name"

    ROBOFLOW_PROJECT = "project_name"

    VERSION = xxx
    ```
    Alternatively, if your dataset is on kaggle, download your `kaggle.json` credentials from kaggle and move to the main directory of your project, ensure that the dataset is organized as shown in this [kaggle dataset](https://www.kaggle.com/datasets/ichimarugin/padel-ball-dataset), finally, go to the `get_dataset.py` script, in the `KagglePadelBallDataDownloader` class, change `self.dataset_url` to the URL of your own kaggle dataset.


5. Run the `get_dataset.py` script, if your data source is roboflow, then run  `get_dataset.py --source="roboflow"`, likewise, set `--source` to "kaggle" if your data source is kaggle. It is mandatory to also include the `--dtype` flag, it can only be (detection or segmentation), Eg: `get_dataset.py --source="roboflow" --dtype="segmentation"`

6. Run the `train_det.py` or `train_seg.py` scripts depending on whether you are training a detection or segmentation model. Use the `--help` flag to view and understand all the CLI arguments. If you wish to train on multiple GPUs, you can use `torchrun` facilitate training, like so: `torchrun --standalone --nproc_per_node=gpu train_seg.py --use_ddp --lr_schedule --batch_size=32`. You can also change some of the configurations in `config/*/yaml` directory, you can change things like the backbone (view the `modules/backbone.py` for all backbones), you can change the neck and head architecture as well (`modules/common.py`) and other configurations.


# Inference

To run inference on your test data, use the `inference_seg.py` or `inference_det.py` scripts depending whether you trained a detection or segmentation model. Use the `--help` flag to view all CLI arguments. To run inference on a directory of images, you can run the script with the follow command: `ython inference_seg.py --path="test_directory"`. If you wish to store the output images in their original size instead of 640x640, use the `--save_og_size` flag. If you wish to generate a summary of the detections in a csv file, use the `--with_summary` flag. If you wish to run inference on a video, just set the video path with `--path="video-path.mp4"`. You can also specify which classes to track with the `--tracked_classes`, Eg: `--tracked_classes="2,4,5"` will only track classes 2, 4 and 5.

Finally, you can create a "classmap" folder to store the class index to label map in a json file called `classmap.json`. This file will be stored in `classmap/detection/classmap.json` or `classmap/segmentation/classmap.json` depending on the task, the creation of this file is optional, if it not created, class indexes will simply be used in the detection outputs. The file format is like so:

```json
[
    { "id": 0, "name": "ball-raket-gender-net-court", "supercategory": "none" },
    { "id": 1, "name": "allcourt", "supercategory": "ball-raket-gender-net-court" },
    ...
    { "id": 17, "name": "youngfemale", "supercategory": "ball-raket-gender-net-court" },
    { "id": 18, "name": "youngmale", "supercategory": "ball-raket-gender-net-court" }
]
```
Each index of this list (after the first) corresponds to the index class index, so index 0 of the list is:
 ```json 
 { "id": 1, "name": "allcourt", "supercategory": "ball-raket-gender-net-court" }
```

import torch
from . import common
from .detection import DetectionNet
from typing import *


class SegmentationNet(DetectionNet):
    def __init__(self, *args, **kwargs):
        super(SegmentationNet, self).__init__(*args, **kwargs)
        self.proto_seg_module = common.ProtoSegModule(
            self.neck.out_fmaps_channels[1], 
            self.config["num_masks"], 
            **self.config["protos_config"]
        )

    def forward(self, *args, **kwargs) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        torch.Tensor,
    ]:  
        # preds (y) shape per scale: (N, ny, nx, num_anchors, 5+C+k)
        # protos (p) shape: (N, k, H/f, W/f); f=2; k=number of segment masks; N=batch size
        # To get the segments from p (prototype segments), we reshape y such that:
        # y shape: (N, r, 5+C+k); r = (ny * nx * num_anchors). Next, we can slice the last
        # dimension of the y tensor to get (N, r, k). Suppose we post process y by passing it
        # through NMS (Non-Max Suppression), the shape of y would become (N, n, k), where n is
        # is the number of accepted bboxes. Finally we can get the masks by linearly combining
        # p and y such that the mask m = sigmoid(y \matmul p), with shape (N, n, H/f, W/f)
        # Note: This technique is inspired from the YOLACT (You Only Look at Transformers) paper
        # https://arxiv.org/pdf/1904.02689. It is not exactly it but it is similar, for example,
        # in the original YOLACT, f=4 and not 2.
        preds, protos = DetectionNet.forward(self, *args, **kwargs)
        return preds, protos
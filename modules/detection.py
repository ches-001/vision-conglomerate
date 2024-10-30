import torch
import torch.nn as nn
from . import backbone, common
from .common import RepVGGBlock
from typing import *


class DetectionNetwork(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        num_classes: int, 
        config: Dict[str, Any],
        anchors: Dict[str, Any]
    ):
        super(DetectionNetwork, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.num_classes = num_classes
        # anchors range from 0 to 1 (they are relative to the image space)
        self.num_anchors = len(anchors["sm"])
        _train_anchors = self.config["train_anchors"]
        self.out_channels = self.num_anchors*(5 + num_classes)
        self.sm_anchors = nn.Parameter(torch.tensor(anchors["sm"]), requires_grad=_train_anchors)
        self.md_anchors = nn.Parameter(torch.tensor(anchors["md"]), requires_grad=_train_anchors)
        self.lg_anchors = nn.Parameter(torch.tensor(anchors["lg"]), requires_grad=_train_anchors)

        self.backbone = getattr(backbone, config["backbone"])(
            in_channels, **config.get(config["backbone"].lower()+"_config", {})
        )
        self.neck = getattr(common, config["neck"])(
            *self.backbone.bbone_out_channels, **config.get(config["neck"].lower()+"_config", {})
        )
        self.head = nn.ModuleList([
            getattr(common, self.config["head"])(
                ch, 
                num_classes=self.num_classes, 
                num_anchors=self.num_anchors, 
                **self.config.get(self.config["head"].lower()+"_config", {})
            )for ch in self.neck.neck_out_channels
        ])
        self.apply(self._xavier_init_weights)

    def forward(
        self,
        x: torch.Tensor, 
        combine_scales: bool=False,
        to_img_scale: bool=False
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        torch.Tensor,
    ]:
        fmaps = self.backbone(x)
        n3, n4, n5 = self.neck(fmaps)
        sm_scale = self.head[0](n3)
        md_scale = self.head[1](n4)
        lg_scale = self.head[2](n5)

        # process predictions at different scales
        sm_preds = self._get_scale_pred(sm_scale, self.sm_anchors, input_shape=x.shape[2:], to_img_scale=to_img_scale)
        md_preds = self._get_scale_pred(md_scale, self.md_anchors, input_shape=x.shape[2:], to_img_scale=to_img_scale)
        lg_preds = self._get_scale_pred(lg_scale, self.lg_anchors, input_shape=x.shape[2:], to_img_scale=to_img_scale)

        if not combine_scales:
            preds = (sm_preds, md_preds, lg_preds)
        else:
            batch_size = x.shape[0]
            k = 0
            if hasattr(self, "proto_seg_module"):
                k = self.config["num_masks"]
            sm_preds = sm_preds.reshape(batch_size, -1, self.num_classes+5+k)
            md_preds = md_preds.reshape(batch_size, -1, self.num_classes+5+k)
            lg_preds = lg_preds.reshape(batch_size, -1, self.num_classes+5+k)
            preds = torch.cat((sm_preds, md_preds, lg_preds), dim=1).flatten(start_dim=1, end_dim=-2)

        if hasattr(self, "proto_seg_module"):
            protos = self.proto_seg_module(n3)
            return preds, protos
        return preds

    def _get_scale_pred(
        self, 
        scale_pred: torch.Tensor, 
        anchors: torch.Tensor, 
        input_shape: Tuple[int, int], 
        to_img_scale: bool=False
    ):
        # shape: [batch_size, ny, nx, num_anchors, 5+num_classes]
        _, ny, nx, _, _ = scale_pred.shape

        # bbox index beginning along the last dimension (dim=-1 or dim=4)
        bbox_i = self.num_classes+1

        # first index corresponds to objectness of each box
        objectness = scale_pred[..., :1]

        # next `num_class` indexes correspond to class probability logits of each bbox
        class_proba = scale_pred[..., 1:bbox_i]

        # the second to the last two indexes of the last dimension corresponds to box centers (x, y)
        xy = (scale_pred[..., bbox_i:bbox_i+2].sigmoid() * 2) - 0.5

        # last two indexes of last dimension corresponds to width and height of boxes (w, h)
        wh = (scale_pred[..., bbox_i+2:bbox_i+4].sigmoid() * 2).pow(2)

        # scale predicted bboxes to full image size
        if to_img_scale:
            # calculate stride values to map feature map h and w to image h and w
            input_shape = torch.tensor(input_shape, device=scale_pred.device)
            stride = torch.tensor(
                [input_shape[0]/ny, input_shape[1]/nx], 
                device=scale_pred.device, 
                dtype=scale_pred.dtype
            )
            grid = self._make_2dgrid(nx, ny, device=scale_pred.device)
            xy = (xy + grid) * stride
            wh = wh * anchors * torch.tensor([nx, ny], device=scale_pred.device) * stride

        if not hasattr(self, "proto_seg_module"):
            pred = torch.cat((objectness, class_proba, xy, wh), dim=-1)
        else:
            # masks coefficients
            masks_coefs = scale_pred[..., bbox_i+4:].tanh()
            pred = torch.cat((objectness, class_proba, xy, wh, masks_coefs), dim=-1)
        return pred

    def _make_2dgrid(self, nx: int, ny: int, device: str="cpu") -> torch.Tensor:
        xindex = torch.arange(nx, device=device)
        yindex = torch.arange(ny, device=device)
        ygrid, xgrid = torch.meshgrid([yindex, xindex], indexing="ij")
        return torch.stack((xgrid, ygrid), dim=2).reshape(1, ny, nx, 1, 2).float()

    def _xavier_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)

    def inference(self):
        self.eval()
        def toggle_inference_mode(m: nn.Module):
            if isinstance(m, RepVGGBlock):
                if (
                    isinstance(m.identity, (nn.BatchNorm2d, nn.Identity)) and 
                    isinstance(m.conv1x1.norm, nn.BatchNorm2d) and 
                    isinstance(m.conv3x3.norm, nn.BatchNorm2d)
                ): m.toggle_inference_mode()
        self.apply(toggle_inference_mode)
import torch
import torch.nn as nn
from . import backbone, common
from .common import RepVGGBlock
from typing import *


class DetectionNet(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        num_classes: int,
        config: Dict[str, Any],
        anchors: Optional[Dict[str, Any]]=None,
        num_keypoints: Optional[int]=None,
    ):
        super(DetectionNet, self).__init__()
        if anchors is None:
            # NOTE: For your own good, ensure that anchors are only NoneType at inference,
            # this is because after training, anchors are saved to state dict and at inference
            # the state dict is loaded and applied to an initialisation of this network. During
            # training however, anchors are expected and if you do not provide them, well...
            anchors = {}
            anchors["sm"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            anchors["md"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            anchors["lg"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

        self.config = config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
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
            *self.backbone.out_fmaps_channels, **config.get(config["neck"].lower()+"_config", {})
        )
        self.head = nn.ModuleList([
            getattr(common, self.config["head"])(
                ch, 
                num_classes=self.num_classes, 
                num_anchors=self.num_anchors,
                num_masks=self.config.get("num_masks", None),
                num_keypoints=self.num_keypoints,
                **self.config.get(self.config["head"].lower()+"_config", {})
            )for ch in self.neck.out_fmaps_channels[1:]
        ])
        self.apply(self._xavier_init_weights)

    def forward(self, x: torch.Tensor, inference: bool=False, og_size: Optional[Tuple[int, int]]=None) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        torch.Tensor,
    ]:
        fmaps = self.backbone(x)
        _, n3, n4, n5 = self.neck(fmaps)
        sm_scale = self.head[0](n3)
        md_scale = self.head[1](n4)
        lg_scale = self.head[2](n5)

        # process predictions at different scales
        sm_preds = self._get_scale_pred(sm_scale, self.sm_anchors, input_shape=x.shape[2:], inference=inference)
        md_preds = self._get_scale_pred(md_scale, self.md_anchors, input_shape=x.shape[2:], inference=inference)
        lg_preds = self._get_scale_pred(lg_scale, self.lg_anchors, input_shape=x.shape[2:], inference=inference)

        if not inference:
            preds = (sm_preds, md_preds, lg_preds)
        else:
            if (og_size is not None) and (og_size[0] != x.shape[2] and og_size[1] != x.shape[3]):
                _from = torch.tensor([x.shape[3], x.shape[2], x.shape[3], x.shape[2]], device=x.device)
                _to = torch.tensor([og_size[1], og_size[0], og_size[1], og_size[0]], device=x.device)
                sm_preds = self._bbox_to_size(sm_preds, _from, _to)
                md_preds = self._bbox_to_size(md_preds, _from, _to)
                lg_preds = self._bbox_to_size(lg_preds, _from, _to)
            batch_size = x.shape[0]
            k = 0 # num mask coefficients (YOLACT Implementation)
            kp = (self.num_keypoints or 0) * 5 # (x, y, p_visible, p_occluded, p_deleted) keypoints
            if hasattr(self, "proto_seg_module"):
                k = self.config["num_masks"]
            final_dim = self.num_classes + 5 + k + kp
            sm_preds = sm_preds.reshape(batch_size, -1, final_dim)
            md_preds = md_preds.reshape(batch_size, -1, final_dim)
            lg_preds = lg_preds.reshape(batch_size, -1, final_dim)
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
        inference: bool=False
    ) -> torch.Tensor:
        # shape: [batch_size, ny, nx, num_anchors, 1 + num_classes + 4 + num_masks + (5*num_keypoints))]
        _, ny, nx, _, _ = scale_pred.shape

        # bbox beginning and ending slice indexes along the last dimension (dim=-1 or dim=4)
        bbox_i = self.num_classes+1
        bbox_j = bbox_i + 4
        
        # begining slice index of the keypoints
        kp_i = bbox_j

        # first index corresponds to objectness of each box
        objectness = scale_pred[..., :1]

        # next `num_class` indexes correspond to class probability logits of each bbox
        class_proba = scale_pred[..., 1:bbox_i]

        # the second to the last two indexes of the last dimension corresponds to box centers (x, y)
        xy = (scale_pred[..., bbox_i:bbox_i+2].sigmoid() * 2) - 0.5

        # last two indexes of last dimension corresponds to width and height of boxes (w, h)
        wh = (scale_pred[..., bbox_i+2:bbox_j].sigmoid() * 2).pow(2)

        # keypoints and mask_coefs
        keypoints = None
        masks_coefs = None

        if hasattr(self, "proto_seg_module"):
            k = self.proto_seg_module.out_channels
            kp_i += k
            masks_coefs = scale_pred[..., bbox_j:kp_i].tanh()

        if self.num_keypoints is not None and self.num_keypoints > 0:
            keypoints = scale_pred[..., kp_i:]
            keypoints = keypoints.reshape(*keypoints.shape[:-1], -1, 5)
            # keypoints coordinates of model predictions will range from 0 to 1 and will not be relative
            # to the image dimensions but rather, to the bbox dimensions that they belong to. Do take note
            # of this fact when reading through this codebase, because the loss function is designed with
            # this in-mind (to avoid getting confused).
            keypoints[..., :2] = keypoints[..., :2].sigmoid()

        if inference:
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
            # scale keypoints back to image scale
            if torch.is_tensor(keypoints):
                # each keypoint is denoted as [x, y, p_visible, p_occluded, p_deleted], x and y are sigmoid outputs
                # and hence will be multiplied by the width and height of their corresponding bboxes and processed further
                # by the img_width and img_height to be brought to image scale.
                keypoints[..., :2] = keypoints[..., :2] * wh.unsqueeze(dim=4)
                keypoints[..., :2] = keypoints[..., :2] + (xy - (wh / 2)).unsqueeze(dim=4)
            
        pred = torch.cat((objectness, class_proba, xy, wh), dim=-1)

        if torch.is_tensor(masks_coefs):
            pred = torch.cat((pred, masks_coefs), dim=-1)

        if torch.is_tensor(keypoints):
            keypoints = keypoints.reshape(*keypoints.shape[:-2], -1)
            pred = torch.cat([pred, keypoints], dim=-1)
            
        return pred
    
    def _bbox_to_size(self, pred: torch.Tensor, _from: torch.Size, _to: torch.Size) -> torch.Tensor:
        box_i = 1 + self.num_classes
        box_j = box_i + 4
        kp_i = box_j
        if hasattr(self, "proto_seg_module"):
            k = self.proto_seg_module.out_channels
            kp_i += k
        pred[..., box_i:box_j] = (pred[..., box_i:box_j] / _from) * _to

        ones = torch.ones(3, device=pred.device)
        pred[..., kp_i:] = (
            (pred[..., kp_i:].reshape(
                *pred.shape[:-1], -1, 5
            )[..., :] / torch.concat([_from[:2], ones])) * torch.concat([_to[:2], ones])
        ).reshape(*pred.shape[:-1], -1)
        return pred.contiguous()

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
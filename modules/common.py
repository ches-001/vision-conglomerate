import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


class ConvBNorm(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: Union[int, Tuple[int, int]], 
            stride: Union[int, Tuple[int, int]]=1, 
            padding: Optional[Union[int, Tuple[int, int]]]=None,
            activation: Optional[Type]=nn.SiLU,
            bias: bool=True,
        ):
        super(ConvBNorm, self).__init__()

        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = [i//2 for i in kernel_size]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=bias
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = None
        if activation:
            self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    

class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *x: torch.Tensor):
        return torch.cat(x, dim=self.dim)
    

class RepVGGBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            activation: Optional[Type]=nn.SiLU, 
            stride: Union[int, Tuple[int, int]]=1,
            padding: Optional[Union[int, Tuple[int, int]]]=None,
            identity_layer: Type=nn.BatchNorm2d
        ):
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding or 3//2
        self.inference_mode = False

        self.conv3x3 = ConvBNorm(
            in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=self.padding, bias=False
        )
        self.conv1x1 = ConvBNorm(
            in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=self.padding-1, bias=False
        )
        if stride == 1 and in_channels == out_channels:
            self.identity = identity_layer(out_channels)
        else:
            self.identity = nn.Identity()
        if activation:
            self.activation = activation()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv_reparam"):
            return self.activation(self.conv_reparam(x))
        
        out = self.conv3x3(x) + self.conv1x1(x)
        if not isinstance(self.identity, nn.Identity):
            out = out + self.identity(x)
        if self.activation:
            out = self.activation(out)
        return out
    
    def reparameterize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w3x3, b3x3 = self._merge_conv_bn(self.conv3x3.conv, self.conv3x3.norm)
        w1x1, b1x1 = self._merge_conv_bn(self.conv1x1.conv, self.conv1x1.norm)
        w = w3x3 + F.pad(w1x1, [1, 1, 1, 1])
        b = b3x3 + b1x1
        if not isinstance(self.identity, nn.Identity):
            wI1x1, bI1x1 = self._merge_conv_bn(nn.Identity(), self.identity)
            w = w + F.pad(wI1x1, [1, 1, 1, 1])
            b = b + bI1x1
        return w, b
    
    def _merge_conv_bn(
            self, 
            conv: Union[nn.Conv2d, nn.Identity], 
            bn: nn.BatchNorm2d,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(bn, nn.BatchNorm2d):
            raise RuntimeError(
                f"RepVGGBlock reparameterization only works with nn.BatchNorm2d layers, got {type(bn)} instead"
            )
        if isinstance(conv, nn.Conv2d):
            w = conv.weight
        elif isinstance(conv, nn.Identity):
            input_dim = self.in_channels//self.conv3x3.conv.groups
            w = torch.zeros((self.in_channels, input_dim, 1, 1), device=self.conv3x3.conv.weight.device)
            for i in range(self.in_channels):
                w[i, i % input_dim, 0, 0] = 1
        else: 
            raise RuntimeError
        gamma = bn.weight
        mu = bn.running_mean
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(bn.running_var + eps)
        weight_n = (gamma / std).reshape(-1, *([1]*(len(w.shape)-1))) * w
        bias_n = ((-mu * gamma) / std) + beta
        return weight_n, bias_n
    
    def toggle_inference_mode(self):
        w, b = self.reparameterize()
        self.conv_reparam = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=(3, 3), stride=self.stride, padding=self.padding
        )
        self.conv_reparam.weight.data = w
        self.conv_reparam.bias.data = b
        if hasattr(self, "conv3x3"): self.__delattr__("conv3x3")
        if hasattr(self, "conv1x1"): self.__delattr__("conv1x1")
        if hasattr(self, "identity"): self.__delattr__("identity")
        self.inference_mode = True
        

class RepBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int=1, e: float=0.5):
        super(RepBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        c_h = int(out_channels * e)
        if n == 1:
            self.conv1 = RepVGGBlock(in_channels, out_channels)
            self.blocks = nn.Identity()
        elif n == 2:
            self.conv1 = RepVGGBlock(in_channels, c_h)
            self.blocks = nn.Sequential(RepVGGBlock(c_h, out_channels))
        elif n > 2:
            self.conv1 = RepVGGBlock(in_channels, c_h)
            _blocks = [RepVGGBlock(c_h, c_h) for i in range(n-2)]
            _blocks.append(RepVGGBlock(c_h, out_channels))
            self.blocks = nn.Sequential(*_blocks)
        else:
            raise Exception(f"n must be > 1, got n={n}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.conv1(x))
    

class BiCwithConvModule(nn.Module):
    def __init__(
            self, 
            c1_in_channels: int,
            c0_in_channels: int, 
            p2_in_channels: int, 
            out_channels: int, 
            e: float=0.5,
            upsample_mode: str="nearest"
        ):
        super(BiCwithConvModule, self).__init__()
        c_h = int(out_channels * e)
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2.0, mode=upsample_mode)
        self.downsample = nn.Upsample(scale_factor=0.5, mode=upsample_mode)
        self.conv_c1 = ConvBNorm(c1_in_channels, c_h, kernel_size=1)
        self.conv_c0 = ConvBNorm(c0_in_channels, c_h, kernel_size=1)
        self.concat = Concat(dim=1)
        self.conv_out = ConvBNorm(c_h+c_h+p2_in_channels, out_channels, kernel_size=1)

    def forward(self, c1: torch.Tensor, c0: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        # c1 -> no upsampling or downsampling
        # c0 -> downsampling
        # p2 -> upsampling
        c1 = self.conv_c1(c1)
        c0 = self.downsample(self.conv_c0(c0))
        p2 = self.upsample(p2)
        output = self.concat(c1, c0, p2)
        output = self.conv_out(output)
        return output
    

class BiCwithNoConvModule(nn.Module):
    def __init__(
        self, 
        c1_in_channels: int, 
        c0_in_channels: int, 
        p2_in_channels: int,
        out_channels: Optional[int]=None,
        upsample_mode: str="nearest"
    ):
        super(BiCwithNoConvModule, self).__init__()
        if not out_channels:
            c_h = None
            self.out_channels = c1_in_channels + c0_in_channels + p2_in_channels
        else:
            c_h = c1_in_channels + c0_in_channels + p2_in_channels
            self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2.0, mode=upsample_mode)
        self.downsample = nn.Upsample(scale_factor=0.5, mode=upsample_mode)
        self.concat = Concat(dim=1)
        if c_h:
            self.conv = ConvBNorm(c_h, self.out_channels, kernel_size=1)

    def forward(self, c1: torch.Tensor, c0: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        # c1 -> no upsampling or downsampling
        # c0 -> downsampling
        # p2 -> upsampling
        c0 = self.downsample(c0)
        p2 = self.upsample(p2)
        output = self.concat(c1, c0, p2)
        if hasattr(self, "conv"):
            output = self.conv(output)
        return output
    

class BottleNeckModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, e: float=0.5, shortcut: bool=True):
        super(BottleNeckModule, self).__init__()
        c_h = int(out_channels * e)
        self.conv1 = ConvBNorm(in_channels, c_h, kernel_size=1, stride=1)
        self.conv2 = ConvBNorm(c_h, out_channels, kernel_size=3, stride=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv2(self.conv1(x))
        if self.shortcut:
            output = x + output
        return output
    

class C3Module(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, e: float=0.5, shortcut: bool=True, num_bottlenecks: int=1):
        super(C3Module, self).__init__()
        c_h = int(out_channels * e)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBNorm(in_channels, c_h, kernel_size=1, stride=1)
        self.conv2 = ConvBNorm(in_channels, c_h, kernel_size=1, stride=1)
        self.bottlenecks = nn.Sequential(
            *(BottleNeckModule(c_h, c_h, e=1.0, shortcut=shortcut) for _ in range(num_bottlenecks))
        )
        self.conv3 = ConvBNorm(2*c_h, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.bottlenecks(self.conv1(x))
        out2 = self.conv2(x)
        output = self.conv3(torch.cat([out1, out2], dim=1))
        return output
    

class SPPFModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, e: float=0.5, pool_kernel_size: int=5):
        super(SPPFModule, self).__init__()
        c_h = int(out_channels * e)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBNorm(in_channels, c_h, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size//2)
        self.conv2 = ConvBNorm(c_h*4, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        p1 = self.pool(y)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        output = self.conv2(torch.cat([y, p2, p2, p3], dim=1))
        return output
        

class CSPSPPFModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, e: float=0.5, pool_kernel_size: int=5):
        super(CSPSPPFModule, self).__init__()
        c_h = int(out_channels * e)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1_3_4 = nn.Sequential(
            ConvBNorm(in_channels, c_h, kernel_size=1),
            ConvBNorm(c_h, c_h, kernel_size=3),
            ConvBNorm(c_h, c_h, kernel_size=1)
        )
        self.conv2 = ConvBNorm(in_channels, c_h, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size//2)
        self.conv5 = ConvBNorm(c_h*4, c_h, kernel_size=1)
        self.conv6 = ConvBNorm(c_h, c_h, kernel_size=3)
        self.concat = Concat(dim=1)
        self.conv7 = ConvBNorm(c_h*2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_1_3_4(x)
        y1 = self.conv2(x)
        x_p1 = self.pool(x1)
        x_p2 = self.pool(x_p1)
        x_p3 = self.pool(x_p2)
        x1 = self.concat(x1, x_p1, x_p2, x_p3)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)
        x_out = self.concat(x1, y1)
        x_out = self.conv7(x_out)
        return x_out
    

class ProtoSegModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int=32, c_h: int=256, upsample_mode: str="nearest"):
        super(ProtoSegModule, self).__init__()

        self.conv1 = ConvBNorm(in_channels, c_h, kernel_size=3, activation=nn.ReLU)
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv2 = ConvBNorm(c_h, c_h, kernel_size=3, activation=nn.ReLU)
        self.conv3 = ConvBNorm(c_h, out_channels, kernel_size=1, activation=nn.ReLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.upsample(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
    

class RepBiPANNeck(nn.Module):
    def __init__(
        self, 
        c2_channels: int,
        c3_channels: int,
        c4_channels: int,
        c5_channels: int,
        width_multiple: float=0.5,
        depth_multiple: float=0.3,
        channel_outs: Optional[List[int]]=None,
        repblock_depths: Optional[List[int]]=None,
        cspsppf_poolk: int=5,
        upsample_mode: str="nearest",
        bic_with_conv: bool=False
    ):
        super(RepBiPANNeck, self).__init__()
        process_rep_depths = lambda x : max(round(x * depth_multiple), 1)
        process_out_channels = lambda x, divisor=8 : (math.ceil((x * width_multiple) / divisor) * divisor) if x else x
        repblock_depths = list(map(process_rep_depths, repblock_depths or [1, 1, 1, 1]))
        assert len(repblock_depths) == 4
        if bic_with_conv:
            channel_outs = list(
                map(process_out_channels, channel_outs or [512, 512, 512, 256, 256, 256, 256, 512, 512, 1024])
            )
            assert len(channel_outs) == 10
            bic_module = BiCwithConvModule
        else:
            channel_outs = list(
                map(process_out_channels, channel_outs or [512, 512, 256, 256, 256, 512, 512, 1024])
            )
            bic_module = BiCwithNoConvModule
            assert len(channel_outs) in [8, 10]
            if len(channel_outs) == 8:
                channel_outs.insert(1, None)
                channel_outs.insert(4, None)

        self.cspsppf0 = CSPSPPFModule(c5_channels, c5_channels, pool_kernel_size=cspsppf_poolk)
        self.conv0 = ConvBNorm(self.cspsppf0.out_channels, channel_outs[0], kernel_size=1)
        self.bic0 = bic_module(
            c4_channels, c3_channels, self.conv0.out_channels, channel_outs[1], upsample_mode=upsample_mode
        )
        self.repblock0 = RepBlock(self.bic0.out_channels, channel_outs[2], n=repblock_depths[0])
        self.conv1 = ConvBNorm(self.repblock0.out_channels, channel_outs[3], kernel_size=1)
        self.bic1 = bic_module(
            c3_channels, c2_channels, self.conv1.out_channels, channel_outs[4], upsample_mode=upsample_mode
        )
        self.repblock1 = RepBlock(self.bic1.out_channels, channel_outs[5], n=repblock_depths[1])
        self.conv2 = ConvBNorm(self.repblock1.out_channels, channel_outs[6], kernel_size=3, stride=2)
        self.concat0 = Concat(dim=1)
        self.repblock2 = RepBlock(
            self.conv2.out_channels+self.repblock0.out_channels, channel_outs[7], n=repblock_depths[2]
        )
        self.conv3 = ConvBNorm(self.repblock2.out_channels, channel_outs[8], kernel_size=3, stride=2)
        self.concat1 = Concat(dim=1)
        self.repblock3 = RepBlock(
            self.conv3.out_channels+self.cspsppf0.out_channels, channel_outs[8], n=repblock_depths[3]
        )

        self.neck_out_channels = [self.repblock1.out_channels, self.repblock2.out_channels, self.repblock3.out_channels]
    
    def forward(self, fmaps: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2, c3, c4, c5 = fmaps
        p5 = self.cspsppf0(c5)
        p4 = self.repblock0(self.bic0(c4, c3, self.conv0(p5)))
        p3 = self.repblock1(self.bic1(c3, c2, self.conv1(p4)))
        n3 = p3
        n4 = self.repblock2(self.concat0(self.conv2(n3), p4))
        n5 = self.repblock3(self.concat1(self.conv3(n4), p5))
        return n3, n4, n5


class BiPANNeck(nn.Module):
    def __init__(
            self, 
            fmap1_channels: int, 
            fmap2_channels: int, 
            fmap3_channels: int, 
            fmap4_channels: int, 
            width_multiple: float=0.5, 
            depth_multiple: float=0.3,
            channel_outs: Optional[List[int]]=None,
            c3_depths: Optional[List[int]]=None,
            sppf_poolk: int=5,
            upsample_mode: str="nearest",
            bic_with_conv: bool=False
        ):
        super(BiPANNeck, self).__init__()
        process_c3_depths = lambda x : max(round(x * depth_multiple), 1)
        process_out_channels = lambda x, divisor=8 : (math.ceil((x * width_multiple) / divisor) * divisor) if x else x
        c3_depths = list(map(process_c3_depths, c3_depths or [3, 6, 9, 3]))
        assert len(c3_depths) == 4
        if bic_with_conv:
            channel_outs = list(
                map(process_out_channels, channel_outs or [512, 512, 512, 256, 256, 256, 256, 512, 512, 1024])
            )
            assert len(channel_outs) == 10
            bic_module = BiCwithConvModule
        else:
            channel_outs = list(
                map(process_out_channels, channel_outs or [512, 512, 256, 256, 256, 512, 512, 1024])
            )
            bic_module = BiCwithNoConvModule
            assert len(channel_outs) in [8, 10]
            if len(channel_outs) == 8:
                channel_outs.insert(1, None)
                channel_outs.insert(4, None)

        self.sppf0 = SPPFModule(fmap4_channels, fmap4_channels, pool_kernel_size=sppf_poolk)
        self.conv0 = ConvBNorm(self.sppf0.out_channels, channel_outs[0], kernel_size=1)
        self.bic_w_no_conv0 = bic_module(
            fmap3_channels, fmap2_channels, self.conv0.out_channels, channel_outs[1], upsample_mode=upsample_mode
        )
        self.c3_0 = C3Module(self.bic_w_no_conv0.out_channels, channel_outs[2], num_bottlenecks=c3_depths[0])
        self.conv1 = ConvBNorm(self.c3_0.out_channels, channel_outs[3], kernel_size=1)
        self.bic_w_no_conv1 = bic_module(
            fmap2_channels, fmap1_channels, self.conv1.out_channels, channel_outs[4], upsample_mode=upsample_mode
        )
        self.c3_1 = C3Module(self.bic_w_no_conv1.out_channels, channel_outs[5], num_bottlenecks=c3_depths[1])
        self.conv2 = ConvBNorm(self.c3_1.out_channels, channel_outs[6], kernel_size=3, stride=2)
        self.concat0 = Concat(dim=1)
        self.c3_2 = C3Module(self.conv2.out_channels+self.conv1.out_channels, channel_outs[7], num_bottlenecks=c3_depths[2])
        self.conv3 = ConvBNorm(self.c3_2.out_channels, channel_outs[8], kernel_size=3, stride=2)
        self.concat1 = Concat(dim=1)
        self.c3_3 = C3Module(self.conv3.out_channels+self.conv0.out_channels, channel_outs[9], num_bottlenecks=c3_depths[3])

        self.neck_out_channels = [self.c3_1.out_channels, self.c3_2.out_channels, self.c3_3.out_channels]
    
    def forward(self, fmaps: Sequence[torch.Tensor]) -> Tuple[torch.Tensor,  torch.Tensor, torch.Tensor]:
        fmap1, fmap2, fmap3, fmap4 = fmaps
        y0 = self.conv0(self.sppf0(fmap4))
        y1 = self.c3_0(self.bic_w_no_conv0(fmap3, fmap2, y0))
        y2 = self.conv1(y1)
        n3 = self.c3_1(self.bic_w_no_conv1(fmap2, fmap1, y2))
        y4 = self.conv2(n3)
        y5 = self.c3_2(self.concat0(y4, y2))
        y6 = self.conv3(y5)
        y7 = self.concat1(y6, y0)
        y8 = self.c3_3(y7)
        return n3, y5, y8


class EffiDeHead(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            num_classes: int, 
            num_anchors: int=3,
            num_masks: Optional[int]=None,
            width_multiple: float=1.0,
            reg_fmap_depth: int=1, 
            cls_fmap_depth: int=1,
            masks_fmap_depth: Optional[int]=None,
        ):
        super(EffiDeHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        stem_out_channels = max(round(in_channels*width_multiple), 1)
        reg_fmap_depth = max(round(reg_fmap_depth), 1)
        cls_fmap_depth = max(round(cls_fmap_depth), 1)
        self.stem_layer = ConvBNorm(in_channels, stem_out_channels, kernel_size=3, stride=1)
        self.regression_fmap_layer = nn.Sequential(
            *[ConvBNorm(stem_out_channels, stem_out_channels, kernel_size=3, stride=1) for _ in range(0, reg_fmap_depth+1)]
        )
        self.classification_fmap_layer = nn.Sequential(
            *[ConvBNorm(stem_out_channels, stem_out_channels, kernel_size=3, stride=1) for _ in range(0, cls_fmap_depth)]
        )
        self.conf_layer = nn.Conv2d(stem_out_channels, num_anchors, kernel_size=1)
        self.cls_layer = nn.Conv2d(stem_out_channels, num_anchors * num_classes, kernel_size=1)
        self.bbox_layer = nn.Conv2d(stem_out_channels, num_anchors * 4, kernel_size=1)
        if num_masks:
            masks_fmap_depth = max(round(masks_fmap_depth or 1), 1)
            self.mask_fmap_layer = nn.Sequential(
                *[ConvBNorm(stem_out_channels, stem_out_channels, kernel_size=3, stride=1) for _ in range(0, masks_fmap_depth)]
            )
            self.masks_layer = nn.Conv2d(stem_out_channels, num_anchors * num_masks, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, ny, nx = x.shape
        stem = self.stem_layer(x)
        conf = self.conf_layer(self.regression_fmap_layer(stem))
        bbox = self.bbox_layer(self.regression_fmap_layer(stem))
        cls = self.cls_layer(self.classification_fmap_layer(stem))
        masks = None
        if hasattr(self, "masks_layer"):
            masks = self.masks_layer(self.mask_fmap_layer(stem))
        conf = conf.permute(0, 2, 3, 1).reshape(batch_size, ny, nx, self.num_anchors, 1)
        cls = cls.permute(0, 2, 3, 1).reshape(batch_size, ny, nx, self.num_anchors, self.num_classes)
        bbox = bbox.permute(0, 2, 3, 1).reshape(batch_size, ny, nx, self.num_anchors, 4)
        if not torch.is_tensor(masks):
            output = torch.cat([conf, cls, bbox], dim=-1)
        else:
            masks = masks.permute(0, 2, 3, 1).reshape(batch_size, ny, nx, self.num_anchors, -1)
            output = torch.cat([conf, cls, bbox, masks], dim=-1)
        # shape: [batch_size, ny, nx, num_anchors, 5+(num_classes+(num_masks or 0))]
        return output
    

class BasicHead(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            num_classes: int, 
            num_anchors: int=3,
            num_masks: Optional[int]=None,
            width_multiple: float=1.0
        ):
        super(BasicHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        stem_out_channels = max(round(in_channels*width_multiple), 1)
        self.stem_layer = ConvBNorm(in_channels, stem_out_channels, kernel_size=3, stride=1)
        self.conv = nn.Conv2d(
            stem_out_channels, 
            out_channels=(num_anchors * (5 + self.num_classes + (num_masks or 0))), 
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, ny, nx = x.shape
        output = self.stem_layer(x)
        output = self.conv(output)
        output = output.permute(0, 2, 3, 1).reshape(batch_size, ny, nx, self.num_anchors, -1)
        # shape: [batch_size, ny, nx, num_anchors, 5+num_classes+(num_mask or 0)]
        return output
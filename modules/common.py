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
            no_batchnorm: bool=False,
            batchnorm_first: bool=True
        ):
        super(ConvBNorm, self).__init__()

        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = [i//2 for i in kernel_size]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._batchnorm_first = batchnorm_first
        self.conv = nn.Conv2d(
            in_channels,
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=bias
        )
        self.norm = nn.BatchNorm2d(out_channels) if (not no_batchnorm) else nn.Identity()
        self.activation = activation() if (activation is not None) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self._batchnorm_first:
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.norm(x)
        return x

    
class ConvTransposeBNorm(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: Union[int, Tuple[int, int]], 
            stride: Union[int, Tuple[int, int]]=1, 
            padding: Optional[Union[int, Tuple[int, int]]]=None,
            activation: Optional[Type]=nn.SiLU,
            bias: bool=True,
            no_batchnorm: bool=False,
            batchnorm_first: bool=True
        ):
        super(ConvTransposeBNorm, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = padding or 0
        self._batchnorm_first = batchnorm_first

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )
        self.norm = nn.BatchNorm2d(out_channels) if (not no_batchnorm) else nn.Identity()
        self.activation = activation() if (activation is not None) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        if self._batchnorm_first:
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.norm(x)
        return x


class ConvBNormUpsample(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            scale: float, 
            upsample_mode: str="nearest",
            activation: Optional[Type]=nn.SiLU, 
            no_batchnorm: bool=False,
            batchnorm_first: bool=True
        ):
        super(ConvBNormUpsample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.upsample = nn.Upsample(scale_factor=self.scale, mode=upsample_mode)
        self.conv = ConvBNorm(
            self.in_channels, 
            self.out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            activation=activation, 
            no_batchnorm=no_batchnorm,
            batchnorm_first=batchnorm_first
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.upsample(x)
        return x


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *x: torch.Tensor):
        return torch.cat(x, dim=self.dim)
    

class RepVGGBlock(nn.Module):
    # RepVGG: Reparameterized Visual Geometry Group
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
    # Rep: Reparameterized
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
    # BIC: Bi-directional Concatenation
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
    # BIC: Bi-directional Concatenation
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
    # C3: Class Conditional Coordinates
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
    # SPPF: Spartial Pyramid Pooling Feature
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
    # CSPSPPF: Cross Stage Partial Spartial Pyramid Pooling Feature
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
    

class CSPNet(nn.Module):
    # CSP: Cross Stage Partial Network
    def __init__(
            self, 
            in_channels: int, 
            width_multiple: float=0.5, 
            depth_multiple: float=0.3,
            dropout: float=0.0
        ):
        super(CSPNet, self).__init__()
        self.in_channels = in_channels
        process_c3_depths = lambda x : max(round(x * depth_multiple), 1)
        process_out_channels = lambda x, divisor=8 : math.ceil((x * width_multiple) / divisor) * divisor
        c3_depths = list(map(process_c3_depths, [3, 6, 9, 3]))
        channel_outs = list(map(process_out_channels, [32, 64, 128, 256, 256, 512, 512, 1024, 1024]))
        assert len(c3_depths) == 4 and len(channel_outs) == 9

        self.conv0 = ConvBNorm(in_channels, channel_outs[0], kernel_size=6, stride=2, padding=2)
        self.conv1 = ConvBNorm(self.conv0.out_channels, channel_outs[1], kernel_size=3, stride=2, padding=1)
        self.c3_0 = C3Module(self.conv1.out_channels, channel_outs[2], num_bottlenecks=c3_depths[0])
        self.conv2 = ConvBNorm(self.c3_0.out_channels, channel_outs[3], kernel_size=3, stride=2, padding=1)
        self.c3_1 = C3Module(self.conv2.out_channels, channel_outs[4], num_bottlenecks=c3_depths[1])
        self.conv3 = ConvBNorm(self.c3_1.out_channels, channel_outs[5], kernel_size=3, stride=2, padding=1)
        self.c3_2 = C3Module(self.conv3.out_channels, channel_outs[6], num_bottlenecks=c3_depths[2])
        self.conv4 = ConvBNorm(self.c3_2.out_channels, channel_outs[7], kernel_size=3, stride=2, padding=1)
        self.c3_3 = C3Module(self.conv4.out_channels, channel_outs[8], num_bottlenecks=c3_depths[3])
        self.dropout = nn.Dropout(dropout)
        self.out_fmaps_channels = (
            self.c3_0.out_channels, self.c3_1.out_channels, self.c3_2.out_channels, self.c3_3.out_channels
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[2] % 32 != 0 or x.shape[3] % 32 != 0:
            raise Exception("input must have width and height divisible by 32")
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.dropout(out)

        fmap1 = self.c3_0(out)
        out = self.conv2(fmap1)
        out = self.dropout(out)

        fmap2 = self.c3_1(out)
        out = self.conv3(fmap2)
        out = self.dropout(out)

        fmap3 = self.c3_2(out)
        out = self.conv4(fmap3)

        fmap4 = self.c3_3(out)

        return fmap1, fmap2, fmap3, fmap4
    

class DeconvCSPNet(nn.Module):
    def __init__(
            self, 
            fmap1_channels: int, 
            fmap2_channels: int,
            fmap3_channels: int,
            fmap4_channels: int,
            out_channels: int,
            width_multiple: float=0.5, 
            depth_multiple: float=0.3,
            dropout: float=0.0
        ):
        super(DeconvCSPNet, self).__init__()
        process_c3_depths = lambda x : max(round(x * depth_multiple), 1)
        process_out_channels = lambda x, divisor=8 : math.ceil((x * width_multiple) / divisor) * divisor
        c3_depths = list(map(process_c3_depths, [3, 9, 6, 3]))
        channel_outs = list(map(process_out_channels, [1024, 1024, 512, 512, 256, 256, 128, 64]))
        assert len(c3_depths) == 4 and len(channel_outs) == 8
        
        self.out_channels = out_channels

        self.c3_0 = C3Module(fmap1_channels, channel_outs[0], num_bottlenecks=c3_depths[0])
        self.deconv0 = ConvBNormUpsample(self.c3_0.out_channels, channel_outs[1], scale=2)
        self.concat0 = Concat(dim=1)

        self.c3_1 = C3Module(self.deconv0.out_channels+fmap2_channels, channel_outs[2], num_bottlenecks=c3_depths[1])
        self.deconv1 = ConvBNormUpsample(self.c3_1.out_channels, channel_outs[3], scale=2)
        self.concat1 = Concat(dim=1)

        self.c3_2 = C3Module(self.deconv1.out_channels+fmap3_channels, channel_outs[4], num_bottlenecks=c3_depths[2])
        self.deconv2 = ConvBNormUpsample(self.c3_2.out_channels, channel_outs[5], scale=2)
        self.concat2 = Concat(dim=1)

        self.c3_3 = C3Module(self.deconv2.out_channels+fmap4_channels, channel_outs[6], num_bottlenecks=c3_depths[3])
        self.deconv3 = ConvBNormUpsample(self.c3_3.out_channels, channel_outs[7], scale=2)

        self.deconv4 = ConvBNormUpsample(
            self.deconv3.out_channels, self.out_channels, scale=2, no_batchnorm=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, fmaps: Sequence[torch.Tensor]) -> torch.Tensor:
        fmap1, fmap2, fmap3, fmap4 = fmaps
        out = self.c3_0(fmap1)
        out = self.deconv0(out)
        out = self.dropout(out)

        out = self.c3_1(self.concat0(out, fmap2))
        out = self.deconv1(out)
        out = self.dropout(out)

        out = self.c3_2(self.concat1(out, fmap3))
        out = self.deconv2(out)
        out = self.dropout(out)

        out = self.c3_3(self.concat2(out, fmap4))
        out = self.deconv3(out)

        out = self.deconv4(out)
        return out


class ProtoSegModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int=32, c_h: int=256, upsample_mode: str="nearest"):
        super(ProtoSegModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBNorm(self.in_channels, c_h, kernel_size=3)
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv2 = ConvBNorm(c_h, c_h, kernel_size=3)
        self.conv3 = ConvBNorm(c_h, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.upsample(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
    

class RepBiPAN(nn.Module):
    # RepBiPAN: Reparameterized Bi-directional Path Aggregated Network
    def __init__(
        self, 
        c2_channels: int,
        c3_channels: int,
        c4_channels: int,
        c5_channels: int,
        width_multiple: float=0.5,
        depth_multiple: float=0.3,
        cspsppf_poolk: int=5,
        upsample_mode: str="nearest",
        bic_with_conv: bool=False
    ):
        super(RepBiPAN, self).__init__()
        process_rep_depths = lambda x : max(round(x * depth_multiple), 1)
        process_out_channels = lambda x, divisor=8 : (math.ceil((x * width_multiple) / divisor) * divisor) if x else x
        repblock_depths = list(map(process_rep_depths, [1, 1, 1, 1]))
        assert len(repblock_depths) == 4
        if bic_with_conv:
            channel_outs = list(
                map(process_out_channels, [512, 512, 512, 256, 256, 256, 256, 512, 512, 1024])
            )
            assert len(channel_outs) == 10
            bic_module = BiCwithConvModule
        else:
            channel_outs = list(
                map(process_out_channels, [512, 512, 256, 256, 256, 512, 512, 1024])
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
            self.conv3.out_channels+self.cspsppf0.out_channels, channel_outs[9], n=repblock_depths[3]
        )

        self.out_fmaps_channels = (
            c2_channels, self.repblock1.out_channels, self.repblock2.out_channels, self.repblock3.out_channels
        )
    
    def forward(self, fmaps: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        c2, c3, c4, c5 = fmaps
        p5 = self.cspsppf0(c5)
        p4 = self.repblock0(self.bic0(c4, c3, self.conv0(p5)))
        p3 = self.repblock1(self.bic1(c3, c2, self.conv1(p4)))
        n3 = p3
        n4 = self.repblock2(self.concat0(self.conv2(n3), p4))
        n5 = self.repblock3(self.concat1(self.conv3(n4), p5))
        return c2, n3, n4, n5
    

class DeconvRepBiPAN(nn.Module):
    # DeconvRepBiPAN: Deconvolution Reparameterized Bi-directional Path Aggregated Network
    def __init__(
        self, 
        c2_channels: int,
        n3_channels: int,
        n4_channels: int,
        n5_channels: int,
        width_multiple: float=0.5,
        depth_multiple: float=0.3,
        cspsppf_poolk: int=5,
        upsample_mode: str="nearest",
        bic_with_conv: bool=False
    ):
        super(DeconvRepBiPAN, self).__init__()
        process_rep_depths = lambda x : max(round(x * depth_multiple), 1)
        process_out_channels = lambda x, divisor=8 : (math.ceil((x * width_multiple) / divisor) * divisor) if x else x
        repblock_depths = list(map(process_rep_depths, [1, 1, 1, 1]))
        assert len(repblock_depths) == 4
        if bic_with_conv:
            channel_outs = list(
                map(process_out_channels, [256, 256, 256, 512, 512, 512, 512, 256, 256, 128])
            )
            assert len(channel_outs) == 10
            bic_module = BiCwithConvModule
        else:
            channel_outs = list(
                map(process_out_channels, [256, 256, 512, 512, 512, 256, 256, 128])
            )
            bic_module = BiCwithNoConvModule
            assert len(channel_outs) in [8, 10]
            if len(channel_outs) == 8:
                channel_outs.insert(1, None)
                channel_outs.insert(4, None)

        self.deconv0 = ConvBNorm(c2_channels, channel_outs[0], kernel_size=1)
        self.bic0 = bic_module(
            n3_channels, self.deconv0.out_channels, n4_channels, channel_outs[1], upsample_mode=upsample_mode
        )
        self.repblock0 = RepBlock(self.bic0.out_channels, channel_outs[2], repblock_depths[0])
        
        self.deconv1 = ConvBNorm(self.repblock0.out_channels, channel_outs[3], kernel_size=1)
        self.bic1 = bic_module(
            n4_channels, self.deconv1.out_channels, n5_channels, channel_outs[4], upsample_mode=upsample_mode
        )
        self.repblock1 = RepBlock(self.bic1.out_channels, channel_outs[5], repblock_depths[1])

        self.cspsppf = CSPSPPFModule(self.repblock1.out_channels, self.repblock1.out_channels, pool_kernel_size=cspsppf_poolk)
        
        self.deconv2 = ConvBNormUpsample(self.cspsppf.out_channels, channel_outs[6], scale=2)
        self.concat0 = Concat(dim=1)

        self.repblock2 =  RepBlock(
            self.deconv2.out_channels+self.repblock0.out_channels, channel_outs[7], repblock_depths[2]
        )
        
        self.deconv3 = ConvBNormUpsample(self.repblock2.out_channels, channel_outs[8], scale=2)
        self.concat1 = Concat(dim=1)
        self.repblock3 = RepBlock(self.deconv3.out_channels+c2_channels, channel_outs[9], repblock_depths[3])

        self.out_fmaps_channels = (
            n5_channels, self.cspsppf.out_channels, self.repblock2.out_channels, self.repblock3.out_channels
        )

    def forward(self, fmaps: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        c2, n3, n4, n5 = fmaps
        q2 = c2
        q3 = self.repblock0(self.bic0(n3, self.deconv0(q2), n4))
        q4 = self.repblock1(self.bic1(n4, self.deconv1(q3), n5))
        f4 = self.cspsppf(q4)
        f3 = self.repblock2(self.concat0(self.deconv2(f4), q3))
        f2 = self.repblock3(self.concat1(self.deconv3(f3), q2))
        return n5, f4, f3, f2


class BiPAN(nn.Module):
    # BiPAN: Bi-directional Path Aggregated Network
    def __init__(
            self, 
            fmap1_channels: int,
            fmap2_channels: int, 
            fmap3_channels: int, 
            fmap4_channels: int, 
            width_multiple: float=0.5, 
            depth_multiple: float=0.3,
            sppf_poolk: int=5,
            upsample_mode: str="nearest",
            bic_with_conv: bool=False
        ):
        super(BiPAN, self).__init__()
        process_c3_depths = lambda x : max(round(x * depth_multiple), 1)
        process_out_channels = lambda x, divisor=8 : (math.ceil((x * width_multiple) / divisor) * divisor) if x else x
        c3_depths = list(map(process_c3_depths, [3, 6, 9, 3]))
        assert len(c3_depths) == 4
        if bic_with_conv:
            channel_outs = list(
                map(process_out_channels, [512, 512, 512, 256, 256, 256, 256, 512, 512, 1024])
            )
            assert len(channel_outs) == 10
            bic_module = BiCwithConvModule
        else:
            channel_outs = list(
                map(process_out_channels, [512, 512, 256, 256, 256, 512, 512, 1024])
            )
            bic_module = BiCwithNoConvModule
            assert len(channel_outs) in [8, 10]
            if len(channel_outs) == 8:
                channel_outs.insert(1, None)
                channel_outs.insert(4, None)

        self.sppf0 = SPPFModule(fmap4_channels, fmap4_channels, pool_kernel_size=sppf_poolk)
        self.conv0 = ConvBNorm(self.sppf0.out_channels, channel_outs[0], kernel_size=1)
        self.bic0 = bic_module(
            fmap3_channels, fmap2_channels, self.conv0.out_channels, channel_outs[1], upsample_mode=upsample_mode
        )
        self.c3_0 = C3Module(self.bic0.out_channels, channel_outs[2], num_bottlenecks=c3_depths[0])
        self.conv1 = ConvBNorm(self.c3_0.out_channels, channel_outs[3], kernel_size=1)
        self.bic1 = bic_module(
            fmap2_channels, fmap1_channels, self.conv1.out_channels, channel_outs[4], upsample_mode=upsample_mode
        )
        self.c3_1 = C3Module(self.bic1.out_channels, channel_outs[5], num_bottlenecks=c3_depths[1])
        self.conv2 = ConvBNorm(self.c3_1.out_channels, channel_outs[6], kernel_size=3, stride=2)
        self.concat0 = Concat(dim=1)
        self.c3_2 = C3Module(self.conv2.out_channels+self.conv1.out_channels, channel_outs[7], num_bottlenecks=c3_depths[2])
        self.conv3 = ConvBNorm(self.c3_2.out_channels, channel_outs[8], kernel_size=3, stride=2)
        self.concat1 = Concat(dim=1)
        self.c3_3 = C3Module(self.conv3.out_channels+self.conv0.out_channels, channel_outs[9], num_bottlenecks=c3_depths[3])

        self.out_fmaps_channels = (
            fmap1_channels, self.c3_1.out_channels, self.c3_2.out_channels, self.c3_3.out_channels
        )
    
    def forward(self, fmaps: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        fmap1, fmap2, fmap3, fmap4 = fmaps
        y0 = self.conv0(self.sppf0(fmap4))
        c2 = self.c3_0(self.bic0(fmap3, fmap2, y0))

        y2 = self.conv1(c2)
        y3 = self.c3_1(self.bic1(fmap2, fmap1, y2))

        y4 = self.conv2(y3)
        y5 = self.c3_2(self.concat0(y4, y2))

        y6 = self.conv3(y5)
        y7 = self.c3_3(self.concat1(y6, y0))

        return fmap1, y3, y5, y7
    

class DeconvBiPAN(nn.Module):
    # DeconvBiPAN: Deconvolution Bi-directional Path Aggregated Network
    def __init__(
            self, 
            fmap1_channels: int,
            y3_channels: int, 
            y5_channels: int, 
            y7_channels: int, 
            width_multiple: float=0.5, 
            depth_multiple: float=0.3,
            sppf_poolk: int=5,
            upsample_mode: str="nearest",
            bic_with_conv: bool=False
        ):
        super(DeconvBiPAN, self).__init__()
        process_c3_depths = lambda x : max(round(x * depth_multiple), 1)
        process_out_channels = lambda x, divisor=8 : (math.ceil((x * width_multiple) / divisor) * divisor) if x else x
        c3_depths = list(map(process_c3_depths, [3, 6, 9, 3]))
        assert len(c3_depths) == 4
        if bic_with_conv:
            channel_outs = list(
                map(process_out_channels, [256, 256, 256, 512, 512, 512, 512, 256, 256, 128])
            )
            assert len(channel_outs) == 10
            bic_module = BiCwithConvModule
        else:
            channel_outs = list(
                map(process_out_channels, [256, 256, 512, 512, 512, 256, 256, 128])
            )
            bic_module = BiCwithNoConvModule
            assert len(channel_outs) in [8, 10]
            if len(channel_outs) == 8:
                channel_outs.insert(1, None)
                channel_outs.insert(4, None)

        self.deconv0 = ConvBNorm(fmap1_channels, channel_outs[0], kernel_size=1)
        self.bic0 = bic_module(
            y3_channels, self.deconv0.out_channels, y5_channels, channel_outs[1], upsample_mode=upsample_mode
        )
        self.c3_0 = C3Module(self.bic0.out_channels, channel_outs[2], num_bottlenecks=c3_depths[0])

        self.deconv1 = ConvBNorm(self.c3_0.out_channels, channel_outs[3], kernel_size=1)
        self.bic1 = bic_module(
            y5_channels, self.deconv1.out_channels, y7_channels, channel_outs[4], upsample_mode=upsample_mode
        )
        self.c3_1 = C3Module(self.bic1.out_channels, channel_outs[5], num_bottlenecks=c3_depths[1])

        self.sppf = SPPFModule(self.c3_1.out_channels, self.c3_1.out_channels, pool_kernel_size=sppf_poolk)

        self.deconv2 = ConvBNormUpsample(self.sppf.out_channels, channel_outs[6], scale=2)
        self.concat0 = Concat(dim=1)
        self.c3_2 = C3Module(
            self.deconv2.out_channels+self.deconv1.out_channels, channel_outs[7], num_bottlenecks=c3_depths[2]
        )

        self.deconv3 = ConvBNormUpsample(self.c3_2.out_channels, channel_outs[8], scale=2)
        self.concat1 = Concat(dim=1)
        self.c3_3 = C3Module(
            self.deconv3.out_channels+self.deconv0.out_channels, channel_outs[9], num_bottlenecks=c3_depths[3]
        )

        self.out_fmaps_channels = (
            y7_channels, self.c3_1.out_channels, self.c3_2.out_channels, self.c3_3.out_channels
        )

    def forward(self, fmaps: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        fmap1, y3, y5, y7 = fmaps
        f0 = self.deconv0(fmap1)
        f1 = self.c3_0(self.bic0(y3, f0, y5))

        f2 = self.deconv1(f1)
        f3 = self.c3_1(self.bic1(y5, f2, y7))

        f4 = self.deconv2(self.sppf(f3))
        f5 = self.c3_2(self.concat0(f4, f2))

        f6 = self.deconv3(f5)
        f7 = self.c3_3(self.concat1(f6, f0))

        return y7, f3, f5, f7


class EffiDecHead(nn.Module):
    # EffiDecHead: Efficient Deoupling Head
    def __init__(
            self, 
            in_channels: int, 
            num_classes: int, 
            num_anchors: int=3,
            num_masks: Optional[int]=None,
            num_keypoints: Optional[int]=None,
            width_multiple: float=1.0,
            reg_fmap_depth: int=1, 
            cls_fmap_depth: int=1,
            masks_fmap_depth: Optional[int]=None,
            keypoints_fmap_depth: Optional[int]=None
        ):
        super(EffiDecHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_masks = num_masks
        self.num_keypoints = num_keypoints
        stem_out_channels = max(round(in_channels*width_multiple), 1)
        reg_fmap_depth = max(round(reg_fmap_depth), 1)
        cls_fmap_depth = max(round(cls_fmap_depth), 1)
        self.stem_layer = ConvBNorm(in_channels, stem_out_channels, kernel_size=3, stride=1)

        _fmap_layer = lambda : ConvBNorm(stem_out_channels, stem_out_channels, 3, 1)

        self.regression_fmap_layer = nn.Sequential(
            *[_fmap_layer() for _ in range(0, reg_fmap_depth+1)]
        )
        self.classification_fmap_layer = nn.Sequential(
            *[_fmap_layer() for _ in range(0, cls_fmap_depth)]
        )
        self.conf_layer = nn.Conv2d(stem_out_channels, num_anchors, kernel_size=1)
        self.cls_layer = nn.Conv2d(stem_out_channels, num_anchors * num_classes, kernel_size=1)
        self.bbox_layer = nn.Conv2d(stem_out_channels, num_anchors * 4, kernel_size=1)

        if num_masks:
            masks_fmap_depth = max(round(masks_fmap_depth or 1), 1)
            self.mask_fmap_layer = nn.Sequential(
                *[_fmap_layer() for _ in range(0, masks_fmap_depth)]
            )
            self.masks_layer = nn.Conv2d(stem_out_channels, num_anchors * num_masks, kernel_size=1)

        if num_keypoints:
            keypoints_fmap_depth = max(round(keypoints_fmap_depth or 1), 1)
            self.keypoints_fmap_layer = nn.Sequential(
                *[_fmap_layer() for _ in range(0, keypoints_fmap_depth)]
            )
            # multiply num_keypoints by 3 because each keypoint comprises [x, y, visible, occluded, deleted]
            self.keypoints_layer = nn.Conv2d(stem_out_channels, num_anchors*5*num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, ny, nx = x.shape
        stem = self.stem_layer(x)
        conf = self.conf_layer(self.regression_fmap_layer(stem))
        bbox = self.bbox_layer(self.regression_fmap_layer(stem))
        cls = self.cls_layer(self.classification_fmap_layer(stem))

        _permute_resize = lambda x, last_dim : (
            x.permute(0, 2, 3, 1)
            .reshape(batch_size, ny, nx, self.num_anchors, last_dim)
        )
        conf = _permute_resize(conf, 1)
        cls = _permute_resize(cls, self.num_classes)
        bbox = _permute_resize(bbox, 4)
        output = torch.cat([conf, cls, bbox], dim=-1)

        if hasattr(self, "masks_layer"):
            masks = self.masks_layer(self.mask_fmap_layer(stem))
            masks =_permute_resize(masks, self.num_masks)
            output = torch.cat([output, masks], dim=-1)
            
        if hasattr(self, "keypoints_layer"):
            keypoints = self.keypoints_layer(self.keypoints_fmap_layer(stem))
            keypoints = _permute_resize(keypoints, 5*self.num_keypoints)
            output = torch.cat([output, keypoints], dim=-1)
        # shape: [batch_size, ny, nx, num_anchors, (1 + num_classes + 4 + num_masks + (5*num_keypoints))]
        return output
    

class BasicHead(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            num_classes: int,
            num_anchors: int=3,
            num_masks: Optional[int]=None,
            num_keypoints: Optional[int]=None,
            width_multiple: float=1.0
        ):
        super(BasicHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_masks = num_masks
        self.num_keypoints = num_keypoints
        stem_out_channels = max(round(in_channels*width_multiple), 1)
        self.stem_layer = ConvBNorm(in_channels, stem_out_channels, kernel_size=3, stride=1)
        # multiply num keypoints by 3 because each keypoint comprises [x, y, visible, occluded, deleted]
        out_channels = num_anchors * (5 + self.num_classes + (self.num_masks or 0) + (self.num_keypoints or 0)*5)
        self.conv = nn.Conv2d(
            stem_out_channels, 
            out_channels=out_channels, 
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, ny, nx = x.shape
        output = self.stem_layer(x)
        output = self.conv(output)
        output = output.permute(0, 2, 3, 1).reshape(batch_size, ny, nx, self.num_anchors, -1)
        # shape: [batch_size, ny, nx, num_anchors, (1 + num_classes + 4 + num_masks + (5*num_keypoints))]
        return output
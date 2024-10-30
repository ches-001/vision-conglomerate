import math
import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from .common import ConvBNorm, C3Module, SPPFModule
from typing import *

class CSPBasedBackBone(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            width_multiple: float=0.5, 
            depth_multiple: float=0.3,
            channel_outs: Optional[List[int]]=None,
            c3_depths: Optional[List[int]]=None,
            dropout: float=0.0
        ):
        super(CSPBasedBackBone, self).__init__()
        self.in_channels = in_channels
        process_c3_depths = lambda x : max(round(x * depth_multiple), 1)
        process_out_channels = lambda x, divisor=8 : math.ceil((x * width_multiple) / divisor) * divisor
        c3_depths = list(map(process_c3_depths, c3_depths or [3, 6, 9, 3]))
        channel_outs = list(map(process_out_channels, channel_outs or [32, 64, 128, 256, 256, 512, 512, 1024, 1024]))
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
        self.bbone_out_channels = (
            self.c3_0.out_channels, self.c3_1.out_channels, self.c3_2.out_channels, self.c3_3.out_channels
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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



class ResNetBackBone(ResNet):
    def __init__(
        self, 
        in_channels: int, 
        dropout: float=0.0, 
        block: Union[str, Type]=BasicBlock, 
        block_layers: Optional[Iterable[int]]=None
    ):
        if isinstance(block, str):
            block = getattr(resnet, block)
        super(ResNetBackBone, self).__init__(block=block, layers=block_layers or [3, 4, 6, 3])
        self.in_channels = in_channels  
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.dropout = nn.Dropout(dropout)

        if block == BasicBlock:
            self.bbone_out_channels = (64, 128, 256, 512)
        elif block == Bottleneck:
            self.bbone_out_channels = (256, 512, 1024, 2048)
        #delete unwanted layers
        del self.maxpool, self.fc, self.avgpool

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        fmap1 = self.layer1(x)
        fmap2 = self.layer2(fmap1)
        fmap3 = self.layer3(fmap2)
        fmap4 = self.layer4(fmap3)
        return fmap1, fmap2, fmap3, fmap4
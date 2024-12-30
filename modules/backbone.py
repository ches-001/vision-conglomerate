import math
import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from .common import CSPNet
from typing import *

class CSPBackBone(CSPNet):
    def __init__(self, *args, **kwargs):
        super(CSPBackBone, self).__init__(*args, **kwargs)
        

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
            self.out_fmaps_channels = (64, 128, 256, 512)
        elif block == Bottleneck:
            self.out_fmaps_channels = (256, 512, 1024, 2048)
        #delete unwanted layers
        del self.maxpool, self.fc, self.avgpool

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[2] % 32 != 0 or x.shape[3] % 32 != 0:
            raise Exception("input must have width and height divisible by 32")
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
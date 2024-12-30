import torch
import torchvision
import torchvision.transforms.functional
from . import common
import torch.nn as nn
from typing import Dict, Any, Sequence, List, Optional, Tuple


class BaseTrackNetEncoder(nn.Module):
    def __init__(self, in_channels: int, width_multiple: int=1):
        super(BaseTrackNetEncoder, self).__init__()
        channel_outs = list(map(
            lambda x : max(round(x * width_multiple), 1), 
            [64, 64, 128, 128, 256, 256, 256, 512, 512, 512]
        ))
        self._enc_modules = nn.ModuleList([
            common.ConvBNorm(in_channels, channel_outs[0], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(channel_outs[0], channel_outs[1], 3, 1, 1, activation=nn.ReLU),
            nn.MaxPool2d(2, 2),
            common.ConvBNorm(channel_outs[1], channel_outs[2], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(channel_outs[2], channel_outs[3], 3, 1, 1, activation=nn.ReLU),
            nn.MaxPool2d(2, 2),
            common.ConvBNorm(channel_outs[3], channel_outs[4], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(channel_outs[4], channel_outs[5], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(channel_outs[5], channel_outs[6], 3, 1, 1, activation=nn.ReLU),
            nn.MaxPool2d(2, 2),
            common.ConvBNorm(channel_outs[6], channel_outs[7], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(channel_outs[7], channel_outs[8], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(channel_outs[8], channel_outs[9], 3, 1, 1, activation=nn.ReLU),
        ])
        self.out_fmaps_channels = [channel_outs[1], channel_outs[3], channel_outs[6], channel_outs[9]]

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        fmaps = []
        fmap_layer_idxs = [1, 3, 6]
        for i in range(0, len(self._enc_modules)):
            x = self._enc_modules[i](x)
            if i in fmap_layer_idxs:
                fmaps.append(x)
        fmaps.append(x)
        return fmaps
    

class BaseTrackNetDecoder(nn.Module):
    def __init__(self, in_fmaps_channels: List[int], out_channels: int, width_multiple: int=1):
        super(BaseTrackNetDecoder, self).__init__()
        channel_outs = list(map(
            lambda x : max(round(x * width_multiple), 1), 
            [256, 256, 256, 126, 128, 64, 64]
        ))
        self._dec_modules = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            common.Concat(dim=1),
            common.ConvBNorm(in_fmaps_channels[3]+in_fmaps_channels[2], channel_outs[0], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(channel_outs[0], channel_outs[1], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(channel_outs[1], channel_outs[2], 3, 1, 1, activation=nn.ReLU),
            nn.Upsample(scale_factor=2),
            common.Concat(dim=1),
            common.ConvBNorm(in_fmaps_channels[1]+channel_outs[2], channel_outs[3], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(channel_outs[3], channel_outs[4], 3, 1, 1, activation=nn.ReLU),
            nn.Upsample(scale_factor=2),
            common.Concat(dim=1),
            common.ConvBNorm(in_fmaps_channels[0]+channel_outs[4], channel_outs[5], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(channel_outs[5], channel_outs[6], 3, 1, 1, activation=nn.ReLU),
            common.ConvBNorm(64, out_channels, 3, 1, 1, activation=nn.ReLU, no_batchnorm=True),
        ])

    def forward(self, fmaps: Sequence[torch.Tensor]) -> torch.Tensor:
        x = fmaps[3]
        concat_idx = 2
        for i in range(0, len(self._dec_modules)):
            if isinstance(self._dec_modules[i], common.Concat):
                x = self._dec_modules[i](x, fmaps[concat_idx])
                concat_idx -= 1
                continue
            x = self._dec_modules[i](x)
        return x


class AdvTrackNetEncoder(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            encoder_modules: List[str], 
            config: Dict[str, Any]
        ):
        super(AdvTrackNetEncoder, self).__init__()
        assert len(encoder_modules) == 2

        self.in_channels = in_channels
        self.enc_module_p1 = getattr(common, encoder_modules[0])(
            self.in_channels, 
            **config.get(encoder_modules[0].lower()+"_config", {})
        )
        self.enc_module_p2 = getattr(common, encoder_modules[1])(
            *self.enc_module_p1.out_fmaps_channels, 
            **config.get(encoder_modules[1].lower()+"_config", {})
        )

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        fmaps = self.enc_module_p1(x)
        fmaps = self.enc_module_p2(fmaps)
        return fmaps


class AdvTrackNetDecoder(nn.Module):
    def __init__(
            self, 
            out_channels: int, 
            in_fmaps_channels: List[int], 
            decoder_modules: List[str], 
            config: Dict[str, Any]
        ):
        super(AdvTrackNetDecoder, self).__init__()
        assert len(decoder_modules) == 2

        self.out_channels = out_channels
        self.dec_module_p1 = getattr(common, decoder_modules[0])(
            *in_fmaps_channels, 
            **config.get(decoder_modules[0].lower()+"_config", {})
        )
        self.dec_module_p2 = getattr(common, decoder_modules[1])(
            *self.dec_module_p1.out_fmaps_channels, 
            out_channels,
            **config.get(decoder_modules[1].lower()+"_config", {})
        )

    def forward(self, fmaps: Sequence[torch.Tensor]) -> torch.Tensor:
        fmaps = self.dec_module_p1(fmaps)
        y = self.dec_module_p2(fmaps)
        return y


class TrackNet(nn.Module):
    def __init__(self, in_channels: int, config: Dict[str, Any]):
        super(TrackNet, self).__init__()
        self.in_channels = in_channels
        architecture = config["architecture"]
        weight_init = config["weight_init"]

        if architecture == "advanced":
            _config = config["advanced_arch_config"]
            self.encoder = AdvTrackNetEncoder(
                self.in_channels, 
                _config["encoder_modules"], 
                _config["encoder_config"]
            )
            self.decoder = AdvTrackNetDecoder(
                256,
                self.encoder.enc_module_p2.out_fmaps_channels, 
                _config["decoder_modules"], 
                _config["decoder_config"]
            )
        elif architecture == "base":
            _config = config["base_arch_config"]
            self.encoder = BaseTrackNetEncoder(
                self.in_channels, **_config["encoder_config"]
            )
            self.decoder = BaseTrackNetDecoder(
                self.encoder.out_fmaps_channels, 256, **_config["decoder_config"]
            )
        else:
            raise Exception(f"Only base and advanced architectures are supported, got {architecture}")
        
        if weight_init == "uniform":
            self.apply(self._uniform_init_weights)
        elif weight_init == "xavier":
            self.apply(self._xavier_init_weights)
        else:
            raise Exception(f"Only 'uniform' and 'xavier' init supported, got {weight_init}")

    def forward(
            self, 
            x: torch.Tensor, 
            inference: bool=False,
            og_size: Optional[Tuple[int, int]]=None
        ) -> torch.Tensor:
        enc_fmaps: torch.Tensor = self.encoder(x)
        y: torch.Tensor = self.decoder(enc_fmaps)
        y = y.permute(0, 2, 3, 1).contiguous()
        if inference:
            y = y.argmax(dim=3).to(device=y.device, dtype=torch.uint8)
            if og_size is not None:
                y = torchvision.transforms.functional.resize(
                    y.unsqueeze(dim=1), size=og_size, antialias=True
                ).squeeze(dim=1)
        return y
    
    def _xavier_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)

    def _uniform_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.uniform_(m.weight, -0.05, 0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def inference(self):
        self.eval()
        def toggle_inference_mode(m: nn.Module):
            if isinstance(m, common.RepVGGBlock):
                if (
                    isinstance(m.identity, (nn.BatchNorm2d, nn.Identity)) and 
                    isinstance(m.conv1x1.norm, nn.BatchNorm2d) and 
                    isinstance(m.conv3x3.norm, nn.BatchNorm2d)
                ): m.toggle_inference_mode()
        self.apply(toggle_inference_mode)
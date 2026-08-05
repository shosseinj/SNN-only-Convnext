from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EffectiveConv2d(nn.Conv2d):
    def __init__(self, *args, force_positive_weights: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_positive_weights = bool(force_positive_weights)

    def effective_weight(self) -> torch.Tensor:
        return self.weight.abs() if self.force_positive_weights else self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.effective_weight(), self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )


class EffectiveLinear(nn.Linear):
    def __init__(self, *args, force_positive_weights: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_positive_weights = bool(force_positive_weights)

    def effective_weight(self) -> torch.Tensor:
        return self.weight.abs() if self.force_positive_weights else self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


class ChannelCurrentNorm(nn.Module):
    def __init__(self, channels: int, enabled: bool = True):
        super().__init__()
        self.norm = (
            nn.GroupNorm(1, channels, eps=1e-3, affine=True)
            if enabled else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ANNDownsample(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        padding: int,
        current_norm: bool = True,
        force_positive_weights: bool = False,
    ):
        super().__init__()
        self.synapse = EffectiveConv2d(
            in_ch, out_ch, kernel_size,
            stride=stride, padding=padding, bias=False,
            force_positive_weights=force_positive_weights,
        )
        self.current_norm = ChannelCurrentNorm(out_ch, current_norm)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.current_norm(self.synapse(x)))


class ReLUConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        current_norm: bool = True,
        force_positive_weights: bool = False,
        residual: bool = True,
    ):
        super().__init__()
        self.dw = EffectiveConv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim,
            bias=False, force_positive_weights=force_positive_weights,
        )
        self.dw_norm = ChannelCurrentNorm(dim, current_norm)

        self.pw1 = EffectiveConv2d(
            dim, 4 * dim, kernel_size=1, bias=False,
            force_positive_weights=force_positive_weights,
        )
        self.pw1_norm = ChannelCurrentNorm(4 * dim, current_norm)
        self.activation = nn.ReLU(inplace=False)

        self.pw2 = EffectiveConv2d(
            4 * dim, dim, kernel_size=1, bias=False,
            force_positive_weights=force_positive_weights,
        )
        self.pw2_norm = ChannelCurrentNorm(dim, current_norm)
        self.residual = bool(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dw_norm(self.dw(x))
        x = self.pw1_norm(self.pw1(x))
        x = self.activation(x)
        x = self.pw2_norm(self.pw2(x))
        if self.residual:
            x = x + identity
        return F.relu(x, inplace=False)


class ANNConvNeXtReLU(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 10,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        current_norm: bool = True,
        cifar_stem: bool = True,
        force_positive_weights: bool = False,
        residual: bool = True,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must each contain four stages")

        self.model_type = "ANN_CONVNEXT_RELU_TTFS_TEACHER"
        self.block_design = "ReLU_teacher_for_two_TTFS_conversion"

        if cifar_stem:
            stem_kernel, stem_stride, stem_padding = 3, 1, 1
        else:
            stem_kernel, stem_stride, stem_padding = 4, 4, 0

        self.downsamples = nn.ModuleList([
            ANNDownsample(in_chans, dims[0], stem_kernel, stem_stride, stem_padding,
                          current_norm, force_positive_weights),
            ANNDownsample(dims[0], dims[1], 2, 2, 0,
                          current_norm, force_positive_weights),
            ANNDownsample(dims[1], dims[2], 2, 2, 0,
                          current_norm, force_positive_weights),
            ANNDownsample(dims[2], dims[3], 2, 2, 0,
                          current_norm, force_positive_weights),
        ])

        self.stages = nn.ModuleList([
            nn.ModuleList([
                ReLUConvNeXtBlock(
                    dims[stage],
                    current_norm=current_norm,
                    force_positive_weights=force_positive_weights,
                    residual=residual,
                )
                for _ in range(depths[stage])
            ])
            for stage in range(4)
        ])

        self.classifier = EffectiveLinear(
            dims[-1], num_classes, bias=True, force_positive_weights=False
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, EffectiveConv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, EffectiveLinear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for stage_index in range(4):
            x = self.downsamples[stage_index](x)
            for block in self.stages[stage_index]:
                x = block(x)
        return x.mean(dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def transferable_state_dict(self) -> Dict[str, torch.Tensor]:
        allowed = (
            ".synapse.", ".current_norm.",
            ".dw.", ".dw_norm.",
            ".pw1.", ".pw1_norm.",
            ".pw2.", ".pw2_norm.",
            "classifier.",
        )
        return {
            name: tensor.detach().clone()
            for name, tensor in self.state_dict().items()
            if any(key in name for key in allowed)
        }


def build_ann_convnext_relu(model_size: str = "tiny", **kwargs) -> ANNConvNeXtReLU:
    configs: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {
        "nano": ((1, 1, 2, 1), (24, 48, 96, 192)),
        "tiny": ((3, 3, 9, 3), (96, 192, 384, 768)),
    }
    if model_size not in configs:
        raise ValueError(
            f"Unknown model_size={model_size!r}; choose from {sorted(configs)}"
        )
    depths, dims = configs[model_size]
    return ANNConvNeXtReLU(depths=depths, dims=dims, **kwargs)


if __name__ == "__main__":
    model = build_ann_convnext_relu(model_size="tiny", num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print({
        "logits_shape": tuple(y.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
    })

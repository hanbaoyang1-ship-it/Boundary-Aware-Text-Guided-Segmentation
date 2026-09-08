from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
from torchvision.models import VGG16_Weights, vgg16


class VGG16Encoder(nn.Module):
    """The three VGG16 feature stages specified in the manuscript."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG16_Weights.DEFAULT if pretrained else None
        layers = list(vgg16(weights=weights).features.children())
        self.stage1 = nn.Sequential(*layers[:5])
        self.stage2 = nn.Sequential(*layers[5:10])
        self.stage3 = nn.Sequential(*layers[10:17])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feature1 = self.stage1(x)
        feature2 = self.stage2(feature1)
        feature3 = self.stage3(feature2)
        return [feature1, feature2, feature3]


class CLIPTextProjector(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        local_files_only: bool = False,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        source = str(Path(model_name_or_path).expanduser()) if Path(model_name_or_path).exists() else model_name_or_path
        self.tokenizer = CLIPTokenizerFast.from_pretrained(source, local_files_only=local_files_only)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            source, local_files_only=local_files_only
        )
        projection_dim = int(self.text_encoder.config.projection_dim)
        self.projection = nn.Sequential(
            nn.Linear(projection_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 256),
        )
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.text_encoder.requires_grad_(False)

    def forward(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        tokens = {name: value.to(device) for name, value in tokens.items()}
        if self.freeze_encoder:
            self.text_encoder.eval()
            with torch.no_grad():
                embeddings = self.text_encoder(**tokens).text_embeds
        else:
            embeddings = self.text_encoder(**tokens).text_embeds
        return self.projection(embeddings)


class SpatialTextExpansion(nn.Module):
    def __init__(self, channels: int = 256, base_size: int = 28) -> None:
        super().__init__()
        self.channels = channels
        self.template = nn.Parameter(torch.randn(1, 1, base_size, base_size))

    def forward(self, text_features: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
        template = torch.nn.functional.interpolate(
            self.template, size=spatial_size, mode="bilinear", align_corners=False
        )
        return text_features[:, :, None, None] * template


class CrossPositionAttention(nn.Module):
    """Paper-aligned A @ V cross-position attention with a residual image path."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image_features.shape
        query = self.query(image_features).flatten(2).transpose(1, 2)
        key = self.key(text_features).flatten(2)
        value = self.value(text_features).flatten(2).transpose(1, 2)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        fused = torch.bmm(attention, value).transpose(1, 2).reshape(batch, channels, height, width)
        return fused + image_features


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


class VGG16Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block3 = ConvBlock(512, 128)
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.block2 = ConvBlock(256, 64)
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.block1 = ConvBlock(128, 32)
        self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, bottleneck: torch.Tensor, encoder_features: Sequence[torch.Tensor]) -> torch.Tensor:
        feature1, feature2, feature3 = encoder_features
        x = self.up3(self.block3(torch.cat([bottleneck, feature3], dim=1)))
        x = self.up2(self.block2(torch.cat([x, feature2], dim=1)))
        x = self.up1(self.block1(torch.cat([x, feature1], dim=1)))
        return self.output(x)


class BACLModel(nn.Module):
    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch32",
        *,
        local_files_only: bool = False,
        freeze_clip: bool = False,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = VGG16Encoder(pretrained=pretrained_backbone)
        self.text_projector = CLIPTextProjector(
            clip_model,
            local_files_only=local_files_only,
            freeze_encoder=freeze_clip,
        )
        self.text_expansion = SpatialTextExpansion()
        self.attention = CrossPositionAttention()
        self.decoder = VGG16Decoder()

    def forward(
        self,
        images: torch.Tensor,
        texts: Sequence[str],
        mask_images: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        image_features = self.encoder(images)
        mask_features = None
        if mask_images is not None:
            if mask_images.shape[1] == 1:
                mask_images = mask_images.repeat(1, 3, 1, 1)
            mask_features = self.encoder(mask_images)
        text_vector = self.text_projector(texts, images.device)
        text_map = self.text_expansion(text_vector, image_features[-1].shape[-2:])
        fused = self.attention(image_features[-1], text_map)
        logits = self.decoder(fused, image_features)
        return {
            "logits": logits,
            "image_features": image_features,
            "mask_features": mask_features,
        }

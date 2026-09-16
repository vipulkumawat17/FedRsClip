from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return super().forward(x.float()).to(dtype)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride > 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(inplanes, planes * self.expansion, 1, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ModifiedResNet(nn.Module):
    def __init__(self, layers: Tuple[int, int, int, int], output_dim: int, heads: int, input_resolution: int, width: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(3, width // 2, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.type(self.conv1.weight.dtype)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.attnpool(x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor | None = None) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.attn_mask
        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor | None = None) -> None:
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width**-0.5
        num_patches = (input_resolution // patch_size) ** 2
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        return x @ self.proj

class AttentionPool2d(nn.Module):
    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spatial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        out, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return out.squeeze(0)


class TextTransformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, width: int, heads: int, layers: int, output_dim: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.transformer = Transformer(width, layers, heads, self.build_attention_mask(context_length))
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        self.initialize_parameters()

    @staticmethod
    def build_attention_mask(context_length: int) -> torch.Tensor:
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def initialize_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = self.transformer.resblocks[0].attn.embed_dim**-0.5 * (2 * len(self.transformer.resblocks))**-0.5
        attn_std = self.transformer.resblocks[0].attn.embed_dim**-0.5
        fc_std = (2 * self.transformer.resblocks[0].attn.embed_dim) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=fc_std)
            nn.init.normal_(block.mlp[2].weight, std=proj_std)
        nn.init.normal_(self.text_projection, std=self.transformer.resblocks[0].attn.embed_dim**-0.5)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text)
        x = x + self.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eos_positions = (text == 2).int().argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), eos_positions] @ self.text_projection


class CLIP(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: TextTransformer,
        embed_dim: int,
        input_resolution: int,
        context_length: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.visual = image_encoder
        self.text = text_encoder
        self.input_resolution = input_resolution
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, logit_scale


def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    image_loss = F.cross_entropy(logits_per_image, labels)
    text_loss = F.cross_entropy(logits_per_text, labels)
    return (image_loss + text_loss) / 2


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int
    image_resolution: int
    vision_layers: Tuple[int, int, int, int] | int
    vision_width: int
    vision_patch_size: int | None
    context_length: int = 76
    vocab_size: int = 259
    transformer_width: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 12


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "RN50": ModelConfig(1024, 224, (3, 4, 6, 3), 64, None),
    "RN101": ModelConfig(512, 224, (3, 4, 23, 3), 64, None),
    "ViT-B-32": ModelConfig(512, 224, 12, 768, 32),
    "ViT-B-16": ModelConfig(512, 224, 12, 768, 16),
}


def build_model(name: str = "RN50", vocab_size: int = 259, context_length: int = 76) -> CLIP:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[name]
    cfg = ModelConfig(
        embed_dim=cfg.embed_dim,
        image_resolution=cfg.image_resolution,
        vision_layers=cfg.vision_layers,
        vision_width=cfg.vision_width,
        vision_patch_size=cfg.vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=cfg.transformer_width,
        transformer_heads=cfg.transformer_heads,
        transformer_layers=cfg.transformer_layers,
    )

    if isinstance(cfg.vision_layers, tuple):
        vision_heads = cfg.vision_width * 32 // 64
        image_encoder = ModifiedResNet(cfg.vision_layers, cfg.embed_dim, vision_heads, cfg.image_resolution, cfg.vision_width)
    else:
        vision_heads = cfg.vision_width // 64
        image_encoder = VisionTransformer(
            cfg.image_resolution,
            cfg.vision_patch_size or 32,
            cfg.vision_width,
            cfg.vision_layers,
            vision_heads,
            cfg.embed_dim,
        )

    text_encoder = TextTransformer(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        width=cfg.transformer_width,
        heads=cfg.transformer_heads,
        layers=cfg.transformer_layers,
        output_dim=cfg.embed_dim,
    )
    return CLIP(image_encoder, text_encoder, cfg.embed_dim, cfg.image_resolution, cfg.context_length, cfg.vocab_size)
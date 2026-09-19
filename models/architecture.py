"""architecture.py - Prithvi 인코더(LoRA) + SR 디코더 모델 정의

Prithvi EO v2 300M 백본(터라토치)의 마지막/중간 레이어 출력을 결합해
디코더에 넘기고, Bicubic 업샘플에 잔차(residual)를 더하는 4x 초해상화 모델.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from terratorch.registry import BACKBONE_REGISTRY

# =====================================================================
# 1. 서브 모듈 (Attention & Feature Extraction)
# =====================================================================

class SpatialAttention(nn.Module):
    """CBAM 스타일의 가벼운 공간 어텐션: 엣지 위치 강조"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(res))


class ChannelAttention(nn.Module):
    """채널 어텐션: 중요한 특징 맵 활성화"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(self.avg_pool(x))


class HybridAttentionBlock(nn.Module):
    """채널 및 공간 정보를 통합 정제하는 잔차 블록"""
    def __init__(self, channels=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.main(x)
        out = self.ca(out)
        out = self.sa(out)
        return x + out  # Residual Learning


class LocalExtractor(nn.Module):
    """LR 이미지에서 로컬 엣지 특징 추출 (경량화 버전)"""
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.main(x)


# =====================================================================
# 2. 메인 디코더 및 통합 모델
# =====================================================================

class UnpatchingLayer(nn.Module):
    """Prithvi 토큰을 단계적으로 업샘플링 (14->56->224)"""
    def __init__(self, in_dim=1024, out_dim=64):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_dim, out_dim * 2, kernel_size=4, stride=4)
        self.up2 = nn.ConvTranspose2d(out_dim * 2, out_dim, kernel_size=4, stride=4)
        self.smooth = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act(self.up1(x))
        x = self.act(self.up2(x))
        return self.act(self.smooth(x))


class PrithviSRDecoder(nn.Module):
    def __init__(self, prithvi_embed_dim=1024, feature_dim=64, out_channels=4):
        super().__init__()
        self.unpatch = UnpatchingLayer(in_dim=prithvi_embed_dim, out_dim=feature_dim)
        self.local_ext = LocalExtractor(in_channels=out_channels, out_channels=feature_dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.refinement = nn.Sequential(
            *[HybridAttentionBlock(channels=feature_dim) for _ in range(4)]
        )

        self.up_conv = nn.Conv2d(feature_dim, out_channels * 16, kernel_size=3, padding=1)

        # [Zero Init] 초기 상태를 Bicubic과 동일하게 강제하여 잔차 학습 가속화
        nn.init.constant_(self.up_conv.weight, 0)
        nn.init.constant_(self.up_conv.bias, 0)

        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, prithvi_tokens, lr_img):
        B = prithvi_tokens.size(0)
        # Reshape: [B, L, D] -> [B, D, 14, 14]
        x = prithvi_tokens[:, 1:, :].transpose(1, 2).reshape(B, -1, 14, 14)

        global_feat = self.unpatch(x)
        local_feat = self.local_ext(lr_img)

        # Global 맥락과 Local 디테일 융합
        fused = self.fusion(torch.cat([global_feat, local_feat], dim=1))
        refined = self.refinement(fused)

        # 모델이 예측한 미세 잔차(Residual)
        residual = self.pixel_shuffle(self.up_conv(refined))

        # 베이스라인(Bicubic)에 잔차를 더해 최종 복원
        bicubic_up = F.interpolate(lr_img, scale_factor=4, mode='bicubic', align_corners=False)
        return residual + bicubic_up


class FullSRModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_6ch, lr_img):
        # 인코더 출력 (Prithvi EO v2의 모든 레이어 아웃풋 가정)
        encoder_outputs = self.encoder(x_6ch)

        # 마지막 레이어(-1)와 중간 레이어(-4)를 결합하여 풍부한 공간 맥락 수혈
        # 데이터가 적은 환경(100개)에서 복원 성능을 높이는 핵심 로직
        combined_feat = (encoder_outputs[-1] + encoder_outputs[-4]) * 0.5

        return self.decoder(combined_feat, lr_img)


# --- 초기화 함수 ---
def get_prithvi_encoder_with_lora(checkpoint_path, model_name='prithvi_eo_v2_300'):
    print("\n[초기화] Prithvi 백본 로드 및 LoRA 어댑터 장착 중...")
    encoder = BACKBONE_REGISTRY.build(model_name, pretrained=False, num_frames=1)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    encoder.load_state_dict(state_dict, strict=False)

    lora_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=["qkv", "proj"],
        lora_dropout=0.1, bias="none"
    )
    lora_encoder = get_peft_model(encoder, lora_config)
    return lora_encoder

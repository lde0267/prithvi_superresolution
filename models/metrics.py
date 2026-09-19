"""metrics.py - SR 평가 지표 (PSNR/SSIM/SAM/Edge RMSE) + Bicubic 베이스라인

기존 노트북 평가 섹션은 별도 `_baseline.tif`(다른 폴더에 미리 생성된 Bicubic 업샘플
결과)와 옛 `_S2.tif`/`_K3A.tif` 네이밍을 요구했는데, 실제로는 존재하지 않는 폴더였다.
Bicubic 베이스라인은 `F.interpolate` 로 그 자리에서 계산 가능하므로 별도 파일이
필요 없다 — 모델(`PrithviSRDecoder`)이 내부적으로 쓰는 것과 동일한 연산이다.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import sobel
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def bicubic_baseline(lr_img: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """LR 텐서 (B, C, H, W) 를 Bicubic 4x 업샘플."""
    return F.interpolate(lr_img, scale_factor=scale, mode="bicubic", align_corners=False)


def to_hwc_numpy(img) -> np.ndarray:
    """(B,C,H,W) 또는 (C,H,W) 텐서/배열을 (H,W,C) numpy 로 변환."""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = np.squeeze(img)
    if img.ndim == 3 and img.shape[0] <= 8:  # (C,H,W) -> (H,W,C)
        img = np.transpose(img, (1, 2, 0))
    return img


def calculate_sam(target: np.ndarray, reference: np.ndarray) -> float:
    """Spectral Angle Mapper (라디안). 낮을수록 분광 보존이 좋음. (H,W,C) 가정."""
    dot_product = np.sum(target * reference, axis=-1)
    norm_target = np.linalg.norm(target, axis=-1)
    norm_ref = np.linalg.norm(reference, axis=-1)
    val = dot_product / (norm_target * norm_ref + 1e-8)
    val = np.clip(val, -1.0, 1.0)
    return float(np.mean(np.arccos(val)))


def calculate_edge_rmse(target: np.ndarray, reference: np.ndarray) -> float:
    """Sobel 엣지 크기 RMSE. 낮을수록 경계선 복원이 정확함. (H,W,C) 가정."""
    def edge_magnitude(img):
        gray = np.mean(img[..., :3], axis=-1) if img.shape[-1] >= 3 else img[..., 0]
        dx = sobel(gray, axis=0)
        dy = sobel(gray, axis=1)
        return np.sqrt(dx**2 + dy**2)

    return float(np.sqrt(np.mean((edge_magnitude(target) - edge_magnitude(reference)) ** 2)))


def compute_all_metrics(pred_hwc: np.ndarray, gt_hwc: np.ndarray) -> dict:
    """(H,W,C), 0~1 범위로 정규화된 pred/gt 에 대해 PSNR/SSIM/SAM/Edge RMSE 를 계산."""
    return {
        "psnr": float(psnr(gt_hwc, pred_hwc, data_range=1.0)),
        "ssim": float(ssim(gt_hwc, pred_hwc, data_range=1.0, channel_axis=-1)),
        "sam": calculate_sam(pred_hwc, gt_hwc),
        "edge_rmse": calculate_edge_rmse(pred_hwc, gt_hwc),
    }

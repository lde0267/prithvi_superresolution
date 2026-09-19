"""evaluate.py - 테스트셋 평가 (모델 vs Bicubic 베이스라인)

data_preprocessing 파이프라인이 만드는 {base}_lr.tif/{base}_hr.tif 페어를 그대로 쓴다.
예전 노트북처럼 별도 '_baseline.tif' 폴더가 필요 없다 — bicubic 베이스라인은
metrics.bicubic_baseline() 으로 그 자리에서 계산한다.

사용법:
    python models/evaluate.py                 # 테스트 분할 전체 평가 + 요약 출력
    python models/evaluate.py --visualize 3    # 추가로 샘플 3개 시각화
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm

from config import DATASET_ROOT, ENCODER_FILE, CHECKPOINT_DIR
from dataset import get_data_split, KOMPSATSRDataset
from architecture import get_prithvi_encoder_with_lora, PrithviSRDecoder, FullSRModel
from metrics import bicubic_baseline, to_hwc_numpy, compute_all_metrics


def load_model(device: torch.device, checkpoint_name: str = "best_sr_model.pth"):
    """Prithvi 인코더(LoRA) + 디코더를 만들고 체크포인트 가중치를 로드."""
    if not os.path.exists(ENCODER_FILE):
        raise FileNotFoundError(
            f"Prithvi 인코더가 없습니다: {ENCODER_FILE}\n"
            f"먼저 실행하세요: python models/download_prithvi.py"
        )
    encoder_lora = get_prithvi_encoder_with_lora(ENCODER_FILE)
    decoder = PrithviSRDecoder()
    model = FullSRModel(encoder_lora, decoder).to(device)

    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"✅ 가중치 로드: {ckpt_path}")
    else:
        print(f"⚠️ 체크포인트가 없습니다: {ckpt_path} — 학습되지 않은 초기 가중치로 평가합니다.")

    model.eval()
    return model


@torch.no_grad()
def run_one(model, sample: dict, device: torch.device) -> dict:
    """샘플 1개에 대해 모델 추론 + Bicubic 베이스라인 + 두 방식의 지표를 계산."""
    x_6ch = sample["x_6ch"].unsqueeze(0).unsqueeze(2).to(device)  # (1, 6, 1, H, W)
    lr_img = sample["lr_img"].unsqueeze(0).to(device)              # (1, 4, H, W)
    gt_img = sample["gt_img"].unsqueeze(0).to(device)              # (1, 4, H_hr, W_hr)

    model_out = model(x_6ch, lr_img)
    bicubic_out = bicubic_baseline(lr_img)

    gt_hwc = to_hwc_numpy(gt_img)
    model_hwc = np.clip(to_hwc_numpy(model_out), 0.0, 1.0)
    bicubic_hwc = np.clip(to_hwc_numpy(bicubic_out), 0.0, 1.0)

    return {
        "model": compute_all_metrics(model_hwc, gt_hwc),
        "bicubic": compute_all_metrics(bicubic_hwc, gt_hwc),
        "raw": {
            "lr": to_hwc_numpy(lr_img),
            "gt": gt_hwc,
            "model": model_hwc,
            "bicubic": bicubic_hwc,
        },
    }


def evaluate_all(model, test_list: list[dict], device: torch.device) -> list[dict]:
    """test_list 전체를 평가하고 페어별 결과 리스트를 반환."""
    ds = KOMPSATSRDataset(test_list, transform=False)
    results = []
    for i in tqdm(range(len(ds)), desc="평가"):
        results.append(run_one(model, ds[i], device))
    return results


def print_summary(results: list[dict]) -> None:
    if not results:
        print("평가할 테스트 샘플이 없습니다.")
        return

    def avg(key_a, key_b):
        return float(np.mean([r[key_a][key_b] for r in results]))

    print(f"\n📊 테스트셋 평가 결과 ({len(results)} samples)")
    print("-" * 70)
    print(f"{'Metric':<12} | {'Bicubic':>12} | {'Model':>12} | {'Gain'}")
    print("-" * 70)
    for metric, higher_better, unit in [
        ("psnr", True, "dB"), ("ssim", True, ""),
        ("sam", False, "rad"), ("edge_rmse", False, ""),
    ]:
        b, m = avg("bicubic", metric), avg("model", metric)
        gain = (m - b) if higher_better else (b - m)
        arrow = "↑" if higher_better else "↓ (작을수록 좋음)"
        print(f"{metric.upper():<12} | {b:>12.4f} | {m:>12.4f} | {gain:+.4f} {unit} ({arrow})")
    print("-" * 70)


def visualize(results: list[dict], num_samples: int = 3) -> None:
    import matplotlib.pyplot as plt

    def stretch(img, pct=2):
        img = img.copy()
        rgb = img[..., [2, 1, 0]] if img.shape[-1] >= 3 else img
        out = np.zeros_like(rgb)
        for c in range(rgb.shape[-1]):
            ch = rgb[..., c]
            lo, hi = np.percentile(ch, pct), np.percentile(ch, 100 - pct)
            out[..., c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1) if hi > lo else ch
        return out

    for i, r in enumerate(results[:num_samples]):
        raw = r["raw"]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, key, title in zip(
            axes, ["lr", "bicubic", "model", "gt"],
            ["1. Input (LR)", "2. Bicubic", "3. Model (Ours)", "4. Ground Truth (HR)"],
        ):
            ax.imshow(stretch(raw[key]))
            ax.set_title(title, fontsize=13)
            ax.axis("off")
        fig.suptitle(f"Test sample {i + 1}/{len(results)}", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.show()


def main(num_visualize: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    _, _, test_list = get_data_split(DATASET_ROOT)
    print(f"테스트 샘플: {len(test_list)}개")
    if not test_list:
        print("⚠️ 테스트 분할이 비어 있습니다 (데이터가 적으면 train/val 로만 나뉠 수 있음). "
              "더 많은 페어를 data_preprocessing 파이프라인으로 생성하세요.")
        return

    model = load_model(device)
    results = evaluate_all(model, test_list, device)
    print_summary(results)

    if num_visualize > 0:
        visualize(results, num_samples=num_visualize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KOMPSAT-3A SR 모델 테스트셋 평가")
    parser.add_argument("--visualize", type=int, default=0, help="시각화할 샘플 수 (기본 0=미시각화)")
    args = parser.parse_args()
    main(num_visualize=args.visualize)

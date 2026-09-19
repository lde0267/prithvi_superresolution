"""dataset.py - KOMPSAT-3A SR 학습용 데이터 스캔 + Dataset 클래스

data_preprocessing/module3_simulation/scripts/run_chip_extraction.py 가 만드는
data/output/chips/{label}_{scene}/{base}_lr.tif(입력) + {base}_hr.tif(정답)
페어를 스캔해 Train/Val/Test 로 분할하고, PyTorch Dataset 으로 로드합니다.
"""
import os
import random

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def count_chip_data_pairs(root_path: str) -> list[dict]:
    """폴더별 매칭 결과를 출력하며 전체 {input, gt} 페어 리스트를 반환합니다 (탐색용)."""
    print(f"🔍 데이터 스캔 시작: {root_path}")

    if not os.path.exists(root_path):
        print("❌ 경로가 존재하지 않습니다. "
              "data_preprocessing/module3_simulation/scripts/run_chip_extraction.py 를 먼저 실행했는지 확인하세요.")
        return []

    video_folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    total_pairs = 0
    all_pairs: list[dict] = []

    print("\n📊 폴더별 칩(Chip) 매칭 결과")
    print("-" * 65)

    for folder in sorted(video_folders):
        folder_path = os.path.join(root_path, folder)
        files = os.listdir(folder_path)
        lr_files = [f for f in files if f.lower().endswith('_lr.tif')]

        match_count = 0
        for lr_file in lr_files:
            ext_index = lr_file.lower().rfind('_lr.tif')
            if ext_index == -1:
                continue
            gt_file = lr_file[:ext_index] + "_hr.tif"
            if gt_file in files:
                match_count += 1
                all_pairs.append({
                    'input': os.path.join(folder_path, lr_file),
                    'gt': os.path.join(folder_path, gt_file)
                })

        if match_count > 0:
            display_folder = folder if len(folder) < 45 else f"{folder[:42]}..."
            print(f"📁 {display_folder} : {match_count:3d} chips")
            total_pairs += match_count

    print("-" * 65)
    print(f"✅ 총 학습 가능한 데이터 쌍(Chips): {total_pairs} 개")
    print("=" * 65)

    return all_pairs


def get_data_split(
    root_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """root_path 를 스캔해 (train, val, test) 페어 리스트로 분할합니다."""
    random.seed(seed)
    all_pairs: list[dict] = []

    if not os.path.exists(root_path):
        print(f"Error: Path not found {root_path}")
        return [], [], []

    video_folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    for folder in video_folders:
        folder_path = os.path.join(root_path, folder)
        files = os.listdir(folder_path)
        lr_files = [f for f in files if f.lower().endswith('_lr.tif')]

        for lr_file in lr_files:
            ext_idx = lr_file.lower().rfind('_lr.tif')
            gt_file = lr_file[:ext_idx] + "_hr.tif"
            if gt_file in files:
                all_pairs.append({
                    'input': os.path.join(folder_path, lr_file),
                    'gt': os.path.join(folder_path, gt_file)
                })

    random.shuffle(all_pairs)
    total = len(all_pairs)
    if total == 0:
        return [], [], []

    train_idx = int(total * train_ratio)
    val_idx = train_idx + int(total * val_ratio)

    # 데이터가 적을 때(소규모 예시 데이터셋) val 이 0개가 되지 않도록 최소 1개 보장
    if total >= 2:
        train_idx = min(train_idx, total - 1)
        val_idx = max(val_idx, train_idx + 1)
        val_idx = min(val_idx, total)

    return all_pairs[:train_idx], all_pairs[train_idx:val_idx], all_pairs[val_idx:]


class KOMPSATSRDataset(Dataset):
    """SR 맞춤형 실시간 증강이 포함된 Dataset 클래스."""

    def __init__(self, data_list: list[dict], transform: bool = False):
        self.data_list = data_list
        self.transform = transform

        # Prithvi Normalize 파라미터 (6채널 기준: Blue,Green,Red,NIR,SWIR1,SWIR2)
        self.mean = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
        self.std = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        paths = self.data_list[idx]

        # 1. 파일 로드 (Rasterio는 C, H, W 반환 -> 증강을 위해 H, W, C 로 변경)
        with rasterio.open(paths['input']) as src:
            lr_img = src.read().transpose(1, 2, 0).astype(np.float32)
        with rasterio.open(paths['gt']) as src:
            gt_img = src.read().transpose(1, 2, 0).astype(np.float32)

        # 2. SR 맞춤형 증강 (이미지 크기가 달라도 안전하게 작동)
        if self.transform:
            if random.random() > 0.5:
                lr_img = lr_img[:, ::-1, :]
                gt_img = gt_img[:, ::-1, :]

            if random.random() > 0.5:
                lr_img = lr_img[::-1, :, :]
                gt_img = gt_img[::-1, :, :]

            k = random.randint(0, 3)
            if k > 0:
                lr_img = np.rot90(lr_img, k, axes=(0, 1))
                gt_img = np.rot90(gt_img, k, axes=(0, 1))

            lr_img = np.ascontiguousarray(lr_img)
            gt_img = np.ascontiguousarray(gt_img)

        # 3. 채널 직접 할당 (4채널 -> 6채널)
        h, w, c = lr_img.shape
        input_6ch = np.zeros((h, w, 6), dtype=np.float32)
        input_6ch[..., :c] = lr_img

        # 4. 정규화 및 Tensor 변환 (C, H, W 순서로 최종 변경)
        input_6ch_norm = (input_6ch - self.mean) / self.std
        input_6ch_norm = torch.from_numpy(input_6ch_norm.transpose(2, 0, 1))

        # Skip connection 및 Ground Truth용 0~1 스케일링 + 이상치 클리핑
        lr_norm = np.clip(lr_img / 10000.0, 0.0, 1.0)
        gt_norm = np.clip(gt_img / 10000.0, 0.0, 1.0)

        lr_norm = torch.from_numpy(lr_norm.transpose(2, 0, 1))
        gt_norm = torch.from_numpy(gt_norm.transpose(2, 0, 1))

        return {
            'x_6ch': input_6ch_norm,  # 모델 입력용 (6, H_lr, W_lr)
            'lr_img': lr_norm,        # Skip connection용 (4, H_lr, W_lr)
            'gt_img': gt_norm         # 정답용 (4, H_gt, W_gt)
        }

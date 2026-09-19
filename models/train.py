"""train.py - KOMPSAT-3A SR 모델 학습 진입점

사용법:
    python models/train.py                  # 기본 200 에폭까지 학습 (체크포인트 이어서)
    python models/train.py --max-epochs 1    # 스모크 테스트 (배선 확인용, 1 에폭만)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import DATASET_ROOT, ENCODER_FILE, CHECKPOINT_DIR
from dataset import get_data_split, KOMPSATSRDataset
from architecture import get_prithvi_encoder_with_lora, PrithviSRDecoder, FullSRModel
from losses import SatelliteSRLoss
from checkpoint import load_checkpoint, save_checkpoint


def main(max_epochs: int = 200, batch_size: int = 8) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    history_path = os.path.join(CHECKPOINT_DIR, 'loss_history.json')

    # 1. 데이터 분할 및 로더 설정
    train_list, val_list, test_list = get_data_split(DATASET_ROOT)
    if not train_list or not val_list:
        print(f"🚨 학습 가능한 데이터가 부족합니다 (train={len(train_list)}, val={len(val_list)}).")
        print(f"   data_preprocessing 파이프라인을 먼저 실행해 data/output/chips/ 에 칩을 생성하세요:")
        print(f"   run_simulation.py -> run_mtf_simulation.py -> run_chip_extraction.py")
        return
    train_ds = KOMPSATSRDataset(train_list, transform=True)
    val_ds = KOMPSATSRDataset(val_list, transform=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=min(4, len(val_ds)), shuffle=False)

    # 2. 모델 초기화
    if not os.path.exists(ENCODER_FILE):
        print(f"🚨 Prithvi 인코더 체크포인트가 없습니다: {ENCODER_FILE}")
        print(f"   먼저 실행하세요: python models/download_prithvi.py")
        return

    encoder_lora = get_prithvi_encoder_with_lora(ENCODER_FILE)
    decoder = PrithviSRDecoder()
    full_model = FullSRModel(encoder_lora, decoder).to(device)

    optimizer = optim.AdamW(full_model.parameters(), lr=1e-4, weight_decay=1e-4)
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)

    # 3. 체크포인트 및 기존 기록 로드
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, 'latest_sr_model.pth')
    start_epoch, _ = load_checkpoint(full_model, optimizer, latest_ckpt_path)

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"📈 기존 {len(history['train_loss'])} 에포크 기록을 불러왔습니다.")
    else:
        history = {'train_loss': [], 'val_loss': []}

    best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')

    print(f"🚀 학습을 시작합니다. (최대 {max_epochs} 에폭, Ctrl+C로 중단하면 안전 저장)")

    try:
        for epoch in range(start_epoch, max_epochs):
            # --- [Train Phase] ---
            full_model.train()
            alpha, beta, gamma = (1.0, 0.15, 0.3)
            criterion = SatelliteSRLoss(alpha=alpha, beta=beta, gamma=gamma).to(device)

            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
                x_6ch = batch['x_6ch'].unsqueeze(2).to(device)
                lr_img = batch['lr_img'].to(device)
                gt_img = batch['gt_img'].to(device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    output = full_model(x_6ch, lr_img)
                    loss, _, _, _ = criterion(output, gt_img)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            # --- [Validation Phase] ---
            full_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_6ch = batch['x_6ch'].unsqueeze(2).to(device)
                    lr_img = batch['lr_img'].to(device)
                    gt_img = batch['gt_img'].to(device)

                    with autocast(device_type=device.type, enabled=use_amp):
                        output = full_model(x_6ch, lr_img)
                        loss, _, _, _ = criterion(output, gt_img)
                    val_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)

            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            with open(history_path, 'w') as f:
                json.dump(history, f)

            print(f"Epoch {epoch + 1} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_checkpoint(full_model, optimizer, epoch, avg_val, os.path.join(CHECKPOINT_DIR, 'best_sr_model.pth'))
                print(f"🌟 Best Model 갱신 (Val Loss: {best_val_loss:.4f})")

            save_checkpoint(full_model, optimizer, epoch, avg_train, latest_ckpt_path)

    except KeyboardInterrupt:
        print("\n🛑 [학습 수동 중단 감지] 사용자에 의해 학습이 멈췄습니다.")
        print("💾 현재까지의 학습 진행 상태를 저장하는 중...")
        current_loss = avg_train if 'avg_train' in locals() else (history['train_loss'][-1] if history['train_loss'] else 0.0)
        current_epoch = epoch if 'epoch' in locals() else start_epoch
        save_checkpoint(full_model, optimizer, current_epoch, current_loss, latest_ckpt_path)
        print(f"✅ 진행 상태가 '{latest_ckpt_path}'에 안전하게 저장되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KOMPSAT-3A SR 모델 학습")
    parser.add_argument('--max-epochs', type=int, default=200, help='최대 에폭 수 (기본 200)')
    parser.add_argument('--batch-size', type=int, default=8, help='배치 크기 (기본 8)')
    args = parser.parse_args()
    main(max_epochs=args.max_epochs, batch_size=args.batch_size)

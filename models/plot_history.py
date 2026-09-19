"""plot_history.py - 학습 손실 곡선 시각화"""
import json
import os

import matplotlib.pyplot as plt


def plot_loss_history(checkpoint_dir: str) -> None:
    history_path = os.path.join(checkpoint_dir, 'loss_history.json')

    if not os.path.exists(history_path):
        print("❌ 저장된 기록 파일이 없습니다.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if len(epochs) > 5:
        plt.axvline(x=5, color='gray', linestyle='--', label='Warm-up End')

    plt.show()


if __name__ == '__main__':
    from config import CHECKPOINT_DIR
    plot_loss_history(CHECKPOINT_DIR)

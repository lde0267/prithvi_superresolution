"""checkpoint.py - 학습 체크포인트 저장/복원"""
import os
import torch


def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"🚀 체크포인트 발견! {start_epoch} 에폭부터 학습을 재개합니다.")
        return start_epoch, checkpoint['loss']
    print("🆕 기존 체크포인트가 없습니다. 처음부터 학습을 시작합니다.")
    return 0, float('inf')


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

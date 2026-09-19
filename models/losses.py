"""losses.py - SR 학습용 손실 함수 (L1 + SAM + Edge 결합)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- [Loss 1] Spectral Angle Mapper Loss (분광 정보 유지) ---
class SAMLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # 0. 수치적 안정을 위해 아주 작은 값을 더함
        dot_product = torch.sum(pred * target, dim=1)
        norm_pred = torch.norm(pred, dim=1)
        norm_target = torch.norm(target, dim=1)

        # 1. 분모가 0이 되는 것 방지
        denominator = norm_pred * norm_target + self.eps
        cos_sim = dot_product / denominator

        # 2. acos의 입력값을 -1+eps ~ 1-eps로 엄격하게 제한 (FP16 nan 방지)
        cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)

        sam_angle = torch.acos(cos_sim)

        loss = torch.mean(sam_angle)
        if torch.isnan(loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return loss


# --- [Loss 2] Edge Loss (세밀한 경계선 복원) ---
class EdgeLoss(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        filter_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        filter_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.weight_x = filter_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.weight_y = filter_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    def forward(self, pred, target):
        weight_x = self.weight_x.to(pred.device).type_as(pred)
        weight_y = self.weight_y.to(pred.device).type_as(pred)

        pred_dx = F.conv2d(pred, weight_x, padding=1, groups=pred.size(1))
        pred_dy = F.conv2d(pred, weight_y, padding=1, groups=pred.size(1))
        target_dx = F.conv2d(target, weight_x, padding=1, groups=target.size(1))
        target_dy = F.conv2d(target, weight_y, padding=1, groups=target.size(1))

        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)
        return loss_dx + loss_dy


# --- [Loss 3] 통합 다중 손실함수 (L1 + SAM + Edge) ---
class SatelliteSRLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.sam_loss = SAMLoss()
        self.edge_loss = EdgeLoss(channels=4)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        sam = self.sam_loss(pred, target)
        edge = self.edge_loss(pred, target)

        total_loss = (self.alpha * l1) + (self.beta * sam) + (self.gamma * edge)
        return total_loss, l1, sam, edge

import torch
import torch.nn as nn
from typing import Optional, List, Union
import numpy as np


class AdaptiveRankLoss(nn.Module):
    """
    自适应排序损失函数，使用上一轮epoch的信息作为上下文。

    将当前批次的分数与内部存储的上一轮分数进行比较，计算自适应损失。
    使用平方损失替代Huber损失以简化计算。

    Args:
        scale_sensitivity (float): 自适应缩放因子的敏感度参数，默认2.0
        max_prev_targets (int): 上一轮target样本子采样上限，默认10000
        max_prev_decoys (int): 上一轮decoy样本子采样上限，默认10000
    """

    def __init__(self,
                 max_prev_targets: int = 10000,
                 max_prev_decoys: int = 10000):
        super().__init__()
        self.max_prev_targets = max_prev_targets
        self.max_prev_decoys = max_prev_decoys

        # 存储上一轮的预测分数和标签
        self.prev_scores: Optional[torch.Tensor] = None
        self.prev_labels: Optional[torch.Tensor] = None

    def initialize_history(self, initial_labels: Union[List[int], np.ndarray, torch.Tensor]):
        """
        初始化历史状态，用于第一个epoch的计算。

        Args:
            initial_labels: 整个训练集的真实标签
        """
        # 统一转换为tensor
        if isinstance(initial_labels, list):
            initial_labels = torch.tensor(initial_labels)
        elif isinstance(initial_labels, np.ndarray):
            initial_labels = torch.from_numpy(initial_labels)

        device = torch.device("cpu")
        initial_labels = initial_labels.float().to(device)
        num_samples = len(initial_labels)

        # 随机初始化预测分数
        self.prev_scores = torch.rand(num_samples).float().to(device) * 0.5
        self.prev_labels = initial_labels

        print(f"AdaptiveRankLoss initialized with {num_samples} samples.")

    def update_history(self, epoch_scores: torch.Tensor, epoch_labels: torch.Tensor):
        """
        更新历史状态。

        Args:
            epoch_scores: 当前epoch的预测分数
            epoch_labels: 当前epoch的真实标签
        """
        device = torch.device("cpu")
        self.prev_scores = epoch_scores.detach().float().to(device)
        self.prev_labels = epoch_labels.detach().float().to(device)

    def _compute_adaptive_scale(self, current_targets: torch.Tensor, prev_decoys: torch.Tensor) -> torch.Tensor:
        """计算自适应缩放因子"""
        if len(prev_decoys) == 0:
            return torch.ones_like(current_targets)

        max_prev_decoy = torch.max(prev_decoys.detach())
        gap = current_targets - max_prev_decoy
        return torch.exp(-torch.clamp(gap, min=0))

    def _subsample_history(self, prev_targets: torch.Tensor, prev_decoys: torch.Tensor, device: torch.device):
        """对历史样本进行子采样"""
        # 计算采样概率
        target_prob = min(1.0, self.max_prev_targets / max(1, len(prev_targets)))
        decoy_prob = min(1.0, self.max_prev_decoys / max(1, len(prev_decoys)))

        # 随机子采样
        sampled_targets = prev_targets[torch.rand(len(prev_targets), device=device) < target_prob]
        sampled_decoys = prev_decoys[torch.rand(len(prev_decoys), device=device) < decoy_prob]

        return sampled_targets, sampled_decoys

    def forward(self,
                current_scores: torch.Tensor,
                current_labels: torch.Tensor,
                margin: torch.Tensor) -> torch.Tensor:
        """
        计算自适应排序损失。

        Args:
            current_scores: 当前批次的预测分数
            current_labels: 当前批次的真实标签 (1 for target, 0 for decoy)
            margin: 可学习的间隔参数

        Returns:
            损失值
        """
        # 检查历史记录是否已初始化
        if self.prev_scores is None or self.prev_labels is None:
            print("Warning: History not initialized. Returning zero loss.")
            return torch.tensor(0.0, device=current_scores.device, requires_grad=True)

        # 移动到相同设备
        device = current_scores.device
        prev_scores = self.prev_scores.to(device)
        prev_labels = self.prev_labels.to(device)

        # 分离target和decoy
        current_is_target = (current_labels >= 0.5)
        prev_is_target = (prev_labels >= 0.5)

        # 边缘情况处理
        num_current_targets = torch.sum(current_is_target)
        if num_current_targets == 0 or num_current_targets == len(current_is_target):
            return torch.sum(current_scores) * 1e-9

        # 提取不同类型的分数
        current_targets = current_scores[current_is_target]
        current_decoys = current_scores[~current_is_target]
        prev_targets = prev_scores[prev_is_target]
        prev_decoys = prev_scores[~prev_is_target]

        # 计算自适应缩放因子
        adaptive_scale = self._compute_adaptive_scale(current_targets, prev_decoys)

        # 对历史样本进行子采样
        sampled_prev_targets, sampled_prev_decoys = self._subsample_history(prev_targets, prev_decoys, device)

        # 计算损失组件
        loss_components = []

        # 损失1: Current Target vs Previous Decoy
        if len(current_targets) > 0 and len(sampled_prev_decoys) > 0:
            # 计算差值矩阵
            diff_matrix = sampled_prev_decoys.view(1, -1) - current_targets.view(-1, 1) + margin
            hinge_loss = torch.clamp(diff_matrix, min=0)

            # 使用平方损失替代Huber损失
            squared_loss = hinge_loss ** 2

            # 应用自适应缩放
            scaled_loss = squared_loss * adaptive_scale.view(-1, 1)
            loss_components.append(torch.sum(scaled_loss) / self.max_prev_targets)

        # 损失2: Current Decoy vs Previous Target
        if len(current_decoys) > 0 and len(sampled_prev_targets) > 0:
            # 计算差值矩阵
            diff_matrix = current_decoys.view(-1, 1) - sampled_prev_targets.view(1, -1) + margin
            hinge_loss = torch.clamp(diff_matrix, min=0)

            # 使用平方损失
            squared_loss = hinge_loss ** 2

            loss_components.append(torch.sum(squared_loss) / self.max_prev_decoys)

        # 计算最终损失
        total_loss = sum(loss_components) if loss_components else torch.tensor(0.0, device=device)

        # NaN处理
        final_loss = torch.where(torch.isnan(total_loss), torch.zeros_like(total_loss), total_loss)

        # 确保梯度传播
        if not final_loss.requires_grad and (current_scores.requires_grad or margin.requires_grad):
            final_loss = final_loss + 0.0 * (current_scores.sum() + margin)

        return final_loss
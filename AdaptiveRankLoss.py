import torch
import torch.nn as nn
from typing import Optional, List ,Union# 增加类型提示
import numpy as np

class AdaptiveRankLoss(nn.Module):
    """
    Adaptive Ranking Loss 类，使用上一轮 (epoch) 的信息作为上下文。

    将当前批次的分数与内部存储的上一轮分数进行比较，以计算自适应损失。

    Attributes:
        huber_delta (float): Huber Loss 的阈值。
        scale_sensitivity (float): 自适应缩放因子的敏感度参数 (原 alpha)。
        max_prev_target_samples (int): 上一轮 Target 样本子采样的数量上限。
        max_prev_decoy_samples (int): 上一轮 Decoy 样本子采样的数量上限。
        prev_scores (Optional[torch.Tensor]): 内部存储的上一轮预测得分。
        prev_labels (Optional[torch.Tensor]): 内部存储的上一轮真实标签。
    """
    def __init__(self,
                 huber_delta: float = 1.0,
                 scale_sensitivity: float = 2.0,
                 max_prev_target_samples: int = 10000,
                 max_prev_decoy_samples: int = 10000):
        """
        初始化 AdaptiveRankLoss 类。

        Args:
            huber_delta (float, optional): Huber Loss 的阈值。默认为 1.0。
            scale_sensitivity (float, optional): 自适应缩放因子的敏感度参数。默认为 2.0。
            max_prev_target_samples (int, optional): 上一轮 Target 样本子采样上限。默认为 10000。
            max_prev_decoy_samples (int, optional): 上一轮 Decoy 样本子采样上限。默认为 10000。
        """
        super().__init__()
        self.huber_delta = huber_delta
        self.scale_sensitivity = scale_sensitivity
        self.max_prev_target_samples = max_prev_target_samples
        self.max_prev_decoy_samples = max_prev_decoy_samples

        # 初始化内部状态
        self.prev_scores: Optional[torch.Tensor] = None
        self.prev_labels: Optional[torch.Tensor] = None

    def initialize_history(self, initial_all_labels: Union[List[int], np.ndarray, torch.Tensor]):
        """
        初始化上一轮的状态，用于第一个 epoch 的计算。

        Args:
            initial_all_labels (Union[List[int], np.ndarray, torch.Tensor]): 整个训练集在第一个 epoch 开始时的真实标签。
        """
        if isinstance(initial_all_labels, list): # 如果是列表，先转为 Tensor
             initial_all_labels = torch.tensor(initial_all_labels)
        elif isinstance(initial_all_labels, np.ndarray): # 如果是 Numpy 数组，也转为 Tensor
             initial_all_labels = torch.from_numpy(initial_all_labels)

        # 确保是 Tensor 并且在正确的设备上 (这里假定 CPU，与原脚本一致)
        device = torch.device("cpu")
        initial_all_labels = initial_all_labels.float().to(device) # 现在可以安全地调用 .float()
        num_samples = len(initial_all_labels)

        # 使用与原脚本相同的逻辑初始化 prev_scores
        self.prev_scores = torch.rand(num_samples).float().to(device) * 0.5
        self.prev_labels = initial_all_labels

        print(f"AdaptiveRankLoss history initialized with {num_samples} samples.")

    def update_history(self, current_epoch_scores: torch.Tensor, current_epoch_labels: torch.Tensor):
        """
        使用当前 epoch 的结果更新内部存储的上一轮状态。

        Args:
            current_epoch_scores (torch.Tensor): 当前完整 epoch 的预测得分。
            current_epoch_labels (torch.Tensor): 当前完整 epoch 的真实标签。
        """
        # 确保是 Float Tensor 并在 CPU 上 (如果需要与其他部分保持一致)
        device = torch.device("cpu") # 或者根据你的训练环境决定
        self.prev_scores = current_epoch_scores.detach().float().to(device)
        self.prev_labels = current_epoch_labels.detach().float().to(device)
        # print(f"AdaptiveRankLoss history updated with {len(self.prev_scores)} samples.") # 可选的调试信息


    def _huber_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 Huber Loss。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 计算得到的 Huber Loss。
        """
        abs_x = torch.abs(x)
        return torch.where(
            abs_x <= self.huber_delta,
            0.5 * x * x,
            self.huber_delta * (abs_x - 0.5 * self.huber_delta)
        )

    def forward(self,
                current_scores: torch.Tensor,
                current_labels: torch.Tensor,
                margin: torch.Tensor # 直接传入 margin (gamma)
               ) -> torch.Tensor:
        """
        计算 Adaptive Ranking Loss。

        Args:
            current_scores (torch.Tensor): 当前批次的模型预测得分。
            current_labels (torch.Tensor): 当前批次的真实标签 (e.g., 1 for target, 0 for decoy)。
            margin (torch.Tensor): 模型的可学习间隔参数 (gamma)。

        Returns:
            torch.Tensor: 计算得到的损失值。
        """
        # --- 检查历史记录是否已初始化 ---
        if self.prev_scores is None or self.prev_labels is None:
            # 可以在这里返回零损失，或者抛出错误，或者进行某种默认处理
            # 返回零损失可能导致第一个 batch 的梯度为零
            # 更好的方法是确保 initialize_history 在训练开始前被调用
            # 这里我们先返回零损失并打印警告，以防忘记初始化
            # 或者根据实际需要抛出错误：
            # raise RuntimeError("AdaptiveRankLoss history not initialized. Call initialize_history() before the first forward pass.")
            print("Warning: AdaptiveRankLoss history not initialized. Returning zero loss for this batch. Call initialize_history() before training.")
            return torch.tensor(0.0, device=current_scores.device, requires_grad=True) # 确保返回的 Tensor 可以求导

        # --- 将内部状态移动到与当前数据相同的设备 ---
        # （如果 initialize_history 和 update_history 保证了设备一致性，这一步可能非必需，但更健壮）
        device = current_scores.device
        prev_scores = self.prev_scores.to(device)
        prev_labels = self.prev_labels.to(device)

        # --- 参数预处理 ---
        current_is_target = (current_labels >= 0.5)
        prev_is_target = (prev_labels >= 0.5)

        # --- 处理边缘情况 ---
        num_current_targets_total = torch.sum(current_is_target)
        if num_current_targets_total == 0 or num_current_targets_total == current_is_target.shape[0]:
            return torch.sum(current_scores) * 1e-9 # 保持与原函数行为一致

        # --- 分离不同批次和类型的得分 ---
        current_target_scores = current_scores[current_is_target]
        current_decoy_scores = current_scores[~current_is_target]
        prev_target_scores = prev_scores[prev_is_target]
        prev_decoy_scores = prev_scores[~prev_is_target]

        # --- 计算自适应缩放因子 ---
        if len(prev_decoy_scores) > 0:
            # 使用 detach() 避免这部分计算影响 prev_scores 的梯度（理论上不应影响，但更安全）
            max_prev_decoy_score = torch.max(prev_decoy_scores.detach())
            # 计算 gap 时 current_target_scores 不需要 detach
            current_target_vs_prev_decoy_gap = current_target_scores - max_prev_decoy_score
            adaptive_scale = torch.exp(-self.scale_sensitivity * torch.clamp(current_target_vs_prev_decoy_gap, min=0))
        else:
            adaptive_scale = torch.ones_like(current_target_scores)

        # --- 对上一轮的样本进行子采样 ---
        num_prev_targets_total = max(1, len(prev_target_scores)) # 使用原始长度计算概率
        num_prev_decoys_total = max(1, len(prev_decoy_scores))

        target_sample_prob = min(1.0, self.max_prev_target_samples / num_prev_targets_total) # 概率不应超过1
        decoy_sample_prob = min(1.0, self.max_prev_decoy_samples / num_prev_decoys_total)

        # 进行随机子采样 (在当前设备上进行)
        sampled_prev_target_scores = prev_target_scores[
            torch.rand(len(prev_target_scores), device=device) < target_sample_prob
        ]
        sampled_prev_decoy_scores = prev_decoy_scores[
            torch.rand(len(prev_decoy_scores), device=device) < decoy_sample_prob
        ]

        # --- 计算损失的各个组成部分 ---
        loss_parts = []

        # --- 损失部分 1: Current Target vs. Sampled Previous Decoy ---
        num_current_targets = len(current_target_scores)
        num_sampled_prev_decoys = len(sampled_prev_decoy_scores)
        if num_current_targets > 0 and num_sampled_prev_decoys > 0:
            curr_target_vs_prev_decoy_diff = (
                sampled_prev_decoy_scores.view(1, -1) - current_target_scores.view(-1, 1) + margin
            )
            curr_target_vs_prev_decoy_hinge = torch.clamp(curr_target_vs_prev_decoy_diff, min=0)
            curr_target_vs_prev_decoy_huber = self._huber_loss(curr_target_vs_prev_decoy_hinge)
            # 应用自适应缩放
            scaled_loss_part1 = curr_target_vs_prev_decoy_huber * adaptive_scale.view(-1, 1)
            # 归一化分母使用子采样上限，与原函数保持一致，或使用实际采样数量？原实现是上限。
            # 使用上限更稳定，避免分母过小。
            loss_parts.append(torch.sum(scaled_loss_part1) / self.max_prev_target_samples)

        # --- 损失部分 2: Current Decoy vs. Sampled Previous Target ---
        num_current_decoys = len(current_decoy_scores)
        num_sampled_prev_targets = len(sampled_prev_target_scores)
        if num_current_decoys > 0 and num_sampled_prev_targets > 0:
            curr_decoy_vs_prev_target_diff = (
                 current_decoy_scores.view(-1, 1) - sampled_prev_target_scores.view(1, -1) + margin
            )
            curr_decoy_vs_prev_target_hinge = torch.clamp(curr_decoy_vs_prev_target_diff, min=0)
            curr_decoy_vs_prev_target_huber = self._huber_loss(curr_decoy_vs_prev_target_hinge)
            # 归一化分母使用子采样上限
            loss_parts.append(torch.sum(curr_decoy_vs_prev_target_huber) / self.max_prev_decoy_samples)

        # --- 计算最终总损失 ---
        total_loss = sum(loss_parts) if loss_parts else torch.tensor(0.0, device=device)

        # 处理 NaN 情况 (与原函数保持一致)
        # 需要确保 total_loss 需要梯度
        final_loss = torch.where(torch.isnan(total_loss), torch.zeros_like(total_loss), total_loss)

        # 确保最终返回的 loss 具有 requires_grad=True（如果输入有的话）
        # 如果 loss_parts 为空，返回的 tensor(0.0) 默认 requires_grad=False
        # 一个简单的处理方法是加一个依赖输入的零项
        if not final_loss.requires_grad and (current_scores.requires_grad or margin.requires_grad):
             final_loss = final_loss + 0.0 * (current_scores.sum() + margin) # 添加一个依赖输入的零项

        return final_loss
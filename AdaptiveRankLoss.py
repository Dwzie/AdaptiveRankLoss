import torch
import torch.nn as nn

class AdaptiveRankLoss(nn.Module):
    """
    Adaptive Ranking Loss 类，使用上一批次的信息作为上下文。

    将当前批次的分数与上一批次的分数进行比较，以计算自适应损失，
    可能更关注相对于上一批次分数分布而言更难区分的样本。

    Attributes:
        huber_delta (float): Huber Loss 的阈值。
        scale_sensitivity (float): 自适应缩放因子的敏感度参数 (原 alpha)。
        max_prev_target_samples (int): 上一批次 Target 样本子采样的数量上限。
        max_prev_decoy_samples (int): 上一批次 Decoy 样本子采样的数量上限。
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
            max_prev_target_samples (int, optional): 上一批次 Target 样本子采样上限。默认为 10000。
            max_prev_decoy_samples (int, optional): 上一批次 Decoy 样本子采样上限。默认为 10000。
        """
        super().__init__()
        self.huber_delta = huber_delta
        self.scale_sensitivity = scale_sensitivity
        self.max_prev_target_samples = max_prev_target_samples
        self.max_prev_decoy_samples = max_prev_decoy_samples

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
                model: nn.Module, # 需要模型来获取 margin (gamma)
                prev_scores: torch.Tensor,
                prev_labels: torch.Tensor
               ) -> torch.Tensor:
        """
        计算 Adaptive Ranking Loss。

        Args:
            current_scores (torch.Tensor): 当前批次的模型预测得分。
            current_labels (torch.Tensor): 当前批次的真实标签 (e.g., 1 for target, 0 for decoy)。
            model (nn.Module): 包含 margin (gamma) 参数的模型对象。
            prev_scores (torch.Tensor): 上一批次的模型预测得分。
            prev_labels (torch.Tensor): 上一批次的真实标签。

        Returns:
            torch.Tensor: 计算得到的损失值。
        """
        # --- 参数预处理 ---
        current_is_target = (current_labels >= 0.5)
        prev_is_target = (prev_labels >= 0.5)

        # --- 处理边缘情况 ---
        num_current_targets_total = torch.sum(current_is_target)
        if num_current_targets_total == 0 or num_current_targets_total == current_is_target.shape[0]:
            # 如果当前批次全为 Target 或全为 Decoy，返回一个接近零的值
            # 保持与原函数行为一致
            return torch.sum(current_scores) * 1e-9

        # --- 分离不同批次和类型的得分 ---
        current_target_scores = current_scores[current_is_target]
        current_decoy_scores = current_scores[~current_is_target]
        prev_target_scores = prev_scores[prev_is_target]
        prev_decoy_scores = prev_scores[~prev_is_target]

        # --- 计算自适应缩放因子 ---
        if len(prev_decoy_scores) > 0:
            max_prev_decoy_score = torch.max(prev_decoy_scores)
            current_target_vs_prev_decoy_gap = current_target_scores - max_prev_decoy_score
            # 应用缩放敏感度参数
            adaptive_scale = torch.exp(-self.scale_sensitivity * torch.clamp(current_target_vs_prev_decoy_gap, min=0))
        else:
            adaptive_scale = torch.ones_like(current_target_scores)

        # --- 对上一批次的样本进行子采样 ---
        num_prev_targets = max(1, len(prev_target_scores))
        num_prev_decoys = max(1, len(prev_decoy_scores))

        # 计算采样概率
        target_sample_prob = self.max_prev_target_samples / num_prev_targets
        decoy_sample_prob = self.max_prev_decoy_samples / num_prev_decoys

        # 进行随机子采样 (注意: torch.rand_like 返回 [0, 1) 区间的值)
        sampled_prev_target_scores = prev_target_scores[
            torch.rand_like(prev_target_scores) < target_sample_prob
        ]
        sampled_prev_decoy_scores = prev_decoy_scores[
            torch.rand_like(prev_decoy_scores) < decoy_sample_prob
        ]

        # --- 计算损失的各个组成部分 ---
        loss_parts = []
        # 从模型中获取间隔参数 (在你的 ResNeXt 类中是 gamma)
        # 注意：原函数注释写的是 model.margin，但代码调用时用的是 model.gamma
        # 这里我们假设模型中有一个名为 gamma 的参数
        if not hasattr(model, 'gamma'):
             raise AttributeError("模型对象必须包含 'gamma' 属性作为 margin 参数。")
        margin = model.gamma

        # --- 损失部分 1: Current Target vs. Previous Decoy ---
        num_current_targets = len(current_target_scores)
        num_sampled_prev_decoys = len(sampled_prev_decoy_scores)
        if num_current_targets > 0 and num_sampled_prev_decoys > 0:
            curr_target_vs_prev_decoy_diff = (
                sampled_prev_decoy_scores.view(1, -1) - current_target_scores.view(-1, 1) + margin
            )
            curr_target_vs_prev_decoy_hinge = torch.clamp(curr_target_vs_prev_decoy_diff, min=0)
            curr_target_vs_prev_decoy_huber = self._huber_loss(curr_target_vs_prev_decoy_hinge)
            scaled_loss_part1 = curr_target_vs_prev_decoy_huber * adaptive_scale.view(-1, 1)
            # 归一化分母使用子采样上限，与原函数保持一致
            loss_parts.append(torch.sum(scaled_loss_part1) / self.max_prev_target_samples)

        # --- 损失部分 2: Current Decoy vs. Previous Target ---
        num_current_decoys = len(current_decoy_scores)
        num_sampled_prev_targets = len(sampled_prev_target_scores)
        if num_current_decoys > 0 and num_sampled_prev_targets > 0:
            curr_decoy_vs_prev_target_diff = (
                 current_decoy_scores.view(-1, 1) - sampled_prev_target_scores.view(1, -1) + margin
            )
            curr_decoy_vs_prev_target_hinge = torch.clamp(curr_decoy_vs_prev_target_diff, min=0)
            curr_decoy_vs_prev_target_huber = self._huber_loss(curr_decoy_vs_prev_target_hinge)
            # 归一化分母使用子采样上限，与原函数保持一致
            loss_parts.append(torch.sum(curr_decoy_vs_prev_target_huber) / self.max_prev_decoy_samples)

        # --- 计算最终总损失 ---
        total_loss = sum(loss_parts) if loss_parts else torch.tensor(0.0, device=current_scores.device)

        # 处理 NaN 情况 (与原函数保持一致)
        return torch.where(torch.isnan(total_loss), torch.zeros_like(total_loss), total_loss)
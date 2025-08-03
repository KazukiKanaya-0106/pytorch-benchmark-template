import torch
import torch.nn as nn
from typing import Literal


class BinaryFocalLoss(nn.Module):
    """
    BinaryFocalLoss
    """

    def __init__(
        self,
        alpha: list[float],
        gamma: float,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """
        BinaryFocalLossを初期化する
        """
        super().__init__()
        self.alpha = torch.tensor(alpha)  # クラスごとの重みα
        self.gamma = torch.tensor(gamma)
        self.reduction = reduction

    def forward(self, predicted_probs: torch.Tensor, collect_labels: torch.Tensor) -> torch.Tensor:
        """
        BinaryFocalLossを計算する

        Args:
            predicted_probs (torch.Tensor): ネットワークの予測確率 (B, 1, H, W) 0~1の確率
            collect_labels (torch.Tensor): 正解ラベル (B, 1, H, W) 0 or 1

        Returns:
            torch.Tensor: 損失値 スカラー値
        """
        # 正解ラベルの確率 (B, 1, H, W)
        correct_label_probs = collect_labels * predicted_probs + (1 - collect_labels) * (1 - predicted_probs)

        # α の適用（正解ラベル -> α[0], 不正解ラベル -> α[1]）
        alpha_t = self.alpha[0] * collect_labels + self.alpha[1] * (1 - collect_labels)

        # 数値安定性のために log の内部を制限
        smoothing = 1e-8
        correct_label_probs = torch.clamp(correct_label_probs, smoothing, 1.0 - smoothing)

        # 損失の計算
        loss = -alpha_t * ((1 - correct_label_probs) ** self.gamma) * torch.log(correct_label_probs)

        match self.reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case "none":
                return loss
            case _:
                raise ValueError(f"Invalid reduction: {self.reduction}")

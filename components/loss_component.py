import torch.nn as nn
from losses.binary_focal_loss import BinaryFocalLoss


class LossComponent:
    def __init__(self, loss_name: str, loss_config: dict) -> None:
        self._loss = self.build_loss(
            loss_name=loss_name,
            loss_config=loss_config,
        )

    def build_loss(self, loss_name: str, loss_config: dict) -> nn.Module:
        match loss_name:
            case "cross_entropy_loss":
                return nn.CrossEntropyLoss(
                    reduction=loss_config["reduction"],
                )
            case "binary_cross_entropy_loss":
                return nn.BCELoss(
                    reduction=loss_config["reduction"],
                )
            case "binary_focal_loss":
                return BinaryFocalLoss(
                    alpha=loss_config["alpha"],
                    gamma=loss_config["gamma"],
                    reduction=loss_config["reduction"],
                )
            case _:
                raise ValueError(f"Unsupported loss: {loss_name}")

    @property
    def loss(self) -> nn.Module:
        return self._loss

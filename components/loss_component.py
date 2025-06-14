import torch.nn as nn


class LossComponent:
    def __init__(self, config: dict) -> None:
        self.loss: nn.Module = self.build_loss(config)

    def build_loss(self, config: dict) -> nn.Module:
        loss_name: str = config["training"]["loss"]
        loss_config: dict = config["loss"][loss_name]
        loss: nn.Module = None
        match loss_name:
            case "cross_entropy":
                loss = nn.CrossEntropyLoss(
                    reduction=loss_config["reduction"],
                )
            case _:
                raise ValueError(f"Unsupported loss: {loss_name}")
        return loss

    def get(self) -> nn.Module:
        return self.loss

import torch.nn as nn
import torch.optim as optim


class OptimizerComponent:
    def __init__(self, config: dict, model: nn.Module) -> None:
        self.optimizer = self.build_optimizer(config, model)

    def build_optimizer(self, config: dict, model: nn.Module) -> optim.Optimizer:
        optimizer_name: str = config["training"]["optimizer"]
        optimizer_config: dict = config["optimizer"][optimizer_name]
        optimizer: optim.Optimizer = None
        match optimizer_name:
            case "adam":
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=optimizer_config["learning_rate"],
                    weight_decay=optimizer_config["weight"],
                )
            case "sgd":
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=optimizer_config["learning_rate"],
                    weight_decay=optimizer_config["weight"],
                )
            case _:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizer

    def get(self) -> optim.Optimizer:
        return self.optimizer

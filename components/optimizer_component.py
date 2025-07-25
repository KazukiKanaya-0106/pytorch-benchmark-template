from torch.nn import Module
import torch.optim as optim


class OptimizerComponent:
    def __init__(self, optimizer_name: str, optimizer_config: dict, model: Module) -> None:
        self._optimizer = self._build_optimizer(
            optimizer_name=optimizer_name,
            optimizer_config=optimizer_config,
            model=model,
        )

    def _build_optimizer(self, optimizer_name: str, optimizer_config: dict, model: Module) -> optim.Optimizer:

        match optimizer_name:
            case "adam":
                return optim.Adam(
                    model.parameters(),
                    lr=float(optimizer_config["learning_rate"]),
                    weight_decay=float(optimizer_config["weight_decay"]),
                )
            case "sgd":
                return optim.SGD(
                    model.parameters(),
                    lr=float(optimizer_config["learning_rate"]),
                    weight_decay=float(optimizer_config["weight_decay"]),
                )
            case _:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

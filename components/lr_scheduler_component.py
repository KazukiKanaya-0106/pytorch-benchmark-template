from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, LRScheduler
from typing import Literal


class LRSchedulerComponent:
    def __init__(
        self, lr_scheduler_name: str, lr_scheduler_config: dict, mode: Literal["min", "max"], optimizer: Optimizer
    ) -> None:
        self._lr_scheduler = self._build_lr_scheduler(
            lr_scheduler_name=lr_scheduler_name,
            lr_scheduler_config=lr_scheduler_config,
            mode=mode,
            optimizer=optimizer,
        )

    def _build_lr_scheduler(
        self, lr_scheduler_name: str, lr_scheduler_config: dict, mode: Literal["min", "max"], optimizer: Optimizer
    ) -> LRScheduler | None:
        if lr_scheduler_name is None:
            return None

        match lr_scheduler_name:
            case "cosine_annealing":
                return CosineAnnealingLR(
                    optimizer,
                    T_max=lr_scheduler_config["T_max"],
                    eta_min=float(lr_scheduler_config["eta_min"]),
                    last_epoch=lr_scheduler_config["last_epoch"],
                )
            case "step_lr":
                return StepLR(
                    optimizer,
                    step_size=lr_scheduler_config["step_size"],
                    gamma=lr_scheduler_config["gamma"],
                    last_epoch=lr_scheduler_config["last_epoch"],
                )
            case "reduce_on_plateau":
                return ReduceLROnPlateau(
                    optimizer,
                    mode=mode,
                    factor=lr_scheduler_config["factor"],
                    patience=lr_scheduler_config["patience"],
                    min_lr=float(lr_scheduler_config["min_lr"]),
                )
            case _:
                raise ValueError(f"Unsupported LR scheduler: {lr_scheduler_name}")

    @property
    def lr_scheduler(self) -> LRScheduler | None:
        return self._lr_scheduler

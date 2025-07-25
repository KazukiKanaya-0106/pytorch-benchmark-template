from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, _LRScheduler
from typing import Optional, Literal


class SchedulerComponent:
    def __init__(
        self, scheduler_name: str, scheduler_config: dict, mode: Literal["min", "max"], optimizer: Optimizer
    ) -> None:
        self._scheduler = self._build_scheduler(
            scheduler_name=scheduler_name,
            scheduler_config=scheduler_config,
            mode=mode,
            optimizer=optimizer,
        )

    def _build_scheduler(
        self, scheduler_name: str, scheduler_config: dict, mode: Literal["min", "max"], optimizer: Optimizer
    ) -> Optional[_LRScheduler]:
        if scheduler_name is None:
            return None

        match scheduler_name:
            case "cosine_annealing":
                return CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_config["T_max"],
                    eta_min=float(scheduler_config["eta_min"]),
                    last_epoch=scheduler_config["last_epoch"],
                    verbose=True,
                )
            case "step_lr":
                return StepLR(
                    optimizer,
                    step_size=scheduler_config["step_size"],
                    gamma=scheduler_config["gamma"],
                    last_epoch=scheduler_config["last_epoch"],
                    verbose=True,
                )
            case "reduce_on_plateau":
                return ReduceLROnPlateau(
                    optimizer,
                    mode=mode,
                    factor=scheduler_config["factor"],
                    patience=scheduler_config["patience"],
                    verbose=True,
                    min_lr=float(scheduler_config["min_lr"]),
                )
            case _:
                raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    @property
    def scheduler(self) -> Optional[_LRScheduler]:
        return self._scheduler

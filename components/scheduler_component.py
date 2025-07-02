from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from typing import Optional


class SchedulerComponent:
    def __init__(self, scheduler_name: str, scheduler_config: dict, optimizer: Optimizer) -> None:
        self._scheduler = self._build_scheduler(
            scheduler_name=scheduler_name,
            scheduler_config=scheduler_config,
            optimizer=optimizer,
        )

    def _build_scheduler(
        self, scheduler_name: str, scheduler_config: dict, optimizer: Optimizer
    ) -> Optional[_LRScheduler]:
        if scheduler_name is None:
            return None

        match scheduler_name:
            case "cosine_annealing":
                return CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_config["T_max"],
                    eta_min=scheduler_config["eta_min"],
                    last_epoch=scheduler_config["last_epoch"],
                    verbose=True,
                )
            case _:
                raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    @property
    def scheduler(self) -> Optional[_LRScheduler]:
        return self._scheduler

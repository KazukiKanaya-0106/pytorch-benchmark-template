from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


class SchedulerComponent:
    def __init__(self, config: dict, optimizer: Optimizer) -> None:
        self.scheduler = self.build_scheduler(config, optimizer)

    def build_scheduler(self, config: dict, optimizer: Optimizer):
        scheduler_name: str = config["training"]["scheduler"]
        scheduler_config: dict = config["scheduler"][scheduler_name]

        match scheduler_name:
            case "cosine_annealing":
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_config["T_max"],
                    eta_min=scheduler_config["eta_min"],
                    last_epoch=-1,
                    verbose=True,
                )
            case None | "none":
                scheduler = None
            case _:
                raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")

        return scheduler

    def get(self):
        return self.scheduler

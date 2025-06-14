from torch.nn import Module
from torch.optim import Optimizer
from torchmetrics import Metric

from .scheduler_component import SchedulerComponent
from .loss_component import LossComponent
from .model_component import ModelComponent
from .optimizer_component import OptimizerComponent
from .metrics_component import MetricsComponent


class ComponentBuilder:
    def __init__(self, config: dict) -> None:
        self.model: Module = ModelComponent(config).get()
        self.loss: Module = LossComponent(config).get()
        self.optimizer: Optimizer = OptimizerComponent(
            config=config,
            model=self.model,
        ).get()
        self.scheduler = SchedulerComponent(
            config=config,
            optimizer=self.optimizer,
        ).get()
        self.metrics: list[Metric] = MetricsComponent(config).get()

    def get_model(self) -> Module:
        return self.model

    def get_loss(self) -> Module:
        return self.loss

    def get_optimizer(self) -> Optimizer:
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

    def get_metrics(self):
        return self.metrics

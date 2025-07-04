from typing import Any, Literal
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchmetrics import Metric
from tqdm import tqdm
import contextlib
from utils import TorchUtils


class EpochTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        metrics: list[Metric],
        optimizer: Optimizer,
        scheduler,
        device: torch.device,
        save_best_monitor: str,
        monitor_task: Literal["min", "max"],
        data_loader: DataLoader,
        mode: Literal["training", "validation"],
    ) -> None:
        self.model: nn.Module = model
        self.loss_fn: nn.Module = loss_fn
        self.metrics: list[Metric] = metrics
        self.optimizer: Optimizer = optimizer
        self.data_loader: DataLoader = data_loader
        self.mode: Literal["training", "validation"] = mode
        self.scheduler: Any = scheduler
        self.device = device
        self.save_best_monitor: str = save_best_monitor

    def fit(self) -> dict:

        current_lr = self.optimizer.param_groups[0]["lr"]

        if self.mode == "training":
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0

        total_grad = 0.0
        num_params = 0.0

        metric_values = {}

        for metric in self.metrics:
            metric.reset()

        loop = tqdm(
            self.data_loader,
            desc=f"{self.mode.capitalize()}",
            leave=True,
            dynamic_ncols=True,
        )

        with torch.no_grad() if self.mode == "validation" else contextlib.nullcontext():
            for inputs, labels in loop:
                inputs = TorchUtils.move_to_device(obj=inputs, device=self.device)
                labels = TorchUtils.move_to_device(obj=labels, device=self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                if self.mode == "training":
                    self.optimizer.zero_grad()
                    loss.backward()

                    for param in self.model.parameters():
                        if param.grad is not None:
                            total_grad += param.grad.abs().mean().item()
                            num_params += 1

                    self.optimizer.step()

                for metric in self.metrics:
                    metric.update(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                postfix = {type(self.loss_fn).__name__: loss.item()}
                metric_values = {
                    metric.__name__: metric.compute().item() for metric in self.metrics
                }
                postfix.update(metric_values)
                loop.set_postfix(postfix)

            avg_loss = total_loss / total_samples
            avg_grad = total_grad / num_params if num_params > 0 else 0.0

            result = {
                "average_loss": float(avg_loss),
                **metric_values,
            }
            if self.mode == "training" and self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    monitor = result[self.save_best_monitor]
                    self.scheduler.step(monitor)
                else:
                    self.scheduler.step()

            # training 固有のログ（gradient, learning rate）
            if self.mode == "training":
                result["average_grad"] = float(avg_grad)
                result["learning_rate"] = float(current_lr)

            return result

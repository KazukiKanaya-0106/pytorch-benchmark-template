from typing import Literal, Callable
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
        device: torch.device,
        data_loader: DataLoader,
        mode: Literal["training", "validation"],
        forward_fn: Callable = lambda model, X: model(X),
    ) -> None:
        self.model: nn.Module = model
        self.loss_fn: nn.Module = loss_fn
        self.metrics: list[Metric] = metrics
        self.optimizer: Optimizer = optimizer
        self.data_loader: DataLoader = data_loader
        self.mode: Literal["training", "validation"] = mode
        self.device = device
        self.forward_fn = forward_fn

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
            for batch in loop:
                X, y = TorchUtils.split_batch(batch)
                X = TorchUtils.move_to_device(obj=X, device=self.device)
                y = TorchUtils.move_to_device(obj=y, device=self.device)

                outputs = self.forward_fn(self.model, X)
                loss = self.loss_fn(outputs, y)

                if self.mode == "training":
                    self.optimizer.zero_grad()
                    loss.backward()

                    for param in self.model.parameters():
                        if param.grad is not None:
                            total_grad += param.grad.abs().mean().item()
                            num_params += 1

                    self.optimizer.step()

                for metric in self.metrics:
                    metric.update(outputs, y)

                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)

                metric_values = {metric.__name__: metric.compute().item() for metric in self.metrics}
                postfix = {
                    type(self.loss_fn).__name__: loss.item(),
                    **metric_values,
                }
                loop.set_postfix(postfix)

            avg_loss = total_loss / total_samples
            avg_grad = total_grad / num_params if num_params > 0 else 0.0

            result = {
                "average_loss": float(avg_loss),
                **metric_values,
            }

            # training 固有のログ（average_grad, learning_rate）
            if self.mode == "training":
                result |= {
                    "average_grad": float(avg_grad),
                    "learning_rate": float(current_lr),
                }

            return result

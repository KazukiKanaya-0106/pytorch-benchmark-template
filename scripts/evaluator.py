from typing import Literal, Callable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchmetrics import Metric
from tqdm import tqdm
from utils import TorchUtils


class _Evaluator:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        metrics: list[Metric],
        optimizer: Optimizer,
        device: torch.device,
        data_loader: DataLoader,
        forward_fn: Callable = lambda model, X: model(X),
        description: str = "Evaluation",
    ) -> None:
        self.model: nn.Module = model
        self.loss_fn: nn.Module = loss_fn
        self.metrics: list[Metric] = metrics
        self.optimizer: Optimizer = optimizer
        self.data_loader: DataLoader = data_loader
        self.device = device
        self.forward_fn = forward_fn
        self.description: str = description

    def evaluate(self) -> dict:

        current_lr = self.optimizer.param_groups[0]["lr"]

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        metric_values = {}

        for metric in self.metrics:
            metric.reset()

        loop = tqdm(
            self.data_loader,
            desc=f"{self.description.capitalize()}",
            leave=True,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch in loop:
                X, y = TorchUtils.split_batch(batch)
                X = TorchUtils.move_to_device(obj=X, device=self.device)
                y = TorchUtils.move_to_device(obj=y, device=self.device)

                outputs = self.forward_fn(self.model, X)
                loss = self.loss_fn(outputs, y)

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

            return {
                "average_loss": float(avg_loss),
                **metric_values,
            }


class Validator(_Evaluator):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        metrics: list[Metric],
        optimizer: Optimizer,
        device: torch.device,
        data_loader: DataLoader,
        forward_fn: Callable = lambda model, X: model(X),
        description: str = "Validation",
    ) -> None:
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            data_loader=data_loader,
            forward_fn=forward_fn,
            description=description,
        )

    def validate(self) -> dict:
        result = super().evaluate()
        return result


class Tester(_Evaluator):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        metrics: list[Metric],
        optimizer: Optimizer,
        device: torch.device,
        data_loader: DataLoader,
        forward_fn: Callable = lambda model, X: model(X),
        description: str = "Testing",
    ) -> None:
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            data_loader=data_loader,
            forward_fn=forward_fn,
            description=description,
        )

    def test(self) -> dict:
        result = super().evaluate()
        return result

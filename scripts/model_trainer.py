import os
from io import BytesIO
from typing import Literal, Callable
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Metric
from scripts.epoch_trainer import EpochTrainer
from scripts.early_stopper import EarlyStopper
from utils import TensorBoard, DataStructureUtils, MLflow, TorchUtils


class ModelTrainer:
    def __init__(
        self,
        device: torch.device,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        model: Module,
        loss_fn: Module,
        metrics: list[Metric],
        optimizer: Optimizer,
        scheduler,
        early_stopper: EarlyStopper | None,
        save_best_monitor: str,
        monitor_task: Literal["min", "max"],
        best_weight_source: str | BytesIO,
        forward_fn: Callable = lambda model, X: model(X),
        tensorboard: TensorBoard | None = None,
        mlflow: MLflow | None = None,
    ) -> None:
        self.train_epoch: EpochTrainer = EpochTrainer(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            mode="training",
            forward_fn=forward_fn,
        )
        self.valid_epoch: EpochTrainer = EpochTrainer(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            data_loader=valid_loader,
            device=device,
            mode="validation",
            forward_fn=forward_fn,
        )
        self.test_epoch: EpochTrainer = EpochTrainer(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            data_loader=test_loader,
            device=device,
            mode="validation",
            forward_fn=forward_fn,
        )
        self.model: Module = model.to(device)
        self.optimizer: Optimizer = optimizer
        self.scheduler: _LRScheduler = scheduler
        self.early_stopper: EarlyStopper | None = early_stopper
        self.best_weight_source: str | BytesIO = best_weight_source
        self.epochs: int = epochs
        self.save_best_monitor: str = save_best_monitor
        self.monitor_task: Literal["min", "max"] = monitor_task
        self.tensorboard: TensorBoard | None = tensorboard
        self.mlflow: MLflow | None = mlflow

    def fit(self) -> dict:

        best_score = float("-inf") if self.monitor_task == "max" else float("inf")
        best_epoch = 0
        for epoch in range(1, self.epochs + 1, 1):
            print(f"\nEpoch {epoch} / {self.epochs}")
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"learning_rate: {current_lr:.8f}")
            train_log = self.train_epoch.fit()
            valid_log = self.valid_epoch.fit()

            monitor_score = valid_log[self.save_best_monitor]

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(monitor_score)
                else:
                    self.scheduler.step()

            is_better: bool = TorchUtils.is_better_score(
                score=monitor_score,
                best=best_score,
                task=self.monitor_task,
            )

            if is_better or epoch == 1:
                best_score = monitor_score
                best_epoch = epoch

                TorchUtils.save_model_state(
                    model=self.model,
                    destination=self.best_weight_source,
                )
                print("\nUpdate best weight!\n")

            for prefix, log in [("train", train_log), ("valid", valid_log)]:
                log_prefixed = DataStructureUtils.add_prefix(log, prefix)
                if self.tensorboard:
                    self.tensorboard.log_metrics(metrics=log_prefixed, step=epoch)
                if self.mlflow:
                    self.mlflow.log_metrics(metrics=log_prefixed, step=epoch)

            if self.early_stopper is not None:
                if self.early_stopper(monitor_score):
                    break

        TorchUtils.load_model_state(model=self.model, source=self.best_weight_source)

        test_log: dict = self.test_epoch.fit()
        test_log = DataStructureUtils.convert_to_builtin_types(test_log)
        test_log_prefixed = DataStructureUtils.add_prefix(test_log, "test")
        test_log_prefixed["best_epoch"] = best_epoch

        return test_log_prefixed

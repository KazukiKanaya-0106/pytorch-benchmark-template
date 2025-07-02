import os
from io import BytesIO
from typing import Literal
import torch
from torch.nn import Module
from torch.optim import Optimizer
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
        tensorboard: TensorBoard | None = None,
        mlflow: MLflow | None = None,
    ) -> None:
        self.train_epoch: EpochTrainer = EpochTrainer(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_loader,
            save_best_monitor=save_best_monitor,
            monitor_task=monitor_task,
            device=device,
            mode="training",
        )
        self.valid_epoch: EpochTrainer = EpochTrainer(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=valid_loader,
            save_best_monitor=save_best_monitor,
            monitor_task=monitor_task,
            device=device,
            mode="validation",
        )
        self.test_epoch: EpochTrainer = EpochTrainer(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=test_loader,
            save_best_monitor=save_best_monitor,
            monitor_task=monitor_task,
            device=device,
            mode="validation",
        )
        self.model: Module = model.to(device)
        self.optimizer: Optimizer = optimizer
        self.early_stopper: EarlyStopper | None = early_stopper
        self.best_weight_source: str | BytesIO = best_weight_source
        self.epochs: int = epochs
        self.save_best_monitor: str = save_best_monitor
        self.monitor_task: Literal["min", "max"] = monitor_task
        self.tensorboard: TensorBoard | None = tensorboard
        self.mlflow: MLflow | None = mlflow

    def fit(self) -> dict:
        if self.mlflow:
            self.mlflow.start_run()

        best_score = 0 if self.monitor_task == "max" else float("-inf")
        best_epoch = 0
        for epoch in range(1, self.epochs + 1, 1):
            print(f"\nEpoch {epoch} / {self.epochs}")
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"learning_rate: {current_lr:.8f}")
            train_log = self.train_epoch.fit()
            valid_log = self.valid_epoch.fit()

            monitor_score = valid_log[self.save_best_monitor]

            is_better: bool = TorchUtils.is_better_score(
                score=monitor_score,
                best=best_score,
                task=self.monitor_task,
            )

            if is_better:
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
                if self.early_stopper(valid_log[self.early_stopper.monitor]):
                    break

        if self.tensorboard:
            self.tensorboard.close()

        TorchUtils.load_model_state(model=self.model, source=self.best_weight_source)

        test_log: dict = self.test_epoch.fit()
        test_log = DataStructureUtils.convert_to_builtin_types(test_log)
        test_log_prefixed = DataStructureUtils.add_prefix(test_log, "test")
        test_log_prefixed["best_epoch"] = best_epoch

        print("\nTest Results:")
        print("=" * 50)
        for key, value in test_log_prefixed.items():
            if isinstance(value, float):
                print(f"{key:<20}: {value:>10.4f}")
            else:
                print(f"{key:<20}: {value:>10}")
        print("=" * 50)

        return test_log_prefixed

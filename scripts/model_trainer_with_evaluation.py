import copy
from io import BytesIO
from typing import Literal, Callable
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Metric
from scripts.epoch_wise_trainer import EpochWiseTrainer
from scripts.evaluator import Validator, Tester
from scripts.early_stopper import EarlyStopper

from utils.torch_utils import TorchUtils
from utils.data_structure_utils import DataStructureUtils
from utils.tensorboard import TensorBoard
from utils.mlflow import MLflow


class ModelTrainerWithEvaluation:
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
        lr_scheduler: LRScheduler | None,
        early_stopper: EarlyStopper | None,
        save_best_monitor: str,
        monitor_task: Literal["min", "max"],
        forward_fn: Callable = lambda model, X: model(X),
        tensorboard: TensorBoard | None = None,
        mlflow: MLflow | None = None,
    ) -> None:
        self.epoch_wise_trainer = EpochWiseTrainer(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            forward_fn=forward_fn,
            description="Training",
        )
        self.validator = Validator(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            data_loader=valid_loader,
            device=device,
            forward_fn=forward_fn,
            description="Validation",
        )
        self.tester = Tester(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            data_loader=test_loader,
            device=device,
            forward_fn=forward_fn,
            description="Testing",
        )
        self.model: Module = model.to(device)
        self.optimizer: Optimizer = optimizer
        self.lr_scheduler: LRScheduler | None = lr_scheduler
        self.early_stopper: EarlyStopper | None = early_stopper
        self.epochs: int = epochs
        self.save_best_monitor: str = save_best_monitor
        self.monitor_task: Literal["min", "max"] = monitor_task
        self.tensorboard: TensorBoard | None = tensorboard
        self.mlflow: MLflow | None = mlflow

    def fit(self) -> dict:

        best_train_log: dict = {}
        best_train_epoch: int = 0
        best_train_score: float = float("-inf") if self.monitor_task == "max" else float("inf")
        best_train_weight: dict = {}

        best_valid_log: dict = {}
        best_valid_epoch: int = 0
        best_valid_score: float = float("-inf") if self.monitor_task == "max" else float("inf")
        best_valid_weight: dict = {}

        for epoch in range(1, self.epochs + 1, 1):
            print(f"\nEpoch {epoch} / {self.epochs}")
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"learning_rate: {current_lr:.8f}")

            train_log: dict = self.epoch_wise_trainer.fit()
            valid_log: dict = self.validator.validate()

            train_monitor_score = train_log[self.save_best_monitor]
            valid_monitor_score = valid_log[self.save_best_monitor]

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(valid_monitor_score)
                else:
                    self.lr_scheduler.step()

            is_better_train: bool = TorchUtils.is_better_score(
                score=train_monitor_score,
                best=best_train_score,
                task=self.monitor_task,
            )
            is_better_valid: bool = TorchUtils.is_better_score(
                score=valid_monitor_score,
                best=best_valid_score,
                task=self.monitor_task,
            )

            if is_better_train or epoch == 1:
                best_train_score = train_monitor_score
                best_train_log = train_log
                best_train_epoch = epoch
                best_train_weight = copy.deepcopy(self.model.state_dict())

            if is_better_valid or epoch == 1:
                best_valid_score = valid_monitor_score
                best_valid_log = valid_log
                best_valid_epoch = epoch
                best_valid_weight = copy.deepcopy(self.model.state_dict())
                print("\nUpdate best validation weight!\n")

            for prefix, log in [("train", train_log), ("valid", valid_log)]:
                log_prefixed = DataStructureUtils.add_prefix(log, prefix)
                if self.tensorboard:
                    self.tensorboard.log_metrics(metrics=log_prefixed, step=epoch)
                if self.mlflow:
                    self.mlflow.log_metrics(metrics=log_prefixed, step=epoch)

            if self.early_stopper is not None:
                if self.early_stopper(valid_monitor_score):
                    break

        return {
            "best_train_log": DataStructureUtils.convert_to_builtin_types(
                best_train_log | {"best_epoch": best_train_epoch}
            ),
            "best_valid_log": DataStructureUtils.convert_to_builtin_types(
                best_valid_log | {"best_epoch": best_valid_epoch}
            ),
            "best_train_weight": best_train_weight,
            "best_valid_weight": best_valid_weight,
        }

    def test(self, weight_source: str | BytesIO | dict | None = None) -> dict:
        if weight_source is not None:
            TorchUtils.load_model_state(model=self.model, source=weight_source)
        test_result: dict = self.tester.test()
        test_result = DataStructureUtils.convert_to_builtin_types(test_result)

        return test_result

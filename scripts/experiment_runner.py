import os
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from scripts.epoch_runner import EpochRunner
from scripts.early_stopper import EarlyStopper
from utils import TensorBoard, DataStructureUtils, FileUtils, MLflow, TorchUtils


class ExperimentRunner:
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
        save_best_metric: str,
        output_file_dir: str,
        tensorboard: TensorBoard,
        mlflow: MLflow,
    ) -> None:
        self.train_runner: EpochRunner = EpochRunner(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_loader,
            save_best_metric=save_best_metric,
            device=device,
            mode="training",
        )
        self.valid_runner: EpochRunner = EpochRunner(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=valid_loader,
            save_best_metric=save_best_metric,
            device=device,
            mode="validation",
        )
        self.test_runner: EpochRunner = EpochRunner(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=test_loader,
            save_best_metric=save_best_metric,
            device=device,
            mode="validation",
        )
        self.model: Module = model.to(device)
        self.optimizer: Optimizer = optimizer
        self.early_stopper = early_stopper
        self.output_file_dir: str = output_file_dir
        self.epochs = epochs
        self.save_best_metric = save_best_metric
        self.tensorboard = tensorboard
        self.mlflow = mlflow

    def run(self) -> None:
        self.mlflow.start_run()
        os.makedirs(self.output_file_dir, exist_ok=True)
        max_score = 0
        best_epoch = 0
        for epoch in range(1, self.epochs + 1, 1):
            print(f"\nEpoch {epoch} / {self.epochs}")
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"learning_rate: {current_lr:.8f}")
            train_log = self.train_runner.run_epoch()
            valid_log = self.valid_runner.run_epoch()

            save_best_metric_score = valid_log[self.save_best_metric]
            if save_best_metric_score > max_score:

                max_score = save_best_metric_score
                best_epoch = epoch

                TorchUtils.save_model_state(
                    self.model,
                    f"{self.output_file_dir}/best_weight.pth",
                )
                print("\nUpdate best weight!\n")

            train_log_prefixed = DataStructureUtils.add_prefix(train_log, "train")
            valid_log_prefixed = DataStructureUtils.add_prefix(valid_log, "valid")

            self.tensorboard.log_metrics(metrics=train_log_prefixed, step=epoch)
            self.tensorboard.log_metrics(metrics=valid_log_prefixed, step=epoch)

            self.mlflow.log_metrics(metrics=train_log_prefixed, step=epoch)
            self.mlflow.log_metrics(metrics=valid_log_prefixed, step=epoch)

            if self.early_stopper is not None:
                if self.early_stopper(train_log["average_loss"]):
                    break

        self.tensorboard.close()

        best_weight_path: str = f"{self.output_file_dir}/best_weight.pth"
        self.model.load_state_dict(torch.load(best_weight_path))

        test_log: dict = self.test_runner.run_epoch()
        test_log["best_epoch"] = best_epoch

        test_log = DataStructureUtils.convert_to_builtin_types(test_log)
        FileUtils.save_dict_to_yaml(
            dictionary=test_log, path=f"{self.output_file_dir}/test_log.yml"
        )

        test_log_prefixed = DataStructureUtils.add_prefix(test_log, "test")
        self.mlflow.log_metrics(metrics=test_log_prefixed)
        self.mlflow.log_artifact(f"{self.output_file_dir}/best_weight.pth")
        self.mlflow.log_artifact(f"{self.output_file_dir}/test_log.yml")
        self.mlflow.log_artifact(f"{self.output_file_dir}/config_backup.yml")
        self.mlflow.end_run()

        for key, value in test_log.items():
            if isinstance(value, float):
                print(f"{key:<15}: {value:.4f}")
            else:
                print(f"{key:<15}: epoch {value}")

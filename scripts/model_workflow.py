import os
import warnings

warnings.filterwarnings("ignore")
from typing import Callable
from contextlib import nullcontext
import torch
from typing import Literal
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LRScheduler

from components.dataloader_component import DataLoaderComponent
from components.dataset_component import DatasetComponent
from components.loss_component import LossComponent
from components.lr_scheduler_component import LRSchedulerComponent
from components.metrics_component import MetricsComponent
from components.model_component import ModelComponent
from components.optimizer_component import OptimizerComponent

from scripts.model_trainer_with_evaluation import ModelTrainerWithEvaluation
from scripts.early_stopper import EarlyStopper

from utils.torch_utils import TorchUtils
from utils.file_utils import FileUtils
from utils.tensorboard import TensorBoard
from utils.mlflow import MLflow
from utils.display_utils import DisplayUtils


class ModelWorkflow:
    def __init__(
        self,
        config: dict,
        key: str | None = None,
        output_dir: str | None = None,
        sub_dir: str | None = None,
    ) -> None:

        self._config = config

        self._key: str = key or config["meta"]["key"]
        self._seed: int = config["meta"]["seed"] or 42
        self._epochs: int = config["training"]["epochs"]
        self._save_best_monitor: str = config["evaluation"]["save_best_monitor"]
        self._monitor_task: Literal["min", "max"] = config["evaluation"]["monitor_task"]
        self._device: torch.device = self._init_device(config["meta"]["device"])

        self._output_dir: str = output_dir or self._prepare_output_dir()
        self._sub_dir: str | None = sub_dir
        if self._sub_dir:
            self._output_dir = os.path.join(self._output_dir, self._sub_dir)
        os.makedirs(self._output_dir, exist_ok=True)
        FileUtils.save_dict_to_yaml(config, os.path.join(self._output_dir, "original_config_backup.yml"))

        self._generator: torch.Generator | None = None

        self._train_loader: DataLoader | None = None
        self._valid_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None
        self._test_dataset: Dataset | None = None
        self._model: nn.Module | None = None
        self._loss_fn: nn.Module | None = None
        self._metrics: list | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._lr_scheduler: LRScheduler | None = None
        self._early_stopper: EarlyStopper | None = None

        self._tensorboard: TensorBoard | None = None
        self._mlflow: MLflow | None = None

    def _init_device(self, requested: str) -> torch.device:
        if requested == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    def _prepare_output_dir(self) -> str:
        path = os.path.join(
            self._config["logging"]["root_dir"],
            self._config["logging"]["experiment_assets"]["dir"],
            self._key,
        )
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def forward_fn(self) -> Callable:
        return TorchUtils.resolve_forward_fn(self._config["training"]["dataset"])

    def setup_seed(self) -> None:
        if self._seed:
            self._generator = TorchUtils.setup_seed_with_generator(self._seed)

    def init_dataloaders(self) -> None:
        c = self._config
        datasets = DatasetComponent(
            c["training"]["dataset"], c["dataset"][c["training"]["dataset"]], self._seed
        ).datasets
        self._test_dataset = datasets[2]
        loaders = DataLoaderComponent(
            loader_config=c["dataset"][c["training"]["dataset"]]["loader"],
            train_dataset=datasets[0],
            valid_dataset=datasets[1],
            test_dataset=datasets[2],
            generator=self._generator,
        ).loaders
        self._train_loader, self._valid_loader, self._test_loader = loaders

    def init_components(
        self,
        model: nn.Module | None = None,
        loss_fn: nn.Module | None = None,
        metrics: list | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: LRScheduler | None = None,
        early_stopper: EarlyStopper | None = None,
    ) -> None:
        c = self._config

        self._model = (
            model
            or ModelComponent(
                model_name=c["training"]["model"],
                model_config=c["model"][c["training"]["model"]],
                device=self._device,
                weight_source=c["training"]["weight"],
            ).model
        )

        self._loss_fn = (
            loss_fn
            or LossComponent(
                loss_name=c["training"]["loss"],
                loss_config=c["loss"][c["training"]["loss"]],
            ).loss
        )

        self._metrics = (
            metrics
            or MetricsComponent(
                evaluation_config=c["evaluation"],
                num_classes=c["dataset"][c["training"]["dataset"]]["num_classes"],
                device=self._device,
            ).metrics
        )

        self._optimizer = (
            optimizer
            or OptimizerComponent(
                optimizer_name=c["training"]["optimizer"],
                optimizer_config=c["optimizer"][c["training"]["optimizer"]],
                model=self._model,
            ).optimizer
        )

        self._lr_scheduler = (
            lr_scheduler
            or LRSchedulerComponent(
                lr_scheduler_name=c["training"]["lr_scheduler"],
                lr_scheduler_config=c["lr_scheduler"].get(c["training"]["lr_scheduler"], None),
                mode=c["evaluation"]["monitor_task"],
                optimizer=self._optimizer,
            ).lr_scheduler
        )

        if c["training"]["early_stopping"]:
            self._early_stopper = early_stopper or EarlyStopper(
                patience=c["early_stopping"]["patience"],
                delta=c["early_stopping"]["delta"],
                task=c["evaluation"]["monitor_task"],
                verbose=c["early_stopping"]["verbose"],
            )

    def init_loggers(self) -> None:
        c = self._config
        tb_dir = os.path.join(
            c["logging"]["root_dir"], c["logging"]["tensorboard"]["dir"], self._key, self._sub_dir or ""
        )
        mlflow_dir = os.path.join(
            c["logging"]["root_dir"],
            c["logging"]["mlflow"]["dir"],
        )
        self._tensorboard = TensorBoard(tb_dir)
        self._mlflow = MLflow(
            project_name=c["meta"]["project"],
            run_name=os.path.join(self._key, self._sub_dir or ""),
            log_dir=mlflow_dir,
        )

    def train_valid_test(self) -> dict:
        if not self._train_loader or not self._valid_loader or not self._test_loader:
            raise ValueError("Data loaders are not initialized.")
        if not self._model or not self._loss_fn or not self._metrics or not self._optimizer:
            raise ValueError("Model, loss function, metrics, and optimizer are not initialized.")
        trainer = ModelTrainerWithEvaluation(
            device=self._device,
            epochs=self._epochs,
            train_loader=self._train_loader,
            valid_loader=self._valid_loader,
            test_loader=self._test_loader,
            model=self._model,
            loss_fn=self._loss_fn,
            metrics=self._metrics,
            optimizer=self._optimizer,
            lr_scheduler=self._lr_scheduler,
            early_stopper=self._early_stopper,
            save_best_monitor=self._save_best_monitor,
            monitor_task=self._monitor_task,
            forward_fn=self.forward_fn,
            tensorboard=self._tensorboard,
            mlflow=self._mlflow,
        )

        with self._mlflow or nullcontext(), self._tensorboard or nullcontext():
            best_train_valid_result: dict = trainer.fit()
            test_result: dict = trainer.test(weight_source=best_train_valid_result["best_valid_weight"])
            DisplayUtils.print_metrics(test_result, title="Test Result")

            FileUtils.save_dict_to_yaml(
                best_train_valid_result["best_train_log"], os.path.join(self._output_dir, "best_train_log.yml")
            )
            FileUtils.save_dict_to_yaml(
                best_train_valid_result["best_valid_log"], os.path.join(self._output_dir, "best_valid_log.yml")
            )
            torch.save(
                best_train_valid_result["best_train_weight"],
                os.path.join(self._output_dir, "best_training_weight.pth"),
            )
            torch.save(
                best_train_valid_result["best_valid_weight"],
                os.path.join(self._output_dir, "best_validation_weight.pth"),
            )

            FileUtils.save_dict_to_yaml(test_result, os.path.join(self._output_dir, "test_result.yml"))

            if self._mlflow:
                for fname in [
                    "best_validation_weight.pth",
                    "original_config_backup.yml",
                    "best_config_backup.yml",
                    "test_result.yml",
                ]:
                    self._mlflow.log_artifact(os.path.join(self._output_dir, fname))

            return {
                "best_train_log": best_train_valid_result["best_train_log"],
                "best_valid_log": best_train_valid_result["best_valid_log"],
                "best_train_weight": best_train_valid_result["best_train_weight"],
                "best_valid_weight": best_train_valid_result["best_valid_weight"],
                "test_result": test_result,
            }

    def predict(self) -> None:
        match self._config["training"]["dataset"]:
            case _:
                pass

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def config(self) -> dict:
        return self._config

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def key(self) -> str:
        return self._key

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def train_loader(self) -> DataLoader | None:
        return self._train_loader

    @property
    def valid_loader(self) -> DataLoader | None:
        return self._valid_loader

    @property
    def test_loader(self) -> DataLoader | None:
        return self._test_loader

    @property
    def test_dataset(self) -> Dataset | None:
        return self._test_dataset

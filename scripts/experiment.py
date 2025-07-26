import os
import warnings

warnings.filterwarnings("ignore")
from typing import Callable
import torch
from torch.utils.data import DataLoader, Dataset

from core import Config
from components import *
from scripts import ModelTrainer, EarlyStopper
from utils import TorchUtils, MLflow, TensorBoard, FileUtils, DisplayUtils


class Experiment:
    def __init__(
        self,
        config_path: str | None = None,
        base_path: str = "configs/base.yml",
        key: str = "",
        config_override: dict | None = None,
        output_dir: str | None = None,
        sub_dir: str | None = None,
    ) -> None:

        if config_override is not None:
            self.config: dict = config_override
        else:
            self.config: dict = Config(base_path=base_path, override_path=config_path, key=key).config

        self.key: str = self.config["meta"]["key"]
        self.seed: int | None = self.config["meta"].get("seed")
        self.device: torch.device = self._init_device(self.config["meta"]["device"])

        self.output_dir: str = output_dir or self._prepare_output_dir()
        self.sub_dir: str | None = sub_dir
        if self.sub_dir:
            self.output_dir = os.path.join(self.output_dir, self.sub_dir)
        FileUtils.save_dict_to_yaml(self.config, f"{self.output_dir}/original_config_backup.yml")

        self.train_loader: DataLoader | None = None
        self.valid_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.test_dataset: Dataset | None = None
        self.model: torch.nn.Module | None = None
        self.loss_fn: torch.nn.Module | None = None
        self.metrics: list | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: object | None = None
        self.early_stopper: EarlyStopper | None = None

    def _init_device(self, requested: str) -> torch.device:
        if requested == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    def _prepare_output_dir(self) -> str:
        path = os.path.join(
            self.config["logging"]["root_dir"],
            self.config["logging"]["experiment_assets"]["dir"],
            self.key,
        )
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def forward_fn(self) -> Callable:
        return TorchUtils.resolve_forward_fn(self.config["training"]["dataset"])

    def setup_seed(self) -> None:
        if self.seed:
            TorchUtils.set_global_seed(self.seed)

    def init_dataloaders(self) -> None:
        c = self.config
        datasets = DatasetComponent(
            c["training"]["dataset"], c["dataset"][c["training"]["dataset"]], self.seed
        ).datasets
        self.test_dataset = datasets[2]
        loaders = DataLoaderComponent(
            loader_config=c["dataset"][c["training"]["dataset"]]["loader"],
            train_dataset=datasets[0],
            valid_dataset=datasets[1],
            test_dataset=datasets[2],
        ).loaders
        self.train_loader, self.valid_loader, self.test_loader = loaders

    def init_components(self) -> None:
        c = self.config

        self.model = ModelComponent(
            model_name=c["training"]["model"],
            model_config=c["model"][c["training"]["model"]],
            device=self.device,
            weight_source=c["training"]["weight"],
        ).model

        self.loss_fn = LossComponent(
            loss_name=c["training"]["loss"],
            loss_config=c["loss"][c["training"]["loss"]],
        ).loss

        self.metrics = MetricsComponent(
            evaluation_config=c["evaluation"],
            num_classes=c["dataset"][c["training"]["dataset"]]["num_classes"],
            device=self.device,
        ).metrics

        self.optimizer = OptimizerComponent(
            optimizer_name=c["training"]["optimizer"],
            optimizer_config=c["optimizer"][c["training"]["optimizer"]],
            model=self.model,
        ).optimizer

        self.scheduler = SchedulerComponent(
            scheduler_name=c["training"]["scheduler"],
            scheduler_config=c["scheduler"].get(c["training"]["scheduler"], None),
            mode=c["evaluation"]["monitor_task"],
            optimizer=self.optimizer,
        ).scheduler

        if c["training"].get("early_stopping"):
            self.early_stopper = EarlyStopper(
                patience=c["early_stopping"]["patience"],
                delta=c["early_stopping"]["delta"],
                task=c["evaluation"]["monitor_task"],
                verbose=c["early_stopping"]["verbose"],
            )

    def run(self) -> None:
        c = self.config
        tb_dir = os.path.join(
            c["logging"]["root_dir"], c["logging"]["tensorboard"]["dir"], self.key, self.sub_dir or ""
        )
        mlflow_dir = os.path.join(
            c["logging"]["root_dir"],
            c["logging"]["mlflow"]["dir"],
        )

        tensorboard = TensorBoard(tb_dir)
        mlflow = MLflow(
            project_name=c["meta"]["project"],
            run_name=os.path.join(self.key, self.sub_dir or ""),
            log_dir=mlflow_dir,
        )

        trainer = ModelTrainer(
            device=self.device,
            epochs=c["training"]["epochs"],
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            test_loader=self.test_loader,
            model=self.model,
            loss_fn=self.loss_fn,
            metrics=self.metrics,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            early_stopper=self.early_stopper,
            save_best_monitor=c["evaluation"]["save_best_monitor"],
            monitor_task=c["evaluation"]["monitor_task"],
            best_weight_source=os.path.join(self.output_dir, "best_weight.pth"),
            forward_fn=self.forward_fn,
            tensorboard=tensorboard,
            mlflow=mlflow,
        )

        with mlflow, tensorboard:
            test_log = trainer.fit()
            DisplayUtils.print_metrics(test_log, title="Test Results")
            FileUtils.save_dict_to_yaml(test_log, os.path.join(self.output_dir, "test_log.yml"))

            for fname in [
                "best_weight.pth",
                "original_config_backup.yml",
                "best_config_backup.yml",
                "test_log.yml",
            ]:
                mlflow.log_artifact(os.path.join(self.output_dir, fname))

    def predict(self) -> None:
        match self.config["training"]["dataset"]:
            case "cifar10":
                pass
            case "sst2":
                pass
            case _:
                pass

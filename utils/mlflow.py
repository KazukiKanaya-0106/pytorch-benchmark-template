import os
from typing import Any
import mlflow
import mlflow.pytorch
from torch.nn import Module

from .data_structure_utils import DataStructureUtils


class MLflow:
    def __init__(
        self,
        project_name: str,
        run_name: str,
        log_dir: str,
    ):
        self.project_name: str = project_name
        self.run_name: str = run_name
        self.log_dir: str = log_dir
        self.active_run = None

    def start_run(self) -> None:
        mlflow.set_tracking_uri(f"file:{self.log_dir}")
        mlflow.set_experiment(self.project_name)
        self.active_run = mlflow.start_run(run_name=self.run_name)

    def log_param(self, key: str, value: Any) -> None:
        value = DataStructureUtils.convert_to_builtin_types(value)
        mlflow.log_param(key, value)

    def log_params(self, params: dict) -> None:
        params = DataStructureUtils.convert_to_builtin_types(params)
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        metrics = DataStructureUtils.convert_to_builtin_types(metrics)
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        if os.path.exists(path):
            mlflow.log_artifact(path)
        else:
            print(f"[Warning] Artifact not found, skipping: {path}")

    def log_model(self, model: Module, artifact_path: str = "model") -> None:
        mlflow.pytorch.log_model(model, artifact_path)

    def end_run(self) -> None:
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_run()
        if exc_type is not None:
            print(f"Exception occurred: {exc_value}")
            return False

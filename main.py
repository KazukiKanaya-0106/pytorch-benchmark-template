import os
import warnings

warnings.filterwarnings("ignore")

import torch
from core import Config
from components import *
from scripts import ExperimentRunner, EarlyStopper
from scripts.deepglobe_roadextraction import Predictor
from utils import TorchUtils, MLflow, TensorBoard, FileUtils, DataStructureUtils

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to config file"
    )
    parser.add_argument("--base", "-b", type=str, default="configs/base.yml")
    parser.add_argument("--key", "-k", type=str, default="")
    parser.add_argument("--skip-plot", "-s", action="store_true")
    args = parser.parse_args()

    config = Config(
        base_path="configs/base.yml",
        override_path=args.config,
        key=args.key,
    ).config

    config = DataStructureUtils.convert_to_builtin_types(config)
    key = config["meta"]["key"]

    output_file_dir = f'{config["logging"]["root_dir"]}/{config["logging"]["experiment_assets"]["dir"]}/{key}'
    os.makedirs(output_file_dir, exist_ok=True)
    FileUtils.save_dict_to_yaml(
        dictionary=config, path=f"{output_file_dir}/config_backup.yml"
    )

    seed = config["meta"]["seed"]
    if seed:
        TorchUtils.set_global_seed(seed)

    device = torch.device(config["meta"]["device"])

    model = ModelComponent(
        model_name=config["training"]["model"],
        model_config=config["model"][config["training"]["model"]],
        device=device,
        weight_source=config["training"]["weight"],
    ).model

    loss_fn = LossComponent(
        loss_name=config["training"]["loss"],
        loss_config=config["loss"][config["training"]["loss"]],
    ).loss

    metrics = MetricsComponent(
        evaluation_config=config["evaluation"],
        num_classes=config["data"]["num_classes"],
        device=device,
    ).metrics

    optimizer = OptimizerComponent(
        optimizer_name=config["training"]["optimizer"],
        optimizer_config=config["optimizer"][config["training"]["optimizer"]],
        model=model,
    ).optimizer

    scheduler = SchedulerComponent(
        scheduler_name=config["training"]["scheduler"],
        scheduler_config=config["scheduler"].get(config["training"]["scheduler"], None),
        optimizer=optimizer,
    ).scheduler

    early_stopper = None
    if config["training"]["early_stopping"]:
        early_stopper = EarlyStopper(
            patience=config["early_stopping"]["patience"],
            delta=config["early_stopping"]["delta"],
            task=config["early_stopping"]["task"],
            verbose=config["early_stopping"]["verbose"],
        )

    save_best_metric = config["evaluation"]["save_best_metric"]

    train_dataset, valid_dataset, test_dataset = DatasetComponent(
        config["data"], config["meta"]["seed"]
    ).datasets
    train_loader, valid_loader, test_loader = DataLoaderComponent(
        loader_config=config["data"]["loader"],
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    ).loaders

    tensorboard_log_dir = f'{config["logging"]["root_dir"]}/{config["logging"]["tensorboard"]["dir"]}/{key}'
    tensorboard = TensorBoard(tensorboard_log_dir)

    project_name: str = config["meta"]["project"]
    mlflow_log_dir: str = (
        f'{config["logging"]["root_dir"]}/{config["logging"]["mlflow"]["dir"]}'
    )
    mlflow = MLflow(
        project_name=project_name,
        run_name=key,
        log_dir=mlflow_log_dir,
    )

    experiment_runner = ExperimentRunner(
        device=device,
        epochs=config["training"]["epochs"],
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=early_stopper,
        save_best_metric=save_best_metric,
        output_file_dir=output_file_dir,
        tensorboard=tensorboard,
        mlflow=mlflow,
    )

    experiment_runner.run()

    if config["data"]["dataset"] == "DeepGlobeRoadExtraction":
        predictor = Predictor(
            model=model,
            predict_dataset=test_dataset,
            output_file_dir=output_file_dir,
        )
        predictor.predict(num_predicts=5, device=device, skip_plot=args.skip_plot)


if __name__ == "__main__":
    main()

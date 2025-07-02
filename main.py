import os
import warnings

warnings.filterwarnings("ignore")

import torch
from core import Config, GridSearchParameters
from components import *
from scripts import ModelTrainer, EarlyStopper, GridSearch
from utils import TorchUtils, MLflow, TensorBoard, FileUtils

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config file")
    parser.add_argument("--base", "-b", type=str, default="configs/base.yml")
    parser.add_argument("--key", "-k", type=str, default="")
    parser.add_argument("--skip-plot", "-s", action="store_true")
    parser.add_argument("--grid-search", "-g", type=str, default="")
    args = parser.parse_args()

    config = Config(
        base_path="configs/base.yml",
        override_path=args.config,
        key=args.key,
    ).config

    key = config["meta"]["key"]

    output_file_dir = (
        f'{config["logging"]["root_dir"]}/{config["logging"]["experiment_assets"]["dir"]}/{key}'
    )
    os.makedirs(output_file_dir, exist_ok=True)
    FileUtils.save_dict_to_yaml(
        dictionary=config, path=f"{output_file_dir}/original_config_backup.yml"
    )

    seed = config["meta"]["seed"]
    if seed:
        TorchUtils.set_global_seed(seed)

    device = torch.device(config["meta"]["device"])

    train_dataset, valid_dataset, test_dataset = DatasetComponent(
        config["data"], config["meta"]["seed"]
    ).datasets
    train_loader, valid_loader, test_loader = DataLoaderComponent(
        loader_config=config["data"]["loader"],
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    ).loaders

    best_config: dict = config
    if args.grid_search:
        grid_search_parameters = GridSearchParameters(args.grid_search)
        FileUtils.save_dict_to_yaml(
            dictionary=grid_search_parameters.parameters,
            path=f"{output_file_dir}/grid_search_backup.yml",
        )

        grid_search = GridSearch(
            config=config,
            parameters=grid_search_parameters.flattened_parameters,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            output_file_dir=output_file_dir,
        )
        best_config, best_result = grid_search.fit()
        FileUtils.save_dict_to_yaml(
            dictionary=best_result, path=f"{output_file_dir}/best_grid_search_result.yml"
        )
        FileUtils.save_dict_to_yaml(
            dictionary=best_config, path=f"{output_file_dir}/best_config_backup.yml"
        )

    model = ModelComponent(
        model_name=best_config["training"]["model"],
        model_config=best_config["model"][best_config["training"]["model"]],
        device=device,
        weight_source=best_config["training"]["weight"],
    ).model

    loss_fn = LossComponent(
        loss_name=best_config["training"]["loss"],
        loss_config=best_config["loss"][best_config["training"]["loss"]],
    ).loss

    metrics = MetricsComponent(
        evaluation_config=best_config["evaluation"],
        num_classes=best_config["data"]["num_classes"],
        device=device,
    ).metrics

    optimizer = OptimizerComponent(
        optimizer_name=best_config["training"]["optimizer"],
        optimizer_config=best_config["optimizer"][best_config["training"]["optimizer"]],
        model=model,
    ).optimizer

    scheduler = SchedulerComponent(
        scheduler_name=best_config["training"]["scheduler"],
        scheduler_config=best_config["scheduler"].get(config["training"]["scheduler"], None),
        optimizer=optimizer,
    ).scheduler

    early_stopper = None
    if best_config["training"]["early_stopping"]:
        early_stopper = EarlyStopper(
            monitor=best_config["early_stopping"]["monitor"],
            patience=best_config["early_stopping"]["patience"],
            delta=best_config["early_stopping"]["delta"],
            task=best_config["early_stopping"]["task"],
            verbose=best_config["early_stopping"]["verbose"],
        )

    tensorboard_log_dir = (
        f'{best_config["logging"]["root_dir"]}/{best_config["logging"]["tensorboard"]["dir"]}/{key}'
    )
    tensorboard = TensorBoard(tensorboard_log_dir)

    project_name: str = best_config["meta"]["project"]
    mlflow_log_dir: str = (
        f'{best_config["logging"]["root_dir"]}/{best_config["logging"]["mlflow"]["dir"]}'
    )
    mlflow = MLflow(
        project_name=project_name,
        run_name=key,
        log_dir=mlflow_log_dir,
    )

    model_trainer = ModelTrainer(
        device=device,
        epochs=best_config["training"]["epochs"],
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=early_stopper,
        save_best_monitor=best_config["evaluation"]["save_best_monitor"],
        monitor_task=best_config["evaluation"]["monitor_task"],
        best_weight_source=f"{output_file_dir}/best_weight.pth",
        tensorboard=tensorboard,
        mlflow=mlflow,
    )

    test_log = model_trainer.fit()

    FileUtils.save_dict_to_yaml(dictionary=test_log, path=f"{output_file_dir}/test_log.yml")
    mlflow.log_artifact(f"{output_file_dir}/best_weight.pth")
    mlflow.log_artifact(f"{output_file_dir}/test_log.yml")
    mlflow.log_artifact(f"{output_file_dir}/original_config_backup.yml")
    mlflow.log_artifact(f"{output_file_dir}/grid_search_backup.yml")
    mlflow.log_artifact(f"{output_file_dir}/best_config_backup.yml")
    mlflow.end_run()


if __name__ == "__main__":
    main()

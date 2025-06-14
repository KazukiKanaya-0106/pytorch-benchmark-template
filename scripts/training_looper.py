import os
import torch
from torch.nn import Module
from torch.optim import Optimizer
from components import ComponentBuilder
from .epoch_runner import EpochRunner
from utils import TensorBoard, DataStructureUtils, FileUtils


class TrainingLooper:
    def __init__(
        self,
        config: dict,
        train_runner: EpochRunner,
        valid_runner: EpochRunner,
        test_runner: EpochRunner,
        component_builder: ComponentBuilder,
    ) -> None:
        self.config = config
        self.train_runner: EpochRunner = train_runner
        self.valid_runner: EpochRunner = valid_runner
        self.test_runner: EpochRunner = test_runner
        self.model: Module = component_builder.get_model()
        self.optimizer: Optimizer = component_builder.get_optimizer()
        self.experiment_assets_dir: str = (
            f'{config["logging"]["root_dir"]}/{config["logging"]["experiment_assets"]["dir"]}'
        )
        self.key: str = config["meta"]["key"]
        self.epochs = config["training"]["epochs"]
        self.save_best_metric = config["evaluation"]["save_best_metric"]
        self.tensorboard = TensorBoard(config)

    def run(self) -> None:
        output_file_dir = f"{self.experiment_assets_dir}/{self.key}"
        os.makedirs(output_file_dir, exist_ok=True)
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

                torch.save(
                    self.model.state_dict(),
                    f"{output_file_dir}/best_weight.pth",
                )
                print("\nUpdate best weight!\n")

            self.tensorboard.log_epoch(
                train_log=train_log,
                valid_log=valid_log,
                epoch=epoch,
            )

        self.tensorboard.close()

        test_log: dict = self.test_runner.run_epoch()
        test_log["best_epoch"] = best_epoch

        test_log = DataStructureUtils.convert_to_builtin_types(test_log)
        FileUtils.save_dict_to_yaml(
            dictionary=test_log, path=f"{output_file_dir}/test_log.yml"
        )

        config = DataStructureUtils.convert_to_builtin_types(self.config)
        FileUtils.save_dict_to_yaml(
            dictionary=config, path=f"{output_file_dir}/config_backup.yml"
        )

        for key, value in test_log.items():
            if isinstance(value, float):
                print(f"{key:<15}: {value:.4f}")
            else:
                print(f"{key:<15}: epoch {value}")

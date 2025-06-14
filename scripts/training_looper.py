import os
import torch
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
        self.model: ComponentBuilder = component_builder.get_model()
        self.log_dir: str = config["logging"]["artifacts"]["log_dir"]
        self.key: str = config["meta"]["key"]
        self.epochs = config["training"]["epochs"]
        self.save_best_metric = config["evaluation"]["save_best_metric"]
        self.tensorboard = TensorBoard(config)

    def run(self) -> None:
        output_file_dir = f"{self.log_dir}/{self.key}"
        os.makedirs(output_file_dir, exist_ok=True)
        max_score = 0
        best_epoch = 0
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1} / {self.epochs}")
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

        FileUtils.save_dict_to_yaml(
            dictionary=test_log, path=f"{output_file_dir}/test_log.yml"
        )
        FileUtils.save_dict_to_yaml(
            dictionary=self.config, path=f"{output_file_dir}/config_backup.yml"
        )

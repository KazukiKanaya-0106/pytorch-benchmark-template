from sklearn.model_selection import ParameterGrid
import os

from scripts.experiment import Experiment
from utils import DataStructureUtils, FileUtils, TorchUtils


class GridSearch:
    def __init__(self, base_experiment: Experiment, param_grid: dict) -> None:
        self.base_experiment = base_experiment
        self.param_grid = param_grid
        self.output_dir = base_experiment.output_dir
        self.best_score: float = (
            float("-inf") if base_experiment.config["evaluation"]["monitor_task"] == "max" else float("inf")
        )
        self.best_config: dict = base_experiment.config
        self.best_result: dict = {}

    def run(self) -> tuple[dict, dict]:
        all_combinations = list(ParameterGrid(self.param_grid))
        save_best_monitor = self.base_experiment.config["evaluation"]["save_best_monitor"]
        monitor_task = self.base_experiment.config["evaluation"]["monitor_task"]

        for i, params in enumerate(all_combinations):
            print(f"=== Grid Search {i+1} / {len(all_combinations)} ===")
            nested_params = DataStructureUtils.unflatten_dict(params)
            candidate_config = DataStructureUtils.deep_merge_dict(
                self.base_experiment.config,
                nested_params,
            )

            sub_dir = f"grid_search_{i}"

            run_output_dir = os.path.join(self.output_dir, sub_dir)
            os.makedirs(run_output_dir, exist_ok=True)
            FileUtils.save_dict_to_yaml(candidate_config, os.path.join(run_output_dir, "config_backup.yml"))

            exp = Experiment(
                config_override=candidate_config,
                key=self.base_experiment.key,
                output_dir=self.base_experiment.output_dir,
                sub_dir=sub_dir,
            )
            exp.device = self.base_experiment.device
            exp.seed = self.base_experiment.seed
            exp.train_loader = self.base_experiment.train_loader
            exp.valid_loader = self.base_experiment.valid_loader
            exp.test_loader = self.base_experiment.test_loader
            exp.test_dataset = self.base_experiment.test_dataset

            exp.init_components()
            exp.run()

            test_log = FileUtils.load_yaml_as_dict(os.path.join(run_output_dir, "test_log.yml"))
            monitor_score = test_log.get(f"test_{save_best_monitor}")

            if TorchUtils.is_better_score(monitor_score, self.best_score, monitor_task):
                self.best_score = monitor_score
                self.best_config = candidate_config
                self.best_result = {
                    "search_id": i,
                    "monitor_score": monitor_score,
                    "params": params,
                }

            result_csv_row = {
                "search_id": i,
                **params,
                **test_log,
            }
            FileUtils.save_dict_to_csv(result_csv_row, os.path.join(self.output_dir, "grid_search_results.csv"))

        FileUtils.save_dict_to_yaml(self.best_result, os.path.join(self.output_dir, "best_grid_search_result.yml"))
        FileUtils.save_dict_to_yaml(self.best_config, os.path.join(self.output_dir, "best_config_backup.yml"))

        return self.best_config, self.best_result

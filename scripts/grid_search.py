from sklearn.model_selection import ParameterGrid
import os

from scripts.model_pipeline import ModelPipeline
from utils import DataStructureUtils, FileUtils, TorchUtils


class GridSearch:
    def __init__(self, base_pipeline: ModelPipeline, param_grid: dict) -> None:
        self.base_pipeline = base_pipeline
        self.param_grid = param_grid
        self.output_dir = base_pipeline.output_dir
        self.best_score: float = (
            float("-inf") if base_pipeline.config["evaluation"]["monitor_task"] == "max" else float("inf")
        )
        self.best_config: dict = base_pipeline.config
        self.best_result: dict = {}

    def run(self) -> tuple[dict, dict]:
        all_combinations = list(ParameterGrid(self.param_grid))
        save_best_monitor = self.base_pipeline.config["evaluation"]["save_best_monitor"]
        monitor_task = self.base_pipeline.config["evaluation"]["monitor_task"]

        for i, params in enumerate(all_combinations):
            print(f"=== Grid Search {i+1} / {len(all_combinations)} ===")
            nested_params = DataStructureUtils.unflatten_dict(params)
            candidate_config = DataStructureUtils.deep_merge_dict(
                self.base_pipeline.config,
                nested_params,
            )

            sub_dir = f"grid_search_{i}"

            run_output_dir = os.path.join(self.output_dir, sub_dir)
            os.makedirs(run_output_dir, exist_ok=True)
            FileUtils.save_dict_to_yaml(candidate_config, os.path.join(run_output_dir, "config_backup.yml"))

            pipeline = ModelPipeline(
                config=candidate_config,
                key=self.base_pipeline.key,
                output_dir=self.base_pipeline.output_dir,
                sub_dir=sub_dir,
            )
            pipeline._device = self.base_pipeline._device
            pipeline._seed = self.base_pipeline._seed
            pipeline._train_loader = self.base_pipeline.train_loader
            pipeline._valid_loader = self.base_pipeline.valid_loader
            pipeline._test_loader = self.base_pipeline.test_loader
            pipeline._test_dataset = self.base_pipeline.test_dataset

            pipeline.init_components()
            result: dict = pipeline.train_valid_test()

            test_result = result["test_result"]
            monitor_score = test_result[save_best_monitor]

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
                **test_result,
            }
            FileUtils.save_dict_to_csv(result_csv_row, os.path.join(self.output_dir, "grid_search_results.csv"))

        FileUtils.save_dict_to_yaml(self.best_result, os.path.join(self.output_dir, "best_grid_search_result.yml"))
        FileUtils.save_dict_to_yaml(self.best_config, os.path.join(self.output_dir, "best_config_backup.yml"))

        return self.best_config, self.best_result

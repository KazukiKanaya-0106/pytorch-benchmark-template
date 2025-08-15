from sklearn.model_selection import ParameterGrid
import os

from scripts.model_workflow import ModelWorkflow
from utils.data_structure_utils import DataStructureUtils
from utils.file_utils import FileUtils
from utils.torch_utils import TorchUtils


class GridSearch:
    def __init__(self, base_workflow: ModelWorkflow, param_grid: dict) -> None:
        self.base_workflow = base_workflow
        self.param_grid = param_grid
        self.output_dir = base_workflow.output_dir
        self.best_score: float = (
            float("-inf") if base_workflow.config["evaluation"]["monitor_task"] == "max" else float("inf")
        )
        self.best_config: dict = base_workflow.config
        self.best_result: dict = {}

    def run(self) -> tuple[dict, dict]:
        all_combinations = list(ParameterGrid(self.param_grid))
        save_best_monitor = self.base_workflow.config["evaluation"]["save_best_monitor"]
        monitor_task = self.base_workflow.config["evaluation"]["monitor_task"]

        for i, params in enumerate(all_combinations):
            print(f"=== Grid Search {i+1} / {len(all_combinations)} ===")
            nested_params = DataStructureUtils.unflatten_dict(params)
            candidate_config = DataStructureUtils.deep_merge_dict(
                self.base_workflow.config,
                nested_params,
            )

            sub_dir = f"grid_search_{i}"

            run_output_dir = os.path.join(self.output_dir, sub_dir)
            os.makedirs(run_output_dir, exist_ok=True)
            FileUtils.save_dict_to_yaml(candidate_config, os.path.join(run_output_dir, "config_backup.yml"))

            workflow = ModelWorkflow(
                config=candidate_config,
                output_dir=self.base_workflow.output_dir,
                sub_dir=sub_dir,
            )
            workflow._device = self.base_workflow._device
            workflow._seed = self.base_workflow._seed
            workflow._train_loader = self.base_workflow.train_loader
            workflow._valid_loader = self.base_workflow.valid_loader
            workflow._test_loader = self.base_workflow.test_loader
            workflow._test_dataset = self.base_workflow.test_dataset

            workflow.init_components()
            result: dict = workflow.train_valid_test()

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

from io import BytesIO
from sklearn.model_selection import ParameterGrid
import torch
from torch.utils.data import DataLoader
from utils import DataStructureUtils
from scripts import ModelTrainer, EarlyStopper
from components import *
from utils import FileUtils
from utils.torch_utils import TorchUtils


class GridSearch:
    def __init__(
        self,
        config: dict,
        parameters: dict,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        output_file_dir: str,
    ) -> None:
        self.config: dict = config
        self.parameters: dict = parameters
        self.device: torch.device = device
        self.output_file_dir: str = output_file_dir

        self.train_loader: DataLoader = DataStructureUtils.truncate_loader(
            loader=train_loader,
            fraction=config["grid_search"]["data"]["fraction"],
            batch_size=config["grid_search"]["data"]["batch_size"],
            num_workers=config["grid_search"]["data"]["num_workers"],
            shuffle=config["grid_search"]["data"]["shuffle_train"],
        )
        self.valid_loader: DataLoader = DataStructureUtils.truncate_loader(
            loader=valid_loader,
            fraction=config["grid_search"]["data"]["fraction"],
            batch_size=config["grid_search"]["data"]["batch_size"],
            num_workers=config["grid_search"]["data"]["num_workers"],
            shuffle=False,
        )
        self.test_loader: DataLoader = DataStructureUtils.truncate_loader(
            loader=test_loader,
            fraction=config["grid_search"]["data"]["fraction"],
            batch_size=config["grid_search"]["data"]["batch_size"],
            num_workers=config["grid_search"]["data"]["num_workers"],
            shuffle=False,
        )

    def fit(self) -> tuple[dict, dict]:
        best_search_id: int = -1
        best_weight_source = BytesIO()
        save_best_monitor = self.config["evaluation"]["save_best_monitor"]
        monitor_task = self.config["evaluation"]["monitor_task"]
        best_score = 0 if monitor_task == "max" else float("-inf")
        best_config: dict = self.config

        all_combinations = list(ParameterGrid(self.parameters))

        for search_id, param_combination in enumerate(ParameterGrid(self.parameters)):
            print("\n" + "=" * 40)
            print(f"[GridSearch] Search {search_id + 1} / {len(all_combinations)}")
            print("=" * 40)
            nested_param_combination: dict = DataStructureUtils.unflatten_dict(param_combination)
            candidate_config: dict = DataStructureUtils.deep_merge_dict(
                base_dict=self.config, override_dict=nested_param_combination
            )

            model = ModelComponent(
                model_name=candidate_config["training"]["model"],
                model_config=candidate_config["model"][candidate_config["training"]["model"]],
                device=self.device,
                weight_source=candidate_config["training"]["weight"],
            ).model

            loss_fn = LossComponent(
                loss_name=candidate_config["training"]["loss"],
                loss_config=candidate_config["loss"][candidate_config["training"]["loss"]],
            ).loss

            metrics = MetricsComponent(
                evaluation_config=candidate_config["evaluation"],
                num_classes=candidate_config["data"]["num_classes"],
                device=self.device,
            ).metrics

            optimizer = OptimizerComponent(
                optimizer_name=candidate_config["training"]["optimizer"],
                optimizer_config=candidate_config["optimizer"][
                    candidate_config["training"]["optimizer"]
                ],
                model=model,
            ).optimizer

            scheduler = SchedulerComponent(
                scheduler_name=candidate_config["training"]["scheduler"],
                scheduler_config=candidate_config["scheduler"].get(
                    candidate_config["training"]["scheduler"], None
                ),
                optimizer=optimizer,
            ).scheduler

            early_stopper = None
            if candidate_config["training"]["early_stopping"]:
                early_stopper = EarlyStopper(
                    monitor=candidate_config["early_stopping"]["monitor"],
                    patience=candidate_config["early_stopping"]["patience"],
                    delta=candidate_config["early_stopping"]["delta"],
                    task=candidate_config["early_stopping"]["task"],
                    verbose=candidate_config["early_stopping"]["verbose"],
                )

            model_trainer = ModelTrainer(
                device=self.device,
                epochs=self.config["grid_search"]["epochs"],
                train_loader=self.train_loader,
                valid_loader=self.valid_loader,
                test_loader=self.test_loader,
                model=model,
                loss_fn=loss_fn,
                metrics=metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopper=early_stopper,
                save_best_monitor=save_best_monitor,
                monitor_task=monitor_task,
                best_weight_source=best_weight_source,
                tensorboard=None,
                mlflow=None,
            )

            test_log: dict = model_trainer.fit()

            monitor_score: float = test_log[f"test_{save_best_monitor}"]

            is_better: bool = TorchUtils.is_better_score(
                score=monitor_score,
                best=best_score,
                task=monitor_task,
            )
            if is_better:
                best_score = monitor_score
                best_config = candidate_config
                best_search_id = search_id

            result_csv_row = {
                "search_id": search_id,
                **param_combination,
                **test_log,
            }

            FileUtils.save_dict_to_csv(
                dictionary=result_csv_row, path=f"{self.output_file_dir}/grid_search_results.csv"
            )

        best_result = {
            "search_id": best_search_id,
            f"monitor_score ({save_best_monitor})": float(best_score),
            "param_combination": (
                list(ParameterGrid(self.parameters))[best_search_id] if best_search_id >= 0 else {}
            ),
        }

        print("\n" + "=" * 40)
        print("[GridSearch] Best Grid Search Result Summary")
        print("=" * 40)
        print(f"Best Search ID       : {best_search_id}")
        print(f"Best Parameters      : {best_result['param_combination']}")
        print(f"Best Score ({save_best_monitor}): {float(best_score):.4f}")
        print("=" * 40)

        return best_config, best_result

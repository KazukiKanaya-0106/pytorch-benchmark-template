import argparse

from core.config import Config
from core.grid_search_parameters import GridSearchParameters

from scripts.model_workflow import ModelWorkflow
from scripts.grid_search import GridSearch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument("--base", "-b", default="configs/base.yml", help="Path to base config file")
    parser.add_argument("--key", "-k", default="", help="Unique experiment key")
    parser.add_argument("--grid-search", "-g", default="", help="Path to grid search config YAML")
    args = parser.parse_args()

    config: dict = Config(base_path=args.base, override_path=args.config, key=args.key).config

    workflow = ModelWorkflow(
        config=config,
        key=args.key,
    )

    if args.grid_search:
        workflow.setup_seed()
        workflow.init_dataloaders()
        param_grid = GridSearchParameters(args.grid_search).flattened_parameters
        gs = GridSearch(workflow, param_grid)
        best_config, _ = gs.run()

        workflow = ModelWorkflow(
            config=best_config,
            key=args.key,
        )

    workflow.setup_seed()
    workflow.init_dataloaders()
    workflow.init_components()
    workflow.init_loggers()
    workflow.train_valid_test()
    workflow.predict()


if __name__ == "__main__":
    main()

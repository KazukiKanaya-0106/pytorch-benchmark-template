import argparse

from core.config import Config
from core.grid_search_parameters import GridSearchParameters

from scripts.model_pipeline import ModelPipeline
from scripts.grid_search import GridSearch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument("--base", "-b", default="configs/base.yml", help="Path to base config file")
    parser.add_argument("--key", "-k", default="", help="Unique experiment key")
    parser.add_argument("--skip-plot", "-s", action="store_true", help="Skip plot generation in prediction")
    parser.add_argument("--grid-search", "-g", default="", help="Path to grid search config YAML")
    args = parser.parse_args()

    config: dict = Config(base_path=args.base, override_path=args.config, key=args.key).config

    pipeline = ModelPipeline(
        config=config,
        key=args.key,
    )

    if args.grid_search:
        pipeline.setup_seed()
        pipeline.init_dataloaders()
        param_grid = GridSearchParameters(args.grid_search).flattened_parameters
        gs = GridSearch(pipeline, param_grid)
        best_config, _ = gs.run()

        pipeline = ModelPipeline(
            config=best_config,
            key=args.key,
        )

    pipeline.setup_seed()
    pipeline.init_dataloaders()
    pipeline.init_components()
    pipeline.init_loggers()
    pipeline.train_valid_test()
    pipeline.predict(skip_plot=args.skip_plot)


if __name__ == "__main__":
    main()

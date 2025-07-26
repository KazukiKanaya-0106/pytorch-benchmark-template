import argparse

from core import Config, GridSearchParameters
from scripts import Experiment, GridSearch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument("--base", "-b", default="configs/base.yml", help="Path to base config file")
    parser.add_argument("--key", "-k", default="", help="Unique experiment key")
    parser.add_argument("--grid-search", "-g", default="", help="Path to grid search config YAML")
    args = parser.parse_args()

    config: dict = Config(base_path=args.base, override_path=args.config, key=args.key).config

    exp = Experiment(
        config=config,
        key=args.key,
    )

    exp.setup_seed()
    exp.init_dataloaders()

    if args.grid_search:
        param_grid = GridSearchParameters(args.grid_search).flattened_parameters
        gs = GridSearch(exp, param_grid)
        best_config, _ = gs.run()
        exp.config = best_config

    exp.init_components()
    exp.run()
    exp.predict()


if __name__ == "__main__":
    main()

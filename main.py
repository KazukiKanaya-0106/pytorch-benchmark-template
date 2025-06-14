from core import Config
from components import ComponentBuilder
from data import DataLoaderBuilder
from scripts import EpochRunner, TrainingLooper

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to config file"
    )
    parser.add_argument("--key", "-k", type=str, default="")
    parser.add_argument("--base", "-b", type=str, default="configs/data.yml")
    args = parser.parse_args()
    config = Config(
        base_path="configs/base.yml",
        override_path=args.config,
        key=args.key,
    ).get()

    component_builder = ComponentBuilder(config)
    dataloader_builder = DataLoaderBuilder(config)
    train_loader, valid_loader, test_loader = dataloader_builder.get_loader()
    train_runner = EpochRunner(
        config=config,
        component_builder=component_builder,
        data_loader=train_loader,
        mode="training",
    )
    valid_runner = EpochRunner(
        config=config,
        component_builder=component_builder,
        data_loader=valid_loader,
        mode="validation",
    )
    test_runner = EpochRunner(
        config=config,
        component_builder=component_builder,
        data_loader=test_loader,
        mode="validation",
    )
    training_looper = TrainingLooper(
        config=config,
        train_runner=train_runner,
        valid_runner=valid_runner,
        test_runner=test_runner,
        component_builder=component_builder,
    )

    training_looper.run()


if __name__ == "__main__":
    main()

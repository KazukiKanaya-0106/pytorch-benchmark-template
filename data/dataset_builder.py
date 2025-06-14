from torch.utils.data import Dataset
from .datasets import CIFAR10


class DatasetBuilder:
    def __init__(self, config: dict):
        self.train_dataset, self.valid_dataset, self.test_dataset = self.build_dataset(
            config
        )

    def build_dataset(self, config: dict) -> tuple[Dataset, Dataset, Dataset]:
        data_config: dict = config["data"]
        dataset_name: str = data_config["dataset"]

        train_dataset, valid_dataset, test_dataset = None, None, None

        match dataset_name:
            case "CIFAR10":
                (train_dataset, valid_dataset, test_dataset) = CIFAR10(config).get()
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

        return train_dataset, valid_dataset, test_dataset

    def get(self) -> tuple[Dataset, Dataset, Dataset]:
        return self.train_dataset, self.valid_dataset, self.test_dataset

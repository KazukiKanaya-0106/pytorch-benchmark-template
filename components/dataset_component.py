from torch.utils.data import Dataset
from data.datasets import CIFAR10Datasets


class DatasetComponent:
    def __init__(self, data_config: dict, seed: int):
        self._train_dataset, self._valid_dataset, self._test_dataset = self._build_dataset(data_config, seed)

    def _build_dataset(self, data_config: dict, seed: int) -> tuple[Dataset, Dataset, Dataset]:
        dataset_name: str = data_config["dataset"]

        match dataset_name:
            case "CIFAR10":
                return CIFAR10Datasets(data_config).datasets
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

    @property
    def datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return self._train_dataset, self._valid_dataset, self._test_dataset

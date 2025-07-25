from torch.utils.data import Dataset
from data.datasets import CIFAR10Datasets, GlueSST2Datasets


class DatasetComponent:
    def __init__(self, dataset_name: str, data_config: dict, seed: int):
        self._train_dataset, self._valid_dataset, self._test_dataset = self._build_dataset(
            dataset_name, data_config, seed
        )

    def _build_dataset(self, dataset_name: str, data_config: dict, seed: int) -> tuple[Dataset, Dataset, Dataset]:
        match dataset_name:
            case "cifar10":
                return CIFAR10Datasets(data_config).datasets
            case "sst2":
                return GlueSST2Datasets(data_config).datasets
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

    @property
    def datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return self._train_dataset, self._valid_dataset, self._test_dataset

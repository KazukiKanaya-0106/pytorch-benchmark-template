import torch
from torch.utils.data import Dataset
from data.datasets.cifar10_datasets import CIFAR10Datasets
from data.datasets.glue_sst2_datasets import GlueSST2Datasets


class DatasetComponent:
    def __init__(self, dataset_name: str, data_config: dict, generator: torch.Generator, model_name: str):
        self._train_dataset, self._valid_dataset, self._test_dataset = self._build_dataset(
            dataset_name, data_config, generator, model_name
        )

    def _build_dataset(
        self, dataset_name: str, data_config: dict, generator: torch.Generator, model_name: str
    ) -> tuple[Dataset, Dataset, Dataset]:
        match dataset_name:
            case "cifar10":
                return CIFAR10Datasets(data_config, generator).datasets
            case "sst2":
                return GlueSST2Datasets(data_config, generator, model_name).datasets
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

    @property
    def datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return self._train_dataset, self._valid_dataset, self._test_dataset

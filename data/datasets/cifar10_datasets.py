import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset, Dataset


class CIFAR10Datasets:
    def __init__(self, data_config: dict):
        download_dir: str = data_config["downloads"]
        os.makedirs(download_dir, exist_ok=True)

        transform = transforms.ToTensor()

        full_train_data: Dataset = datasets.CIFAR10(root=download_dir, train=True, download=True, transform=transform)
        full_test_data: Dataset = datasets.CIFAR10(root=download_dir, train=False, download=True, transform=transform)

        fraction: float = data_config["fraction"]

        total_train_size = int(len(full_train_data) * fraction)
        partial_train_data = Subset(full_train_data, range(total_train_size))

        total_test_size = int(len(full_test_data) * fraction)
        partial_test_data = Subset(full_test_data, range(total_test_size))

        train_ratio: float = data_config["split"]["training"]
        valid_ratio: float = data_config["split"]["validation"]
        assert abs(train_ratio + valid_ratio - 1.0) < 1e-5, "Train/Validation ratio must sum to 1.0."

        train_size = int(total_train_size * train_ratio)
        valid_size = total_train_size - train_size

        self._train_dataset, self._valid_dataset = random_split(partial_train_data, [train_size, valid_size])
        self._test_dataset = partial_test_data

    @property
    def datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return self._train_dataset, self._valid_dataset, self._test_dataset

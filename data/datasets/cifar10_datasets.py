import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset, Dataset


class CIFAR10Datasets:
    def __init__(self, data_config: dict):
        download_dir: str = data_config["downloads"]
        os.makedirs(download_dir, exist_ok=True)

        transform = transforms.ToTensor()

        full_train_data: Dataset = datasets.CIFAR10(root=download_dir, train=True, download=True, transform=transform)
        test_data: Dataset = datasets.CIFAR10(root=download_dir, train=False, download=True, transform=transform)

        # 部分データの抽出
        data_frac: float = data_config["fraction"]
        total_size: int = int(len(full_train_data) * data_frac)
        partial_data: Dataset = Subset(full_train_data, range(total_size))

        # 分割比チェック
        train_ratio: float = data_config["split"]["training"]
        valid_ratio: float = data_config["split"]["validation"]
        assert abs(train_ratio + valid_ratio - 1.0) < 1e-5, "Train/Validation ratio must sum to 1.0."

        # train / val 分割
        train_size: int = int(total_size * train_ratio)
        valid_size: int = total_size - train_size

        self._train_dataset, self._valid_dataset = random_split(partial_data, [train_size, valid_size])

        self._test_dataset = test_data

    @property
    def datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return self._train_dataset, self._valid_dataset, self._test_dataset

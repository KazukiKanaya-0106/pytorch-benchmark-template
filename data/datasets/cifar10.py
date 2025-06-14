import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset, Dataset


class CIFAR10:
    def __init__(self, config: dict):
        data_config: dict = config["data"]
        download_dir: str = data_config["downloads"]

        transform = transforms.ToTensor()

        os.makedirs(download_dir, exist_ok=True)
        train_valid_data: Dataset = datasets.CIFAR10(
            root=download_dir, train=True, download=True, transform=transform
        )
        test_data: Dataset = datasets.CIFAR10(
            root=download_dir, train=False, download=True, transform=transform
        )

        # 使用割合の制限
        data_frac: float = data_config["data_frac"]
        total_size: int = int(len(train_valid_data) * data_frac)
        dataset: Dataset = Subset(train_valid_data, range(total_size))

        # 分割比の確認
        train_ratio: float = data_config["split"]["train"]
        valid_ratio: float = data_config["split"]["validation"]
        assert (
            abs(train_ratio + valid_ratio - 1.0) < 1e-5
        ), "Please ensure that the sum of 'train' and 'validation' is 1.0."

        # train/val のみ分割、test は test_data をそのまま使う
        train_size: int = int(total_size * train_ratio / (train_ratio + valid_ratio))
        valid_size: int = total_size - train_size

        self.train_dataset, self.valid_dataset = random_split(
            dataset,
            [train_size, valid_size],
        )

        self.test_dataset: Dataset = test_data

    def get(self) -> tuple[Dataset, Dataset, Dataset]:
        return self.train_dataset, self.valid_dataset, self.test_dataset

from .dataset_builder import DatasetBuilder
from torch.utils.data import DataLoader, Dataset


class DataLoaderBuilder:
    def __init__(self, config: dict):
        self.train_dataset, self.valid_dataset, self.test_dataset = DatasetBuilder(
            config
        ).get()
        self.train_loader, self.valid_loader, self.test_loader = self.build_dataloader(
            config
        )

    def build_dataloader(
        self, config: dict
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        batch_size = config["data"]["loader"]["batch_size"]
        num_workers = config["data"]["loader"]["num_workers"]
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, valid_loader, test_loader

    def get_loader(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        return self.train_loader, self.valid_loader, self.test_loader

    def get_dataset(self) -> tuple[Dataset, Dataset, Dataset]:
        return self.train_dataset, self.valid_dataset, self.test_dataset

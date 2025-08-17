import torch
from torch.utils.data import DataLoader, Dataset


class DataLoaderComponent:
    def __init__(
        self,
        loader_config: dict,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        generator: torch.Generator,
    ):
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset

        self._train_loader, self._valid_loader, self._test_loader = self._build_dataloader(loader_config, generator)

    def _build_dataloader(
        self, loader_config: dict, generator: torch.Generator | None
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        batch_size: int = loader_config["batch_size"]
        num_workers: int = loader_config["num_workers"]
        shuffle: bool = loader_config["shuffle_train"]

        train_loader = DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
        )
        valid_loader = DataLoader(
            self._valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            generator=generator,
        )
        test_loader = DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            generator=generator,
        )
        return train_loader, valid_loader, test_loader

    @property
    def loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        return self._train_loader, self._valid_loader, self._test_loader

    @property
    def datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return self._train_dataset, self._valid_dataset, self._test_dataset

import os
from typing import Callable
import pandas as pd
from torch.utils.data import Dataset
from data.transforms import Transforms

from .deepglobe_roadextraction_dataset import DeepGlobeRoadExtractionDataset

from utils import DataStructureUtils


class DeepGlobeRoadExtractionDatasets:
    def __init__(self, data_config: dict, seed: int):
        download_dir: str = data_config["downloads"]
        input_dir: str = os.path.join(download_dir, "deepglobe_road_extraction")

        data_frac: float = data_config["data_frac"]

        train_ratio: float = data_config["split"]["training"]
        valid_ratio: float = data_config["split"]["validation"]
        test_ratio: float = 1.0 - (train_ratio + valid_ratio)

        assert (
            abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-5
        ), "Train/Validation/Test ratio must sum to 1.0."

        # --- CSV読込とパス解決 ---
        df = pd.read_csv(os.path.join(input_dir, "metadata.csv"))
        df = df[df["split"] == "train"][["image_id", "sat_image_path", "mask_path"]]
        df["sat_image_path"] = df["sat_image_path"].apply(
            lambda p: os.path.join(input_dir, p)
        )
        df["mask_path"] = df["mask_path"].apply(lambda p: os.path.join(input_dir, p))

        df = df.sample(frac=data_frac, random_state=seed)

        # --- DataFrameを3分割 ---
        train_df, valid_df, test_df = DataStructureUtils.split_dataframe_three_ways(
            df, train_ratio, valid_ratio, test_ratio, seed
        )

        transform = Transforms.resize()

        # --- Dataset化 ---
        self._train_dataset = DeepGlobeRoadExtractionDataset(
            train_df, transform=transform
        )
        self._valid_dataset = DeepGlobeRoadExtractionDataset(
            valid_df, transform=transform
        )
        self._test_dataset = DeepGlobeRoadExtractionDataset(
            test_df, transform=transform
        )

    @property
    def datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return self._train_dataset, self._valid_dataset, self._test_dataset

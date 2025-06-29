import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable


class DeepGlobeRoadExtractionDataset(Dataset):
    """
    衛星画像と道路マスクのペアを読み込むカスタムデータセット
    """

    def __init__(
        self, data_frame: pd.DataFrame, transform: Callable | None = None
    ) -> None:
        """
        データセットの初期化

        Args:
            data_frame (pd.DataFrame): 画像とマスクのペアを含むデータフレーム
            transform (callable, optional): 画像とマスクに適用する変換
        """

        self.data_frame = data_frame
        self.transform = transform

    def __len__(self) -> int:
        """
        データセットのサイズを返す

        Returns:
            int: データセットのサイズ
        """

        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスに対応する衛星画像と道路マスクを返す

        Args:
            idx (int): インデックス

        Returns:
            tuple: 衛星画像と道路マスクのペア
        """

        sat_img_path = self.data_frame.iloc[idx, 1]
        mask_img_path = self.data_frame.iloc[idx, 2]

        sat_img = cv2.imread(sat_img_path)
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
        sat_img = sat_img.astype("uint8")  # 0~255

        mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        mask_img = mask_img.astype("uint8")  # 0 or 255
        mask_img = np.expand_dims(mask_img, axis=-1)  # C のための次元「1」を追加

        if self.transform:
            transformed_imgs = self.transform(image=sat_img, mask=mask_img)
            sat_img = transformed_imgs["image"]
            mask_img = transformed_imgs["mask"]

        # numpy -> torch
        sat_img = torch.from_numpy(sat_img)
        mask_img = torch.from_numpy(mask_img)

        sat_img = sat_img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        sat_img = sat_img.float() / 255  # 0~1

        mask_img = mask_img.permute(2, 0, 1)  # (H, W, 1) -> (1, H, W)
        mask_img = mask_img.float() / 255  # 0 or 1

        return (sat_img, mask_img)

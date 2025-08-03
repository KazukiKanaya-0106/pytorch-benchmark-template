import math
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Any
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from collections.abc import Sized


class DataStructureUtils:
    """
    データ構造に関する処理を提供するクラス
    """

    @staticmethod
    def deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
        """
        再帰的に2つの辞書をマージし、同じキーがある場合は上書き辞書の値で更新する関数
        """
        merged = base_dict.copy()
        for k, v in override_dict.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = DataStructureUtils.deep_merge_dict(merged[k], v)
            else:
                merged[k] = v
        return merged

    @staticmethod
    def flatten_dict(d: dict, parent_key: str = "", sep: str = "."):
        """
        ネストされた辞書をドット区切りの1階層の辞書に変換する関数
        """
        flattened = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flattened.update(DataStructureUtils.flatten_dict(v, new_key, sep=sep))
            else:
                flattened[new_key] = v
        return flattened

    @staticmethod
    def unflatten_dict(flat: dict, sep: str = "."):
        """
        ドット区切りの平坦な辞書をネストされた辞書に変換する関数
        """
        unflattend = {}
        for k, v in flat.items():
            keys = k.split(sep)
            d = unflattend
            for part in keys[:-1]:
                d = d.setdefault(part, {})
            d[keys[-1]] = v
        return unflattend

    @staticmethod
    def convert_to_builtin_types(obj: Any) -> Any:
        """
        Pythonの組み込み型に再帰的に変換する関数。
        """
        if isinstance(obj, torch.Tensor):
            return obj.item()
        elif isinstance(obj, np.generic):
            val = obj.item()
            return round(val, 5) if isinstance(val, float) else val
        elif isinstance(obj, float):
            return round(obj, 5)
        elif isinstance(obj, dict):
            return {k: DataStructureUtils.convert_to_builtin_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DataStructureUtils.convert_to_builtin_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(DataStructureUtils.convert_to_builtin_types(v) for v in obj)
        return obj

    @staticmethod
    def add_prefix(d: dict, prefix: str) -> dict:
        """
        辞書のキーにプレフィックスを加える関数。
        """
        return {f"{prefix}_{k}": v for k, v in d.items()}

    @staticmethod
    def split_dataframe_three_ways(df, train_ratio, valid_ratio, test_ratio, random_state=42):
        """
        pandas Dataframe を三分割する関数。
        """
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-5

        train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=random_state)
        valid_rel = valid_ratio / (valid_ratio + test_ratio)
        valid_df, test_df = train_test_split(temp_df, train_size=valid_rel, random_state=random_state)

        return train_df, valid_df, test_df

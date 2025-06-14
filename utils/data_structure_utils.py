from typing import Any
import torch
import numpy as np


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
            return {
                k: DataStructureUtils.convert_to_builtin_types(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [DataStructureUtils.convert_to_builtin_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(DataStructureUtils.convert_to_builtin_types(v) for v in obj)
        return obj

    @staticmethod
    def add_prefix(d: dict, prefix: str) -> dict:
        return {f"{prefix}_{k}": v for k, v in d.items()}

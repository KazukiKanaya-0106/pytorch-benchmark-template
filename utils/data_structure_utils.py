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
    def convert_to_builtin_types(obj: Any) -> int | float | str | dict | list:
        """
        Python独自の組み込み型にデータを変換する関数
        """
        if isinstance(obj, torch.Tensor):
            return obj.item()
        elif isinstance(obj, np.generic):
            return round(obj.item(), 5) if isinstance(obj.item(), float) else obj.item()
        elif isinstance(obj, dict):
            return {
                k: DataStructureUtils.convert_to_builtin_types(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [DataStructureUtils.convert_to_builtin_types(v) for v in obj]
        return obj

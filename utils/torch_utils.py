from io import BytesIO
from typing import Any, Callable, Literal
import torch
from torch import nn, Tensor
import random
import numpy as np
import os
import time

from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle


class TorchUtils:
    """
    PyTorchを主とする各種処理を提供するutilクラス
    """

    @staticmethod
    def move_to_device(obj, device):
        """
        再帰的に任意のオブジェクト（Tensor, dict, list, tuple）を指定されたデバイスに移動
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: TorchUtils.move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(TorchUtils.move_to_device(x, device) for x in obj)
        else:
            return obj

    @staticmethod
    def set_global_seed(seed: int) -> None:
        """
        ランダムシードを全フレームワークに適用し、再現性を確保する
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["PYTHONHASHSEED"] = str(seed)

        print(f"[TorchUtils] Global seed set to {seed}")

    @staticmethod
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def count_trainable_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def save_model_state(model: nn.Module, destination: str | BytesIO):
        """
        モデルのstate_dictをファイルパスまたはBytesIOに保存する

        Args:
            model (nn.Module): 保存対象のモデル
            dest (str | BytesIO): 保存先（パスまたはBytesIO）
        """
        if isinstance(destination, str):
            torch.save(model.state_dict(), destination)
        elif isinstance(destination, BytesIO):
            destination.seek(0)
            destination.truncate(0)
            torch.save(model.state_dict(), destination)
            destination.seek(0)
        else:
            raise TypeError("`destination` must be a file path (str) or BytesIO object.")

    @staticmethod
    def load_model_state(model: nn.Module, source: str | BytesIO):
        """
        ファイルパスまたはBytesIOからstate_dictを読み込み、モデルに適用する

        Args:
            model (nn.Module): ロード対象のモデル
            source (str | BytesIO): 読み込み元（パスまたはBytesIO）
        """
        if isinstance(source, str):
            state_dict = torch.load(source)
        elif isinstance(source, BytesIO):
            source.seek(0)
            state_dict = torch.load(source)
        else:
            raise TypeError("`source` must be a file path (str) or BytesIO object.")

        model.load_state_dict(state_dict)

    @staticmethod
    def is_better_score(score: float, best: float, task: Literal["min", "max"]) -> bool:
        """
        モデルのスコアが改善したかを真偽値で返す
        """
        if task == "max":
            return score > best
        elif task == "min":
            return score < best
        else:
            raise ValueError(f"Unknown monitor_task: {task}")

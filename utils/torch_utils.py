from io import BytesIO
from typing import Literal, Callable
import torch
from torch import nn
import random
import numpy as np
import os
import time


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
    def setup_seed_with_generator(master_seed: int) -> torch.Generator:
        """
        ランダムシードを全フレームワークに適用し、再現性を確保する
        """
        os.environ["PYTHONHASHSEED"] = str(master_seed)
        rng = np.random.RandomState(master_seed)

        random.seed(rng.randint(0, 2**32))
        np.random.seed(rng.randint(0, 2**32))
        torch.manual_seed(rng.randint(0, 2**32))

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng.randint(0, 2**32))

        generator_seed = rng.randint(0, 2**32)
        print(f"[TorchUtils] Global seed set to {master_seed}, DataLoader generator seed: {generator_seed}")

        return torch.Generator().manual_seed(generator_seed)

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def count_trainable_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def load_model_state(model: nn.Module, source: str | BytesIO | dict):
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
        elif isinstance(source, dict):
            state_dict = source
        else:
            raise TypeError("`source` must be a file path (str) or BytesIO or dict object.")

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

    @staticmethod
    def split_batch(batch) -> tuple:
        """
        バッチを入力とラベルに分割する
        """
        if isinstance(batch, dict):
            # "labels" または "label" をサポート
            label_key = "labels" if "labels" in batch else "label"
            inputs = {k: v for k, v in batch.items() if k != label_key}
            labels = batch.get(label_key, None)
            return inputs, labels

        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]

        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    @staticmethod
    def resolve_forward_fn(dataset_name: str) -> Callable:
        def forward_image(model, X):
            return model(X)

        def forward_nlp(model, X):
            return model(**X).logits  # transformers系は logits を返す必要あり

        match dataset_name:
            case "cifar10":
                return forward_image
            case "sst2":
                return forward_nlp
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

import torch
import random
import numpy as np
import os


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
        ランダムシードを全フレームワークに適用し、再現性を確保する。
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

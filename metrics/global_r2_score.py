from torch import Tensor
from torchmetrics import Metric, R2Score


class GlobalR2(Metric):
    """
    R²を global_mode（flatten）または通常モードで計算する Metric。
    """

    higher_is_better = True
    full_state_update = False

    def __init__(self, global_mode: bool = True, **kwargs):
        """
        Args:
            global_mode: Trueなら全出力をflattenして1本のR²を計算
                         Falseなら通常のR²計算
            **kwargs: R2Scoreにそのまま渡す追加引数
        """
        super().__init__()
        self.global_mode = global_mode
        self._r2 = R2Score(**kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.global_mode:
            self._r2.update(preds.reshape(-1), target.reshape(-1))
        else:
            self._r2.update(preds, target)

    def compute(self) -> Tensor:
        return self._r2.compute()

    def reset(self) -> None:
        self._r2.reset()

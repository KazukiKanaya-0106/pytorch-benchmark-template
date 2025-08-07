from typing import Any, Literal

from utils.data_structure_utils import DataStructureUtils


class EarlyStopper:
    def __init__(
        self,
        patience: int = 10,
        delta: float = 0.0,
        verbose: bool = False,
        task: Literal["min", "max"] = "min",
    ):
        """
        Args:
            patience (int): 改善が見られなくても許容するエポック数
            delta (float): 改善とみなす最小の差
            verbose (bool): Trueにすると、改善がないたびにカウントを表示する
            task (Literal["min", "max"]):
                "min"なら損失値のようにスコアが小さいほど良いとみなし、"max"なら精度のようにスコアが大きいほど良いとみなす
        """
        self.patience: int = patience
        self.delta: float = delta
        self.verbose: bool = verbose
        self.task: Literal["min", "max"] = task

        self.counter: int = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, raw_score: Any) -> bool:
        score: float = DataStructureUtils.convert_to_builtin_types(raw_score)
        adjusted_score = -score if self.task == "min" else score

        if self.best_score is None:
            self.best_score = adjusted_score
        elif adjusted_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = adjusted_score
            self.counter = 0

        return self.early_stop

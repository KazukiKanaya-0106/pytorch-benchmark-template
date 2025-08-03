from typing import Literal
import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    F1Score,
    Accuracy,
    Precision,
    Recall,
    JaccardIndex,
)


class MetricsComponent:
    def __init__(self, evaluation_config: dict, num_classes: int, device: torch.device) -> None:
        self._metrics = self._build_metrics(
            evaluation_config=evaluation_config,
            num_classes=num_classes,
            device=device,
        )

    def _build_metrics(self, evaluation_config: dict, num_classes: int, device: torch.device) -> list[Metric]:
        metric_list: list[str] = evaluation_config["metrics"]
        average: Literal["micro", "macro", "weighted", "none"] | None = evaluation_config["average"]
        task: Literal["binary", "multiclass", "multilabel"] = evaluation_config["task"]

        metrics: list[Metric] = []

        for metric_name in metric_list:
            match metric_name:
                case "f1":
                    metric = F1Score(task=task, num_classes=num_classes, average=average)
                case "iou":
                    metric = JaccardIndex(task=task, num_classes=num_classes, average=average)
                case "precision":
                    metric = Precision(task=task, num_classes=num_classes, average=average)
                case "recall":
                    metric = Recall(task=task, num_classes=num_classes, average=average)
                case "accuracy":
                    metric = Accuracy(task=task, num_classes=num_classes, average=average)
                case _:
                    raise ValueError(f"Unsupported metric: {metric_name}")
            metric.__name__ = metric_name
            metrics.append(metric.to(device))

        return metrics

    @property
    def metrics(self) -> list[Metric]:
        return self._metrics

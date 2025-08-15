import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    F1Score,
    Accuracy,
    Precision,
    Recall,
    JaccardIndex,
)

from metrics.r2_score import R2Score
from metrics.relative_l2_error import RelativeL2Error


class MetricsComponent:
    def __init__(self, metrics_config: dict, metrics_list: list[str], num_classes: int, device: torch.device) -> None:
        self._metrics = self._build_metrics(
            metrics_config=metrics_config,
            metrics_list=metrics_list,
            num_classes=num_classes,
            device=device,
        )

    def _build_metrics(
        self, metrics_config: dict, metrics_list: list[str], num_classes: int, device: torch.device
    ) -> list[Metric]:
        metrics: list[Metric] = []

        for metric_name in metrics_list:
            c = metrics_config[metric_name]
            match metric_name:
                case "f1":
                    metric = F1Score(task=c["task"], num_classes=num_classes, average=c["average"])
                case "iou":
                    metric = JaccardIndex(task=c["task"], num_classes=num_classes, average=c["average"])
                case "precision":
                    metric = Precision(task=c["task"], num_classes=num_classes, average=c["average"])
                case "recall":
                    metric = Recall(task=c["task"], num_classes=num_classes, average=c["average"])
                case "accuracy":
                    metric = Accuracy(task=c["task"], num_classes=num_classes, average=c["average"])
                case "r2":
                    metric = R2Score(global_mode=c["global_mode"])
                case "relative_l2_error":
                    metric = RelativeL2Error()
                case _:
                    raise ValueError(f"Unsupported metric: {metric_name}")
            metric.__name__ = metric_name
            metrics.append(metric.to(device))

        return metrics

    @property
    def metrics(self) -> list[Metric]:
        return self._metrics

from torchmetrics import Metric
from torchmetrics.classification import (
    F1Score,
    Accuracy,
    Precision,
    Recall,
    MulticlassJaccardIndex,
)


class MetricsComponent:
    def __init__(self, config: dict) -> None:
        self.metrics: list[Metric] = self.build_metrics(config)

    def build_metrics(self, config: dict) -> list[Metric]:
        evaluation_config: str = config["evaluation"]
        num_classes: int = config["data"]["num_classes"]
        metric_list: list[str] = evaluation_config["metrics"]
        average: str = evaluation_config["average"]
        task: str = evaluation_config["task"]

        metrics: list[Metric] = []

        for metric_name in metric_list:
            match metric_name:
                case "f1":
                    metric = F1Score(
                        task=task, num_classes=num_classes, average=average
                    )
                case "iou":
                    metric = MulticlassJaccardIndex(num_classes=num_classes)
                case "precision":
                    metric = Precision(
                        task=task, num_classes=num_classes, average=average
                    )
                case "recall":
                    metric = Recall(task=task, num_classes=num_classes, average=average)
                case "accuracy":
                    metric = Accuracy(task=task, num_classes=num_classes)
                case _:
                    raise ValueError(f"Unsupported metric: {metric_name}")

            metrics.append(metric)

        return metrics

    def get(self) -> list[Metric]:
        return self.metrics

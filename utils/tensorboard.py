from .data_structure_utils import DataStructureUtils
from torch.utils.tensorboard.writer import SummaryWriter


class TensorBoard:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: dict, step: int | None = None):
        metrics = DataStructureUtils.convert_to_builtin_types(metrics)
        for key, value in metrics.items():
            self.writer.add_scalar(f"/{key}", value, step)

    def close(self) -> None:
        self.writer.close()

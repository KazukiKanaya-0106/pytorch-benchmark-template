from utils.data_structure_utils import DataStructureUtils
from torch.utils.tensorboard.writer import SummaryWriter


class TensorBoard:
    def __init__(self, log_dir: str, root_prefix: str = ""):
        self.log_dir: str = log_dir
        self.root_prefix: str = root_prefix
        self.writer = None

    def log_metrics(self, metrics: dict, step: int | None = None):
        if self.writer is None:
            raise RuntimeError("TensorBoard writer is not opened. Call open() first.")
        metrics = DataStructureUtils.convert_to_builtin_types(metrics)
        for key, value in metrics.items():
            self.writer.add_scalar(f"/{self.root_prefix}/{key}" if self.root_prefix else f"/{key}", value, step)

    def open(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            print(f"[TensorBoard] Exception occurred: {exc_value}")

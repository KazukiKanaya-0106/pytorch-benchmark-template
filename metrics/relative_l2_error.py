import torch
from torchmetrics import Metric


class RelativeL2Error(Metric):
    higher_is_better = False
    full_state_update = False

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.add_state("num_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("den_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        err = preds - target
        self.num_sq += torch.sum(err**2)
        self.den_sq += torch.sum(target**2)

    def compute(self):
        return torch.sqrt(self.num_sq) / (torch.sqrt(self.den_sq) + self.eps)

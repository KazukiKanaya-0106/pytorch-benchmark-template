import torch


class TorchUtils:
    @staticmethod
    def move_to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self.move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.move_to_device(x, device) for x in obj)
        else:
            return obj

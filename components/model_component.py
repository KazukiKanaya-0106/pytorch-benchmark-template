import torch
from torch.nn import Module
from models import Resnet18
from models import MLPLayer7


class ModelComponent:
    def __init__(self, config: dict):
        device = torch.device(config["meta"]["device"])
        self.model: Module = self.build_model(config).to(device)

    def build_model(self, config: dict) -> Module:
        model_name: str = config["training"]["model"]
        model_config: dict = config["model"][model_name]
        model: Module = None
        match model_name:
            case "resnet18":
                model = Resnet18(
                    in_channels=model_config["in_channels"],
                    output_dim=model_config["output_dim"],
                    pretrained=model_config["pretrained"],
                )
            case "mlp_layer7":
                model = MLPLayer7(
                    input_dim=model_config["input_dim"],
                    output_dim=model_config["output_dim"],
                )
            case _:
                raise ValueError(f"Unsupported model: {model_name}")

        return model

    def get(self) -> Module:
        return self.model

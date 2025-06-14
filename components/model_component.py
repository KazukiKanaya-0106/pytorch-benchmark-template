from torch.nn import Module
from models import Resnet18


class ModelComponent:
    def __init__(self, config: dict):
        self.model: Module = self.build_model(config)

    def build_model(self, config: dict) -> Module:
        model_name: str = config["training"]["loss"]
        model_config: dict = config["training"][model_name]
        model: Module = None
        match model_name:
            case "resnet18":
                model = Resnet18(
                    in_channels=model_config["in_channels"],
                    output_dim=model_config["output_dim"],
                    pretrained=model_config["pretrained"],
                )
            case _:
                raise ValueError(f"Unsupported model: {model_name}")

        return model

    def get(self) -> Module:
        return self.model

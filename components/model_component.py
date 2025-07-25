from io import BytesIO
import torch
from torch.nn import Module
from models import Resnet18, MLPLayer7, MLPLayer3
from utils.torch_utils import TorchUtils


class ModelComponent:
    def __init__(
        self,
        model_name: str,
        model_config: dict,
        device: torch.device,
        weight_source: str | BytesIO | None = None,
    ):
        self._model = self._build_model(
            model_name=model_name,
            model_config=model_config,
            device=device,
            weight_source=weight_source,
        )

    def _build_model(
        self,
        model_name: str,
        model_config: dict,
        device: torch.device,
        weight_source: str | BytesIO | None = None,
    ) -> Module:
        model: Module | None = None

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
            case "mlp_layer3":
                model = MLPLayer3(
                    input_dim=model_config["input_dim"],
                    output_dim=model_config["output_dim"],
                )
            case _:
                raise ValueError(f"Unsupported model: {model_name}")

        if weight_source:
            print("Load pretrained weight")
            TorchUtils.load_model_state(model=model, source=weight_source)

        return TorchUtils.move_to_device(model, device)

    @property
    def model(self) -> Module:
        return self._model

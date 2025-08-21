from io import BytesIO
import segmentation_models_pytorch as smp
import torch
from torch.nn import Module
from transformers import BertForSequenceClassification
from models.resnet18 import Resnet18
from models.mlp_layer7 import MLPLayer7
from models.mlp_layer3 import MLPLayer3
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
            case "deeplabv3plus_backbone":
                model = smp.DeepLabV3Plus(
                    encoder_name=model_config["encoder_name"],
                    encoder_weights=model_config["encoder_weights"],
                    classes=model_config["classes"],
                    activation=model_config["activation"],
                )
            case "bert-base-uncased":
                model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", num_labels=model_config["num_labels"]
                )
            case "distilbert-base-uncased":
                model = BertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=model_config["num_labels"]
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

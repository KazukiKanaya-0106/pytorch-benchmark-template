import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Callable


class Transforms:
    @staticmethod
    def resnet_transform() -> Callable:
        return A.Compose(
            [
                A.Resize(1024, 1024),
                A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

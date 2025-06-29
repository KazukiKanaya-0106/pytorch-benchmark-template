from typing import Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Transforms:
    @staticmethod
    def resize() -> Callable:
        return A.Compose(
            [
                A.Resize(1024, 1024),
                A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=4),
            ]
        )

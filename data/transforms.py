import albumentations as A
from albumentations.pytorch import ToTensorV2


class Transforms:
    @staticmethod
    def resnet_transform() -> callable:
        return A.Compose(
            [
                A.Resize(1024, 1024),
                A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

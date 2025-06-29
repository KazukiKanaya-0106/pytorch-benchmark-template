import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import TorchUtils


class Predictor:

    def __init__(
        self, model: nn.Module, predict_dataset: Dataset, output_file_dir: str
    ):
        self.model = model
        self.predict_dataset = predict_dataset
        self.output_file_dir = output_file_dir

    def predict(self, num_predicts: int, device: torch.device, skip_plot: bool = False):
        TorchUtils.move_to_device(self.model, device)
        self.model.eval()

        for i in range(1, num_predicts + 1, 1):
            # データセットから取り出されたためtensor型である
            sat_img, mask_img = self.predict_dataset[i]
            # バッチ次元を追加しモデルに入力可能な形[B, C, H, W]にする
            x_tensor = TorchUtils.move_to_device(sat_img, device).unsqueeze(0)
            with torch.no_grad():
                pred_mask = self.model(x_tensor).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.float32)

            sat_img = (sat_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask_img = mask_img.squeeze().numpy()

            self._visualizer(
                id=i,
                skip_plot=skip_plot,
                satellite_image=sat_img,
                predicted_mask=pred_mask,
                ground_truth_mask=mask_img,
            )

    def _visualizer(self, id: int, skip_plot: bool, **images) -> None:
        """
        衛星画像、道路マスク、予測マスクを可視化し保存する

        Args:
            id (int): 保存するフィギュア（予測結果）の識別子
            save (bool): 予測を保存するフラグ
            skip_plot (bool): 予測を描画しないフラグ
            **images: 可視化する画像
        """

        len_image = len(images)
        plt.figure(figsize=(16, 10))
        for i, (name, image) in enumerate(images.items()):

            plt.subplot(1, len_image, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(name.replace("_", " ").title())

            if "mask" in name:
                plt.imshow(image, cmap="gray")
            else:
                plt.imshow(image)

        os.makedirs(self.output_file_dir, exist_ok=True)  # ディレクトリがない場合は作成
        plt.savefig(
            f"{self.output_file_dir}/predict_{id}.png",
            dpi=300,
            bbox_inches="tight",
        )

        # 描画をスキップするかを判断
        if not skip_plot:
            plt.show()

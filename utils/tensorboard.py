from torch.utils.tensorboard import SummaryWriter


class TensorBoard:
    def __init__(self, config: dict):
        tensorboard_dir: str = (
            f'{config["logging"]["root_dir"]}/{config["logging"]["tensorboard"]["dir"]}'
        )
        key = config["meta"]["key"]

        log_dir = f"{tensorboard_dir}/{key}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_epoch(self, train_log: dict, valid_log: dict, epoch: int):

        for key, value in train_log.items():
            self.writer.add_scalar(f"Train/{key}", value, epoch)
        for key, value in valid_log.items():
            self.writer.add_scalar(f"Validation/{key}", value, epoch)

    def close(self) -> None:
        self.writer.close()

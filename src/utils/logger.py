import os
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir: str = "runs/"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_console(self, step: int, metrics: dict):
        msg = f"Step {step:7d} | " + " | ".join(f"{k}: {v:8.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
        print(msg)

    def close(self):
        self.writer.close()

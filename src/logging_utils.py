from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _confusion_matrix_figure(matrix: list[list[int]], class_names: list[str]) -> Any:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(np.array(matrix), cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return fig


class RunLoggers:
    def __init__(self, config: dict[str, Any], output_dir: Path, run_name: str) -> None:
        self.config = config
        self.output_dir = output_dir
        self.run_name = run_name
        self.tb_writer = None
        self.wandb_run = None

    def start(self) -> None:
        if self.config["logging"]["tensorboard"]["enabled"]:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except ImportError as exc:
                raise ImportError("TensorBoard logging requested but tensorboard is not installed.") from exc

        if self.config["logging"]["wandb"]["enabled"]:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError("W&B logging requested but wandb is not installed.") from exc

            wandb_cfg = self.config["logging"]["wandb"]
            self.wandb_run = wandb.init(
                project=wandb_cfg.get("project"),
                entity=wandb_cfg.get("entity"),
                name=self.run_name,
                tags=wandb_cfg.get("tags") or None,
                config=self.config,
                dir=str(self.output_dir),
                mode=wandb_cfg.get("mode") or None,
            )

    def log(self, metrics: dict[str, Any], step: int, prefix: str = "") -> None:
        scalars = scalar_metrics(metrics)
        if prefix:
            scalars = {f"{prefix}/{key}": value for key, value in scalars.items()}
        if self.tb_writer is not None:
            for key, value in scalars.items():
                self.tb_writer.add_scalar(key, value, global_step=step)
        if self.wandb_run is not None:
            self.wandb_run.log(scalars, step=step)

    def log_confusion_matrix(
        self,
        matrix: list[list[int]],
        class_names: list[str],
        step: int,
    ) -> None:
        if self.tb_writer is not None:
            try:
                import matplotlib.pyplot as plt

                fig = _confusion_matrix_figure(matrix, class_names)
                self.tb_writer.add_figure("validation/confusion_matrix", fig, global_step=step)
                plt.close(fig)
            except ImportError:
                LOGGER.warning("matplotlib not installed; skipping TensorBoard confusion matrix figure")

        if self.wandb_run is not None:
            try:
                import matplotlib.pyplot as plt
                import wandb

                fig = _confusion_matrix_figure(matrix, class_names)
                table_rows = [
                    [true_class, predicted_class, int(matrix[true_idx][pred_idx])]
                    for true_idx, true_class in enumerate(class_names)
                    for pred_idx, predicted_class in enumerate(class_names)
                ]
                self.wandb_run.log(
                    {
                        "validation/confusion_matrix": wandb.Image(fig),
                        "validation/confusion_matrix_counts": wandb.Table(
                            columns=["true_class", "predicted_class", "count"],
                            data=table_rows,
                        ),
                    },
                    step=step,
                )
                plt.close(fig)
            except Exception as exc:  # pragma: no cover - logging must not kill training
                LOGGER.warning("Failed to log confusion matrix to W&B: %s", exc)

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()

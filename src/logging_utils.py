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

    matrix_array = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(matrix_array, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    threshold = matrix_array.max() / 2.0 if matrix_array.size and matrix_array.max() > 0 else 0
    for row_idx in range(matrix_array.shape[0]):
        for col_idx in range(matrix_array.shape[1]):
            value = int(matrix_array[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    return fig


class RunLoggers:
    def __init__(self, config: dict[str, Any], output_dir: Path, run_name: str) -> None:
        self.config = config
        self.output_dir = output_dir
        self.run_name = run_name
        self.tb_writer = None
        self.wandb_run = None
        self._comparison_history: dict[str, list[float]] = {
            "epoch": [],
            "train_accuracy": [],
            "validation_accuracy": [],
            "train_loss": [],
            "validation_loss": [],
        }

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
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")

    def log(self, metrics: dict[str, Any], step: int, prefix: str = "") -> None:
        scalars = scalar_metrics(metrics)
        if prefix:
            scalars = {f"{prefix}/{key}": value for key, value in scalars.items()}
        if self.tb_writer is not None:
            for key, value in scalars.items():
                self.tb_writer.add_scalar(key, value, global_step=step)
        if self.wandb_run is not None:
            self.wandb_run.log(scalars, step=step)

    def log_epoch_comparisons(
        self,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        step: int,
    ) -> None:
        train_accuracy = train_metrics.get("accuracy")
        val_accuracy = val_metrics.get("accuracy")
        train_loss = train_metrics.get("loss")
        val_loss = val_metrics.get("loss")

        if self.tb_writer is not None:
            if train_accuracy is not None and val_accuracy is not None:
                self.tb_writer.add_scalars(
                    "accuracy",
                    {"train": float(train_accuracy), "validation": float(val_accuracy)},
                    global_step=step,
                )
            if train_loss is not None and val_loss is not None:
                self.tb_writer.add_scalars(
                    "loss",
                    {"train": float(train_loss), "validation": float(val_loss)},
                    global_step=step,
                )

        if self.wandb_run is None:
            return

        self._comparison_history["epoch"].append(float(step))
        self._comparison_history["train_accuracy"].append(float(train_accuracy))
        self._comparison_history["validation_accuracy"].append(float(val_accuracy))
        self._comparison_history["train_loss"].append(float(train_loss))
        self._comparison_history["validation_loss"].append(float(val_loss))

        try:
            import wandb

            epochs = self._comparison_history["epoch"]
            self.wandb_run.log(
                {
                    "epoch": step,
                    "comparison/accuracy": wandb.plot.line_series(
                        xs=epochs,
                        ys=[
                            self._comparison_history["train_accuracy"],
                            self._comparison_history["validation_accuracy"],
                        ],
                        keys=["train", "validation"],
                        title="Train vs Validation Accuracy",
                        xname="epoch",
                    ),
                    "comparison/loss": wandb.plot.line_series(
                        xs=epochs,
                        ys=[
                            self._comparison_history["train_loss"],
                            self._comparison_history["validation_loss"],
                        ],
                        keys=["train", "validation"],
                        title="Train vs Validation Loss",
                        xname="epoch",
                    ),
                },
                step=step,
            )
        except Exception as exc:  # pragma: no cover - logging must not kill training
            LOGGER.warning("Failed to log W&B comparison charts: %s", exc)

    def log_wrong_examples(self, examples: list[dict[str, Any]], step: int) -> None:
        if self.wandb_run is None or not examples:
            return
        try:
            import wandb

            images = []
            for example in examples:
                caption = (
                    f"true={example['true_label']} | pred={example['predicted_label']} | "
                    f"confidence={example['confidence']:.3f}"
                )
                images.append(wandb.Image(example["image_path"], caption=caption))
            self.wandb_run.log({"validation/wrong_examples": images}, step=step)
        except Exception as exc:  # pragma: no cover - logging must not kill training
            LOGGER.warning("Failed to log wrong examples to W&B: %s", exc)

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

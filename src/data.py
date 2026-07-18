from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Allows CSV validation without a PyTorch install.
    class Dataset:  # type: ignore[no-redef]
        pass


IMAGE_COLUMN = "image_path"
NORMALIZED_LABEL_COLUMN = "category"
DEFAULT_LABEL_COLUMNS = ("class_category", "category")
MISSING_FILE_POLICIES = ("error", "skip")


def resolve_csv_files(csv_source: str | Path) -> list[Path]:
    csv_source = Path(csv_source)
    if not csv_source.exists():
        raise FileNotFoundError(f"CSV path does not exist: {csv_source}")
    if csv_source.is_file():
        return [csv_source]
    if csv_source.is_dir():
        csv_files = sorted(csv_source.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No .csv files found in directory: {csv_source}")
        return csv_files
    raise ValueError(f"CSV path must be a file or directory: {csv_source}")


def select_label_column(
    fieldnames: list[str] | None,
    csv_path: Path,
    label_column: str | None = None,
) -> str:
    if not fieldnames:
        raise ValueError(f"{csv_path} has no CSV header")
    if IMAGE_COLUMN not in fieldnames:
        raise ValueError(f"{csv_path} must contain an {IMAGE_COLUMN!r} column. Found: {fieldnames}")
    if label_column:
        if label_column not in fieldnames:
            raise ValueError(f"{csv_path} must contain label column {label_column!r}. Found: {fieldnames}")
        return label_column
    for candidate in DEFAULT_LABEL_COLUMNS:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        f"{csv_path} must contain one of label columns {DEFAULT_LABEL_COLUMNS}, "
        f"or pass/configure a label column. Found: {fieldnames}"
    )


def read_single_classification_csv(
    csv_path: str | Path,
    label_column: str | None = None,
) -> list[dict[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        selected_label_column = select_label_column(reader.fieldnames, csv_path, label_column)
        rows = [
            {
                IMAGE_COLUMN: (row.get(IMAGE_COLUMN) or "").strip(),
                NORMALIZED_LABEL_COLUMN: (row.get(selected_label_column) or "").strip(),
                "source_csv": str(csv_path),
            }
            for row in reader
        ]

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")
    for idx, row in enumerate(rows, start=2):
        if not row[IMAGE_COLUMN] or not row[NORMALIZED_LABEL_COLUMN]:
            raise ValueError(
                f"Missing {IMAGE_COLUMN}/{selected_label_column} in {csv_path} at line {idx}"
            )
    return rows


def read_classification_csv(
    csv_source: str | Path,
    label_column: str | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in resolve_csv_files(csv_source):
        rows.extend(read_single_classification_csv(csv_path, label_column=label_column))
    if not rows:
        raise ValueError(f"No rows found in CSV source: {csv_source}")
    return rows


def infer_class_to_idx(
    rows: list[dict[str, str]],
    allowed_classes: list[str] | None = None,
) -> dict[str, int]:
    classes = sorted({row[NORMALIZED_LABEL_COLUMN] for row in rows})
    if allowed_classes:
        unknown = sorted(set(classes) - set(allowed_classes))
        if unknown:
            raise ValueError(f"Training CSV contains unknown categories: {unknown}")
    return {category: idx for idx, category in enumerate(classes)}


def split_missing_rows(
    rows: list[dict[str, str]],
    dataset_root: str | Path,
    class_to_idx: dict[str, int],
    validate_files: bool = True,
    split_name: str = "dataset",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dataset_root = Path(dataset_root)
    unknown = sorted({row[NORMALIZED_LABEL_COLUMN] for row in rows} - set(class_to_idx))
    if unknown:
        raise ValueError(f"{split_name} CSV contains classes not found in training CSV: {unknown}")

    valid_rows = []
    missing_rows = []
    if validate_files:
        for row in rows:
            image_path = dataset_root / row[IMAGE_COLUMN]
            if not image_path.is_file():
                missing_row = dict(row)
                missing_row["resolved_path"] = str(image_path)
                missing_rows.append(missing_row)
            else:
                valid_rows.append(row)
    else:
        valid_rows = list(rows)
    return valid_rows, missing_rows


def write_missing_files_log(missing_rows: list[dict[str, str]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_csv", IMAGE_COLUMN, "resolved_path", NORMALIZED_LABEL_COLUMN]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in missing_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def validate_rows(
    rows: list[dict[str, str]],
    dataset_root: str | Path,
    class_to_idx: dict[str, int],
    validate_files: bool = True,
    split_name: str = "dataset",
    missing_files_policy: str = "error",
) -> list[dict[str, str]]:
    if missing_files_policy not in MISSING_FILE_POLICIES:
        raise ValueError(
            f"missing_files_policy must be one of {MISSING_FILE_POLICIES}, got: {missing_files_policy}"
        )
    valid_rows, missing_rows = split_missing_rows(
        rows,
        dataset_root,
        class_to_idx,
        validate_files=validate_files,
        split_name=split_name,
    )
    if missing_rows and missing_files_policy == "error":
        first_missing = [row["resolved_path"] for row in missing_rows[:10]]
        raise FileNotFoundError(
            f"{split_name} CSV references {len(missing_rows)} missing image files. "
            f"First missing paths: {first_missing}"
        )
    if not valid_rows:
        raise ValueError(
            f"{split_name} dataset has no usable rows after checking image files. "
            f"Missing rows: {len(missing_rows)}"
        )
    return valid_rows


class ImageCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        dataset_root: str | Path,
        class_to_idx: dict[str, int],
        transform: Any | None = None,
        validate_files: bool = True,
        return_paths: bool = False,
        split_name: str = "dataset",
        label_column: str | None = None,
        missing_files_policy: str = "error",
        missing_files_log_path: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        rows = read_classification_csv(self.csv_path, label_column=label_column)
        valid_rows, missing_rows = split_missing_rows(
            rows,
            self.dataset_root,
            class_to_idx,
            validate_files=validate_files,
            split_name=split_name,
        )
        if missing_rows and missing_files_policy == "error":
            first_missing = [row["resolved_path"] for row in missing_rows[:10]]
            raise FileNotFoundError(
                f"{split_name} CSV references {len(missing_rows)} missing image files. "
                f"First missing paths: {first_missing}"
            )
        if not valid_rows:
            raise ValueError(
                f"{split_name} dataset has no usable rows after skipping missing files. "
                f"Missing rows: {len(missing_rows)}"
            )
        if missing_rows and missing_files_log_path is not None:
            write_missing_files_log(missing_rows, missing_files_log_path)
        self.rows = valid_rows
        self.missing_rows = missing_rows
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_paths = return_paths
        self.targets = [class_to_idx[row[NORMALIZED_LABEL_COLUMN]] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int] | tuple[Any, int, str]:
        from PIL import Image

        row = self.rows[index]
        path = self.dataset_root / row[IMAGE_COLUMN]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row[NORMALIZED_LABEL_COLUMN]]
        if self.return_paths:
            return image, label, str(path)
        return image, label


def class_counts(targets: list[int], num_classes: int) -> Any:
    import torch

    counts = Counter(targets)
    return torch.tensor([counts.get(i, 0) for i in range(num_classes)], dtype=torch.long)


def make_class_weights(counts: Any, mode: str = "inverse_frequency") -> Any:
    import torch

    if torch.any(counts == 0):
        raise ValueError(f"Cannot compute class weights because some classes have zero samples: {counts}")
    counts_f = counts.float()
    if mode == "inverse_frequency":
        weights = counts_f.sum() / (len(counts_f) * counts_f)
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")
    return weights


def make_weighted_sampler(
    targets: list[int],
    counts: Any,
    replacement: bool = True,
) -> Any:
    import torch
    from torch.utils.data import WeightedRandomSampler

    class_weights = make_class_weights(counts)
    sample_weights = torch.tensor([class_weights[target].item() for target in targets], dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=replacement,
    )

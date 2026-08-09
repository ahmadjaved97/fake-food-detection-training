from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    from torch.utils.data import Dataset, Sampler
except ModuleNotFoundError:  # Allows CSV validation without a PyTorch install.
    class Dataset:  # type: ignore[no-redef]
        pass

    class Sampler:  # type: ignore[no-redef]
        def __class_getitem__(cls, item: Any) -> Any:
            return cls

        pass


IMAGE_COLUMN = "image_path"
NORMALIZED_LABEL_COLUMN = "category"
DEFAULT_LABEL_COLUMNS = ("class_category", "category")
MISSING_FILE_POLICIES = ("error", "skip")


def load_rgb_image(path: str | Path) -> Any:
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode == "P" and "transparency" in img.info:
            return img.convert("RGBA").convert("RGB")
        return img.convert("RGB")


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
        rows = []
        for row in reader:
            cleaned_row = {
                str(key): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            cleaned_row[IMAGE_COLUMN] = (row.get(IMAGE_COLUMN) or "").strip()
            cleaned_row[NORMALIZED_LABEL_COLUMN] = (row.get(selected_label_column) or "").strip()
            cleaned_row["source_csv"] = str(csv_path)
            rows.append(cleaned_row)

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
    if not classes:
        raise ValueError("Cannot infer classes because the filtered training data has no labels.")
    if allowed_classes:
        allowed_classes = [str(category) for category in allowed_classes]
        unknown = sorted(set(classes) - set(allowed_classes))
        if unknown:
            raise ValueError(f"Training CSV contains unknown categories: {unknown}")
        missing = [category for category in allowed_classes if category not in classes]
        if missing:
            raise ValueError(f"Configured allowed_classes are absent from filtered training data: {missing}")
        return {category: idx for idx, category in enumerate(allowed_classes)}
    return {category: idx for idx, category in enumerate(classes)}


def normalize_filter_values(values: Any) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    normalized = [str(value).strip() for value in values if str(value).strip()]
    return normalized or None


def normalize_column_filters(filters: Mapping[str, Any] | None) -> dict[str, set[str]]:
    if not filters:
        return {}
    normalized = {}
    for column, values in filters.items():
        normalized_values = normalize_filter_values(values)
        if normalized_values:
            normalized[str(column)] = set(normalized_values)
    return normalized


def filter_rows_by_label_values(
    rows: list[dict[str, str]],
    include: Any = None,
    exclude: Any = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    include_values = normalize_filter_values(include)
    exclude_values = normalize_filter_values(exclude)
    include_set = set(include_values) if include_values else None
    exclude_set = set(exclude_values) if exclude_values else set()

    kept_rows = []
    dropped_rows = []
    for row in rows:
        label = row[NORMALIZED_LABEL_COLUMN]
        keep = (include_set is None or label in include_set) and label not in exclude_set
        if keep:
            kept_rows.append(row)
        else:
            dropped_rows.append(row)
    return kept_rows, dropped_rows


def filter_rows_by_column_values(
    rows: list[dict[str, str]],
    include: Mapping[str, Any] | None = None,
    exclude: Mapping[str, Any] | None = None,
    split_name: str = "dataset",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    include_filters = normalize_column_filters(include)
    exclude_filters = normalize_column_filters(exclude)
    if not include_filters and not exclude_filters:
        return rows, []

    known_columns = set().union(*(row.keys() for row in rows)) if rows else set()
    missing_columns = sorted((set(include_filters) | set(exclude_filters)) - known_columns)
    if missing_columns:
        raise ValueError(f"{split_name} filters reference missing CSV columns: {missing_columns}")

    kept_rows = []
    dropped_rows = []
    for row in rows:
        included = all(row.get(column, "") in values for column, values in include_filters.items())
        excluded = any(row.get(column, "") in values for column, values in exclude_filters.items())
        if included and not excluded:
            kept_rows.append(row)
        else:
            dropped_rows.append(row)
    return kept_rows, dropped_rows


def split_filter_config(data_cfg: dict[str, Any], key: str, split_name: str) -> dict[str, Any]:
    filters = data_cfg.get(key) or {}
    if split_name == "validation":
        return filters.get("validation") or filters.get("val") or {}
    return filters.get(split_name) or {}


def apply_configured_row_filters(
    rows: list[dict[str, str]],
    data_cfg: dict[str, Any],
    split_name: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    original_count = len(rows)
    label_filter_cfg = split_filter_config(data_cfg, "label_filters", split_name)
    include_labels = normalize_filter_values(label_filter_cfg.get("include"))
    allowed_classes = normalize_filter_values(data_cfg.get("allowed_classes"))
    if allowed_classes is not None:
        include_labels = (
            [label for label in include_labels if label in set(allowed_classes)]
            if include_labels is not None
            else allowed_classes
        )

    rows, dropped_by_label = filter_rows_by_label_values(
        rows,
        include=include_labels,
        exclude=label_filter_cfg.get("exclude"),
    )

    row_filter_cfg = split_filter_config(data_cfg, "row_filters", split_name)
    rows, dropped_by_metadata = filter_rows_by_column_values(
        rows,
        include=row_filter_cfg.get("include"),
        exclude=row_filter_cfg.get("exclude"),
        split_name=split_name,
    )
    if not rows:
        raise ValueError(
            f"{split_name} dataset has no rows after applying label/metadata filters. "
            f"Started with {original_count} rows."
        )
    return rows, {
        "raw_rows": original_count,
        "filtered_label_rows": len(dropped_by_label),
        "filtered_metadata_rows": len(dropped_by_metadata),
        "selected_rows": len(rows),
    }


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
        rows: list[dict[str, str]] | None = None,
        row_filter_report: dict[str, int] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        rows = rows if rows is not None else read_classification_csv(self.csv_path, label_column=label_column)
        row_filter_report = row_filter_report or {
            "raw_rows": len(rows),
            "filtered_label_rows": 0,
            "filtered_metadata_rows": 0,
            "selected_rows": len(rows),
        }
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
        self.total_rows = int(row_filter_report.get("raw_rows", len(rows)))
        self.selected_rows = int(row_filter_report.get("selected_rows", len(rows)))
        self.filtered_label_rows = int(row_filter_report.get("filtered_label_rows", 0))
        self.filtered_metadata_rows = int(row_filter_report.get("filtered_metadata_rows", 0))
        self.rows = valid_rows
        self.missing_rows = missing_rows
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_paths = return_paths
        self.targets = [class_to_idx[row[NORMALIZED_LABEL_COLUMN]] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int] | tuple[Any, int, str]:
        row = self.rows[index]
        path = self.dataset_root / row[IMAGE_COLUMN]
        image = load_rgb_image(path)
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
    seed: int = 0,
) -> Any:
    import torch

    class_weights = make_class_weights(counts)
    sample_weights = torch.tensor([class_weights[target].item() for target in targets], dtype=torch.double)
    return CoordinatedWeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=replacement,
        seed=seed,
    )


class CoordinatedWeightedRandomSampler(Sampler[int]):
    """Weighted sampler with a shared epoch seed for distributed Accelerate runs.

    Every process builds the same weighted index stream for a given epoch. Accelerate
    then shards that stream across processes, avoiding each GPU independently rolling
    a different weighted sample sequence.
    """

    def __init__(
        self,
        weights: Any,
        num_samples: int,
        replacement: bool = True,
        seed: int = 0,
    ) -> None:
        import torch

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        import torch

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=generator,
        ).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

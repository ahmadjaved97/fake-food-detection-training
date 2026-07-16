from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def setup_logging(is_main_process: bool = True) -> None:
    level = logging.INFO if is_main_process else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at top level: {path}")
    return data


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(data: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_jsonl(data: Any, path: str | Path) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def get_env_info() -> dict[str, Any]:
    return {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

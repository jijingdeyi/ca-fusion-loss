"""Project paths. Set IVIF_DATA_ROOT (or a local .env) to your dataset root."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.environ.get("IVIF_DATA_ROOT", ROOT / "data")).expanduser().resolve()

TRAIN_PATH = str(DATA_ROOT / "MSRS" / "train")
TEST_PATH = str(DATA_ROOT / "MSRS" / "test")
DETECT_PATH = str(DATA_ROOT / "MSRS" / "detection")

# Layout expected under IVIF_DATA_ROOT. Each entry is a folder that contains ir/ and vi/.
EVAL_DATASETS = [
    ("MSRS", DATA_ROOT / "MSRS" / "test"),
    ("LLVIP", DATA_ROOT / "llvip_test"),
    ("RoadScene", DATA_ROOT / "RoadScene" / "test"),
    ("FMB", DATA_ROOT / "FMB" / "test"),
    ("M3FD", DATA_ROOT / "M3FD" / "M3FD_Fusion"),
    ("OpIVF", DATA_ROOT / "OpIVF" / "test"),
]

"""Configuration for the 5(+1)-method AWSRE dataset rerun.

Same experimental design as experiments/scientific_config.py (the
dissertation's run): same alpha grid, same 17 attack conditions, same
32x32 requested watermark size, 2 watermarks per host, same seed. Only the
method list is longer, and every path can be set from an environment
variable so the run works in Google Colab (data on Google Drive) as well
as on a local PC.

Environment variables (all optional):
    AWSRE_MS_COCO_DIR, AWSRE_OXFORD_PET_DIR   host image folders
    AWSRE_WATERMARK_DIR                       watermark bank folder
    AWSRE_OUTPUT_DIR                          results folder (SQLite + CSV)
    AWSRE_IMAGES_PER_DATASET                  hosts per dataset (default 400)
    AWSRE_INCLUDE_DCT_QIM                     "1" adds DCT-QIM (default "1")
    AWSRE_WORKERS                             parallel worker processes
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


MS_COCO_DIR = _path("AWSRE_MS_COCO_DIR", Path("/content/drive/MyDrive/MS_COCO_2017_val_800"))
OXFORD_PET_DIR = _path("AWSRE_OXFORD_PET_DIR", Path("/content/drive/MyDrive/Oxford_IIIT_Pet_800"))
WATERMARK_DIR = _path("AWSRE_WATERMARK_DIR", PROJECT_ROOT / "experiments" / "watermark_bank")
OUTPUT_DIR = _path("AWSRE_OUTPUT_DIR", PROJECT_ROOT / "experiments" / "scientific_results")

# The dissertation's 3-method models were trained on 400 + 400 hosts.
# Using the same number keeps the new results directly comparable; the
# runner is resumable, so this can be raised to 800 later and only the
# new hosts are computed.
MAX_IMAGES_PER_DATASET = int(os.environ.get("AWSRE_IMAGES_PER_DATASET", "400"))
WATERMARKS_PER_HOST = 2

METHODS = ["DCT", "DWT", "DCT-SVD", "DWT-SVD", "BLOCK-SVD"]
if os.environ.get("AWSRE_INCLUDE_DCT_QIM", "1") == "1":
    METHODS.append("DCT-QIM")

ALPHAS = [10.0, 15.0, 20.0, 25.0, 30.0]
REQUESTED_WATERMARK_SIZE = 32

ATTACKS = [
    ("none", None),
    ("jpeg", 90), ("jpeg", 70), ("jpeg", 50),
    ("gaussian_noise", 0.005), ("gaussian_noise", 0.010), ("gaussian_noise", 0.020),
    ("gaussian_blur", 3), ("gaussian_blur", 5),
    ("crop", 0.95), ("crop", 0.90),
    ("rotation", 2.0), ("rotation", 5.0),
    ("brightness", -20.0), ("brightness", 20.0),
    ("contrast", -20.0), ("contrast", 20.0),
]

DATABASE_PATH = OUTPUT_DIR / "awsre_scientific_v2.sqlite3"
CSV_PATH = OUTPUT_DIR / "master_scientific_results.csv"
ERROR_LOG_PATH = OUTPUT_DIR / "errors.log"

RANDOM_SEED = 2026
WORKERS = int(os.environ.get("AWSRE_WORKERS", str(max(1, (os.cpu_count() or 2)))))
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LOCAL_BLOCK_SIZE = 32

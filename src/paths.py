# src/paths.py: lấy đường dẫn từ biến môi trường, không viết cứng trong mã
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("ADNI_DATA_DIR", "data/selected"))
OUT_DIR = Path(os.environ.get("ADNI_OUT_DIR", DATA_DIR / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

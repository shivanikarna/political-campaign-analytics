"""Download the Kaggle dataset into data/ using the official Kaggle API.

Setup (one time):
  1. pip install kaggle
  2. Create a token at https://www.kaggle.com/settings -> "Create New Token" (downloads kaggle.json)
  3. Put kaggle.json in  %USERPROFILE%\\.kaggle\\kaggle.json   (Windows)  or  ~/.kaggle/kaggle.json
Then:  python scripts/download_data.py
No Kaggle account? Download the zip manually from the dataset page - see README.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "manchunhui/us-election-2020-tweets"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR)]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Could not run Python.")
        return 1
    except subprocess.CalledProcessError:
        print("\nDownload failed. Check that 'pip install kaggle' succeeded and kaggle.json is in place "
              "(see the notes at the top of this file), or download manually from "
              f"https://www.kaggle.com/datasets/{DATASET}")
        return 1
    for z in DATA_DIR.glob("*.zip"):
        print("Unzipping", z.name)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DATA_DIR)
        z.unlink()
    csvs = sorted(p.name for p in DATA_DIR.glob("*.csv"))
    print("Done. CSV files in data/:", ", ".join(csvs) or "none found")
    return 0 if csvs else 1


if __name__ == "__main__":
    raise SystemExit(main())

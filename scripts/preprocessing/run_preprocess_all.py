from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(script: str) -> None:
    path = ROOT / script
    print(f"Running {path}")
    subprocess.run([sys.executable, str(path)], check=True)


def main() -> None:
    _run("scripts/preprocessing/preprocess_ethics.py")
    _run("scripts/preprocessing/preprocess_normbank.py")
    _run("scripts/preprocessing/preprocess_mfrc.py")
    _run("scripts/preprocessing/preprocess_mfd2.py")
    _run("scripts/preprocessing/preprocess_moralbench.py")
    _run("scripts/EDA/eda_processed.py")


if __name__ == "__main__":
    main()

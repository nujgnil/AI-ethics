from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(script: str) -> None:
    path = ROOT / "scripts" / script
    print(f"Running {path}")
    subprocess.run([sys.executable, str(path)], check=True)


def main() -> None:
    _run("preprocess_ethics.py")
    _run("preprocess_normbank.py")
    _run("preprocess_mfrc.py")
    _run("preprocess_mfd2.py")
    _run("preprocess_moralbench.py")
    _run("eda_processed.py")


if __name__ == "__main__":
    main()

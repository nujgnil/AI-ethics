from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _python_executable() -> Path:
    venv_python = ROOT / "venv" / "Scripts" / "python.exe"
    current_python = Path(sys.executable).resolve()
    if venv_python.exists() and current_python != venv_python.resolve():
        return venv_python
    return current_python


def _run(module: str) -> None:
    python_exe = _python_executable()
    print(f"Running module {module} with {python_exe}")
    env = dict(os.environ)
    src_root = str(ROOT / "src")
    env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
    try:
        subprocess.run([str(python_exe), "-m", module], check=True, cwd=ROOT, env=env)
    except subprocess.CalledProcessError as exc:
        if "preprocess_mfrc" in module:
            raise RuntimeError(
                "MFRC preprocessing requires parquet support. "
                "Use the project virtual environment or install 'pyarrow'."
            ) from exc
        raise


def main() -> None:
    _run("ai_ethics.preprocess.preprocess_hendryicks_ethics")
    _run("ai_ethics.preprocess.preprocess_normbank")
    _run("ai_ethics.preprocess.preprocess_mfrc")
    _run("ai_ethics.preprocess.preprocess_mfd2")
    _run("ai_ethics.preprocess.preprocess_moralbench")
    _run("ai_ethics.analysis.eda_processed")


if __name__ == "__main__":
    main()

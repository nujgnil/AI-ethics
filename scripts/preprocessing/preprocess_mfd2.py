from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "mfd2"
OUT_DIR = ROOT / "data" / "processed" / "mfd2"


def main() -> None:
    if not RAW_DIR.exists():
        print(f"Missing dataset root: {RAW_DIR}")
        return

    dic_files = list(RAW_DIR.glob("*.dic"))
    if not dic_files:
        print("No .dic file found in data/raw/mfd2")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dic_path = dic_files[0]
    out_dic = OUT_DIR / "mfd2.dic"
    shutil.copyfile(dic_path, out_dic)

    docx_files = list(RAW_DIR.glob("*.docx"))
    if docx_files:
        out_docx = OUT_DIR / "mfd2_summary.docx"
        shutil.copyfile(docx_files[0], out_docx)

    print(f"Standardized MFD2 -> {OUT_DIR}")


if __name__ == "__main__":
    main()

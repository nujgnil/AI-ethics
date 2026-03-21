from __future__ import annotations

import argparse
from urllib.request import urlopen
from pathlib import Path

DATASET_REPO = "morebench/morebench"
FILES = {
    "public": "morebench_public.csv",
    "theory": "morebench_theory.csv",
}
BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    root = project_root()
    for candidate in (root / "Data", root / "data"):
        if candidate.exists():
            return candidate
    return root / "Data"


def raw_morebench_dir() -> Path:
    path = data_root() / "raw" / "morebench"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MoReBench CSV files into Data/raw/morebench.")
    parser.add_argument(
        "--subset",
        choices=["all", *FILES.keys()],
        default="all",
        help="Which MoReBench file to download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files.",
    )
    return parser.parse_args()


def download_one(config_name: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return

    url = f"{BASE_URL}/{config_name}"
    print(f"Downloading {config_name} from {DATASET_REPO}")
    with urlopen(url) as response:
        data = response.read()
    output_path.write_bytes(data)
    print(f"Wrote {len(data)} bytes to {output_path}")


def main() -> None:
    args = parse_args()
    out_dir = raw_morebench_dir()

    targets = FILES.items() if args.subset == "all" else [(args.subset, FILES[args.subset])]
    for _, filename in targets:
        download_one(filename, out_dir / filename, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

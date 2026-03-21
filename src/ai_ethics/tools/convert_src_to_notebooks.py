from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
NOTEBOOK_ROOT = REPO_ROOT / "notebooks" / "src"

MAIN_GUARD_RE = re.compile(
    r"\nif __name__ == [\"']__main__[\"']:\n(?P<body>(?:    .*(?:\n|$))*)\s*$",
    re.MULTILINE,
)


def iter_source_files() -> Iterable[Path]:
    yield from sorted(
        path
        for path in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def notebook_markdown(rel_path: Path) -> str:
    title = rel_path.stem.replace("_", " ").title()
    return (
        f"# {title}\n\n"
        f"Generated from `src/{rel_path.as_posix()}`.\n"
        "The first code cell recreates script-like path behavior for notebook execution."
    )


def bootstrap_source(rel_path: Path) -> str:
    rel_posix = rel_path.as_posix()
    return textwrap.dedent(
        f"""
        from pathlib import Path
        import sys

        _NOTEBOOK_BOOTSTRAP_VERBOSE = True

        if _NOTEBOOK_BOOTSTRAP_VERBOSE:
            print("[bootstrap] cwd =", Path.cwd().resolve())

        if "ipykernel" in sys.modules:
            # Avoid argparse failures from Jupyter kernel launch flags.
            sys.argv = [sys.argv[0]]
            if _NOTEBOOK_BOOTSTRAP_VERBOSE:
                print("[bootstrap] detected ipykernel, trimmed sys.argv to:", sys.argv)

        _SOURCE_RELATIVE_PATH = Path("src/{rel_posix}")
        _repo_root = None
        for _candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
            if _NOTEBOOK_BOOTSTRAP_VERBOSE:
                print("[bootstrap] checking candidate:", _candidate)
            if (_candidate / _SOURCE_RELATIVE_PATH).exists():
                _repo_root = _candidate
                if _NOTEBOOK_BOOTSTRAP_VERBOSE:
                    print("[bootstrap] matched repo root:", _repo_root)
                break

        if _repo_root is None:
            _repo_root = Path.cwd().resolve()
            if _NOTEBOOK_BOOTSTRAP_VERBOSE:
                print("[bootstrap] no match found, falling back to cwd:", _repo_root)

        _source_file = (_repo_root / _SOURCE_RELATIVE_PATH).resolve()
        __file__ = str(_source_file)
        if _NOTEBOOK_BOOTSTRAP_VERBOSE:
            print("[bootstrap] source relative path =", _SOURCE_RELATIVE_PATH)
            print("[bootstrap] resolved __file__ =", __file__)

        for _path in (str(_repo_root), str(_source_file.parent)):
            if _path not in sys.path:
                sys.path.insert(0, _path)
                if _NOTEBOOK_BOOTSTRAP_VERBOSE:
                    print("[bootstrap] added to sys.path:", _path)
            elif _NOTEBOOK_BOOTSTRAP_VERBOSE:
                print("[bootstrap] already on sys.path:", _path)
        """
    ).strip()


def split_main_guard(source: str) -> tuple[str, str | None]:
    match = MAIN_GUARD_RE.search(source)
    if not match:
        return source.rstrip(), None

    main_body = textwrap.dedent(match.group("body")).strip()
    return source[: match.start()].rstrip(), main_body or None


def make_cell(cell_type: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if source and not source.endswith("\n"):
        lines[-1] = lines[-1] + "\n"

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell.update(
            {
                "execution_count": None,
                "outputs": [],
            }
        )
    return cell


def build_notebook(rel_path: Path, source: str) -> dict:
    body_source, main_body = split_main_guard(source)
    cells = [
        make_cell("markdown", notebook_markdown(rel_path)),
        make_cell("code", bootstrap_source(rel_path)),
    ]

    if body_source.strip():
        cells.append(make_cell("code", body_source))
    if main_body:
        cells.append(make_cell("markdown", "## Entrypoint\n\nRun the original script entrypoint when needed."))
        cells.append(make_cell("code", main_body))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert_file(source_path: Path) -> Path:
    rel_path = source_path.relative_to(SRC_ROOT)
    target_path = NOTEBOOK_ROOT / rel_path.with_suffix(".ipynb")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    source = source_path.read_text(encoding="utf-8")
    notebook = build_notebook(rel_path, source)
    target_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return target_path


def main() -> None:
    generated = [convert_file(path) for path in iter_source_files()]
    print(f"Generated {len(generated)} notebooks under {NOTEBOOK_ROOT}")
    for path in generated:
        print(path.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()

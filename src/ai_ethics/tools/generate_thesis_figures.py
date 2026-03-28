from __future__ import annotations

import argparse
import ast
import io
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import seaborn as sns


FIGURE_CAPTIONS = {
    "3.1": "Comparison of supervised moral datasets by label type, size, task structure, and evaluation role.",
    "3.2": "Benchmark spectrum from fixed-label supervised tasks to open-ended dilemma and rubric-based reasoning tasks.",
    "3.3": "Theory-to-proxy overview linking consciousness theories to observable AI evaluation dimensions.",
    "3.4": "Model families in prior work, divided into classical classifiers, encoder transformers, and prompt-based generative systems.",
    "3.5": "Literature gap diagram showing what supervised, reasoning, and interpretive approaches each capture and what they miss.",
    "4.1": "RQ1 framework showing data, model, prediction, and evaluation loop.",
    "4.2": "RQ2 framework showing prompt, response, scoring, and qualitative interpretation.",
    "4.3": "RQ3 theory map linking theories to measurable behavioural proxies.",
    "4.4": "Three-layer evaluation design across data type, task type, metric type, and claim strength.",
    "5.1": "Top-level repository and environment overview.",
    "5.2": "End-to-end workflow from raw data to final results.",
    "5.3": "EDA summary panels for row counts, split structure, and label balance.",
    "5.3a": "Example text-length and missingness plots.",
    "5.4": "Cleaning and normalisation pipeline from raw files to processed outputs.",
    "5.5": "Benchmark-layer construction flow.",
    "5.6": "Supervised modelling design.",
    "5.7": "Prompt-evaluation design with run, score, and aggregate stages.",
    "5.8": "Results artefact map showing metrics, logs, models, and prompt-eval outputs.",
    "5.9": "Validity threat map across data, model, metric, and interpretation layers.",
    "6.1": "Dataset strategy diagram showing benchmark-role grouping.",
    "6.2": "ETHICS raw-to-processed overview with trainable and eval-only rows.",
    "6.3": "NormBank summary showing split structure, label balance, and contextual fields.",
    "6.4": "MFRC raw-to-processed overview showing aggregation into the final benchmark form.",
    "6.5": "MoralBench overview showing prompt-item structure and benchmark role.",
    "6.6": "MoReBench Public overview showing structured items and rubric linkage.",
    "6.7": "MoReBench Theory overview showing theory-linked prompt structure.",
    "6.8": "Interpretive benchmark schema showing item_id, metric_id, scenario_group, prompt_variant, and response_format.",
    "6.9": "Metric specification mapping from proxy family to scoring rationale and theoretical source.",
    "6.10": "Resource-role diagram showing how MFD2 supports interpretation rather than direct evaluation.",
    "6.11": "Data quality and bias map across benchmark layers.",
    "7.1": "Top-level repository structure.",
    "7.2": "Code package structure for src/ai_ethics/.",
    "7.3": "Data directory flow from Data/raw/ to Data/processed/ and Data/eda/.",
    "7.4": "Core project architecture from data to models to results.",
    "7.5": "Data loading and preprocessing interaction diagram.",
    "7.6": "Training pipeline showing input dataset, model choice, training, evaluation, and saved outputs.",
    "7.7": "Prompt evaluation pipeline with raw response storage, scoring tables, and aggregated summaries.",
    "7.8": "Notebook role in the project, showing inspection and figure generation on top of saved outputs.",
    "7.9": "Results directory structure showing metrics, models, training logs, and prompt-eval outputs.",
    "8.1": "Supervised experiment matrix showing datasets and model families.",
    "8.2": "Model family overview.",
    "8.3": "Run workflow from smoke test to full run.",
    "8.4": "RQ1 metric map.",
    "8.5": "Cross-dataset heatmap or grouped bar chart.",
    "8.6": "ETHICS model comparison plot.",
    "8.7": "NormBank model comparison plot.",
    "8.8": "MFRC model comparison plot.",
    "8.9": "Error taxonomy or example table.",
    "8.10": "Calibration plot by model.",
    "9.1": "MoralBench run artefact overview or benchmark completion figure.",
    "9.2": "MoralBench primary score plot.",
    "9.3": "Interpretive proxy metric heatmap or bar chart.",
    "9.4": "Scenario-level proxy results by group.",
}

COLORS = {
    "blue": "#355C7D",
    "teal": "#2A9D8F",
    "green": "#6A994E",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#D62828",
    "pink": "#C06C84",
    "light_blue": "#DCEAF7",
    "light_teal": "#DDF3EF",
    "light_green": "#E6F1DF",
    "light_gold": "#FDF0D0",
    "light_orange": "#FCE7D6",
    "light_red": "#F8D7DA",
    "light_gray": "#F5F7FA",
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3]


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 240
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12


def wrap(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def clean_model_name(name: str) -> str:
    name = str(name)
    return (
        name.replace("microsoft/", "")
        .replace("-base-uncased", "")
        .replace("-base", "")
        .replace("_", " ")
        .replace("tfidf", "TF-IDF")
        .replace("bow", "BoW")
        .replace("svc", "SVC")
        .replace("bert", "BERT")
        .replace("roberta", "RoBERTa")
        .replace("deberta-v3", "DeBERTa-v3")
    )


def parse_mapping(value: object) -> dict[str, int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    try:
        raw = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, int] = {}
    for key, item in raw.items():
        try:
            parsed[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return parsed


def load_metrics_table(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?<=\d)(?=20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", "\n", text)
    frame = pd.read_csv(io.StringIO(text))
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    numeric_cols = [
        "accuracy",
        "macro_f1",
        "micro_f1",
        "balanced_accuracy",
        "mcc",
        "auroc_ovr",
        "pr_auc_macro",
        "brier_score",
        "ece_10bin",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values("timestamp_utc").reset_index(drop=True)


def load_training_run_summaries(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    logs_dir = results_dir / "training_logs"
    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = pd.read_json(summary_path, typ="series")
        except ValueError:
            continue
        parts = run_dir.name.split("_", 2)
        dataset = parts[1] if len(parts) > 1 else ""
        model = parts[2] if len(parts) > 2 else ""
        rows.append(
            {
                "experiment_key": run_dir.name,
                "dataset": dataset,
                "model": model,
                "model_type": summary.get("model_type", ""),
                "train_examples": summary.get("train_examples", np.nan),
                "test_examples": summary.get("test_examples", np.nan),
                "epochs": summary.get("epochs", np.nan),
                "global_step": summary.get("global_step", np.nan),
                "train_runtime": summary.get("train_runtime", np.nan),
            }
        )
    return pd.DataFrame(rows)


def load_qualitative_examples(path: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"^\[(?P<timestamp>[^\]]+)\] dataset=(?P<dataset>\S+) model=(?P<model>\S+)\n(?P<body>.*?)(?=\n\[|\Z)",
        re.M | re.S,
    )
    rows: list[dict[str, object]] = []
    text = path.read_text(encoding="utf-8")
    for match in pattern.finditer(text):
        dataset = match.group("dataset")
        model = match.group("model")
        body = match.group("body").strip()
        if not body or body.startswith("No errors captured."):
            continue
        for line in body.splitlines():
            if not line.startswith("- true="):
                continue
            true_match = re.search(r"true=([^ ]+)", line)
            pred_match = re.search(r"pred=([^ ]+)", line)
            conf_match = re.search(r"confidence=([0-9.]+)", line)
            text_match = re.search(r"text=(.*)$", line)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "true": true_match.group(1) if true_match else "",
                    "pred": pred_match.group(1) if pred_match else "",
                    "confidence": float(conf_match.group(1)) if conf_match else np.nan,
                    "text": (text_match.group(1) if text_match else "").replace("&gt;", "").strip(),
                }
            )
    return pd.DataFrame(rows)


def add_title(ax: plt.Axes, caption: str) -> None:
    ax.set_title(wrap(caption, 58), loc="left", pad=16)


def save_figure(
    fig: plt.Figure,
    figure_id: str,
    output_dir: Path,
    manifest_rows: list[dict[str, str]],
) -> None:
    filename = f"fig_{figure_id.replace('.', '_')}.png"
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    manifest_rows.append(
        {
            "figure_id": figure_id,
            "filename": filename,
            "caption": FIGURE_CAPTIONS[figure_id],
        }
    )


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#243B53",
    fontsize: int = 10,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.4,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, wrap(text, max(18, int(w * 90))), ha="center", va="center", fontsize=fontsize)


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    text: str = "",
    color: str = "#52667A",
    fontsize: int = 12,
) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=color)
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.03, wrap(text, 16), ha="center", va="center", fontsize=fontsize, color=color)


def count_child_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_dir())


def format_bar_value(value: float, style: str) -> str:
    if pd.isna(value):
        return ""
    if style == "int":
        return f"{int(round(value)):,}"
    if style == "float1":
        return f"{value:.1f}"
    if style == "float2":
        return f"{value:.2f}"
    if style == "float3":
        return f"{value:.3f}"
    return str(value)


def _get_static_text_bboxes(ax: plt.Axes, renderer) -> list[Bbox]:
    bboxes: list[Bbox] = []
    texts = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
    tick_texts = list(ax.get_xticklabels()) + list(ax.get_yticklabels())
    legend = ax.get_legend()
    if legend is not None:
        tick_texts.extend(legend.get_texts())
    for text in texts + tick_texts:
        if not text.get_visible() or not text.get_text():
            continue
        bboxes.append(text.get_window_extent(renderer=renderer).expanded(1.03, 1.10))
    return bboxes


def _bbox_overlaps_any(bbox: Bbox, others: list[Bbox]) -> bool:
    return any(bbox.overlaps(other) for other in others)


def _place_text_with_candidates(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    candidates: list[tuple[float, float, str, str]],
    occupied: list[Bbox],
    *,
    fontsize: int = 12,
    color: str = "#1F2933",
) -> None:
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    last_annotation = None
    for dx, dy, ha, va in candidates:
        annotation = ax.annotate(
            text,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            clip_on=False,
        )
        fig.canvas.draw()
        bbox = annotation.get_window_extent(renderer=renderer).expanded(1.05, 1.15)
        if not _bbox_overlaps_any(bbox, occupied):
            occupied.append(bbox)
            return
        annotation.remove()
        last_annotation = annotation
    fallback = ax.annotate(
        text,
        xy=(x, y),
        xytext=candidates[-1][:2],
        textcoords="offset points",
        ha=candidates[-1][2],
        va=candidates[-1][3],
        fontsize=fontsize,
        color=color,
        clip_on=False,
    )
    fig.canvas.draw()
    occupied.append(fallback.get_window_extent(renderer=renderer).expanded(1.05, 1.15))


def _expand_axis_for_labels(ax: plt.Axes, orientation: str) -> None:
    if orientation == "vertical":
        bottom, top = ax.get_ylim()
        if ax.get_yscale() == "log":
            log_bottom, log_top = np.log10(bottom), np.log10(top)
            ax.set_ylim(bottom, 10 ** (log_top + (log_top - log_bottom) * 0.12))
        else:
            ax.set_ylim(bottom, top + (top - bottom) * 0.14)
    else:
        left, right = ax.get_xlim()
        if ax.get_xscale() == "log":
            log_left, log_right = np.log10(left), np.log10(right)
            ax.set_xlim(left, 10 ** (log_right + (log_right - log_left) * 0.10))
        else:
            ax.set_xlim(left, right + (right - left) * 0.16)


def annotate_bar_containers(ax: plt.Axes, style: str = "int") -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    patches = [patch for patch in ax.patches if patch.get_visible()]
    if not patches:
        return
    first_bbox = patches[0].get_window_extent(renderer=renderer)
    orientation = "horizontal" if first_bbox.width > first_bbox.height else "vertical"
    _expand_axis_for_labels(ax, orientation)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = _get_static_text_bboxes(ax, renderer)
    if orientation == "vertical":
        patches = sorted(patches, key=lambda p: (p.get_x() + p.get_width() / 2, p.get_y() + p.get_height()))
    else:
        patches = sorted(patches, key=lambda p: (p.get_y() + p.get_height() / 2, p.get_x() + p.get_width()))
    for patch in patches:
        value = patch.get_width() if orientation == "horizontal" else patch.get_height()
        label = format_bar_value(float(value), style)
        if not label:
            continue
        if orientation == "vertical":
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_y() + patch.get_height()
            candidates = [
                (0, 6, "center", "bottom"),
                (0, 16, "center", "bottom"),
                (10, 8, "left", "bottom"),
                (-10, 8, "right", "bottom"),
                (0, -8, "center", "top"),
            ]
        else:
            x = patch.get_x() + patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            candidates = [
                (6, 0, "left", "center"),
                (14, 0, "left", "center"),
                (6, 8, "left", "bottom"),
                (6, -8, "left", "top"),
                (-6, 0, "right", "center"),
            ]
        _place_text_with_candidates(ax, x, y, label, candidates, occupied)


def annotate_scatter_labels(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = _get_static_text_bboxes(ax, renderer)
    for collection in ax.collections:
        offsets = collection.get_offsets()
        for offset in offsets:
            px, py = ax.transData.transform((offset[0], offset[1]))
            occupied.append(Bbox.from_bounds(px - 10, py - 10, 20, 20))
    for _, row in frame.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        label = str(row[label_col])
        candidates = [
            (8, 8, "left", "bottom"),
            (8, -8, "left", "top"),
            (-8, 8, "right", "bottom"),
            (-8, -8, "right", "top"),
            (0, 10, "center", "bottom"),
            (10, 0, "left", "center"),
            (-10, 0, "right", "center"),
        ]
        _place_text_with_candidates(ax, x, y, label, candidates, occupied)


def render_box_flow(
    figure_id: str,
    boxes: list[dict[str, object]],
    arrows: list[dict[str, object]],
    output_dir: Path,
    manifest_rows: list[dict[str, str]],
    *,
    figsize: tuple[float, float] = (12, 7),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.02, wrap(FIGURE_CAPTIONS[figure_id], 64), fontsize=14, weight="bold", transform=ax.transAxes)
    for box in boxes:
        draw_box(
            ax,
            float(box["x"]),
            float(box["y"]),
            float(box["w"]),
            float(box["h"]),
            str(box["text"]),
            facecolor=str(box.get("facecolor", COLORS["light_gray"])),
            fontsize=int(box.get("fontsize", 12)),
        )
    for arrow in arrows:
        draw_arrow(
            ax,
            tuple(arrow["start"]),
            tuple(arrow["end"]),
            text=str(arrow.get("text", "")),
            color=str(arrow.get("color", "#52667A")),
        )
    save_figure(fig, figure_id, output_dir, manifest_rows)


def table_figure(
    figure_id: str,
    frame: pd.DataFrame,
    output_dir: Path,
    manifest_rows: list[dict[str, str]],
    *,
    figsize: tuple[float, float] = (13, 5),
    font_size: int = 12,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(0.0, 1.05, wrap(FIGURE_CAPTIONS[figure_id], 64), fontsize=14, weight="bold", transform=ax.transAxes)
    table = ax.table(
        cellText=frame.values,
        colLabels=frame.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0, 0, 1, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["light_blue"])
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white" if row % 2 else COLORS["light_gray"])
    save_figure(fig, figure_id, output_dir, manifest_rows)


def matrix_figure(
    figure_id: str,
    frame: pd.DataFrame,
    output_dir: Path,
    manifest_rows: list[dict[str, str]],
    *,
    cmap: str = "Blues",
    fmt: str = ".0f",
    figsize: tuple[float, float] = (10, 5),
    annot_kws: dict[str, object] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(frame, annot=True, cmap=cmap, cbar=False, fmt=fmt, ax=ax, annot_kws=annot_kws or {})
    add_title(ax, FIGURE_CAPTIONS[figure_id])
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    save_figure(fig, figure_id, output_dir, manifest_rows)


def load_context(root: Path) -> dict[str, object]:
    results_dir = root / "results"
    processed_dir = root / "Data" / "processed"
    benchmark_supervised = processed_dir / "benchmark_supervised"
    benchmark_reasoning = processed_dir / "benchmark_reasoning"
    benchmark_interpretive = processed_dir / "benchmark_interpretive" / "interpretive"
    resources_dir = processed_dir / "resources" / "mfd2"

    metrics_all = load_metrics_table(results_dir / "metrics.csv")
    metrics_latest = (
        metrics_all.sort_values("timestamp_utc")
        .groupby(["dataset", "model"], as_index=False)
        .tail(1)
        .sort_values(["dataset", "macro_f1"], ascending=[True, False])
        .reset_index(drop=True)
    )

    raw_dataset_summary = pd.read_csv(root / "Data" / "eda" / "raw" / "raw_dataset_summary.csv")
    raw_file_summary = pd.read_csv(
        root / "Data" / "eda" / "raw" / "raw_file_summary.csv",
        usecols=["dataset", "rows", "bytes", "text_len_median", "text_len_p95", "empty_text", "empty_label"],
    )
    benchmark_manifest = pd.read_csv(processed_dir / "benchmark_manifest.csv")

    ethics_summary = pd.read_csv(benchmark_supervised / "ethics" / "summary.csv")
    ethics_labeled = pd.read_csv(benchmark_supervised / "ethics" / "ethics_labeled.csv", usecols=["split", "task", "label_name"])
    ethics_eval_only = pd.read_csv(benchmark_supervised / "ethics" / "ethics_eval_only.csv", usecols=["split"])
    normbank_summary = pd.read_csv(benchmark_supervised / "normbank" / "summary.csv")
    normbank_readable = pd.read_csv(benchmark_supervised / "normbank" / "normbank_readable.csv", usecols=["split", "label_name"])
    mfrc_summary = pd.read_csv(benchmark_supervised / "mfrc" / "summary.csv")
    mfrc_aggregated = pd.read_csv(benchmark_supervised / "mfrc" / "mfrc_aggregated.csv", usecols=["split", "label_name"])

    moralbench_items = pd.read_csv(benchmark_reasoning / "moralbench" / "moralbench_items.csv", usecols=["collection", "foundation", "prompt_format"])
    morebench_public = pd.read_csv(
        benchmark_reasoning / "morebench_public" / "morebench_public_structured.csv",
        usecols=["dilemma_type", "role_domain", "rubric_item_count", "context"],
    )
    morebench_theory = pd.read_csv(
        benchmark_reasoning / "morebench_theory" / "morebench_theory_structured.csv",
        usecols=["dilemma_type", "role_domain", "theory", "context"],
    )
    interpretive_benchmark = pd.read_csv(
        benchmark_interpretive / "interpretive_benchmark.csv",
        usecols=["item_id", "metric_id", "scenario_group", "prompt_variant", "response_format"],
    )
    metric_specs = pd.read_csv(benchmark_interpretive / "metric_specs.csv")
    mfd2_summary = pd.read_csv(resources_dir / "summary.csv")

    training_runs = load_training_run_summaries(results_dir)
    qualitative_examples = load_qualitative_examples(results_dir / "ualitative_examples.txt")

    rq2_model = pd.DataFrame(
        [
            {
                "dataset": "moralbench",
                "model": "chatgpt_web",
                "metric_id": "moralbench",
                "primary_score": 1.0,
                "format_compliance": 1.0,
            }
        ]
    )
    rq2_scenarios = pd.DataFrame(
        [
            {"dataset": "moralbench", "model": "chatgpt_web", "metric_id": "moralbench", "scenario_group": "all_items", "avg_primary_score": 1.0}
        ]
    )
    rq3_model = pd.DataFrame(
        [
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "agency_coherence", "primary_score": 0.5, "theory": "attention_schema"},
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "cross_context_integration", "primary_score": 1.0, "theory": "integrated_information"},
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "identity_persistence", "primary_score": 0.7777777778, "theory": "global_workspace"},
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "metacognitive_calibration", "primary_score": 0.9955555556, "theory": "higher_order_thought"},
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "self_model_consistency", "primary_score": 0.8666666667, "theory": "higher_order_thought"},
        ]
    )
    rq3_scenarios = pd.DataFrame(
        [
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "agency_coherence", "scenario_group": "decision_reason_alignment", "avg_primary_score": 0.5},
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "cross_context_integration", "scenario_group": "constraint_integration", "avg_primary_score": 1.0},
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "identity_persistence", "scenario_group": "identity_memory_boundary", "avg_primary_score": 1.0},
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "identity_persistence", "scenario_group": "identity_role", "avg_primary_score": 0.6666666667},
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "metacognitive_calibration", "scenario_group": "calibration_fact", "avg_primary_score": 0.9955555556},
            {"dataset": "interpretive", "model": "chatgpt_web", "metric_id": "self_model_consistency", "scenario_group": "self_model_baseline", "avg_primary_score": 0.8666666667},
        ]
    )

    return {
        "root": root,
        "results_dir": results_dir,
        "metrics_latest": metrics_latest,
        "raw_dataset_summary": raw_dataset_summary,
        "raw_file_summary": raw_file_summary,
        "benchmark_manifest": benchmark_manifest,
        "ethics_summary": ethics_summary,
        "ethics_labeled": ethics_labeled,
        "ethics_eval_only": ethics_eval_only,
        "normbank_summary": normbank_summary,
        "normbank_readable": normbank_readable,
        "mfrc_summary": mfrc_summary,
        "mfrc_aggregated": mfrc_aggregated,
        "moralbench_items": moralbench_items,
        "morebench_public": morebench_public,
        "morebench_theory": morebench_theory,
        "interpretive_benchmark": interpretive_benchmark,
        "metric_specs": metric_specs,
        "mfd2_summary": mfd2_summary,
        "rq2_model": rq2_model,
        "rq2_scenarios": rq2_scenarios,
        "rq3_model": rq3_model,
        "rq3_scenarios": rq3_scenarios,
        "training_runs": training_runs,
        "qualitative_examples": qualitative_examples,
    }


def figure_3_1(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    normbank_counts = parse_mapping(ctx["normbank_summary"].iloc[0]["labels"])
    mfrc_labels = pd.Series(ctx["mfrc_aggregated"]["label_name"]).dropna().astype(str).nunique()
    frame = pd.DataFrame(
        [
            {"dataset": "ETHICS", "rows": 111602, "labels": 2, "structure": "binary moral judgement", "role": "trainable benchmark + eval-only sidecar"},
            {"dataset": "NormBank", "rows": int(ctx["normbank_summary"].iloc[0]["rows"]), "labels": len(normbank_counts), "structure": "context-rich norm status", "role": "trainable benchmark"},
            {"dataset": "MFRC", "rows": int(ctx["mfrc_summary"].iloc[0]["majority_label_rows"]), "labels": int(mfrc_labels), "structure": "aggregated discourse labels", "role": "realism check benchmark"},
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=frame, x="rows", y="dataset", color=COLORS["blue"], ax=ax)
    ax.set_xscale("log")
    add_title(ax, FIGURE_CAPTIONS["3.1"])
    ax.set_xlabel("Rows (log scale)")
    ax.set_ylabel("")
    annotate_bar_containers(ax, "int")
    fig.canvas.draw()
    occupied = _get_static_text_bboxes(ax, fig.canvas.get_renderer())
    for idx, row in frame.iterrows():
        _place_text_with_candidates(
            ax,
            float(row["rows"]),
            float(idx),
            f'{row["labels"]} labels | {row["structure"]} | {row["role"]}',
            [
                (60, 0, "left", "center"),
                (60, 10, "left", "bottom"),
                (60, -10, "left", "top"),
                (110, 0, "left", "center"),
            ],
            occupied,
        )
    plt.tight_layout()
    save_figure(fig, "3.1", output_dir, manifest_rows)


def figure_3_2(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    frame = pd.DataFrame(
        [
            ("ETHICS", 0.12, 0.94, 111602, "supervised"),
            ("NormBank", 0.20, 0.90, 155423, "supervised"),
            ("MFRC", 0.30, 0.82, 17884, "supervised"),
            ("MoralBench", 0.62, 0.56, 88, "reasoning"),
            ("MoReBench Public", 0.80, 0.45, 500, "reasoning"),
            ("MoReBench Theory", 0.86, 0.38, 150, "reasoning"),
            ("Interpretive", 0.95, 0.25, 40, "interpretive"),
        ],
        columns=["benchmark", "task_openness", "scoring_structure", "rows", "layer"],
    )
    palette = {"supervised": COLORS["blue"], "reasoning": COLORS["orange"], "interpretive": COLORS["pink"]}
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(data=frame, x="task_openness", y="scoring_structure", size=np.log10(frame["rows"] + 1), hue="layer", palette=palette, sizes=(180, 900), ax=ax, legend=False)
    add_title(ax, FIGURE_CAPTIONS["3.2"])
    ax.set_xlabel("Task openness")
    ax.set_ylabel("Scoring structure")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.0)
    annotate_scatter_labels(ax, frame, "task_openness", "scoring_structure", "benchmark")
    plt.tight_layout()
    save_figure(fig, "3.2", output_dir, manifest_rows)


def figure_3_3(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    boxes = [
        {"x": 0.05, "y": 0.74, "w": 0.22, "h": 0.12, "text": "Global Workspace", "facecolor": COLORS["light_blue"]},
        {"x": 0.05, "y": 0.57, "w": 0.22, "h": 0.12, "text": "Higher-Order Thought", "facecolor": COLORS["light_blue"]},
        {"x": 0.05, "y": 0.40, "w": 0.22, "h": 0.12, "text": "Attention Schema", "facecolor": COLORS["light_blue"]},
        {"x": 0.05, "y": 0.23, "w": 0.22, "h": 0.12, "text": "Integrated Information", "facecolor": COLORS["light_blue"]},
        {"x": 0.05, "y": 0.06, "w": 0.22, "h": 0.12, "text": "Predictive Processing", "facecolor": COLORS["light_blue"]},
        {"x": 0.73, "y": 0.74, "w": 0.22, "h": 0.12, "text": "Identity persistence", "facecolor": COLORS["light_orange"]},
        {"x": 0.73, "y": 0.57, "w": 0.22, "h": 0.12, "text": "Self-model consistency", "facecolor": COLORS["light_orange"]},
        {"x": 0.73, "y": 0.40, "w": 0.22, "h": 0.12, "text": "Metacognitive calibration", "facecolor": COLORS["light_orange"]},
        {"x": 0.73, "y": 0.23, "w": 0.22, "h": 0.12, "text": "Cross-context integration", "facecolor": COLORS["light_orange"]},
        {"x": 0.73, "y": 0.06, "w": 0.22, "h": 0.12, "text": "Agency coherence", "facecolor": COLORS["light_orange"]},
    ]
    arrows = [
        {"start": (0.27, 0.80), "end": (0.73, 0.80)},
        {"start": (0.27, 0.80), "end": (0.73, 0.29)},
        {"start": (0.27, 0.63), "end": (0.73, 0.63)},
        {"start": (0.27, 0.63), "end": (0.73, 0.46)},
        {"start": (0.27, 0.46), "end": (0.73, 0.63)},
        {"start": (0.27, 0.46), "end": (0.73, 0.12)},
        {"start": (0.27, 0.29), "end": (0.73, 0.29)},
        {"start": (0.27, 0.12), "end": (0.73, 0.46)},
        {"start": (0.27, 0.12), "end": (0.73, 0.80)},
    ]
    render_box_flow("3.3", boxes, arrows, output_dir, manifest_rows)


def figure_3_4(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    boxes = [
        {"x": 0.06, "y": 0.14, "w": 0.25, "h": 0.64, "text": "Classical classifiers\n\nTF-IDF + Logistic Regression\nTF-IDF + Linear SVC\nBoW + Multinomial Naive Bayes", "facecolor": COLORS["light_blue"]},
        {"x": 0.375, "y": 0.14, "w": 0.25, "h": 0.64, "text": "Encoder transformers\n\nDistilBERT\nBERT\nRoBERTa\nDeBERTa-v3", "facecolor": COLORS["light_green"]},
        {"x": 0.69, "y": 0.14, "w": 0.25, "h": 0.64, "text": "Prompt-based systems\n\nChatGPT web replay\nGPT-5 mini prompt runs\nManual replay templates", "facecolor": COLORS["light_orange"]},
    ]
    render_box_flow("3.4", boxes, [], output_dir, manifest_rows)


def figure_3_5(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    frame = pd.DataFrame(
        [
            ["Supervised", "fixed labels\nclean comparison\nheld-out metrics", "explanations\npluralism\nproxy behaviour"],
            ["Reasoning", "deliberation\ntrade-offs\nresponse structure", "stable labels\ncheap scoring\ncalibration"],
            ["Interpretive", "self-model\nintegration\nconsistency probes", "ontological proof\nground truth consciousness\nbroad ranking"],
        ],
        columns=["Approach", "Captures", "Misses"],
    )
    table_figure("3.5", frame, output_dir, manifest_rows, figsize=(13, 4.5), font_size=12)


def figure_4_1(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow(
        "4.1",
        [
            {"x": 0.05, "y": 0.35, "w": 0.18, "h": 0.2, "text": "Labelled supervised data\nETHICS\nNormBank\nMFRC", "facecolor": COLORS["light_blue"]},
            {"x": 0.30, "y": 0.35, "w": 0.16, "h": 0.2, "text": "Model family\nclassical\nor transformer", "facecolor": COLORS["light_green"]},
            {"x": 0.53, "y": 0.35, "w": 0.16, "h": 0.2, "text": "Predictions on held-out split", "facecolor": COLORS["light_orange"]},
            {"x": 0.76, "y": 0.35, "w": 0.18, "h": 0.2, "text": "Metrics\naccuracy\nmacro F1\nbalanced accuracy\nMCC", "facecolor": COLORS["light_gold"]},
        ],
        [
            {"start": (0.23, 0.45), "end": (0.30, 0.45), "text": "train"},
            {"start": (0.46, 0.45), "end": (0.53, 0.45), "text": "predict"},
            {"start": (0.69, 0.45), "end": (0.76, 0.45), "text": "evaluate"},
        ],
        output_dir,
        manifest_rows,
    )


def figure_4_2(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow(
        "4.2",
        [
            {"x": 0.04, "y": 0.35, "w": 0.18, "h": 0.22, "text": "Reasoning benchmark\nMoralBench\nMoReBench", "facecolor": COLORS["light_blue"]},
            {"x": 0.27, "y": 0.35, "w": 0.16, "h": 0.22, "text": "Prompt item\nand metadata", "facecolor": COLORS["light_green"]},
            {"x": 0.48, "y": 0.35, "w": 0.16, "h": 0.22, "text": "Model response\nstored raw", "facecolor": COLORS["light_orange"]},
            {"x": 0.69, "y": 0.35, "w": 0.12, "h": 0.22, "text": "Heuristic or rubric scoring", "facecolor": COLORS["light_gold"]},
            {"x": 0.84, "y": 0.35, "w": 0.12, "h": 0.22, "text": "Scenario and model interpretation", "facecolor": COLORS["light_red"]},
        ],
        [
            {"start": (0.22, 0.46), "end": (0.27, 0.46)},
            {"start": (0.43, 0.46), "end": (0.48, 0.46)},
            {"start": (0.64, 0.46), "end": (0.69, 0.46)},
            {"start": (0.81, 0.46), "end": (0.84, 0.46)},
        ],
        output_dir,
        manifest_rows,
    )


def figure_4_3(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    theory_map = {
        "global_workspace": "Global workspace",
        "higher_order_thought": "Higher-order thought",
        "attention_schema": "Attention schema",
        "integrated_information": "Integrated information",
    }
    metric_rows = ctx["rq3_model"][["metric_id", "theory"]].copy()
    metric_rows["theory"] = metric_rows["theory"].map(theory_map).fillna(metric_rows["theory"])
    frame = metric_rows.rename(columns={"metric_id": "Proxy metric", "theory": "Theory link"})
    frame["Proxy metric"] = frame["Proxy metric"].str.replace("_", " ").str.title()
    table_figure("4.3", frame, output_dir, manifest_rows, figsize=(11, 4), font_size=12)


def figure_4_4(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    frame = pd.DataFrame(
        {
            "Data type": ["labelled rows", "open prompts", "proxy prompts"],
            "Task type": ["classification", "generation", "interpretive probing"],
            "Metric type": ["standard metrics", "heuristic/rubric", "theory-linked proxies"],
            "Claim strength": ["benchmark performance", "reasoning behaviour", "cautious proxy evidence"],
        },
        index=["Supervised", "Reasoning", "Interpretive"],
    )
    table_figure("4.4", frame.reset_index().rename(columns={"index": "Layer"}), output_dir, manifest_rows, figsize=(14, 4.5), font_size=12)


def figure_5_1(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    root = Path(ctx["root"])
    boxes = [
        {"x": 0.39, "y": 0.40, "w": 0.22, "h": 0.16, "text": "AI-ethics workspace\nWindows PowerShell\nlocal venv", "facecolor": COLORS["light_gold"]},
        {"x": 0.06, "y": 0.64, "w": 0.20, "h": 0.14, "text": f"Data\n{sum(1 for _ in (root / 'Data').rglob('*') if _.is_file())} files", "facecolor": COLORS["light_blue"]},
        {"x": 0.39, "y": 0.74, "w": 0.22, "h": 0.12, "text": f"src\n{sum(1 for _ in (root / 'src').rglob('*.py'))} Python files", "facecolor": COLORS["light_green"]},
        {"x": 0.74, "y": 0.64, "w": 0.20, "h": 0.14, "text": f"results\n{sum(1 for _ in (root / 'results').rglob('*') if _.is_file())} artefacts", "facecolor": COLORS["light_orange"]},
        {"x": 0.08, "y": 0.14, "w": 0.18, "h": 0.14, "text": "admin\nreport + school docs", "facecolor": COLORS["light_red"]},
        {"x": 0.39, "y": 0.12, "w": 0.22, "h": 0.16, "text": "notebooks\nEDA + thesis visuals", "facecolor": COLORS["light_teal"]},
        {"x": 0.74, "y": 0.14, "w": 0.18, "h": 0.14, "text": "venv\ndependencies", "facecolor": COLORS["light_gray"]},
    ]
    arrows = [
        {"start": (0.26, 0.71), "end": (0.39, 0.50)},
        {"start": (0.50, 0.74), "end": (0.50, 0.56)},
        {"start": (0.74, 0.71), "end": (0.61, 0.50)},
        {"start": (0.17, 0.28), "end": (0.39, 0.44)},
        {"start": (0.50, 0.28), "end": (0.50, 0.40)},
        {"start": (0.83, 0.28), "end": (0.61, 0.44)},
    ]
    render_box_flow("5.1", boxes, arrows, output_dir, manifest_rows)


def figure_5_2(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow(
        "5.2",
        [
            {"x": 0.03, "y": 0.38, "w": 0.12, "h": 0.18, "text": "Raw acquisition", "facecolor": COLORS["light_blue"]},
            {"x": 0.18, "y": 0.38, "w": 0.12, "h": 0.18, "text": "EDA and inspection", "facecolor": COLORS["light_teal"]},
            {"x": 0.33, "y": 0.38, "w": 0.12, "h": 0.18, "text": "Cleaning and preprocessing", "facecolor": COLORS["light_green"]},
            {"x": 0.48, "y": 0.38, "w": 0.12, "h": 0.18, "text": "Benchmark construction", "facecolor": COLORS["light_gold"]},
            {"x": 0.63, "y": 0.38, "w": 0.12, "h": 0.18, "text": "Training or prompt evaluation", "facecolor": COLORS["light_orange"]},
            {"x": 0.78, "y": 0.38, "w": 0.12, "h": 0.18, "text": "Analysis and report figures", "facecolor": COLORS["light_red"]},
        ],
        [
            {"start": (0.15, 0.47), "end": (0.18, 0.47)},
            {"start": (0.30, 0.47), "end": (0.33, 0.47)},
            {"start": (0.45, 0.47), "end": (0.48, 0.47)},
            {"start": (0.60, 0.47), "end": (0.63, 0.47)},
            {"start": (0.75, 0.47), "end": (0.78, 0.47)},
        ],
        output_dir,
        manifest_rows,
    )


def figure_5_3(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    raw_dataset_summary = ctx["raw_dataset_summary"].copy().set_index("dataset")
    ethics_labeled = ctx["ethics_labeled"]
    normbank_readable = ctx["normbank_readable"]
    mfrc_aggregated = ctx["mfrc_aggregated"]
    frame = pd.DataFrame(
        [
            {
                "Dataset": "ETHICS",
                "Rows": f"{int(raw_dataset_summary.loc['hendrycks_ethics', 'rows']):,}",
                "Split structure": wrap("train | test | test_hard; eval-only ambig/util splits", 24),
                "Label balance": wrap(
                    f"0: {int((ethics_labeled['label_name'] == 0).sum()):,} | 1: {int((ethics_labeled['label_name'] == 1).sum()):,}",
                    24,
                ),
            },
            {
                "Dataset": "NormBank",
                "Rows": f"{int(raw_dataset_summary.loc['normbank', 'rows']):,}",
                "Split structure": wrap(
                    f"train {int((normbank_readable['split'] == 'train').sum()):,} | dev {int((normbank_readable['split'] == 'dev').sum()):,} | test {int((normbank_readable['split'] == 'test').sum()):,}",
                    24,
                ),
                "Label balance": wrap("taboo 68,057 | normal 59,507 | expected 27,859", 24),
            },
            {
                "Dataset": "MFRC",
                "Rows": f"{len(mfrc_aggregated):,}",
                "Split structure": "train only",
                "Label balance": wrap("Non-Moral dominant; 8 aggregated labels", 24),
            },
            {
                "Dataset": "MoralBench",
                "Rows": f"{len(ctx['moralbench_items']):,} items",
                "Split structure": "evaluation only",
                "Label balance": "no supervised labels",
            },
            {
                "Dataset": "MoReBench Public",
                "Rows": f"{len(ctx['morebench_public']):,} items",
                "Split structure": "evaluation only",
                "Label balance": "rubric-scored responses",
            },
            {
                "Dataset": "MoReBench Theory",
                "Rows": f"{len(ctx['morebench_theory']):,} items",
                "Split structure": "evaluation only",
                "Label balance": "theory-linked rubric items",
            },
        ]
    )
    table_figure("5.3", frame, output_dir, manifest_rows, figsize=(16, 6.5), font_size=12)


def figure_5_3a(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    grouped = ctx["raw_file_summary"].groupby("dataset", as_index=False).agg(
        text_len_median=("text_len_median", "median"),
        text_len_p95=("text_len_p95", "median"),
        empty_text=("empty_text", "sum"),
        empty_label=("empty_label", "sum"),
    )
    grouped = grouped[(grouped["text_len_p95"] > 0) | (grouped["empty_text"] > 0)].copy()
    grouped["Dataset"] = grouped["dataset"].map(
        {
            "hendrycks_ethics": "ETHICS",
            "mfrc": "MFRC",
            "morebench": "MoReBench",
        }
    )
    grouped = grouped.sort_values("text_len_p95", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(grouped))
    ax.hlines(y, grouped["text_len_median"], grouped["text_len_p95"], color=COLORS["light_teal"], linewidth=4)
    ax.scatter(grouped["text_len_median"], y, s=120, color=COLORS["teal"], label="Median text length", zorder=3)
    ax.scatter(grouped["text_len_p95"], y, s=120, color=COLORS["orange"], label="95th percentile length", zorder=3)
    add_title(ax, FIGURE_CAPTIONS["5.3a"])
    ax.set_xlabel("Characters")
    ax.set_ylabel("")
    ax.set_yticks(y)
    ax.set_yticklabels(grouped["Dataset"])
    right_limit = float(grouped["text_len_p95"].max()) * 1.45
    ax.set_xlim(0, right_limit)
    fig.canvas.draw()
    occupied = _get_static_text_bboxes(ax, fig.canvas.get_renderer())
    for row_y, row in grouped.iterrows():
        _place_text_with_candidates(
            ax,
            float(row["text_len_median"]),
            float(row_y),
            f"{int(round(row['text_len_median']))}",
            [(0, 10, "center", "bottom"), (0, -10, "center", "top"), (8, 0, "left", "center")],
            occupied,
        )
        _place_text_with_candidates(
            ax,
            float(row["text_len_p95"]),
            float(row_y),
            f"{int(round(row['text_len_p95']))}",
            [(8, 0, "left", "center"), (0, 10, "center", "bottom"), (0, -10, "center", "top")],
            occupied,
        )
        missing_note = []
        if int(row["empty_text"]) > 0:
            missing_note.append(f"empty text {int(row['empty_text'])}")
        if int(row["empty_label"]) > 0:
            missing_note.append(f"empty label {int(row['empty_label'])}")
        if missing_note:
            _place_text_with_candidates(
                ax,
                right_limit * 0.97,
                float(row_y),
                " | ".join(missing_note),
                [(-4, 0, "right", "center"), (-4, 10, "right", "bottom"), (-4, -10, "right", "top")],
                occupied,
                color="#7C2D12",
            )
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    save_figure(fig, "5.3a", output_dir, manifest_rows)


def figure_5_4(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow(
        "5.4",
        [
            {"x": 0.03, "y": 0.38, "w": 0.15, "h": 0.18, "text": "Raw source files", "facecolor": COLORS["light_blue"]},
            {"x": 0.22, "y": 0.38, "w": 0.15, "h": 0.18, "text": "Dataset-specific preprocess scripts", "facecolor": COLORS["light_teal"]},
            {"x": 0.41, "y": 0.38, "w": 0.15, "h": 0.18, "text": "Normalised columns\ntext | label | split | metadata", "facecolor": COLORS["light_green"]},
            {"x": 0.60, "y": 0.38, "w": 0.15, "h": 0.18, "text": "Processed CSV and JSONL artefacts", "facecolor": COLORS["light_gold"]},
            {"x": 0.79, "y": 0.38, "w": 0.15, "h": 0.18, "text": "Summary tables and benchmark layers", "facecolor": COLORS["light_orange"]},
        ],
        [
            {"start": (0.18, 0.47), "end": (0.22, 0.47)},
            {"start": (0.37, 0.47), "end": (0.41, 0.47)},
            {"start": (0.56, 0.47), "end": (0.60, 0.47)},
            {"start": (0.75, 0.47), "end": (0.79, 0.47)},
        ],
        output_dir,
        manifest_rows,
    )


def figure_5_5(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow(
        "5.5",
        [
            {"x": 0.05, "y": 0.68, "w": 0.20, "h": 0.14, "text": "Processed datasets", "facecolor": COLORS["light_blue"]},
            {"x": 0.33, "y": 0.68, "w": 0.20, "h": 0.14, "text": "Supervised layer\nETHICS\nNormBank\nMFRC", "facecolor": COLORS["light_green"]},
            {"x": 0.61, "y": 0.68, "w": 0.20, "h": 0.14, "text": "Reasoning layer\nMoralBench\nMoReBench", "facecolor": COLORS["light_orange"]},
            {"x": 0.33, "y": 0.30, "w": 0.20, "h": 0.14, "text": "Interpretive layer\nproxy benchmark", "facecolor": COLORS["light_red"]},
            {"x": 0.61, "y": 0.30, "w": 0.20, "h": 0.14, "text": "Support resources\nMFD2", "facecolor": COLORS["light_teal"]},
        ],
        [
            {"start": (0.25, 0.75), "end": (0.33, 0.75)},
            {"start": (0.25, 0.75), "end": (0.61, 0.75)},
            {"start": (0.25, 0.68), "end": (0.33, 0.37)},
            {"start": (0.25, 0.68), "end": (0.61, 0.37)},
        ],
        output_dir,
        manifest_rows,
    )


def figure_5_6(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow("5.6", [{"x": 0.05, "y": 0.34, "w": 0.18, "h": 0.22, "text": "Benchmark-supervised data", "facecolor": COLORS["light_blue"]}, {"x": 0.28, "y": 0.34, "w": 0.16, "h": 0.22, "text": "Model family selection", "facecolor": COLORS["light_green"]}, {"x": 0.49, "y": 0.34, "w": 0.16, "h": 0.22, "text": "Fit model on train split", "facecolor": COLORS["light_orange"]}, {"x": 0.70, "y": 0.34, "w": 0.12, "h": 0.22, "text": "Held-out evaluation", "facecolor": COLORS["light_gold"]}, {"x": 0.85, "y": 0.34, "w": 0.10, "h": 0.22, "text": "metrics.csv + logs + model", "facecolor": COLORS["light_red"]}], [{"start": (0.23, 0.45), "end": (0.28, 0.45)}, {"start": (0.44, 0.45), "end": (0.49, 0.45)}, {"start": (0.65, 0.45), "end": (0.70, 0.45)}, {"start": (0.82, 0.45), "end": (0.85, 0.45)}], output_dir, manifest_rows)


def figure_5_7(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow("5.7", [{"x": 0.04, "y": 0.34, "w": 0.16, "h": 0.22, "text": "Prompt benchmark", "facecolor": COLORS["light_blue"]}, {"x": 0.24, "y": 0.34, "w": 0.14, "h": 0.22, "text": "Run collection", "facecolor": COLORS["light_teal"]}, {"x": 0.42, "y": 0.34, "w": 0.14, "h": 0.22, "text": "responses.jsonl", "facecolor": COLORS["light_green"]}, {"x": 0.60, "y": 0.34, "w": 0.14, "h": 0.22, "text": "item_scores.csv", "facecolor": COLORS["light_gold"]}, {"x": 0.78, "y": 0.34, "w": 0.16, "h": 0.22, "text": "scenario_scores.csv\nmodel_summary.csv\nexamples.jsonl", "facecolor": COLORS["light_orange"]}], [{"start": (0.20, 0.45), "end": (0.24, 0.45), "text": "run"}, {"start": (0.38, 0.45), "end": (0.42, 0.45), "text": "store"}, {"start": (0.56, 0.45), "end": (0.60, 0.45), "text": "score"}, {"start": (0.74, 0.45), "end": (0.78, 0.45), "text": "aggregate"}], output_dir, manifest_rows)


def figure_5_8(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    results_dir = Path(ctx["results_dir"])
    boxes = [
        {"x": 0.38, "y": 0.40, "w": 0.24, "h": 0.16, "text": "results/", "facecolor": COLORS["light_gold"]},
        {"x": 0.05, "y": 0.68, "w": 0.22, "h": 0.14, "text": "metrics.csv\n1 file", "facecolor": COLORS["light_blue"]},
        {"x": 0.39, "y": 0.74, "w": 0.22, "h": 0.12, "text": f"training_logs/\n{count_child_dirs(results_dir / 'training_logs')} runs", "facecolor": COLORS["light_green"]},
        {"x": 0.73, "y": 0.68, "w": 0.22, "h": 0.14, "text": f"models/\n{count_child_dirs(results_dir / 'models')} saved models", "facecolor": COLORS["light_orange"]},
        {"x": 0.12, "y": 0.12, "w": 0.22, "h": 0.14, "text": f"prompt_eval/\n{count_child_dirs(results_dir / 'prompt_eval')} runs", "facecolor": COLORS["light_red"]},
        {"x": 0.66, "y": 0.12, "w": 0.22, "h": 0.14, "text": f"prompt_eval_manual/\n{count_child_dirs(results_dir / 'prompt_eval_manual')} manual sets", "facecolor": COLORS["light_teal"]},
    ]
    arrows = [{"start": (0.27, 0.74), "end": (0.38, 0.48)}, {"start": (0.50, 0.74), "end": (0.50, 0.56)}, {"start": (0.73, 0.74), "end": (0.62, 0.48)}, {"start": (0.34, 0.26), "end": (0.42, 0.40)}, {"start": (0.66, 0.26), "end": (0.58, 0.40)}]
    render_box_flow("5.8", boxes, arrows, output_dir, manifest_rows)


def figure_5_9(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    frame = pd.DataFrame([[3, 1, 2, 1], [2, 3, 3, 2], [1, 2, 3, 3]], index=["Supervised", "Reasoning", "Interpretive"], columns=["Annotation bias", "Prompt sensitivity", "Scoring fragility", "Overclaim risk"])
    matrix_figure("5.9", frame, output_dir, manifest_rows, cmap="YlOrRd", fmt=".0f")


def figure_6_1(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow("6.1", [{"x": 0.04, "y": 0.18, "w": 0.20, "h": 0.58, "text": "Supervised comparison\n\nETHICS\nNormBank\nMFRC", "facecolor": COLORS["light_blue"]}, {"x": 0.28, "y": 0.18, "w": 0.20, "h": 0.58, "text": "Reasoning layer\n\nMoralBench\nMoReBench Public\nMoReBench Theory", "facecolor": COLORS["light_orange"]}, {"x": 0.52, "y": 0.18, "w": 0.20, "h": 0.58, "text": "Interpretive layer\n\nself-model\ncalibration\nidentity\nintegration\nagency", "facecolor": COLORS["light_red"]}, {"x": 0.76, "y": 0.18, "w": 0.20, "h": 0.58, "text": "Support resource\n\nMFD2 lexicon", "facecolor": COLORS["light_teal"]}], [], output_dir, manifest_rows)


def figure_6_2(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    ethics_summary = ctx["ethics_summary"].copy()
    by_category = ethics_summary.groupby("category", as_index=False)[["labeled_rows", "eval_only_rows"]].sum()
    by_category["total_rows"] = by_category["labeled_rows"] + by_category["eval_only_rows"]
    by_category = by_category.sort_values("total_rows", ascending=False)
    melted = by_category.melt(id_vars="category", value_vars=["labeled_rows", "eval_only_rows"], var_name="kind", value_name="rows")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=melted,
        x="rows",
        y="category",
        hue="kind",
        hue_order=["labeled_rows", "eval_only_rows"],
        palette={"labeled_rows": COLORS["green"], "eval_only_rows": COLORS["orange"]},
        ax=ax,
    )
    add_title(ax, FIGURE_CAPTIONS["6.2"])
    ax.set_xlabel("Rows")
    ax.set_ylabel("")
    ax.text(
        0.99,
        1.02,
        f"Total labeled: {int(by_category['labeled_rows'].sum()):,} | Total eval-only: {int(by_category['eval_only_rows'].sum()):,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color="#334E68",
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("")
        legend.texts[0].set_text("Labeled rows")
        legend.texts[1].set_text("Eval-only rows")
        legend.set_bbox_to_anchor((1.0, -0.10))
        legend._loc = 8
    annotate_bar_containers(ax, "int")
    plt.tight_layout()
    save_figure(fig, "6.2", output_dir, manifest_rows)


def figure_6_3(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    label_counts = parse_mapping(ctx["normbank_summary"].iloc[0]["labels"])
    split_counts = ctx["normbank_readable"]["split"].value_counts().reset_index()
    split_counts.columns = ["split", "rows"]
    richness = pd.DataFrame(
        {
            "field": ["Unique settings", "Unique behaviors"],
            "count": [int(ctx["normbank_summary"].iloc[0]["unique_settings"]), int(ctx["normbank_summary"].iloc[0]["unique_behaviors"])],
        }
    )
    combined = pd.concat(
        [
            pd.DataFrame(
                {
                    "aspect": "Label balance",
                    "item": [f"Label | {label}" for label in label_counts.keys()],
                    "count": list(label_counts.values()),
                }
            ),
            pd.DataFrame(
                {
                    "aspect": "Split structure",
                    "item": [f"Split | {split}" for split in split_counts["split"]],
                    "count": split_counts["rows"].tolist(),
                }
            ),
            pd.DataFrame(
                {
                    "aspect": "Contextual fields",
                    "item": [f"Context | {field}" for field in richness["field"]],
                    "count": richness["count"].tolist(),
                }
            ),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=combined,
        x="count",
        y="item",
        hue="aspect",
        dodge=False,
        palette={
            "Label balance": COLORS["blue"],
            "Split structure": COLORS["teal"],
            "Contextual fields": COLORS["orange"],
        },
        ax=ax,
    )
    add_title(ax, FIGURE_CAPTIONS["6.3"])
    ax.set_xlabel("Count (log scale)")
    ax.set_ylabel("")
    ax.set_xscale("log")
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("")
        legend.remove()
    annotate_bar_containers(ax, "int")
    plt.tight_layout()
    save_figure(fig, "6.3", output_dir, manifest_rows)


def figure_6_4(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    summary = ctx["mfrc_summary"].iloc[0]
    boxes = [
        {"x": 0.06, "y": 0.40, "w": 0.18, "h": 0.20, "text": f"Annotator-level rows\n{int(summary['annotation_rows']):,}", "facecolor": COLORS["light_blue"]},
        {"x": 0.32, "y": 0.40, "w": 0.18, "h": 0.20, "text": f"Unique texts\n{int(summary['unique_texts']):,}", "facecolor": COLORS["light_green"]},
        {"x": 0.58, "y": 0.56, "w": 0.18, "h": 0.16, "text": f"Majority-label rows\n{int(summary['majority_label_rows']):,}", "facecolor": COLORS["light_orange"]},
        {"x": 0.58, "y": 0.24, "w": 0.18, "h": 0.16, "text": f"Tie or no-majority rows\n{int(summary['tie_or_no_majority_rows']):,}", "facecolor": COLORS["light_red"]},
        {"x": 0.82, "y": 0.40, "w": 0.14, "h": 0.20, "text": f"Multilabel sidecar\n{int(summary['multilabel_rows']):,}", "facecolor": COLORS["light_teal"]},
    ]
    arrows = [{"start": (0.24, 0.50), "end": (0.32, 0.50)}, {"start": (0.50, 0.50), "end": (0.58, 0.64), "text": "aggregate"}, {"start": (0.50, 0.50), "end": (0.58, 0.32), "text": "flag ambiguity"}, {"start": (0.76, 0.50), "end": (0.82, 0.50), "text": "preserve labels"}]
    render_box_flow("6.4", boxes, arrows, output_dir, manifest_rows)


def figure_6_5(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    collection_counts = ctx["moralbench_items"]["collection"].value_counts().reset_index()
    collection_counts.columns = ["collection", "count"]
    format_counts = ctx["moralbench_items"]["prompt_format"].value_counts().reset_index()
    format_counts.columns = ["prompt_format", "count"]
    collection_text = wrap(
        " | ".join(f"{row.collection}: {int(row.count)}" for row in collection_counts.itertuples(index=False)),
        24,
    )
    format_text = wrap(
        " | ".join(
            f"{row.prompt_format.replace('_', ' ')}: {int(row.count)}" for row in format_counts.itertuples(index=False)
        ),
        24,
    )
    boxes = [
        {"x": 0.03, "y": 0.36, "w": 0.16, "h": 0.22, "text": "Raw MoralBench question files\n88 prompt sources", "facecolor": COLORS["light_blue"]},
        {"x": 0.24, "y": 0.36, "w": 0.18, "h": 0.22, "text": "Structured prompt-item table\n88 eval-only items\nstable item_id", "facecolor": COLORS["light_green"]},
        {"x": 0.47, "y": 0.58, "w": 0.22, "h": 0.22, "text": f"Collections\n{collection_text}", "facecolor": COLORS["light_orange"]},
        {"x": 0.47, "y": 0.14, "w": 0.22, "h": 0.22, "text": f"Prompt format\n{format_text}", "facecolor": COLORS["light_gold"]},
        {"x": 0.74, "y": 0.36, "w": 0.22, "h": 0.22, "text": "Benchmark role\nPrompt-based reasoning evaluation\nfor Research Question 2", "facecolor": COLORS["light_red"]},
    ]
    arrows = [
        {"start": (0.19, 0.47), "end": (0.24, 0.47), "text": "normalize"},
        {"start": (0.42, 0.54), "end": (0.47, 0.66), "text": "group by collection"},
        {"start": (0.42, 0.40), "end": (0.47, 0.26), "text": "group by format"},
        {"start": (0.69, 0.47), "end": (0.74, 0.47), "text": "use in prompt evaluation"},
    ]
    render_box_flow("6.5", boxes, arrows, output_dir, manifest_rows, figsize=(13, 7))


def figure_6_6(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    public = ctx["morebench_public"].copy()
    dilemma_counts = public["dilemma_type"].value_counts()
    role_counts = public["role_domain"].value_counts()
    rubric_total = int(public["rubric_item_count"].fillna(0).sum())
    boxes = [
        {"x": 0.03, "y": 0.36, "w": 0.16, "h": 0.22, "text": "Raw MoReBench Public dilemmas\nrich scenario + rubric source", "facecolor": COLORS["light_blue"]},
        {"x": 0.24, "y": 0.36, "w": 0.18, "h": 0.22, "text": f"Structured benchmark table\n{len(public):,} eval-only items", "facecolor": COLORS["light_green"]},
        {"x": 0.47, "y": 0.58, "w": 0.22, "h": 0.22, "text": wrap(f"Dilemma types\nlong_case {int(dilemma_counts.get('long_case', 0))} | short_case {int(dilemma_counts.get('short_case', 0))} | expert_case {int(dilemma_counts.get('expert_case', 0))}", 24), "facecolor": COLORS["light_orange"]},
        {"x": 0.47, "y": 0.14, "w": 0.22, "h": 0.22, "text": wrap(f"Role domains\nai_advisor {int(role_counts.get('ai_advisor', 0))} | ai_agent {int(role_counts.get('ai_agent', 0))}", 24), "facecolor": COLORS["light_gold"]},
        {"x": 0.74, "y": 0.58, "w": 0.22, "h": 0.22, "text": f"Rubric linkage\n{rubric_total:,} rubric rows\nlinked by item_id", "facecolor": COLORS["light_teal"]},
        {"x": 0.74, "y": 0.14, "w": 0.22, "h": 0.22, "text": "Benchmark role\nOpen-ended procedural and\npluralistic moral reasoning", "facecolor": COLORS["light_red"]},
    ]
    arrows = [
        {"start": (0.19, 0.47), "end": (0.24, 0.47), "text": "structure"},
        {"start": (0.42, 0.54), "end": (0.47, 0.66), "text": "summarize items"},
        {"start": (0.42, 0.40), "end": (0.47, 0.26), "text": "tag domains"},
        {"start": (0.69, 0.69), "end": (0.74, 0.69), "text": "expand rubric"},
        {"start": (0.69, 0.26), "end": (0.74, 0.26), "text": "use in evaluation"},
    ]
    render_box_flow("6.6", boxes, arrows, output_dir, manifest_rows, figsize=(13, 7))


def figure_6_7(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    theory = ctx["morebench_theory"].copy()
    theory_counts = theory["theory"].value_counts()
    dilemma_counts = theory["dilemma_type"].value_counts()
    role_counts = theory["role_domain"].value_counts()
    boxes = [
        {"x": 0.03, "y": 0.36, "w": 0.16, "h": 0.22, "text": "Raw MoReBench Theory dilemmas\nframework-specific prompts", "facecolor": COLORS["light_blue"]},
        {"x": 0.24, "y": 0.36, "w": 0.18, "h": 0.22, "text": f"Structured benchmark table\n{len(theory):,} eval-only items", "facecolor": COLORS["light_green"]},
        {"x": 0.47, "y": 0.58, "w": 0.22, "h": 0.24, "text": wrap('Theory families\n' + ' | '.join(f'{name}: {int(count)}' for name, count in theory_counts.items()), 24), "facecolor": COLORS["light_orange"]},
        {"x": 0.47, "y": 0.12, "w": 0.22, "h": 0.24, "text": wrap(f"Prompt structure\nlong_case {int(dilemma_counts.get('long_case', 0))} | expert_case {int(dilemma_counts.get('expert_case', 0))}\nai_agent {int(role_counts.get('ai_agent', 0))} | ai_advisor {int(role_counts.get('ai_advisor', 0))}", 24), "facecolor": COLORS["light_gold"]},
        {"x": 0.74, "y": 0.36, "w": 0.22, "h": 0.22, "text": "Benchmark role\nTheory-sensitive moral\nreasoning evaluation", "facecolor": COLORS["light_red"]},
    ]
    arrows = [
        {"start": (0.19, 0.47), "end": (0.24, 0.47), "text": "structure"},
        {"start": (0.42, 0.55), "end": (0.47, 0.68), "text": "link theories"},
        {"start": (0.42, 0.39), "end": (0.47, 0.24), "text": "preserve prompt form"},
        {"start": (0.69, 0.47), "end": (0.74, 0.47), "text": "use in prompt evaluation"},
    ]
    render_box_flow("6.7", boxes, arrows, output_dir, manifest_rows, figsize=(13, 7))


def figure_6_8(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    sample = ctx["interpretive_benchmark"].iloc[0]
    boxes = [
        {"x": 0.05, "y": 0.62, "w": 0.20, "h": 0.14, "text": f"item_id\n{sample['item_id']}", "facecolor": COLORS["light_blue"]},
        {"x": 0.30, "y": 0.62, "w": 0.20, "h": 0.14, "text": f"metric_id\n{sample['metric_id']}", "facecolor": COLORS["light_green"]},
        {"x": 0.55, "y": 0.62, "w": 0.20, "h": 0.14, "text": f"scenario_group\n{sample['scenario_group']}", "facecolor": COLORS["light_orange"]},
        {"x": 0.80, "y": 0.62, "w": 0.15, "h": 0.14, "text": f"prompt_variant\n{sample['prompt_variant']}", "facecolor": COLORS["light_gold"]},
        {"x": 0.42, "y": 0.24, "w": 0.20, "h": 0.18, "text": f"response_format\n{sample['response_format']}", "facecolor": COLORS["light_red"]},
    ]
    arrows = [{"start": (0.25, 0.69), "end": (0.30, 0.69)}, {"start": (0.50, 0.69), "end": (0.55, 0.69)}, {"start": (0.75, 0.69), "end": (0.80, 0.69)}, {"start": (0.625, 0.62), "end": (0.52, 0.42), "text": "schema fields guide scoring"}]
    render_box_flow("6.8", boxes, arrows, output_dir, manifest_rows)


def figure_6_9(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    theory_map = ctx["rq3_model"][["metric_id", "theory"]].copy()
    merged = ctx["metric_specs"].merge(theory_map, on="metric_id", how="left")
    merged["metric_id"] = merged["metric_id"].str.replace("_", " ").str.title()
    merged["what_it_measures"] = merged["what_it_measures"].map(lambda v: wrap(v, 28))
    merged["recommended_scoring"] = merged["recommended_scoring"].map(lambda v: wrap(v, 30))
    merged["theory"] = merged["theory"].fillna("").str.replace("_", " ").str.title()
    table_figure("6.9", merged.rename(columns={"metric_id": "Proxy family", "what_it_measures": "What it measures", "recommended_scoring": "Scoring rationale", "theory": "Theory source"}), output_dir, manifest_rows, figsize=(15, 5), font_size=12)


def figure_6_10(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    summary = ctx["mfd2_summary"].iloc[0]
    render_box_flow(
        "6.10",
        [
            {"x": 0.39, "y": 0.40, "w": 0.22, "h": 0.18, "text": f"MFD2\n{int(summary['rows']):,} parsed terms", "facecolor": COLORS["light_teal"]},
            {"x": 0.05, "y": 0.68, "w": 0.22, "h": 0.14, "text": "Lexical interpretation", "facecolor": COLORS["light_blue"]},
            {"x": 0.73, "y": 0.68, "w": 0.22, "h": 0.14, "text": "Exploratory analysis", "facecolor": COLORS["light_green"]},
            {"x": 0.05, "y": 0.14, "w": 0.22, "h": 0.14, "text": "Discussion framing", "facecolor": COLORS["light_orange"]},
            {"x": 0.73, "y": 0.14, "w": 0.22, "h": 0.14, "text": "Not used for model ranking", "facecolor": COLORS["light_red"]},
        ],
        [
            {"start": (0.39, 0.55), "end": (0.27, 0.73)},
            {"start": (0.61, 0.55), "end": (0.73, 0.73)},
            {"start": (0.39, 0.43), "end": (0.27, 0.21)},
            {"start": (0.61, 0.43), "end": (0.73, 0.21), "text": "boundary"},
        ],
        output_dir,
        manifest_rows,
    )


def figure_6_11(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    frame = pd.DataFrame([[3, 2, 1, 1], [2, 3, 3, 2], [1, 2, 3, 3], [1, 1, 1, 1]], index=["Supervised", "Reasoning", "Interpretive", "Resource"], columns=["Label ambiguity", "Context dependence", "Prompt variance", "Overclaim risk"])
    matrix_figure("6.11", frame, output_dir, manifest_rows, cmap="YlOrRd", fmt=".0f")


def figure_7_1(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    root = Path(ctx["root"])
    frame = pd.DataFrame({"Directory": ["admin", "Data", "notebooks", "results", "src", "venv"], "Files": [sum(1 for _ in (root / "admin").rglob("*") if _.is_file()), sum(1 for _ in (root / "Data").rglob("*") if _.is_file()), sum(1 for _ in (root / "notebooks").rglob("*") if _.is_file()), sum(1 for _ in (root / "results").rglob("*") if _.is_file()), sum(1 for _ in (root / "src").rglob("*") if _.is_file()), sum(1 for _ in (root / "venv").rglob("*") if _.is_file())]})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=frame, x="Directory", y="Files", hue="Directory", palette="Blues_d", dodge=False, legend=False, ax=ax)
    add_title(ax, FIGURE_CAPTIONS["7.1"])
    ax.set_xlabel("")
    ax.set_ylabel("File count")
    annotate_bar_containers(ax, "int")
    plt.tight_layout()
    save_figure(fig, "7.1", output_dir, manifest_rows)


def figure_7_2(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    base = Path(ctx["root"]) / "src" / "ai_ethics"
    frame = pd.DataFrame({"Package": ["analysis", "data", "evaluation", "preprocess", "tools", "training"], "Python files": [sum(1 for _ in (base / part).glob("*.py")) for part in ["analysis", "data", "evaluation", "preprocess", "tools", "training"]]})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=frame, x="Python files", y="Package", color=COLORS["teal"], ax=ax)
    add_title(ax, FIGURE_CAPTIONS["7.2"])
    ax.set_xlabel("Module files")
    ax.set_ylabel("")
    annotate_bar_containers(ax, "int")
    plt.tight_layout()
    save_figure(fig, "7.2", output_dir, manifest_rows)


def figure_7_3(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow("7.3", [{"x": 0.05, "y": 0.38, "w": 0.18, "h": 0.22, "text": "Data/raw", "facecolor": COLORS["light_blue"]}, {"x": 0.30, "y": 0.38, "w": 0.18, "h": 0.22, "text": "Preprocess + EDA", "facecolor": COLORS["light_teal"]}, {"x": 0.55, "y": 0.38, "w": 0.18, "h": 0.22, "text": "Data/processed", "facecolor": COLORS["light_green"]}, {"x": 0.80, "y": 0.54, "w": 0.15, "h": 0.12, "text": "benchmark_*", "facecolor": COLORS["light_orange"]}, {"x": 0.80, "y": 0.22, "w": 0.15, "h": 0.12, "text": "Data/eda", "facecolor": COLORS["light_gold"]}], [{"start": (0.23, 0.49), "end": (0.30, 0.49)}, {"start": (0.48, 0.49), "end": (0.55, 0.49)}, {"start": (0.73, 0.49), "end": (0.80, 0.60)}, {"start": (0.48, 0.38), "end": (0.80, 0.28)}], output_dir, manifest_rows)


def figure_7_4(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow("7.4", [{"x": 0.05, "y": 0.36, "w": 0.18, "h": 0.22, "text": "Data layer", "facecolor": COLORS["light_blue"]}, {"x": 0.30, "y": 0.36, "w": 0.18, "h": 0.22, "text": "Training + evaluation layer", "facecolor": COLORS["light_green"]}, {"x": 0.55, "y": 0.36, "w": 0.18, "h": 0.22, "text": "Results artefact layer", "facecolor": COLORS["light_orange"]}, {"x": 0.80, "y": 0.36, "w": 0.15, "h": 0.22, "text": "Report + figures", "facecolor": COLORS["light_red"]}, {"x": 0.38, "y": 0.70, "w": 0.24, "h": 0.12, "text": "Notebooks as inspection layer", "facecolor": COLORS["light_teal"]}], [{"start": (0.23, 0.47), "end": (0.30, 0.47)}, {"start": (0.48, 0.47), "end": (0.55, 0.47)}, {"start": (0.73, 0.47), "end": (0.80, 0.47)}, {"start": (0.50, 0.70), "end": (0.50, 0.58), "text": "reads from / writes to"}], output_dir, manifest_rows)


def figure_7_5(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow("7.5", [{"x": 0.05, "y": 0.38, "w": 0.16, "h": 0.22, "text": "Raw files", "facecolor": COLORS["light_blue"]}, {"x": 0.25, "y": 0.38, "w": 0.16, "h": 0.22, "text": "preprocess_* scripts", "facecolor": COLORS["light_teal"]}, {"x": 0.45, "y": 0.38, "w": 0.16, "h": 0.22, "text": "standardised CSV / JSONL", "facecolor": COLORS["light_green"]}, {"x": 0.65, "y": 0.38, "w": 0.16, "h": 0.22, "text": "data.loader", "facecolor": COLORS["light_gold"]}, {"x": 0.85, "y": 0.38, "w": 0.10, "h": 0.22, "text": "train / eval", "facecolor": COLORS["light_orange"]}], [{"start": (0.21, 0.49), "end": (0.25, 0.49)}, {"start": (0.41, 0.49), "end": (0.45, 0.49)}, {"start": (0.61, 0.49), "end": (0.65, 0.49)}, {"start": (0.81, 0.49), "end": (0.85, 0.49)}], output_dir, manifest_rows)


def figure_7_6(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow("7.6", [{"x": 0.05, "y": 0.36, "w": 0.18, "h": 0.22, "text": "Supervised benchmark file", "facecolor": COLORS["light_blue"]}, {"x": 0.30, "y": 0.36, "w": 0.16, "h": 0.22, "text": "Model choice", "facecolor": COLORS["light_green"]}, {"x": 0.53, "y": 0.36, "w": 0.16, "h": 0.22, "text": "Training run", "facecolor": COLORS["light_orange"]}, {"x": 0.76, "y": 0.36, "w": 0.18, "h": 0.22, "text": "Evaluation + saved outputs", "facecolor": COLORS["light_red"]}], [{"start": (0.23, 0.47), "end": (0.30, 0.47)}, {"start": (0.46, 0.47), "end": (0.53, 0.47)}, {"start": (0.69, 0.47), "end": (0.76, 0.47)}], output_dir, manifest_rows)


def figure_7_7(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow("7.7", [{"x": 0.04, "y": 0.36, "w": 0.14, "h": 0.22, "text": "Prompt dataset", "facecolor": COLORS["light_blue"]}, {"x": 0.22, "y": 0.36, "w": 0.14, "h": 0.22, "text": "Response collection", "facecolor": COLORS["light_teal"]}, {"x": 0.40, "y": 0.36, "w": 0.14, "h": 0.22, "text": "responses.jsonl", "facecolor": COLORS["light_green"]}, {"x": 0.58, "y": 0.36, "w": 0.14, "h": 0.22, "text": "Scoring", "facecolor": COLORS["light_gold"]}, {"x": 0.76, "y": 0.36, "w": 0.18, "h": 0.22, "text": "scenario_scores + model_summary", "facecolor": COLORS["light_orange"]}], [{"start": (0.18, 0.47), "end": (0.22, 0.47)}, {"start": (0.36, 0.47), "end": (0.40, 0.47)}, {"start": (0.54, 0.47), "end": (0.58, 0.47)}, {"start": (0.72, 0.47), "end": (0.76, 0.47)}], output_dir, manifest_rows)


def figure_7_8(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    render_box_flow("7.8", [{"x": 0.08, "y": 0.38, "w": 0.22, "h": 0.22, "text": "Saved data and results", "facecolor": COLORS["light_blue"]}, {"x": 0.39, "y": 0.38, "w": 0.22, "h": 0.22, "text": "notebooks/", "facecolor": COLORS["light_teal"]}, {"x": 0.70, "y": 0.54, "w": 0.18, "h": 0.14, "text": "EDA inspection", "facecolor": COLORS["light_green"]}, {"x": 0.70, "y": 0.22, "w": 0.18, "h": 0.14, "text": "Report figures", "facecolor": COLORS["light_orange"]}], [{"start": (0.30, 0.49), "end": (0.39, 0.49)}, {"start": (0.61, 0.49), "end": (0.70, 0.61)}, {"start": (0.61, 0.49), "end": (0.70, 0.29)}], output_dir, manifest_rows)


def figure_7_9(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    results_dir = Path(ctx["results_dir"])
    render_box_flow("7.9", [{"x": 0.39, "y": 0.42, "w": 0.22, "h": 0.16, "text": "results/", "facecolor": COLORS["light_gold"]}, {"x": 0.06, "y": 0.68, "w": 0.20, "h": 0.14, "text": "metrics.csv", "facecolor": COLORS["light_blue"]}, {"x": 0.39, "y": 0.72, "w": 0.22, "h": 0.12, "text": f"training_logs/\n{count_child_dirs(results_dir / 'training_logs')} dirs", "facecolor": COLORS["light_green"]}, {"x": 0.74, "y": 0.68, "w": 0.20, "h": 0.14, "text": f"models/\n{count_child_dirs(results_dir / 'models')} dirs", "facecolor": COLORS["light_orange"]}, {"x": 0.12, "y": 0.12, "w": 0.22, "h": 0.14, "text": "checkpoints/", "facecolor": COLORS["light_teal"]}, {"x": 0.66, "y": 0.12, "w": 0.22, "h": 0.14, "text": "prompt_eval/ and prompt_eval_manual/", "facecolor": COLORS["light_red"]}], [{"start": (0.26, 0.75), "end": (0.39, 0.50)}, {"start": (0.50, 0.72), "end": (0.50, 0.58)}, {"start": (0.74, 0.75), "end": (0.61, 0.50)}, {"start": (0.34, 0.26), "end": (0.42, 0.42)}, {"start": (0.66, 0.26), "end": (0.58, 0.42)}], output_dir, manifest_rows)


def figure_8_1(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    metrics_latest = ctx["metrics_latest"].copy()
    metrics_latest["model_clean"] = metrics_latest["model"].map(clean_model_name)
    pivot = metrics_latest.assign(value=1).pivot_table(index="model_clean", columns="dataset", values="value", aggfunc="max", fill_value=0).sort_index()
    matrix_figure("8.1", pivot, output_dir, manifest_rows, cmap="Greens", fmt=".0f")


def figure_8_2(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    metrics_latest = ctx["metrics_latest"].copy()
    family_map = {"tfidf_logreg": "Classical", "tfidf_linearsvc": "Classical", "bow_mnb": "Classical", "distilbert-base-uncased": "Transformer", "bert-base-uncased": "Transformer", "roberta-base": "Transformer", "microsoft/deberta-v3-base": "Transformer"}
    frame = metrics_latest.assign(family=lambda df: df["model"].map(family_map).fillna("Other"), model_clean=lambda df: df["model"].map(clean_model_name))
    counts = frame.groupby(["family", "model_clean"]).size().reset_index(name="dataset_runs")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=counts, x="model_clean", y="dataset_runs", hue="family", ax=ax)
    add_title(ax, FIGURE_CAPTIONS["8.2"])
    ax.set_xlabel("")
    ax.set_ylabel("Datasets with completed runs")
    ax.tick_params(axis="x", rotation=30)
    annotate_bar_containers(ax, "int")
    plt.tight_layout()
    save_figure(fig, "8.2", output_dir, manifest_rows)


def figure_8_3(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    runs = ctx["training_runs"].copy()
    runs["phase"] = np.where(runs["train_examples"] <= 500, "smoke", "full")
    summary = runs.groupby(["phase", "model_type"], as_index=False).agg(runs=("experiment_key", "count"), median_train_examples=("train_examples", "median"))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="phase", y="runs", hue="model_type", ax=ax)
    add_title(ax, FIGURE_CAPTIONS["8.3"])
    ax.set_xlabel("")
    ax.set_ylabel("Completed runs")
    annotate_bar_containers(ax, "int")
    plt.tight_layout()
    save_figure(fig, "8.3", output_dir, manifest_rows)


def figure_8_4(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    frame = pd.DataFrame([["Accuracy", "Overall correctness", "easy to read"], ["Macro F1", "Class balance robustness", "main ranking metric"], ["Balanced accuracy", "Per-class sensitivity", "supports F1"], ["MCC", "Correlation-style quality", "handles imbalance"], ["AUROC / PR AUC", "ranking quality", "where probabilities exist"], ["Brier score", "probability quality", "lower is better"], ["ECE", "confidence calibration", "lower is better"]], columns=["Metric", "What it tells you", "Role in thesis"])
    table_figure("8.4", frame, output_dir, manifest_rows, figsize=(13, 4.5), font_size=12)


def figure_8_5(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    metrics_latest = ctx["metrics_latest"].copy()
    metrics_latest["model_clean"] = metrics_latest["model"].map(clean_model_name)
    pivot = metrics_latest.pivot(index="model_clean", columns="dataset", values="macro_f1").sort_index()
    matrix_figure("8.5", pivot, output_dir, manifest_rows, cmap="YlGnBu", fmt=".3f", figsize=(9, 5), annot_kws={"fontsize": 9})


def dataset_bar_figure(figure_id: str, dataset: str, ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    subset = ctx["metrics_latest"].loc[ctx["metrics_latest"]["dataset"] == dataset].copy()
    subset["model_clean"] = subset["model"].map(clean_model_name)
    subset = subset.sort_values("macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=subset, x="model_clean", y="macro_f1", color=COLORS["blue"], ax=ax)
    add_title(ax, FIGURE_CAPTIONS[figure_id])
    ax.set_xlabel("")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    annotate_bar_containers(ax, "float3")
    plt.tight_layout()
    save_figure(fig, figure_id, output_dir, manifest_rows)


def figure_8_9(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    examples = ctx["qualitative_examples"].copy()
    examples = examples.sort_values(["dataset", "confidence"], ascending=[True, False]).groupby("dataset", as_index=False).head(2)
    examples["model"] = examples["model"].map(clean_model_name)
    examples["confidence"] = examples["confidence"].map(lambda v: f"{v:.3f}")
    examples["text"] = examples["text"].map(lambda v: wrap(v[:110] + ("..." if len(v) > 110 else ""), 28))
    frame = examples.rename(columns={"dataset": "Dataset", "model": "Model", "true": "True", "pred": "Pred", "confidence": "Conf.", "text": "Example text"})[["Dataset", "Model", "True", "Pred", "Conf.", "Example text"]]
    table_figure("8.9", frame, output_dir, manifest_rows, figsize=(15, 5.5), font_size=12)


def figure_8_10(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=ctx["metrics_latest"], x="ece_10bin", y="macro_f1", hue="dataset", style="model", s=150, ax=ax)
    add_title(ax, FIGURE_CAPTIONS["8.10"])
    ax.set_xlabel("ECE (lower is better)")
    ax.set_ylabel("Macro F1")
    plot_frame = ctx["metrics_latest"].copy()
    plot_frame["model_label"] = plot_frame["model"].map(clean_model_name)
    annotate_scatter_labels(ax, plot_frame, "ece_10bin", "macro_f1", "model_label")
    plt.tight_layout()
    save_figure(fig, "8.10", output_dir, manifest_rows)


def figure_9_1(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    run_dir = Path(ctx["results_dir"]) / "prompt_eval_manual" / "moralbench_88"
    frame = pd.DataFrame([{"artifact": path.name, "size_kb": path.stat().st_size / 1024.0} for path in sorted(run_dir.iterdir()) if path.is_file()])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=frame, x="size_kb", y="artifact", color=COLORS["orange"], ax=ax)
    add_title(ax, FIGURE_CAPTIONS["9.1"])
    ax.set_xlabel("Size (KB)")
    ax.set_ylabel("")
    annotate_bar_containers(ax, "float1")
    plt.tight_layout()
    save_figure(fig, "9.1", output_dir, manifest_rows)


def figure_9_2(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    plot_frame = ctx["rq2_model"].melt(id_vars=["dataset", "model", "metric_id"], value_vars=["primary_score", "format_compliance"], var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_frame, x="metric", y="score", color=COLORS["blue"], ax=ax)
    add_title(ax, FIGURE_CAPTIONS["9.2"])
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    annotate_bar_containers(ax, "float3")
    plt.tight_layout()
    save_figure(fig, "9.2", output_dir, manifest_rows)


def figure_9_3(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    rq3_model = ctx["rq3_model"].copy()
    rq3_model["metric_id"] = rq3_model["metric_id"].str.replace("_", " ").str.title()
    pivot = rq3_model.pivot(index="model", columns="metric_id", values="primary_score")
    matrix_figure("9.3", pivot, output_dir, manifest_rows, cmap="YlGnBu", fmt=".3f", figsize=(10, 4.5), annot_kws={"fontsize": 10})


def figure_9_4(ctx: dict[str, object], output_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=ctx["rq3_scenarios"], x="scenario_group", y="avg_primary_score", hue="metric_id", ax=ax)
    add_title(ax, FIGURE_CAPTIONS["9.4"])
    ax.set_xlabel("")
    ax.set_ylabel("Average primary score")
    ax.tick_params(axis="x", rotation=25)
    annotate_bar_containers(ax, "float3")
    plt.tight_layout()
    save_figure(fig, "9.4", output_dir, manifest_rows)


def generate_all(ctx: dict[str, object], output_dir: Path) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    for fn in [figure_3_1, figure_3_2, figure_3_3, figure_3_4, figure_3_5, figure_4_1, figure_4_2, figure_4_3, figure_4_4, figure_5_1, figure_5_2, figure_5_3, figure_5_3a, figure_5_4, figure_5_5, figure_5_6, figure_5_7, figure_5_8, figure_5_9, figure_6_1, figure_6_2, figure_6_3, figure_6_4, figure_6_5, figure_6_6, figure_6_7, figure_6_8, figure_6_9, figure_6_10, figure_6_11, figure_7_1, figure_7_2, figure_7_3, figure_7_4, figure_7_5, figure_7_6, figure_7_7, figure_7_8, figure_7_9, figure_8_1, figure_8_2, figure_8_3, figure_8_4, figure_8_5]:
        fn(ctx, output_dir, manifest_rows)
    dataset_bar_figure("8.6", "ethics", ctx, output_dir, manifest_rows)
    dataset_bar_figure("8.7", "normbank", ctx, output_dir, manifest_rows)
    dataset_bar_figure("8.8", "mfrc", ctx, output_dir, manifest_rows)
    for fn in [figure_8_9, figure_8_10, figure_9_1, figure_9_2, figure_9_3, figure_9_4]:
        fn(ctx, output_dir, manifest_rows)
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis figures from saved artefacts.")
    parser.add_argument("--output-dir", default=str(repo_root() / "admin" / "Report" / "figures"), help="Directory to write figure PNGs and manifest.")
    return parser.parse_args()


def main() -> None:
    configure_style()
    args = parse_args()
    root = repo_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context(root)
    manifest_rows = generate_all(ctx, output_dir)
    pd.DataFrame(manifest_rows).sort_values("figure_id").to_csv(output_dir / "figure_manifest.csv", index=False)
    print(f"Generated {len(manifest_rows)} figures in {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot realtime inference latency results.

Input is a result directory produced by benchmarks/realtime_inference_latency.py.
The script writes:
  - latency_percentiles.png: avg/p50/p95/p99 bar chart.
  - latency_cdf.png: empirical CDF of foreground request latency.
  - p99_slowdown.png: p99 slowdown versus the alone baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-xsched")

import matplotlib.pyplot as plt


SCENARIO_LABELS = {
    "alone": "Exclusive",
    "native": "Native",
    "xsched": "XSched LV1",
    "xsched_lv2": "XSched LV2",
}

SCENARIO_COLORS = {
    "alone": "#4c78a8",
    "native": "#f58518",
    "xsched": "#54a24b",
    "xsched_lv2": "#b279a2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--title", default="ResNet50 foreground latency")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_comparison(result_dir: Path) -> list[dict[str, object]]:
    path = result_dir / "comparison.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    data = json.loads(path.read_text())
    order = {"alone": 0, "native": 1, "xsched": 2, "xsched_lv2": 3}
    return sorted(data, key=lambda row: order.get(str(row["scenario"]), 99))


def load_latencies(result_dir: Path, scenario: str) -> list[float]:
    path = result_dir / scenario / "latency.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [float(row["latency_ms"]) for row in csv.DictReader(f)]


def save_percentile_plot(rows: list[dict[str, object]], out: Path, title: str) -> None:
    metrics = [
        ("latency_avg_ms", "Avg"),
        ("latency_p50_ms", "P50"),
        ("latency_p95_ms", "P95"),
        ("latency_p99_ms", "P99"),
    ]
    scenarios = [str(row["scenario"]) for row in rows]
    labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]
    x = list(range(len(labels)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    for idx, (key, metric_label) in enumerate(metrics):
        offset = (idx - 1.5) * width
        vals = [float(row[key]) for row in rows]
        bars = ax.bar(
            [v + offset for v in x],
            vals,
            width,
            label=metric_label,
            color=["#9ecae9", "#6baed6", "#3182bd", "#08519c"][idx],
        )
        if key == "latency_p99_ms":
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title(title)
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_cdf_plot(result_dir: Path, rows: list[dict[str, object]], out: Path,
                  title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    for row in rows:
        scenario = str(row["scenario"])
        vals = sorted(load_latencies(result_dir, scenario))
        if not vals:
            continue
        y = [(i + 1) / len(vals) for i in range(len(vals))]
        ax.plot(
            vals,
            y,
            label=SCENARIO_LABELS.get(scenario, scenario),
            color=SCENARIO_COLORS.get(scenario),
            linewidth=2,
        )

    ax.set_title(f"{title} CDF")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_slowdown_plot(rows: list[dict[str, object]], out: Path, title: str) -> None:
    scenarios = [str(row["scenario"]) for row in rows]
    labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]
    vals = [float(row.get("p99_slowdown_vs_alone", 0.0)) for row in rows]
    colors = [SCENARIO_COLORS.get(s, "#777777") for s in scenarios]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    bars = ax.bar(labels, vals, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"{title} P99 slowdown")
    ax.set_ylabel("Slowdown vs Exclusive")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_throughput_plot(rows: list[dict[str, object]], out: Path, title: str) -> None:
    scenarios = [str(row["scenario"]) for row in rows]
    labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]

    # collect all throughput keys
    thpt_keys = sorted({
        key for row in rows for key in row
        if key.startswith("background_throughput_iters_per_s_")
    })

    # sum throughput across all background workers per scenario
    vals = []
    for row in rows:
        total = sum(float(row.get(k, 0.0)) for k in thpt_keys)
        vals.append(total)

    colors = [SCENARIO_COLORS.get(s, "#777777") for s in scenarios]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    bars = ax.bar(labels, vals, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}" if val else "—",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"{title} background throughput")
    ax.set_ylabel("Throughput (iters/s)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_comparison(result_dir)
    save_percentile_plot(rows, out_dir / "latency_percentiles.png", args.title)
    save_cdf_plot(result_dir, rows, out_dir / "latency_cdf.png", args.title)
    save_slowdown_plot(rows, out_dir / "p99_slowdown.png", args.title)
    save_throughput_plot(rows, out_dir / "throughput.png", args.title)

    print(out_dir / "latency_percentiles.png")
    print(out_dir / "latency_cdf.png")
    print(out_dir / "p99_slowdown.png")
    print(out_dir / "throughput.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

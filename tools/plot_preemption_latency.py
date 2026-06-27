#!/usr/bin/env python3
"""Plot mechanism-level XSched preemption latency benchmark results.

Input is a result directory produced by benchmarks/preemption_latency_scaling.py.
The script writes a single overview figure:
  - preemption_overview.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-xsched")

import matplotlib.pyplot as plt


LEVEL_LABELS = {
    1: "XSched LV1",
    2: "XSched LV2",
}

LEVEL_COLORS = {
    1: "#4c78a8",
    2: "#f58518",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_rows(result_dir: Path) -> list[dict[str, object]]:
    path = result_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    data = json.loads(path.read_text())
    rows = list(data.get("rows", []))
    if not rows:
        raise ValueError(f"no rows in {path}")
    rows.sort(key=lambda row: (int(row["kernel_us"]), int(row["level"])))
    return rows


def group_by_level(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        level = int(row["level"])
        grouped.setdefault(level, []).append(row)
    for level_rows in grouped.values():
        level_rows.sort(key=lambda row: int(row["kernel_us"]))
    return grouped


def save_overview_plot(rows: list[dict[str, object]], out: Path, title: str) -> None:
    kernel_us = sorted({int(row["kernel_us"]) for row in rows})
    metrics = [
        ("preempt_avg_us", "Avg"),
        ("preempt_p50_us", "P50"),
        ("preempt_p95_us", "P95"),
        ("preempt_p99_us", "P99"),
    ]
    grouped = group_by_level(rows)

    fig, axes = plt.subplots(1, len(kernel_us), figsize=(4.5 * len(kernel_us), 5.4), dpi=180)
    if len(kernel_us) == 1:
        axes = [axes]

    for ax, kernel in zip(axes, kernel_us):
        level_rows = {int(row["level"]): row for row in rows if int(row["kernel_us"]) == kernel}
        x = list(range(len(metrics)))
        width = 0.34
        for idx, level in enumerate(sorted(grouped)):
            row = level_rows.get(level)
            if not row:
                continue
            vals = [float(row[key]) for key, _ in metrics]
            offset = (-0.5 + idx) * width
            bars = ax.bar(
                [v + offset for v in x],
                vals,
                width,
                label=LEVEL_LABELS.get(level, f"Level {level}"),
                color=LEVEL_COLORS.get(level),
                alpha=0.92,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

        ax.set_title(f"Kernel {kernel} us")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics])
        ax.grid(axis="y", alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("Latency (us)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(result_dir)
    save_overview_plot(rows, out_dir / "preemption_overview.png", args.title)

    print(out_dir / "preemption_overview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

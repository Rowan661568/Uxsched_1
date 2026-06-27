#!/usr/bin/env python3
import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def split_sort_key(value):
    return -1 if value == "Unsplit" else int(value)


def load_data(result_dir):
    required = ["summary.csv", "normalized_metrics.csv", "tradeoff.csv", "repeat_metrics.csv"]
    missing = [name for name in required if not (result_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required CSV files: {', '.join(missing)}")
    summary = read_csv(result_dir / "summary.csv")
    normalized = read_csv(result_dir / "normalized_metrics.csv")
    tradeoff = read_csv(result_dir / "tradeoff.csv")
    repeat = read_csv(result_dir / "repeat_metrics.csv")
    return summary, normalized, tradeoff, repeat


def save_all(fig, out_base, formats, dpi):
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi if fmt == "png" else None, bbox_inches="tight")


def style_ax(ax):
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)


def labels_for(summary):
    rows = sorted(summary, key=lambda r: split_sort_key(r["split_blocks"]))
    labels = [r["split_blocks"] for r in rows]
    display = ["Unsplit" if x == "Unsplit" else str(x) for x in labels]
    return rows, display


def bar_plot(summary, metric_mean, metric_std, ylabel, title, out_base, formats, dpi, annotate_suffix="", highlight_52=True):
    import matplotlib.pyplot as plt
    rows, display = labels_for(summary)
    values = [fnum(r[metric_mean]) for r in rows]
    errs = [fnum(r[metric_std]) for r in rows]
    colors = ["#4c78a8" if r["split_blocks"] == "Unsplit" else "#f58518" for r in rows]
    if highlight_52:
        colors = ["#d62728" if r["split_blocks"] == "52" else c for r, c in zip(rows, colors)]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(display, values, yerr=errs, capsize=4, color=colors, edgecolor="#222222", linewidth=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}{annotate_suffix}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    style_ax(ax)
    fig.text(0.5, 0.01, "Error bars show cross-repeat standard deviation. Split=52 is highlighted as the formula-derived candidate.", ha="center", fontsize=8)
    save_all(fig, out_base, formats, dpi)
    plt.close(fig)


def plot_tradeoff(tradeoff, out_base, formats, dpi):
    import matplotlib.pyplot as plt
    rows = sorted(tradeoff, key=lambda r: int(r["split_blocks"]))
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter([100.0], [0.0], color="#4c78a8", s=65, label="Unsplit baseline")
    ax.text(100.0, 0.0, " Unsplit", va="center", fontsize=9)
    for row in rows:
        split = row["split_blocks"]
        x = fnum(row["lp_throughput_retention_pct"])
        y = fnum(row["hp_p99_reduction_pct"])
        color = "#d62728" if split == "52" else "#f58518"
        size = 90 if split == "52" else 60
        ax.scatter([x], [y], color=color, s=size)
        ax.text(x, y, f" {split}", va="center", fontsize=9, weight="bold" if split == "52" else "normal")
    ax.annotate("better", xy=(0.92, 0.9), xycoords="axes fraction", ha="right", fontsize=9)
    ax.annotate("", xy=(0.88, 0.86), xytext=(0.70, 0.68), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.set_xlabel("LP throughput retention vs Unsplit (%)")
    ax.set_ylabel("HP P99 reduction vs Unsplit (%)")
    ax.set_title("Latency/Throughput Trade-off by Split Size")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    style_ax(ax)
    fig.text(0.5, 0.01, "The trade-off uses paired repeat ratios. 52 is not claimed as a global optimum.", ha="center", fontsize=8)
    save_all(fig, out_base, formats, dpi)
    plt.close(fig)


def plot_normalized(normalized, out_base, formats, dpi):
    import matplotlib.pyplot as plt
    rows = sorted(normalized, key=lambda r: int(r["split_blocks"]))
    labels = [r["split_blocks"] for r in rows]
    p99 = [fnum(r["hp_p99_ratio_relative_to_unsplit"]) for r in rows]
    lp = [fnum(r["lp_throughput_ratio_relative_to_unsplit"]) for r in rows]
    x = list(range(len(rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar([i - width / 2 for i in x], p99, width, label="HP P99 ratio", color="#4c78a8")
    ax.bar([i + width / 2 for i in x], lp, width, label="LP throughput ratio", color="#f58518")
    ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Ratio relative to paired Unsplit baseline")
    ax.set_title("Normalized Split-size Metrics")
    ax.legend()
    style_ax(ax)
    save_all(fig, out_base, formats, dpi)
    plt.close(fig)


def plot_repeat(repeat_rows, out_base, formats, dpi):
    import matplotlib.pyplot as plt
    by_split = {}
    native = {}
    for row in repeat_rows:
        split = str(row["split_blocks"])
        rep = int(row["repeat"])
        by_split.setdefault(split, []).append((rep, fnum(row["hb_hp_p99_us"])))
        native.setdefault(rep, []).append(fnum(row["native_hp_p99_us"]))
    repeats = sorted(native)
    native_vals = [sum(native[r]) / len(native[r]) for r in repeats]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(repeats, native_vals, marker="o", linewidth=2.0, label="Unsplit baseline", color="#4c78a8")
    for split in sorted(by_split, key=lambda x: int(x)):
        pairs = sorted(by_split[split])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=f"HB_FIXED-{split}",
                color="#d62728" if split == "52" else None)
    ax.set_xlabel("Repeat")
    ax.set_ylabel("HP P99 latency (us)")
    ax.set_title("HP P99 by Repeat and Split Size")
    ax.set_ylim(bottom=0)
    ax.legend()
    style_ax(ax)
    save_all(fig, out_base, formats, dpi)
    plt.close(fig)


def run(result_dir, output_dir, formats, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, normalized, tradeoff, repeat = load_data(result_dir)
    bar_plot(summary, "hp_p99_us_mean", "hp_p99_us_stddev", "HP P99 latency (us)",
             "HP P99 Latency by Split Size", output_dir / "split_size_hp_p99", formats, dpi)
    bar_plot(summary, "lp_throughput_rps_mean", "lp_throughput_rps_stddev", "LP throughput (requests/s)",
             "LP Throughput by Split Size", output_dir / "split_size_lp_throughput", formats, dpi)
    plot_tradeoff(tradeoff, output_dir / "split_size_tradeoff", formats, dpi)
    plot_normalized(normalized, output_dir / "split_size_normalized_metrics", formats, dpi)
    plot_repeat(repeat, output_dir / "split_size_hp_p99_by_repeat", formats, dpi)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def self_test():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        fields = [
            "split_blocks", "kind", "repeat_count", "hp_mean_us_mean", "hp_mean_us_stddev",
            "hp_p95_us_mean", "hp_p95_us_stddev", "hp_p99_us_mean", "hp_p99_us_stddev",
            "lp_throughput_rps_mean", "lp_throughput_rps_stddev", "hp_p99_reduction_vs_unsplit",
            "lp_throughput_retention_vs_unsplit", "lp_throughput_loss_vs_unsplit", "parent_count",
            "child_count", "children_per_parent", "correctness_pass", "global_scheduler_pass", "hb_gate_pass",
        ]
        write_csv(base / "summary.csv", [
            {"split_blocks": "Unsplit", "kind": "unsplit_baseline", "repeat_count": 10, "hp_p99_us_mean": 5000, "hp_p99_us_stddev": 100, "lp_throughput_rps_mean": 500, "lp_throughput_rps_stddev": 10},
            {"split_blocks": "32", "kind": "hb_fixed", "repeat_count": 5, "hp_p99_us_mean": 3000, "hp_p99_us_stddev": 80, "lp_throughput_rps_mean": 220, "lp_throughput_rps_stddev": 5},
            {"split_blocks": "52", "kind": "hb_fixed", "repeat_count": 5, "hp_p99_us_mean": 2500, "hp_p99_us_stddev": 50, "lp_throughput_rps_mean": 300, "lp_throughput_rps_stddev": 6},
        ], fields)
        write_csv(base / "normalized_metrics.csv", [
            {"split_blocks": "32", "hp_p99_ratio_relative_to_unsplit": 0.6, "lp_throughput_ratio_relative_to_unsplit": 0.44},
            {"split_blocks": "52", "hp_p99_ratio_relative_to_unsplit": 0.5, "lp_throughput_ratio_relative_to_unsplit": 0.60},
        ], ["split_blocks", "hp_p99_ratio_relative_to_unsplit", "lp_throughput_ratio_relative_to_unsplit"])
        write_csv(base / "tradeoff.csv", [
            {"split_blocks": "32", "hp_p99_reduction_pct": 40, "lp_throughput_retention_pct": 44},
            {"split_blocks": "52", "hp_p99_reduction_pct": 50, "lp_throughput_retention_pct": 60},
        ], ["split_blocks", "hp_p99_reduction_pct", "lp_throughput_retention_pct"])
        write_csv(base / "repeat_metrics.csv", [
            {"split_blocks": "32", "repeat": 0, "native_hp_p99_us": 5000, "hb_hp_p99_us": 3000},
            {"split_blocks": "32", "repeat": 1, "native_hp_p99_us": 5100, "hb_hp_p99_us": 3100},
            {"split_blocks": "52", "repeat": 0, "native_hp_p99_us": 5000, "hb_hp_p99_us": 2500},
            {"split_blocks": "52", "repeat": 1, "native_hp_p99_us": 5100, "hb_hp_p99_us": 2600},
        ], ["split_blocks", "repeat", "native_hp_p99_us", "hb_hp_p99_us"])
        run(base, base / "figures", ["png", "pdf", "svg"], 120)
        for stem in [
            "split_size_hp_p99", "split_size_lp_throughput", "split_size_tradeoff",
            "split_size_normalized_metrics", "split_size_hp_p99_by_repeat",
        ]:
            assert (base / "figures" / f"{stem}.png").stat().st_size > 0
            assert (base / "figures" / f"{stem}.pdf").stat().st_size > 0
            assert (base / "figures" / f"{stem}.svg").stat().st_size > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.result_dir:
        parser.error("--result-dir is required unless --self-test is used")
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    out = args.output_dir or (args.result_dir / "figures")
    run(args.result_dir, out, formats, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

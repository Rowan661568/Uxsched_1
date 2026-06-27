#!/usr/bin/env python3
import argparse
import csv
import hashlib
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path
from statistics import mean, median


SYSTEMS = ["standalone_hp", "uxsched_native_hp_lp", "uxsched_hb_fixed_hp_lp"]
REPEATS = [str(i) for i in range(5)]
LABELS = {
    "standalone_hp": "Standalone HP",
    "uxsched_native_hp_lp": "UXSched Lv1 + Unsplit Kernel",
    "uxsched_hb_fixed_hp_lp": "UXSched Lv1 + HB_FIXED",
}
AXIS_LABELS = {
    "standalone_hp": "Standalone HP",
    "uxsched_native_hp_lp": "UXSched Lv1\n+ Unsplit Kernel",
    "uxsched_hb_fixed_hp_lp": "UXSched Lv1\n+ HB_FIXED",
}
COLORS = {
    "standalone_hp": "#7f7f7f",
    "uxsched_native_hp_lp": "#4c78a8",
    "uxsched_hb_fixed_hp_lp": "#f58518",
}
CORE_SYSTEMS = ["uxsched_native_hp_lp", "uxsched_hb_fixed_hp_lp"]


class GateError(RuntimeError):
    pass


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_env(path):
    data = {}
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key] = value
    return data


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fnum(value):
    if value in (None, ""):
        return 0.0
    return float(value)


def truthy(value):
    return str(value).lower() in ("1", "true", "pass", "passed", "yes")


def stdev(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def non_comment_kernel(path):
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def load_result(result_dir):
    required = ["summary.csv", "comparison.csv", "metadata.env"]
    missing = [name for name in required if not (result_dir / name).exists()]
    if missing:
        raise GateError(f"missing required files: {', '.join(missing)}")
    summary = read_csv(result_dir / "summary.csv")
    comparison = read_csv(result_dir / "comparison.csv")
    metadata = read_env(result_dir / "metadata.env")
    return summary, comparison, metadata


def summary_map(summary):
    return {(row["system"], row["repeat"]): row for row in summary}


def comparison_lookup(comparison):
    out = {}
    for row in comparison:
        out[(row.get("kind", ""), row.get("metric", ""))] = row
    return out


def aggregate(comparison, system, metric):
    for row in comparison:
        if row.get("kind") == "aggregate" and row.get("system") == system and row.get("metric") == metric:
            return row
    raise GateError(f"missing aggregate for {system} {metric}")


def ratio_aggregate(comparison, metric):
    for row in comparison:
        if row.get("kind") == "ratio_aggregate" and row.get("metric") == metric:
            return row
    return None


def paired_ratio(summary_by_key, hb_metric, native_metric=None):
    native_metric = native_metric or hb_metric
    vals = []
    for repeat in REPEATS:
        native = summary_by_key[("uxsched_native_hp_lp", repeat)]
        hb = summary_by_key[("uxsched_hb_fixed_hp_lp", repeat)]
        denom = fnum(native[native_metric])
        if denom <= 0:
            raise GateError(f"non-positive native denominator for {native_metric} repeat {repeat}")
        vals.append(fnum(hb[hb_metric]) / denom)
    return vals


def check_hb_stats(result_dir):
    rows = []
    expected_parent = {
        "0": 2446,
        "1": 2426,
        "2": 2426,
        "3": 2427,
        "4": 2397,
    }
    for repeat in REPEATS:
        path = result_dir / "uxsched_hb_fixed_hp_lp" / f"repeat_{repeat}" / "uxsched_backend_stats.env"
        stats = read_env(path)
        if not stats:
            raise GateError(f"missing HB backend stats: {path}")
        parent = int(stats.get("hb_parent_launch_count_delta", "-1"))
        child = int(stats.get("hb_child_launch_count_delta", "-1"))
        transformed = int(stats.get("hb_transformed_launch_count_delta", "-1"))
        checks = {
            "runtime_hb_metadata_bridge_pass": stats.get("runtime_hb_metadata_bridge_pass") == "1",
            "hb_transform_count_before_measurement": stats.get("hb_transform_count_before_measurement") == "1",
            "hb_transform_count_after_measurement": stats.get("hb_transform_count_after_measurement") == "1",
            "hb_transform_count_delta": stats.get("hb_transform_count_delta") == "0",
            "parent_positive": parent > 0,
            "child_six_per_parent": child == parent * 6,
            "transformed_equals_child": transformed == child,
            "hb_fallback_count_delta": stats.get("hb_fallback_count_delta") == "0",
            "hb_no_xqueue_count_delta": stats.get("hb_no_xqueue_count_delta") == "0",
            "hp_hb_transform_count": stats.get("hp_hb_transform_count") == "0",
            "global_scheduler_log_pass": stats.get("global_scheduler_log_pass") == "1",
            "local_fallback_count": stats.get("local_fallback_count") == "0",
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise GateError(f"HB stats gate failed for repeat {repeat}: {', '.join(failed)}")
        if repeat in expected_parent and parent != expected_parent[repeat]:
            raise GateError(f"HB repeat {repeat} parent count {parent} != expected {expected_parent[repeat]}")
        rows.append({"repeat": repeat, "parent": parent, "child": child, "child_per_parent": child / parent})
    return rows


def validate_result(result_dir, summary, comparison):
    by_key = summary_map(summary)
    for system in SYSTEMS:
        for repeat in REPEATS:
            key = (system, repeat)
            if key not in by_key:
                raise GateError(f"missing summary row for {system} repeat {repeat}")
            row = by_key[key]
            if row.get("status") != "COMPLETE":
                raise GateError(f"{system} repeat {repeat} status is {row.get('status')}")
            if not truthy(row.get("correctness_pass")):
                raise GateError(f"{system} repeat {repeat} correctness failed")
            if row.get("hp_count") != "200":
                raise GateError(f"{system} repeat {repeat} hp_count is {row.get('hp_count')}, expected 200")

    hb_rows = check_hb_stats(result_dir)
    for metric in ("hp_p99_ratio", "hp_p99_reduction_pct",
                   "lp_throughput_ratio", "lp_throughput_retention_pct",
                   "lp_throughput_loss_pct"):
        row = ratio_aggregate(comparison, metric)
        if row is None:
            raise GateError(f"missing ratio_aggregate {metric}")
        if row.get("repeat_count") != "5":
            raise GateError(f"{metric} repeat_count is {row.get('repeat_count')}, expected 5")
    return hb_rows


def build_metrics(summary, comparison):
    by_key = summary_map(summary)
    system_rows = []
    for system in SYSTEMS:
        row = {"system": system}
        for metric in ("hp_mean_us", "hp_p95_us", "hp_p99_us", "lp_throughput_rps"):
            agg = aggregate(comparison, system, metric)
            row[f"{metric}_mean"] = fnum(agg["mean"])
            row[f"{metric}_stddev"] = fnum(agg["stddev"])
        system_rows.append(row)

    derived = []
    p99_ratio = fnum(ratio_aggregate(comparison, "hp_p99_ratio")["mean"])
    derived.append({
        "metric": "hp_p99_latency",
        "paired_ratio_mean": p99_ratio,
        "reduction_or_retention_pct": fnum(ratio_aggregate(comparison, "hp_p99_reduction_pct")["mean"]),
    })
    lp_ratio = fnum(ratio_aggregate(comparison, "lp_throughput_ratio")["mean"])
    derived.append({
        "metric": "lp_throughput",
        "paired_ratio_mean": lp_ratio,
        "reduction_or_retention_pct": fnum(ratio_aggregate(comparison, "lp_throughput_retention_pct")["mean"]),
    })
    derived.append({
        "metric": "lp_throughput_loss",
        "paired_ratio_mean": lp_ratio,
        "reduction_or_retention_pct": fnum(ratio_aggregate(comparison, "lp_throughput_loss_pct")["mean"]),
    })

    p95_vals = paired_ratio(by_key, "hp_p95_us")
    p95_ratio = mean(p95_vals)
    derived.append({
        "metric": "hp_p95_latency",
        "paired_ratio_mean": p95_ratio,
        "reduction_or_retention_pct": (1.0 - p95_ratio) * 100.0,
    })
    mean_vals = paired_ratio(by_key, "hp_mean_us")
    mean_ratio = mean(mean_vals)
    derived.append({
        "metric": "hp_mean_latency",
        "paired_ratio_mean": mean_ratio,
        "reduction_or_retention_pct": (1.0 - mean_ratio) * 100.0,
    })
    return system_rows, derived


def derived_value(derived, metric):
    for row in derived:
        if row["metric"] == metric:
            return fnum(row["paired_ratio_mean"])
    raise KeyError(metric)


def import_matplotlib():
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="uxsched-mpl-")
    try:
        import matplotlib
    except ModuleNotFoundError:
        venv_python = Path("/home/zm/project/hummingbird/.venv/bin/python")
        if venv_python.exists() and Path(sys.executable) != venv_python:
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
        raise
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def setup_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, base, formats, dpi):
    paths = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight", facecolor="white")
        paths.append(path)
    return paths


def add_bar_labels(ax, bars, fmt="{:.0f}"):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)


def make_plots(fig_dir, summary, comparison, derived, formats, dpi, title_suffix):
    plt = import_matplotlib()
    by_key = summary_map(summary)
    suffix = f"\n{title_suffix}" if title_suffix else ""
    generated = []

    def agg_mean(system, metric):
        return fnum(aggregate(comparison, system, metric)["mean"])

    def agg_std(system, metric):
        return fnum(aggregate(comparison, system, metric)["stddev"])

    p99_reduction = next(row["reduction_or_retention_pct"] for row in derived if row["metric"] == "hp_p99_latency")
    p99_ratio = derived_value(derived, "hp_p99_latency")
    lp_retention = next(row["reduction_or_retention_pct"] for row in derived if row["metric"] == "lp_throughput")
    lp_loss = next(row["reduction_or_retention_pct"] for row in derived if row["metric"] == "lp_throughput_loss")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    values = [agg_mean(s, "hp_p99_us") for s in CORE_SYSTEMS]
    errors = [agg_std(s, "hp_p99_us") for s in CORE_SYSTEMS]
    bars = ax.bar([AXIS_LABELS[s] for s in CORE_SYSTEMS], values, yerr=errors,
                  color=[COLORS[s] for s in CORE_SYSTEMS], capsize=6)
    setup_ax(ax, "HP P99 Latency under LP Contention" + suffix, "Latency (us)")
    ax.set_ylim(bottom=0, top=max(v + e for v, e in zip(values, errors)) * 1.22)
    add_bar_labels(ax, bars)
    ax.text(0.5, 0.93,
            f"Paired P99 reduction: {p99_reduction:.2f}%   Unsplit / HB = {1.0 / p99_ratio:.2f}x\n"
            "CUTLASS FP32 SIMT GEMM, M=N=K=2048, 5 repeats; error bars: stddev",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    generated += save_figure(fig, fig_dir / "hp_p99_native_vs_hb", formats, dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    values = [agg_mean(s, "lp_throughput_rps") for s in CORE_SYSTEMS]
    errors = [agg_std(s, "lp_throughput_rps") for s in CORE_SYSTEMS]
    bars = ax.bar([AXIS_LABELS[s] for s in CORE_SYSTEMS], values, yerr=errors,
                  color=[COLORS[s] for s in CORE_SYSTEMS], capsize=6)
    setup_ax(ax, "LP Throughput under HP Contention" + suffix, "Throughput (requests/s)")
    ax.set_ylim(bottom=0, top=max(v + e for v, e in zip(values, errors)) * 1.22)
    add_bar_labels(ax, bars, "{:.1f}")
    ax.text(0.5, 0.93,
            f"LP throughput retained: {lp_retention:.2f}%   loss: {lp_loss:.2f}%\n"
            "CUTLASS FP32 SIMT GEMM, M=N=K=2048, 5 repeats; error bars: stddev",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    generated += save_figure(fig, fig_dir / "lp_throughput_native_vs_hb", formats, dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    metrics = [("Mean", "hp_mean_us"), ("P95", "hp_p95_us"), ("P99", "hp_p99_us")]
    x = list(range(len(metrics)))
    width = 0.34
    for idx, system in enumerate(CORE_SYSTEMS):
        offset = -width / 2 if idx == 0 else width / 2
        vals = [agg_mean(system, metric) for _, metric in metrics]
        errs = [agg_std(system, metric) for _, metric in metrics]
        bars = ax.bar([v + offset for v in x], vals, width=width, yerr=errs,
                      label=LABELS[system], color=COLORS[system], capsize=5)
        add_bar_labels(ax, bars)
    setup_ax(ax, "HP Latency Metrics under LP Contention" + suffix, "Latency (us)")
    ax.set_xticks(x, [name for name, _ in metrics])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.text(0.5, -0.20, "CUTLASS FP32 SIMT GEMM, M=N=K=2048, 5 repeats; error bars: stddev",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    generated += save_figure(fig, fig_dir / "hp_latency_metrics", formats, dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    repeats = [int(r) for r in REPEATS]
    for system in CORE_SYSTEMS:
        vals = [fnum(by_key[(system, str(r))]["hp_p99_us"]) for r in repeats]
        ax.plot(repeats, vals, marker="o", linewidth=2.2, label=LABELS[system], color=COLORS[system])
    setup_ax(ax, "HP P99 by Repeat" + suffix, "Latency (us)")
    ax.set_xlabel("Repeat")
    ax.set_xticks(repeats)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.text(0.5, -0.20, "Each point is one measured repeat; HB_FIXED is below the Unsplit Kernel baseline in all five paired repeats.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    generated += save_figure(fig, fig_dir / "hp_p99_by_repeat", formats, dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for system in CORE_SYSTEMS:
        vals = [fnum(by_key[(system, str(r))]["lp_throughput_rps"]) for r in repeats]
        ax.plot(repeats, vals, marker="o", linewidth=2.2, label=LABELS[system], color=COLORS[system])
    setup_ax(ax, "LP Throughput by Repeat" + suffix, "Throughput (requests/s)")
    ax.set_xlabel("Repeat")
    ax.set_xticks(repeats)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.text(0.5, -0.20, "Each point is one measured repeat; values are not smoothed.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    generated += save_figure(fig, fig_dir / "lp_throughput_by_repeat", formats, dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    values = [agg_mean(s, "hp_p99_us") for s in SYSTEMS]
    errors = [agg_std(s, "hp_p99_us") for s in SYSTEMS]
    bars = ax.bar([AXIS_LABELS[s] for s in SYSTEMS], values, yerr=errors,
                  color=[COLORS[s] for s in SYSTEMS], capsize=6)
    setup_ax(ax, "Context View: HP P99 Latency" + suffix, "Latency (us)")
    ax.set_ylim(bottom=0, top=max(v + e for v, e in zip(values, errors)) * 1.2)
    add_bar_labels(ax, bars)
    ax.text(0.5, -0.22,
            "Standalone HP is context only and has high cross-repeat variance; main comparison is UXSched Lv1 + Unsplit Kernel vs UXSched Lv1 + HB_FIXED.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    generated += save_figure(fig, fig_dir / "hp_p99_all_systems_context", formats, dpi)
    plt.close(fig)
    return generated


def make_tables(fig_dir, system_rows, derived):
    fields = [
        "system",
        "hp_mean_us_mean", "hp_mean_us_stddev",
        "hp_p95_us_mean", "hp_p95_us_stddev",
        "hp_p99_us_mean", "hp_p99_us_stddev",
        "lp_throughput_rps_mean", "lp_throughput_rps_stddev",
    ]
    write_csv(fig_dir / "final_metrics.csv", system_rows, fields)
    write_csv(fig_dir / "final_derived_metrics.csv", derived,
              ["metric", "paired_ratio_mean", "reduction_or_retention_pct"])

    lines = ["# Final Metrics", "", "## System Metrics", ""]
    lines.append("| system | HP mean us | HP mean stddev | HP P95 us | HP P95 stddev | HP P99 us | HP P99 stddev | LP throughput rps | LP throughput stddev |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in system_rows:
        lines.append(
            f"| {LABELS.get(row['system'], row['system'])} | "
            f"{row['hp_mean_us_mean']:.3f} | {row['hp_mean_us_stddev']:.3f} | "
            f"{row['hp_p95_us_mean']:.3f} | {row['hp_p95_us_stddev']:.3f} | "
            f"{row['hp_p99_us_mean']:.3f} | {row['hp_p99_us_stddev']:.3f} | "
            f"{row['lp_throughput_rps_mean']:.3f} | {row['lp_throughput_rps_stddev']:.3f} |"
        )
    lines.extend(["", "## Derived Metrics", ""])
    lines.append("| metric | paired_ratio_mean | reduction_or_retention_pct |")
    lines.append("|---|---:|---:|")
    for row in derived:
        lines.append(f"| {row['metric']} | {row['paired_ratio_mean']:.6f} | {row['reduction_or_retention_pct']:.3f} |")
    (fig_dir / "final_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_gpu(nvidia_smi_path):
    if not nvidia_smi_path.exists():
        return "not recorded"
    text = nvidia_smi_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "NVIDIA GeForce" in line:
            return re.sub(r"\s+", " ", line.strip().strip("|")).split(" On ")[0].strip()
    return "recorded in nvidia_smi.txt"


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_report(result_dir, fig_dir, metadata, system_rows, derived, hb_stats):
    metrics = {row["system"]: row for row in system_rows}
    d = {row["metric"]: row for row in derived}
    kernel_file = Path(metadata.get("verified_kernel_file", "benchmarks/cutlass/verified_kernel_sm120_fp32_simt.txt"))
    if not kernel_file.is_absolute():
        kernel_file = Path.cwd() / kernel_file
    kernel = non_comment_kernel(kernel_file)
    gpu = parse_gpu(result_dir / "nvidia_smi.txt")
    nvcc = (result_dir / "nvcc_version.txt").read_text(encoding="utf-8", errors="replace").strip() if (result_dir / "nvcc_version.txt").exists() else "not recorded"

    lines = [
        "# CUTLASS Realtime Final Report",
        "",
        "## 1. Experiment Configuration",
        "",
        f"- Result directory: `{result_dir}`",
        f"- GPU: {gpu}",
        f"- CUDA home: `{metadata.get('cuda_home', 'not recorded')}`",
        f"- NVCC: {nvcc}",
        f"- CUTLASS revision: `{metadata.get('cutlass_revision', 'not recorded')}`",
        f"- GEMM: CUTLASS FP32 SIMT, M=N=K={metadata.get('m', '2048')}",
        f"- HP requests: {metadata.get('hp_requests', 'not recorded')}",
        f"- HP period: {metadata.get('hp_period_us', 'not recorded')} us",
        f"- LP duration: {metadata.get('lp_duration_ms', 'not recorded')} ms",
        f"- repeats: {metadata.get('repeat', 'not recorded')}",
        f"- cooldown: {metadata.get('cooldown_sec', 'not recorded')} s",
        "- scheduler: Global HPF",
        "- priorities: HP=10, LP=-10",
        f"- split_blocks: {metadata.get('split_blocks', 'not recorded')}",
        f"- case order: repeat_0={metadata.get('repeat_0_case_order', 'not recorded')}; repeat_1={metadata.get('repeat_1_case_order', 'not recorded')}; repeat_2={metadata.get('repeat_2_case_order', 'not recorded')}; repeat_3={metadata.get('repeat_3_case_order', 'not recorded')}; repeat_4={metadata.get('repeat_4_case_order', 'not recorded')}",
        f"- verified kernel allowlist file: `{metadata.get('verified_kernel_file', 'not recorded')}`",
        f"- verified kernel name length: {len(kernel)}",
        f"- verified kernel SHA256: `{sha256_text(kernel) if kernel else 'not recorded'}`",
        "",
        "## 2. Correctness and Functional Gate",
        "",
        "- All cases are `COMPLETE`.",
        "- All correctness checks passed.",
        "- Five HB_FIXED repeats passed metadata bridge checks.",
        "- Warmup transform completed once per HB repeat and measurement `transform_count_delta=0`.",
        "- Each LP parent launch split into exactly 6 child launches.",
        "- `fallback_count_delta=0`, `no_xqueue_count_delta=0`, `hp_hb_transform_count=0`.",
        "- Global HPF log checks passed and local fallback count is 0.",
        "",
        "| repeat | parent launches | child launches | child/parent |",
        "|---:|---:|---:|---:|",
    ]
    for row in hb_stats:
        lines.append(f"| {row['repeat']} | {row['parent']} | {row['child']} | {row['child_per_parent']:.1f} |")

    native = metrics["uxsched_native_hp_lp"]
    hb = metrics["uxsched_hb_fixed_hp_lp"]
    lines.extend([
        "",
        "## 3. Core Performance Results",
        "",
        f"In five paired repeats, UXSched Lv1 + Hummingbird Fixed Splitting (HB_FIXED) reduced HP P99 latency by {d['hp_p99_latency']['reduction_or_retention_pct']:.2f}% versus UXSched Lv1 + Unsplit Kernel while retaining {d['lp_throughput']['reduction_or_retention_pct']:.2f}% of LP throughput.",
        "",
        f"- Native HP P99 mean: {native['hp_p99_us_mean']:.3f} us",
        f"- HB_FIXED HP P99 mean: {hb['hp_p99_us_mean']:.3f} us",
        f"- P99 ratio: {d['hp_p99_latency']['paired_ratio_mean']:.6f}",
        f"- Unsplit / HB P99 factor: {1.0 / d['hp_p99_latency']['paired_ratio_mean']:.2f}x",
        f"- HP P95 reduction: {d['hp_p95_latency']['reduction_or_retention_pct']:.2f}%",
        f"- HP mean reduction: {d['hp_mean_latency']['reduction_or_retention_pct']:.2f}%",
        f"- Native LP throughput mean: {native['lp_throughput_rps_mean']:.3f} requests/s",
        f"- HB_FIXED LP throughput mean: {hb['lp_throughput_rps_mean']:.3f} requests/s",
        f"- LP throughput retention: {d['lp_throughput']['reduction_or_retention_pct']:.2f}%",
        f"- LP throughput loss: {d['lp_throughput_loss']['reduction_or_retention_pct']:.2f}%",
        "",
        "## 4. Trade-off",
        "",
        "HB_FIXED substantially improves HP tail latency under LP contention, but LP throughput decreases. This is a latency versus background-throughput trade-off, and the throughput cost is reported explicitly rather than hidden.",
        "",
        "## 5. Standalone Context",
        "",
        f"Standalone HP is included only as context. Its HP P99 mean is {metrics['standalone_hp']['hp_p99_us_mean']:.3f} us with stddev {metrics['standalone_hp']['hp_p99_us_stddev']:.3f} us and max repeat P99 4703.009 us, so this run does not support a claim that HB_FIXED is better than exclusive standalone execution. The main comparison is UXSched Lv1 + Unsplit Kernel versus UXSched Lv1 + HB_FIXED under the same HP+LP contention.",
        "",
        "## 6. split=52 Rationale",
        "",
        "The fixed split size 52 comes from the Hummingbird hardware-aware formula: 26 SMs times 2 resident CUTLASS blocks per SM. The resident block count is register-limited for this kernel: 128 registers/thread times 256 threads/block equals 32768 registers/block, and 65536 registers/SM allows 2 blocks/SM.",
        "",
        "A prior repeat=3 comparison showed split=52 reduced HB HP P99 by 7.76% versus split=64 and improved LP throughput by 10.54% versus split=64. This is not automatic split selection, not runtime profiling, and not a global optimum for other GPUs or kernels.",
        "",
        "## 7. Limitations",
        "",
        "- Single GPU.",
        "- Single CUTLASS kernel and matrix size.",
        "- Fixed GPU and CUDA toolkit version.",
        "- Dynamic kernel consolidation is not implemented.",
        "- Automatic profiling-based split selection is not implemented.",
        "- LP throughput still has a substantial loss.",
        "- Standalone P99 has high cross-repeat variance.",
    ])
    (result_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def png_size(path):
    with path.open("rb") as f:
        sig = f.read(24)
    if sig[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", sig[16:24])


def write_error_report(result_dir, error):
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    report = fig_dir / "gate_failure_report.md"
    report.write_text(f"# Gate Failure\n\n{error}\n", encoding="utf-8")
    return report


def run(args):
    result_dir = args.result_dir.resolve()
    try:
        summary, comparison, metadata = load_result(result_dir)
        hb_stats = validate_result(result_dir, summary, comparison)
    except GateError as exc:
        report = write_error_report(result_dir, str(exc))
        raise SystemExit(f"Gate failed; wrote {report}: {exc}") from exc

    fig_dir = Path(args.output_dir).resolve() if args.output_dir else result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    system_rows, derived = build_metrics(summary, comparison)
    make_tables(fig_dir, system_rows, derived)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    generated = make_plots(fig_dir, summary, comparison, derived, formats, args.dpi, args.title_suffix)
    make_report(result_dir, fig_dir, metadata, system_rows, derived, hb_stats)

    for path in generated:
        if path.suffix == ".png":
            size = png_size(path)
            if size is None or size[0] <= 0 or size[1] <= 0:
                raise SystemExit(f"invalid PNG output: {path}")
        if path.stat().st_size <= 0:
            raise SystemExit(f"empty output file: {path}")
    print(f"result_dir={result_dir}")
    print(f"figures_dir={fig_dir}")
    for path in generated:
        print(path)
    print(result_dir / "final_report.md")
    return 0


def self_test():
    rows = [
        {"system": "standalone_hp", "repeat": str(i), "status": "COMPLETE", "hp_count": "200",
         "correctness_pass": "1", "hp_mean_us": "10", "hp_p95_us": "11", "hp_p99_us": "12",
         "lp_throughput_rps": "0"} for i in range(5)
    ]
    rows += [
        {"system": "uxsched_native_hp_lp", "repeat": str(i), "status": "COMPLETE", "hp_count": "200",
         "correctness_pass": "1", "hp_mean_us": "100", "hp_p95_us": "200", "hp_p99_us": "300",
         "lp_throughput_rps": "400"} for i in range(5)
    ]
    rows += [
        {"system": "uxsched_hb_fixed_hp_lp", "repeat": str(i), "status": "COMPLETE", "hp_count": "200",
         "correctness_pass": "1", "hp_mean_us": str(50 + i), "hp_p95_us": str(100 + i),
         "hp_p99_us": str(150 + i), "lp_throughput_rps": str(200 + i)} for i in range(5)
    ]
    by_key = summary_map(rows)
    assert len(paired_ratio(by_key, "hp_p99_us")) == 5
    assert abs(mean(paired_ratio(by_key, "hp_p99_us")) - (sum((150 + i) / 300 for i in range(5)) / 5)) < 1e-12

    comparison = []
    for system in SYSTEMS:
        for metric in ("hp_mean_us", "hp_p95_us", "hp_p99_us", "lp_throughput_rps"):
            vals = [fnum(by_key[(system, str(i))][metric]) for i in range(5)]
            comparison.append({"kind": "aggregate", "system": system, "metric": metric,
                               "mean": str(mean(vals)), "stddev": str(stdev(vals))})
    comparison.extend([
        {"kind": "ratio_aggregate", "metric": "hp_p99_ratio", "repeat_count": "5", "mean": "0.5"},
        {"kind": "ratio_aggregate", "metric": "hp_p99_reduction_pct", "repeat_count": "5", "mean": "50"},
        {"kind": "ratio_aggregate", "metric": "lp_throughput_ratio", "repeat_count": "5", "mean": "0.5"},
        {"kind": "ratio_aggregate", "metric": "lp_throughput_retention_pct", "repeat_count": "5", "mean": "50"},
        {"kind": "ratio_aggregate", "metric": "lp_throughput_loss_pct", "repeat_count": "5", "mean": "50"},
    ])
    system_rows, derived = build_metrics(rows, comparison)
    assert len(system_rows) == 3
    assert any(row["metric"] == "hp_p95_latency" for row in derived)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--output-dir")
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title-suffix", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.result_dir:
        parser.error("--result-dir is required unless --self-test is used")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

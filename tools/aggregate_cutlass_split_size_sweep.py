#!/usr/bin/env python3
import argparse
import csv
import math
import tempfile
from pathlib import Path
from statistics import mean, median


SUMMARY_FIELDS = [
    "split_blocks",
    "kind",
    "repeat_count",
    "hp_mean_us_mean",
    "hp_mean_us_stddev",
    "hp_p95_us_mean",
    "hp_p95_us_stddev",
    "hp_p99_us_mean",
    "hp_p99_us_stddev",
    "lp_throughput_rps_mean",
    "lp_throughput_rps_stddev",
    "hp_p99_reduction_vs_unsplit",
    "lp_throughput_retention_vs_unsplit",
    "lp_throughput_loss_vs_unsplit",
    "parent_count",
    "child_count",
    "children_per_parent",
    "correctness_pass",
    "global_scheduler_pass",
    "hb_gate_pass",
]

REPEAT_FIELDS = [
    "split_blocks",
    "repeat",
    "native_hp_p99_us",
    "hb_hp_p99_us",
    "native_lp_throughput_rps",
    "hb_lp_throughput_rps",
    "hp_p99_ratio",
    "hp_p99_reduction_pct",
    "lp_throughput_ratio",
    "lp_throughput_retention_pct",
    "lp_throughput_loss_pct",
    "parent_count",
    "child_count",
    "children_per_parent",
    "correctness_pass",
    "global_scheduler_pass",
    "hb_gate_pass",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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


def fnum(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value):
    return str(value).lower() in ("1", "true", "pass", "passed", "yes")


def stdev(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def aggregate(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "stddev": stdev(values),
    }


def load_single_run(run_dir):
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path}")
    rows = read_csv(summary_path)
    by_system = {row.get("system"): row for row in rows}
    native = by_system.get("uxsched_native_hp_lp")
    hb = by_system.get("uxsched_hb_fixed_hp_lp")
    if not native or not hb:
        raise ValueError(f"missing native/HB rows in {summary_path}")
    stats = read_env(run_dir / "uxsched_hb_fixed_hp_lp" / "repeat_0" / "uxsched_backend_stats.env")

    parent = inum(stats.get("hb_parent_launch_count_delta"), 0)
    child = inum(stats.get("hb_child_launch_count_delta"), 0)
    transformed = inum(stats.get("hb_transformed_launch_count_delta"), 0)
    fallback = inum(stats.get("hb_fallback_count_delta"), 999)
    no_xqueue = inum(stats.get("hb_no_xqueue_count_delta"), 999)
    transform_delta = inum(stats.get("hb_transform_count_delta"), 999)
    metadata = stats.get("runtime_hb_metadata_bridge_pass") == "1"
    hp_transform = inum(stats.get("hp_hb_transform_count"), 999)
    global_pass = stats.get("global_scheduler_log_pass") == "1"
    local_fallback = inum(stats.get("local_fallback_count"), 999)
    correctness = truthy(native.get("correctness_pass")) and truthy(hb.get("correctness_pass"))
    status_pass = native.get("status") == "COMPLETE" and hb.get("status") == "COMPLETE"
    hb_gate = (
        status_pass
        and correctness
        and metadata
        and parent > 0
        and child > parent
        and transformed == child
        and fallback == 0
        and no_xqueue == 0
        and transform_delta == 0
        and hp_transform == 0
        and global_pass
        and local_fallback == 0
    )
    return {
        "native": native,
        "hb": hb,
        "parent": parent,
        "child": child,
        "children_per_parent": (child / parent) if parent else 0.0,
        "correctness_pass": correctness,
        "global_scheduler_pass": global_pass,
        "hb_gate_pass": hb_gate,
    }


def discover_runs(result_dir):
    runs = []
    for split_dir in sorted(result_dir.glob("split_*")):
        if not split_dir.is_dir():
            continue
        split_text = split_dir.name.replace("split_", "")
        try:
            split = int(split_text)
        except ValueError:
            continue
        repeat_dirs = sorted(split_dir.glob("repeat_*"))
        if repeat_dirs:
            for repeat_dir in repeat_dirs:
                if not (repeat_dir / "summary.csv").exists():
                    continue
                repeat = repeat_dir.name.replace("repeat_", "")
                runs.append((split, repeat, repeat_dir))
        elif (split_dir / "summary.csv").exists():
            runs.append((split, "0", split_dir))
    return runs


def summarize(result_dir):
    repeat_rows = []
    native_all = []
    split_values = {}
    for split, repeat, run_dir in discover_runs(result_dir):
        loaded = load_single_run(run_dir)
        native = loaded["native"]
        hb = loaded["hb"]
        native_p99 = fnum(native.get("hp_p99_us"))
        hb_p99 = fnum(hb.get("hp_p99_us"))
        native_lp = fnum(native.get("lp_throughput_rps"))
        hb_lp = fnum(hb.get("lp_throughput_rps"))
        hp_ratio = hb_p99 / native_p99 if native_p99 > 0 else 0.0
        lp_ratio = hb_lp / native_lp if native_lp > 0 else 0.0
        row = {
            "split_blocks": split,
            "repeat": repeat,
            "native_hp_p99_us": native_p99,
            "hb_hp_p99_us": hb_p99,
            "native_lp_throughput_rps": native_lp,
            "hb_lp_throughput_rps": hb_lp,
            "hp_p99_ratio": hp_ratio,
            "hp_p99_reduction_pct": (1.0 - hp_ratio) * 100.0 if native_p99 > 0 else 0.0,
            "lp_throughput_ratio": lp_ratio,
            "lp_throughput_retention_pct": lp_ratio * 100.0 if native_lp > 0 else 0.0,
            "lp_throughput_loss_pct": (1.0 - lp_ratio) * 100.0 if native_lp > 0 else 0.0,
            "parent_count": loaded["parent"],
            "child_count": loaded["child"],
            "children_per_parent": loaded["children_per_parent"],
            "correctness_pass": 1 if loaded["correctness_pass"] else 0,
            "global_scheduler_pass": 1 if loaded["global_scheduler_pass"] else 0,
            "hb_gate_pass": 1 if loaded["hb_gate_pass"] else 0,
        }
        repeat_rows.append(row)
        native_all.append(native)
        split_values.setdefault(split, []).append((hb, row))

    if not repeat_rows:
        raise RuntimeError(f"no split sweep runs found under {result_dir}")

    summary_rows = []
    native_metrics = {
        "hp_mean_us": [fnum(r.get("hp_mean_us")) for r in native_all],
        "hp_p95_us": [fnum(r.get("hp_p95_us")) for r in native_all],
        "hp_p99_us": [fnum(r.get("hp_p99_us")) for r in native_all],
        "lp_throughput_rps": [fnum(r.get("lp_throughput_rps")) for r in native_all],
    }
    native_agg = {key: aggregate(vals) for key, vals in native_metrics.items()}
    summary_rows.append({
        "split_blocks": "Unsplit",
        "kind": "unsplit_baseline",
        "repeat_count": len(native_all),
        "hp_mean_us_mean": native_agg["hp_mean_us"]["mean"],
        "hp_mean_us_stddev": native_agg["hp_mean_us"]["stddev"],
        "hp_p95_us_mean": native_agg["hp_p95_us"]["mean"],
        "hp_p95_us_stddev": native_agg["hp_p95_us"]["stddev"],
        "hp_p99_us_mean": native_agg["hp_p99_us"]["mean"],
        "hp_p99_us_stddev": native_agg["hp_p99_us"]["stddev"],
        "lp_throughput_rps_mean": native_agg["lp_throughput_rps"]["mean"],
        "lp_throughput_rps_stddev": native_agg["lp_throughput_rps"]["stddev"],
        "hp_p99_reduction_vs_unsplit": 0.0,
        "lp_throughput_retention_vs_unsplit": 100.0,
        "lp_throughput_loss_vs_unsplit": 0.0,
        "parent_count": "",
        "child_count": "",
        "children_per_parent": "",
        "correctness_pass": 1 if all(truthy(r.get("correctness_pass")) for r in native_all) else 0,
        "global_scheduler_pass": "",
        "hb_gate_pass": "",
    })

    normalized_rows = []
    tradeoff_rows = []
    for split in sorted(split_values):
        entries = split_values[split]
        hb_rows = [item[0] for item in entries]
        pair_rows = [item[1] for item in entries]
        metrics = {
            "hp_mean_us": [fnum(r.get("hp_mean_us")) for r in hb_rows],
            "hp_p95_us": [fnum(r.get("hp_p95_us")) for r in hb_rows],
            "hp_p99_us": [fnum(r.get("hp_p99_us")) for r in hb_rows],
            "lp_throughput_rps": [fnum(r.get("lp_throughput_rps")) for r in hb_rows],
        }
        aggs = {key: aggregate(vals) for key, vals in metrics.items()}
        p99_ratios = [fnum(r["hp_p99_ratio"]) for r in pair_rows if inum(r["hb_gate_pass"], 0) == 1]
        lp_ratios = [fnum(r["lp_throughput_ratio"]) for r in pair_rows if inum(r["hb_gate_pass"], 0) == 1]
        p99_ratio_agg = aggregate(p99_ratios)
        lp_ratio_agg = aggregate(lp_ratios)
        parent_count = sum(inum(r["parent_count"]) for r in pair_rows)
        child_count = sum(inum(r["child_count"]) for r in pair_rows)
        children_per_parent = (child_count / parent_count) if parent_count else 0.0
        correctness_pass = all(inum(r["correctness_pass"]) == 1 for r in pair_rows)
        global_pass = all(inum(r["global_scheduler_pass"]) == 1 for r in pair_rows)
        hb_gate = all(inum(r["hb_gate_pass"]) == 1 for r in pair_rows)
        summary_rows.append({
            "split_blocks": split,
            "kind": "hb_fixed",
            "repeat_count": len(pair_rows),
            "hp_mean_us_mean": aggs["hp_mean_us"]["mean"],
            "hp_mean_us_stddev": aggs["hp_mean_us"]["stddev"],
            "hp_p95_us_mean": aggs["hp_p95_us"]["mean"],
            "hp_p95_us_stddev": aggs["hp_p95_us"]["stddev"],
            "hp_p99_us_mean": aggs["hp_p99_us"]["mean"],
            "hp_p99_us_stddev": aggs["hp_p99_us"]["stddev"],
            "lp_throughput_rps_mean": aggs["lp_throughput_rps"]["mean"],
            "lp_throughput_rps_stddev": aggs["lp_throughput_rps"]["stddev"],
            "hp_p99_reduction_vs_unsplit": (1.0 - p99_ratio_agg["mean"]) * 100.0 if p99_ratios else 0.0,
            "lp_throughput_retention_vs_unsplit": lp_ratio_agg["mean"] * 100.0 if lp_ratios else 0.0,
            "lp_throughput_loss_vs_unsplit": (1.0 - lp_ratio_agg["mean"]) * 100.0 if lp_ratios else 0.0,
            "parent_count": parent_count,
            "child_count": child_count,
            "children_per_parent": children_per_parent,
            "correctness_pass": 1 if correctness_pass else 0,
            "global_scheduler_pass": 1 if global_pass else 0,
            "hb_gate_pass": 1 if hb_gate else 0,
        })
        normalized_rows.append({
            "split_blocks": split,
            "hp_p99_ratio_relative_to_unsplit": p99_ratio_agg["mean"],
            "hp_p99_reduction_pct_paired": (1.0 - p99_ratio_agg["mean"]) * 100.0 if p99_ratios else 0.0,
            "lp_throughput_ratio_relative_to_unsplit": lp_ratio_agg["mean"],
            "lp_throughput_retention_pct_paired": lp_ratio_agg["mean"] * 100.0 if lp_ratios else 0.0,
            "lp_throughput_loss_pct_paired": (1.0 - lp_ratio_agg["mean"]) * 100.0 if lp_ratios else 0.0,
            "repeat_count": len(p99_ratios),
        })
        tradeoff_rows.append({
            "split_blocks": split,
            "hp_p99_reduction_pct": (1.0 - p99_ratio_agg["mean"]) * 100.0 if p99_ratios else 0.0,
            "lp_throughput_retention_pct": lp_ratio_agg["mean"] * 100.0 if lp_ratios else 0.0,
            "lp_throughput_loss_pct": (1.0 - lp_ratio_agg["mean"]) * 100.0 if lp_ratios else 0.0,
            "children_per_parent": children_per_parent,
            "hb_gate_pass": 1 if hb_gate else 0,
        })

    return summary_rows, normalized_rows, tradeoff_rows, repeat_rows


def write_report(result_dir, summary_rows, normalized_rows, tradeoff_rows, repeat_rows):
    lines = [
        "# CUTLASS Split Size Sweep Report",
        "",
        "This report is generated from CSV/JSONL/log artifacts already present in the result directory. It does not run benchmarks.",
        "",
        "## Configuration",
        "",
        "- Workload: CUTLASS FP32 SIMT GEMM, M=N=K=2048",
        "- Systems: UXSched Lv1 + Unsplit Kernel versus UXSched Lv1 + HB_FIXED",
        "- Ratios: paired HB / Unsplit values within the same split-size repeat.",
        "- The Unsplit summary aggregates all native baseline runs in this sweep.",
        "",
        "## Summary",
        "",
        "| split_blocks | HP P99 mean (us) | HP P99 reduction vs Unsplit | LP throughput mean (rps) | LP retention | children/parent | HB gate |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['split_blocks']} | {fnum(row['hp_p99_us_mean']):.3f} | "
            f"{fnum(row['hp_p99_reduction_vs_unsplit']):.2f}% | {fnum(row['lp_throughput_rps_mean']):.3f} | "
            f"{fnum(row['lp_throughput_retention_vs_unsplit']):.2f}% | {row.get('children_per_parent', '')} | {row.get('hb_gate_pass', '')} |"
        )
    lines.extend([
        "",
        "## Interpretation Template",
        "",
        "Use this table only after all HB gate checks pass. If split=52 is not the best latency/throughput trade-off in the new repeat=5 sweep, update the competition document accordingly instead of preserving the previous claim.",
        "",
        "Correct wording: split=52 is derived from the current GPU and CUTLASS kernel resource constraints, then validated by the sweep. It is not automatic profiling and not a global optimum.",
    ])
    (result_dir / "split_size_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(result_dir):
    summary_rows, normalized_rows, tradeoff_rows, repeat_rows = summarize(result_dir)
    write_csv(result_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(result_dir / "normalized_metrics.csv", normalized_rows, [
        "split_blocks",
        "hp_p99_ratio_relative_to_unsplit",
        "hp_p99_reduction_pct_paired",
        "lp_throughput_ratio_relative_to_unsplit",
        "lp_throughput_retention_pct_paired",
        "lp_throughput_loss_pct_paired",
        "repeat_count",
    ])
    write_csv(result_dir / "tradeoff.csv", tradeoff_rows, [
        "split_blocks",
        "hp_p99_reduction_pct",
        "lp_throughput_retention_pct",
        "lp_throughput_loss_pct",
        "children_per_parent",
        "hb_gate_pass",
    ])
    write_csv(result_dir / "repeat_metrics.csv", repeat_rows, REPEAT_FIELDS)
    write_report(result_dir, summary_rows, normalized_rows, tradeoff_rows, repeat_rows)
    return 0


def make_fake_run(base, split, repeat, native_p99, hb_p99, native_lp, hb_lp, child_factor=6, failed=False):
    run_dir = base / f"split_{split}" / f"repeat_{repeat}"
    run_dir.mkdir(parents=True)
    summary = [
        {
            "system": "uxsched_native_hp_lp", "repeat": "0", "status": "COMPLETE",
            "hp_count": "200", "hp_mean_us": str(native_p99 * 0.7), "hp_p95_us": str(native_p99 * 0.95),
            "hp_p99_us": str(native_p99), "lp_throughput_rps": str(native_lp), "correctness_pass": "1",
        },
        {
            "system": "uxsched_hb_fixed_hp_lp", "repeat": "0", "status": "FAILED" if failed else "COMPLETE",
            "hp_count": "200", "hp_mean_us": str(hb_p99 * 0.7), "hp_p95_us": str(hb_p99 * 0.95),
            "hp_p99_us": str(hb_p99), "lp_throughput_rps": str(hb_lp), "correctness_pass": "1",
        },
    ]
    write_csv(run_dir / "summary.csv", summary, [
        "system", "repeat", "status", "hp_count", "hp_mean_us", "hp_p95_us", "hp_p99_us",
        "lp_throughput_rps", "correctness_pass",
    ])
    stats_dir = run_dir / "uxsched_hb_fixed_hp_lp" / "repeat_0"
    stats_dir.mkdir(parents=True)
    parent = 10
    child = parent * child_factor
    stats_dir.joinpath("uxsched_backend_stats.env").write_text(
        "\n".join([
            "runtime_hb_metadata_bridge_pass=1",
            "hb_parent_launch_count_delta=10",
            f"hb_child_launch_count_delta={child}",
            f"hb_transformed_launch_count_delta={child}",
            "hb_fallback_count_delta=0",
            "hb_no_xqueue_count_delta=0",
            "hb_transform_count_delta=0",
            "hp_hb_transform_count=0",
            "global_scheduler_log_pass=1",
            "local_fallback_count=0",
        ]) + "\n",
        encoding="utf-8",
    )


def self_test():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        make_fake_run(base, 32, 0, 100.0, 60.0, 200.0, 100.0, child_factor=10)
        make_fake_run(base, 32, 1, 120.0, 72.0, 220.0, 110.0, child_factor=10)
        make_fake_run(base, 52, 0, 100.0, 50.0, 200.0, 120.0, child_factor=6)
        run(base)
        summary = read_csv(base / "summary.csv")
        norm = read_csv(base / "normalized_metrics.csv")
        assert len(summary) == 3
        row32 = next(r for r in norm if r["split_blocks"] == "32")
        assert abs(fnum(row32["hp_p99_ratio_relative_to_unsplit"]) - 0.6) < 1e-12
        assert abs(fnum(row32["lp_throughput_ratio_relative_to_unsplit"]) - 0.5) < 1e-12
        row52 = next(r for r in summary if r["split_blocks"] == "52")
        assert abs(fnum(row52["children_per_parent"]) - 6.0) < 1e-12
        assert Path(base / "split_size_report.md").exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.result_dir:
        parser.error("--result-dir is required unless --self-test is used")
    return run(args.result_dir)


if __name__ == "__main__":
    raise SystemExit(main())

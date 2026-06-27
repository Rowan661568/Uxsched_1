#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


SUMMARY_FIELDS = [
    "system",
    "repeat",
    "status",
    "hp_count",
    "hp_mean_us",
    "hp_p50_us",
    "hp_p95_us",
    "hp_p99_us",
    "hp_max_us",
    "hp_release_lateness_p99_us",
    "lp_completed",
    "lp_duration_us",
    "lp_throughput_rps",
    "lp_mean_us",
    "correctness_pass",
    "output_hash",
    "runtime_launch_intercepted_count",
    "runtime_sync_intercepted_count",
    "runtime_hb_metadata_bridge_pass",
    "hb_transform_count_before_measurement",
    "hb_transform_count_after_measurement",
    "hb_transform_count_delta",
    "hb_parent_launch_count_delta",
    "hb_child_launch_count_delta",
    "hb_transformed_launch_count_delta",
    "hb_fallback_count_delta",
    "hb_no_xqueue_count_delta",
    "hp_hb_transform_count",
    "global_scheduler_log_pass",
    "local_fallback_count",
]


def percentile(vals, pct):
    if not vals:
        return 0.0
    ordered = sorted(vals)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


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


def last_by_type(rows, typ):
    found = None
    for row in rows:
        if row.get("type") == typ:
            found = row
    return found or {}


def truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "pass", "passed", "yes")


def summarize_repeat(system_dir):
    system = system_dir.parent.name
    repeat_name = system_dir.name
    repeat = repeat_name.replace("repeat_", "")
    hp_rows = read_jsonl(system_dir / "hp" / "output.jsonl")
    lp_rows = read_jsonl(system_dir / "lp" / "output.jsonl")
    stats = read_env(system_dir / "uxsched_backend_stats.env")
    status_env = read_env(system_dir / "status.env")

    hp_requests = [r for r in hp_rows if r.get("type") == "request"]
    hp_latencies = [float(r.get("latency_us", 0.0)) for r in hp_requests if r.get("status") == "RAN"]
    hp_lateness = [float(r.get("release_lateness_us", 0.0)) for r in hp_requests if r.get("status") == "RAN"]

    lp_summary = last_by_type(lp_rows, "summary")
    lp_requests = [r for r in lp_rows if r.get("type") == "request" and r.get("status") == "RAN"]
    lp_latencies = [float(r.get("latency_us", 0.0)) for r in lp_requests]

    warmups = []
    for rows in (hp_rows, lp_rows):
        warm = last_by_type(rows, "warmup_summary")
        if warm:
            warmups.append(warm)

    correctness_pass = bool(warmups) and all(truthy(w.get("correctness_pass")) for w in warmups)
    output_hashes = sorted({str(w.get("output_hash", "")) for w in warmups if w.get("output_hash")})
    status = status_env.get("status", "UNKNOWN")
    if any(r.get("status") == "FAILED" for r in hp_requests + lp_requests):
        status = "FAILED"
    if not correctness_pass:
        status = "FAILED"

    row = {
        "system": system,
        "repeat": repeat,
        "status": status,
        "hp_count": len(hp_latencies),
        "hp_mean_us": mean(hp_latencies) if hp_latencies else 0.0,
        "hp_p50_us": percentile(hp_latencies, 50),
        "hp_p95_us": percentile(hp_latencies, 95),
        "hp_p99_us": percentile(hp_latencies, 99),
        "hp_max_us": max(hp_latencies) if hp_latencies else 0.0,
        "hp_release_lateness_p99_us": percentile(hp_lateness, 99),
        "lp_completed": int(lp_summary.get("completed_count", len(lp_latencies)) or 0),
        "lp_duration_us": float(lp_summary.get("duration_us", 0.0) or 0.0),
        "lp_throughput_rps": float(lp_summary.get("throughput_requests_per_second", 0.0) or 0.0),
        "lp_mean_us": float(lp_summary.get("mean_request_us", mean(lp_latencies) if lp_latencies else 0.0) or 0.0),
        "correctness_pass": 1 if correctness_pass else 0,
        "output_hash": ";".join(output_hashes),
    }
    for field in SUMMARY_FIELDS:
        if field not in row:
            row[field] = stats.get(field, "0")
    return row


def numeric(row, key):
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def valid_row(row):
    return str(row.get("status")) in ("COMPLETE", "RAN")


def stdev(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def aggregate_rows(summary_rows):
    out = []
    systems = sorted({r["system"] for r in summary_rows})
    for system in systems:
        rows = [r for r in summary_rows if r["system"] == system]
        for metric in ("hp_p99_us", "hp_p95_us", "hp_mean_us", "lp_throughput_rps"):
            values = [numeric(r, metric) for r in rows if valid_row(r)]
            if not values:
                continue
            out.append({
                "kind": "aggregate",
                "repeat": "",
                "system": system,
                "metric": metric,
                "repeat_count": len(values),
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
                "stddev": stdev(values),
                "value": "",
            })

    native = {r["repeat"]: r for r in summary_rows
              if r["system"] == "uxsched_native_hp_lp" and valid_row(r)}
    hb = {r["repeat"]: r for r in summary_rows
          if r["system"] == "uxsched_hb_fixed_hp_lp" and valid_row(r)}
    ratio_values = {
        "hp_p99_ratio": [],
        "hp_p99_reduction_pct": [],
        "lp_throughput_ratio": [],
        "lp_throughput_retention_pct": [],
        "lp_throughput_loss_pct": [],
    }
    for repeat in sorted(set(native) & set(hb)):
        native_p99 = numeric(native[repeat], "hp_p99_us")
        hb_p99 = numeric(hb[repeat], "hp_p99_us")
        native_lp = numeric(native[repeat], "lp_throughput_rps")
        hb_lp = numeric(hb[repeat], "lp_throughput_rps")
        paired = {}
        if native_p99 > 0:
            hp_ratio = hb_p99 / native_p99
            paired["hp_p99_ratio"] = hp_ratio
            paired["hp_p99_reduction_pct"] = (1.0 - hp_ratio) * 100.0
        if native_lp > 0:
            lp_ratio = hb_lp / native_lp
            paired["lp_throughput_ratio"] = lp_ratio
            paired["lp_throughput_retention_pct"] = lp_ratio * 100.0
            paired["lp_throughput_loss_pct"] = (1.0 - lp_ratio) * 100.0

        for metric, value in paired.items():
            ratio_values[metric].append(value)
            out.append({
                "kind": "ratio_repeat",
                "repeat": repeat,
                "system": "uxsched_hb_fixed_hp_lp/uxsched_native_hp_lp",
                "metric": metric,
                "repeat_count": 1,
                "mean": "",
                "median": "",
                "min": "",
                "max": "",
                "stddev": "",
                "value": value,
            })

    for metric, values in ratio_values.items():
        if not values:
            continue
        out.append({
            "kind": "ratio_aggregate",
            "repeat": "",
            "system": "uxsched_hb_fixed_hp_lp/uxsched_native_hp_lp",
            "metric": metric,
            "repeat_count": len(values),
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "stddev": stdev(values),
            "value": "",
        })
    return out


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run(result_dir):
    repeats = sorted(result_dir.glob("*/repeat_*"))
    summary_rows = [summarize_repeat(path) for path in repeats if path.is_dir()]
    write_csv(result_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    comparison_rows = aggregate_rows(summary_rows)
    write_csv(result_dir / "comparison.csv", comparison_rows,
              ["kind", "repeat", "system", "metric", "repeat_count", "mean", "median", "min", "max", "stddev", "value"])
    return 0


def self_test():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert abs(percentile([1, 2, 3, 4], 95) - 3.8499999999999996) < 1e-9
    assert percentile([7], 99) == 7.0
    assert truthy(True)
    assert truthy("1")
    assert not truthy("0")
    rows = [
        {"system": "uxsched_native_hp_lp", "repeat": "0", "status": "COMPLETE",
         "hp_p99_us": 100.0, "lp_throughput_rps": 200.0},
        {"system": "uxsched_hb_fixed_hp_lp", "repeat": "0", "status": "COMPLETE",
         "hp_p99_us": 50.0, "lp_throughput_rps": 100.0},
        {"system": "uxsched_native_hp_lp", "repeat": "1", "status": "COMPLETE",
         "hp_p99_us": 100.0, "lp_throughput_rps": 200.0},
        {"system": "uxsched_hb_fixed_hp_lp", "repeat": "1", "status": "COMPLETE",
         "hp_p99_us": 60.0, "lp_throughput_rps": 120.0},
        {"system": "uxsched_native_hp_lp", "repeat": "2", "status": "COMPLETE",
         "hp_p99_us": 100.0, "lp_throughput_rps": 200.0},
        {"system": "uxsched_hb_fixed_hp_lp", "repeat": "2", "status": "COMPLETE",
         "hp_p99_us": 70.0, "lp_throughput_rps": 140.0},
        {"system": "uxsched_native_hp_lp", "repeat": "3", "status": "COMPLETE",
         "hp_p99_us": 100.0, "lp_throughput_rps": 200.0},
        {"system": "uxsched_hb_fixed_hp_lp", "repeat": "3", "status": "FAILED",
         "hp_p99_us": 1.0, "lp_throughput_rps": 1.0},
        {"system": "uxsched_hb_fixed_hp_lp", "repeat": "4", "status": "COMPLETE",
         "hp_p99_us": 1.0, "lp_throughput_rps": 1.0},
    ]
    comparison = aggregate_rows(rows)
    hp_repeat = [r for r in comparison if r["kind"] == "ratio_repeat" and r["metric"] == "hp_p99_ratio"]
    assert [r["repeat"] for r in hp_repeat] == ["0", "1", "2"]
    assert [r["value"] for r in hp_repeat] == [0.5, 0.6, 0.7]
    hp_agg = [r for r in comparison if r["kind"] == "ratio_aggregate" and r["metric"] == "hp_p99_ratio"][0]
    assert hp_agg["repeat_count"] == 3
    assert abs(hp_agg["mean"] - 0.6) < 1e-12
    assert abs(hp_agg["median"] - 0.6) < 1e-12
    assert abs(hp_agg["min"] - 0.5) < 1e-12
    assert abs(hp_agg["max"] - 0.7) < 1e-12
    assert abs(hp_agg["stddev"] - 0.1) < 1e-12
    reduction = [r for r in comparison if r["kind"] == "ratio_repeat" and r["metric"] == "hp_p99_reduction_pct"]
    assert [r["value"] for r in reduction] == [50.0, 40.0, 30.000000000000004]


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

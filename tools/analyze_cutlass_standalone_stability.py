#!/usr/bin/env python3
import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from statistics import mean, median


REQUEST_FIELDS = [
    "repeat",
    "request_index",
    "latency_us",
    "gpu_event_us",
    "release_lateness_us",
    "scheduled_time_us",
    "actual_start_time_us",
    "finish_time_us",
    "classification",
    "notes",
]

SUMMARY_FIELDS = [
    "repeat",
    "hp_count",
    "mean",
    "p50",
    "p95",
    "p99",
    "max",
    "release_lateness_p95",
    "release_lateness_p99",
    "release_lateness_max",
    "gpu_event_mean",
    "gpu_event_p95",
    "gpu_event_p99",
    "gpu_event_max",
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


def stdev(vals):
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "request":
                rows.append(row)
    return rows


def load_repeats(result_dir):
    repeats = {}
    base = result_dir / "standalone_hp"
    for repeat_dir in sorted(base.glob("repeat_*")):
        repeat = repeat_dir.name.replace("repeat_", "")
        path = repeat_dir / "hp" / "output.jsonl"
        if path.exists():
            repeats[repeat] = read_jsonl(path)
    return repeats


def f(row, key):
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def req_idx(row):
    try:
        return int(row.get("request_index", -1))
    except (TypeError, ValueError):
        return -1


def summarize_repeat(repeat, rows):
    lat = [f(r, "latency_us") for r in rows if f(r, "latency_us") is not None]
    rel = [f(r, "release_lateness_us") for r in rows if f(r, "release_lateness_us") is not None]
    gpu = [f(r, "gpu_event_us") for r in rows if f(r, "gpu_event_us") is not None]
    return {
        "repeat": repeat,
        "hp_count": len(lat),
        "mean": mean(lat) if lat else "unavailable",
        "p50": percentile(lat, 50) if lat else "unavailable",
        "p95": percentile(lat, 95) if lat else "unavailable",
        "p99": percentile(lat, 99) if lat else "unavailable",
        "max": max(lat) if lat else "unavailable",
        "release_lateness_p95": percentile(rel, 95) if rel else "unavailable",
        "release_lateness_p99": percentile(rel, 99) if rel else "unavailable",
        "release_lateness_max": max(rel) if rel else "unavailable",
        "gpu_event_mean": mean(gpu) if gpu else "unavailable",
        "gpu_event_p95": percentile(gpu, 95) if gpu else "unavailable",
        "gpu_event_p99": percentile(gpu, 99) if gpu else "unavailable",
        "gpu_event_max": max(gpu) if gpu else "unavailable",
    }


def classify_request(row, global_stats, repeat_count):
    latency = f(row, "latency_us")
    gpu = f(row, "gpu_event_us")
    release = f(row, "release_lateness_us")
    idx = req_idx(row)
    notes = []
    if latency is None:
        return "UNAVAILABLE", "latency_us unavailable"

    latency_high = latency >= global_stats["latency_p99"]
    gpu_high = gpu is not None and gpu >= max(global_stats["gpu_p99"], global_stats["gpu_median"] * 1.25)
    release_high = release is not None and release >= max(global_stats["release_p99"], global_stats["release_median"] * 2.0, 500.0)
    early = idx >= 0 and idx < max(5, int(repeat_count * 0.05))
    if early and latency_high:
        notes.append("early_request")

    if latency_high and release_high and not gpu_high:
        classification = "HOST_RELEASE_JITTER"
    elif latency_high and not release_high and not gpu_high:
        classification = "HOST_COMPLETION_JITTER"
    elif latency_high and gpu_high:
        classification = "GPU_EXECUTION_JITTER"
    elif early and latency_high:
        classification = "INSUFFICIENT_STEADY_STATE_WARMUP"
    else:
        classification = "NORMAL_RANGE_TOP_REQUEST"

    if early and classification != "INSUFFICIENT_STEADY_STATE_WARMUP":
        notes.append("also_matches_INSUFFICIENT_STEADY_STATE_WARMUP")
    return classification, ";".join(notes)


def build_analysis(result_dir, top_k):
    repeats = load_repeats(result_dir)
    if not repeats:
        raise SystemExit(f"no standalone repeats found under {result_dir}")
    summaries = [summarize_repeat(repeat, rows) for repeat, rows in sorted(repeats.items(), key=lambda kv: int(kv[0]))]
    all_rows = [row for rows in repeats.values() for row in rows]
    lat = [f(r, "latency_us") for r in all_rows if f(r, "latency_us") is not None]
    gpu = [f(r, "gpu_event_us") for r in all_rows if f(r, "gpu_event_us") is not None]
    rel = [f(r, "release_lateness_us") for r in all_rows if f(r, "release_lateness_us") is not None]
    global_stats = {
        "latency_p99": percentile(lat, 99),
        "latency_median": percentile(lat, 50),
        "gpu_p99": percentile(gpu, 99) if gpu else float("inf"),
        "gpu_median": percentile(gpu, 50) if gpu else float("inf"),
        "release_p99": percentile(rel, 99) if rel else float("inf"),
        "release_median": percentile(rel, 50) if rel else float("inf"),
    }

    outliers = []
    for repeat, rows in sorted(repeats.items(), key=lambda kv: int(kv[0])):
        top = sorted(rows, key=lambda r: f(r, "latency_us") or -1.0, reverse=True)[:top_k]
        for row in top:
            classification, notes = classify_request(row, global_stats, len(rows))
            outliers.append({
                "repeat": repeat,
                "request_index": row.get("request_index", "unavailable"),
                "latency_us": row.get("latency_us", "unavailable"),
                "gpu_event_us": row.get("gpu_event_us", "unavailable"),
                "release_lateness_us": row.get("release_lateness_us", "unavailable"),
                "scheduled_time_us": row.get("scheduled_time_us", "unavailable"),
                "actual_start_time_us": row.get("actual_start_time_us", "unavailable"),
                "finish_time_us": row.get("finish_time_us", "unavailable"),
                "classification": classification,
                "notes": notes,
            })
    return summaries, outliers, global_stats


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value):
    if isinstance(value, str):
        return value
    return f"{float(value):.3f}"


def write_markdown(path, result_dir, summaries, outliers, global_stats):
    worst_summary = max(summaries, key=lambda r: float(r["p99"]) if r["p99"] != "unavailable" else -1)
    worst_request = max(outliers, key=lambda r: float(r["latency_us"]) if r["latency_us"] != "unavailable" else -1)
    p99_values = [float(r["p99"]) for r in summaries if r["p99"] != "unavailable"]
    p95_values = [float(r["p95"]) for r in summaries if r["p95"] != "unavailable"]
    mean_values = [float(r["mean"]) for r in summaries if r["mean"] != "unavailable"]
    lines = [
        "# CUTLASS Standalone HP Stability Analysis",
        "",
        f"Result directory: `{result_dir}`",
        "",
        "No request was removed, smoothed, or modified. Percentiles use sorted linear interpolation.",
        "",
        "## Summary",
        "",
        f"- Abnormal repeat by P99: repeat `{worst_summary['repeat']}` with P99 `{fmt(worst_summary['p99'])}` us.",
        f"- Slowest request: repeat `{worst_request['repeat']}`, request `{worst_request['request_index']}`, latency `{fmt(worst_request['latency_us'])}` us.",
        f"- Slowest request classification: `{worst_request['classification']}`.",
        f"- P99 CV: `{(stdev(p99_values) / mean(p99_values) if p99_values else 0.0):.4f}`.",
        f"- P95 CV: `{(stdev(p95_values) / mean(p95_values) if p95_values else 0.0):.4f}`.",
        f"- Mean CV: `{(stdev(mean_values) / mean(mean_values) if mean_values else 0.0):.4f}`.",
        "",
        "Engineering reference: P99 CV <= 10% is relatively stable; 10%-20% shows some variance; >20% needs more diagnosis. This is a practical guideline, not a theoretical standard.",
        "",
        "## Repeat Metrics",
        "",
        "| repeat | hp_count | mean us | p50 us | p95 us | p99 us | max us | release p95 us | release p99 us | release max us | GPU mean us | GPU p95 us | GPU p99 us | GPU max us |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['repeat']} | {row['hp_count']} | {fmt(row['mean'])} | {fmt(row['p50'])} | "
            f"{fmt(row['p95'])} | {fmt(row['p99'])} | {fmt(row['max'])} | "
            f"{fmt(row['release_lateness_p95'])} | {fmt(row['release_lateness_p99'])} | "
            f"{fmt(row['release_lateness_max'])} | {fmt(row['gpu_event_mean'])} | "
            f"{fmt(row['gpu_event_p95'])} | {fmt(row['gpu_event_p99'])} | {fmt(row['gpu_event_max'])} |"
        )
    lines.extend([
        "",
        "## Slowest 20 Requests per Repeat",
        "",
        "| repeat | request | latency us | GPU event us | release lateness us | scheduled us | actual start us | finish us | classification | notes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in outliers:
        lines.append(
            f"| {row['repeat']} | {row['request_index']} | {fmt(row['latency_us'])} | "
            f"{fmt(row['gpu_event_us'])} | {fmt(row['release_lateness_us'])} | "
            f"{row['scheduled_time_us']} | {row['actual_start_time_us']} | {row['finish_time_us']} | "
            f"{row['classification']} | {row['notes']} |"
        )
    lines.extend([
        "",
        "## Classification Rules",
        "",
        "- `HOST_RELEASE_JITTER`: latency and release lateness are high while GPU event time is normal.",
        "- `HOST_COMPLETION_JITTER`: latency is high while release lateness and GPU event time are normal.",
        "- `GPU_EXECUTION_JITTER`: latency and GPU event time are both high.",
        "- `INSUFFICIENT_STEADY_STATE_WARMUP`: high-latency request is concentrated in the first 5% of requests.",
        "",
        "## Global Reference Statistics",
        "",
        f"- global latency p99: `{global_stats['latency_p99']:.3f}` us",
        f"- global GPU event p99: `{global_stats['gpu_p99']:.3f}` us",
        f"- global release lateness p99: `{global_stats['release_p99']:.3f}` us",
        "",
        "## Next Diagnosis",
        "",
        "Run the standalone-only 1000-request experiment with optional telemetry. If GPU event time stays stable but host latency spikes, focus on host completion/scheduling jitter. If GPU event spikes align with telemetry P-state or clock changes, focus on GPU execution or power-state stability.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    result_dir = args.result_dir.resolve()
    summaries, outliers, global_stats = build_analysis(result_dir, args.top_k)
    output_csv = Path(args.output_csv) if args.output_csv else result_dir / "standalone_outliers.csv"
    output_md = Path(args.output_md) if args.output_md else result_dir / "standalone_stability_analysis.md"
    summary_csv = result_dir / "standalone_stability_summary.csv"
    write_csv(output_csv, outliers, REQUEST_FIELDS)
    write_csv(summary_csv, summaries, SUMMARY_FIELDS)
    write_markdown(output_md, result_dir, summaries, outliers, global_stats)
    print(f"summary_csv={summary_csv}")
    print(f"outliers_csv={output_csv}")
    print(f"report_md={output_md}")


def self_test():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rows_by_repeat = {}
        for repeat in range(3):
            rows_by_repeat[repeat] = [(idx, 1000, 900, 10) for idx in range(50)]
        rows_by_repeat[0][25] = (25, 8000, 900, 6000)
        rows_by_repeat[1][25] = (25, 7000, 6500, 10)
        rows_by_repeat[2][0] = (0, 6600, 900, 10)
        for repeat, rows in rows_by_repeat.items():
            out = root / "standalone_hp" / f"repeat_{repeat}" / "hp"
            out.mkdir(parents=True)
            with (out / "output.jsonl").open("w", encoding="utf-8") as fobj:
                for idx, latency, gpu, release in rows:
                    fobj.write(json.dumps({
                        "type": "request",
                        "request_index": idx,
                        "latency_us": latency,
                        "gpu_event_us": gpu,
                        "release_lateness_us": release,
                        "scheduled_time_us": idx * 30000,
                        "actual_start_time_us": idx * 30000 + release,
                        "finish_time_us": idx * 30000 + release + latency,
                    }) + "\n")
        summaries, outliers, _ = build_analysis(root, 3)
        classes = {row["classification"] for row in outliers}
        assert "HOST_RELEASE_JITTER" in classes
        assert "GPU_EXECUTION_JITTER" in classes
        assert any("INSUFFICIENT_STEADY_STATE_WARMUP" in row["notes"] or
                   row["classification"] == "INSUFFICIENT_STEADY_STATE_WARMUP"
                   for row in outliers)
        assert len(summaries) == 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-csv")
    parser.add_argument("--output-md")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.result_dir:
        parser.error("--result-dir is required unless --self-test is used")
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_history(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("history.json must contain a list of saved analyses.")
    return data


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    weight = idx - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def study_rows(items: list[dict], max_rows: int) -> list[dict]:
    rows = []
    seen_domains: set[str] = set()
    for item in sorted(items, key=lambda x: x.get("createdAt", ""), reverse=True):
        report = item.get("report", {})
        profiling = report.get("profiling") or {}
        if not profiling:
            continue
        domain = report.get("detectedDomain", "unknown")
        if domain in seen_domains:
            continue
        stages = profiling.get("stages") or []
        slowest = None
        if stages:
            slowest = max(stages, key=lambda x: x.get("elapsedMs", 0.0))
        rows.append(
            {
                "domain": domain,
                "label": report.get("detectedLabel", "unknown"),
                "device": profiling.get("deviceIdentifier", "unknown"),
                "perception_ms": profiling.get("perceptionMs"),
                "reasoning_ms": profiling.get("reasoningMs"),
                "end_to_end_ms": profiling.get("endToEndMs"),
                "slowest_stage": slowest.get("title", "--") if slowest else "--",
                "slowest_stage_ms": slowest.get("elapsedMs") if slowest else None,
            }
        )
        seen_domains.add(domain)
        if len(rows) >= max_rows:
            break
    return rows


def domain_summary(items: list[dict]) -> list[dict]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[str, str] = {}
    devices: dict[str, str] = {}
    for item in items:
        report = item.get("report", {})
        profiling = report.get("profiling") or {}
        if not profiling:
            continue
        domain = report.get("detectedDomain", "unknown")
        labels[domain] = report.get("detectedLabel", "unknown")
        devices[domain] = profiling.get("deviceIdentifier", "unknown")
        for key in ("perceptionMs", "reasoningMs", "endToEndMs"):
            value = profiling.get(key)
            if isinstance(value, (int, float)):
                buckets[domain][key].append(float(value))
    rows = []
    for domain, metrics in buckets.items():
        end_values = metrics.get("endToEndMs", [])
        rows.append(
            {
                "domain": domain,
                "label": labels.get(domain, "unknown"),
                "device": devices.get(domain, "unknown"),
                "count": max(len(end_values), len(metrics.get("perceptionMs", [])), len(metrics.get("reasoningMs", []))),
                "perception_mean": sum(metrics.get("perceptionMs", [0.0])) / max(len(metrics.get("perceptionMs", [])), 1),
                "reasoning_mean": sum(metrics.get("reasoningMs", [0.0])) / max(len(metrics.get("reasoningMs", [])), 1),
                "end_mean": sum(end_values) / max(len(end_values), 1),
                "end_p95": percentile(end_values, 0.95) if end_values else None,
            }
        )
    rows.sort(key=lambda x: x["end_mean"], reverse=True)
    return rows


def stage_summary(items: list[dict]) -> list[dict]:
    bucket: dict[str, list[float]] = defaultdict(list)
    titles: dict[str, str] = {}
    for item in items:
        profiling = item.get("report", {}).get("profiling") or {}
        for stage in profiling.get("stages") or []:
            stage_id = stage.get("stageID", "unknown")
            titles[stage_id] = stage.get("title", stage_id)
            elapsed = stage.get("elapsedMs")
            if isinstance(elapsed, (int, float)):
                bucket[stage_id].append(float(elapsed))
    rows = []
    for stage_id, values in bucket.items():
        rows.append(
            {
                "stage_id": stage_id,
                "title": titles.get(stage_id, stage_id),
                "count": len(values),
                "mean_ms": sum(values) / len(values),
                "p50_ms": percentile(values, 0.50),
                "p95_ms": percentile(values, 0.95),
            }
        )
    rows.sort(key=lambda x: x["mean_ms"], reverse=True)
    return rows


def build_latex(items: list[dict], max_rows: int) -> str:
    representative = study_rows(items, max_rows=max_rows)
    domains = domain_summary(items)
    summary = stage_summary(items)
    lines: list[str] = []
    lines.append("% Auto-generated from PULSE iPhone history.json profiling records.")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Representative on-device iPhone latency measurements from saved PULSE analyses.}")
    lines.append("\\label{tab:iphone_latency_representative}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrl}")
    lines.append("\\toprule")
    lines.append("\\textbf{Domain} & \\textbf{Primary result} & \\textbf{Perception (ms)} & \\textbf{Reasoning (ms)} & \\textbf{End-to-end (ms)} & \\textbf{Slowest stage (ms)} & \\textbf{Device} \\\\")
    lines.append("\\midrule")
    for row in representative:
        lines.append(
            f"{latex_escape(row['domain'])} & "
            f"{latex_escape(row['label'])} & "
            f"{fmt_ms(row['perception_ms'])} & "
            f"{fmt_ms(row['reasoning_ms'])} & "
            f"{fmt_ms(row['end_to_end_ms'])} & "
            f"{latex_escape(row['slowest_stage'])} ({fmt_ms(row['slowest_stage_ms'])}) & "
            f"{latex_escape(row['device'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Domain-level iPhone latency summary over saved PULSE analyses.}")
    lines.append("\\label{tab:iphone_latency_domains}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Domain} & \\textbf{Representative output} & \\textbf{n} & \\textbf{Perception mean (ms)} & \\textbf{Reasoning mean (ms)} & \\textbf{End-to-end mean / p95 (ms)} \\\\")
    lines.append("\\midrule")
    for row in domains:
        lines.append(
            f"{latex_escape(row['domain'])} & "
            f"{latex_escape(row['label'])} & "
            f"{row['count']} & "
            f"{fmt_ms(row['perception_mean'])} & "
            f"{fmt_ms(row['reasoning_mean'])} & "
            f"{fmt_ms(row['end_mean'])} / {fmt_ms(row['end_p95'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Stage-level latency summary over profiled iPhone runs.}")
    lines.append("\\label{tab:iphone_latency_stages}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Stage} & \\textbf{Count} & \\textbf{Mean (ms)} & \\textbf{p50 (ms)} & \\textbf{p95 (ms)} \\\\")
    lines.append("\\midrule")
    for row in summary:
        lines.append(
            f"{latex_escape(row['title'])} & {row['count']} & {fmt_ms(row['mean_ms'])} & {fmt_ms(row['p50_ms'])} & {fmt_ms(row['p95_ms'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LaTeX iPhone profiling tables from PULSE history.json.")
    parser.add_argument("--history-json", required=True, help="Path to exported PULSE history.json from the iPhone app container.")
    parser.add_argument("--output-tex", required=True, help="Where to write the LaTeX table snippet.")
    parser.add_argument("--max-rows", type=int, default=8, help="Maximum number of representative study rows.")
    args = parser.parse_args()

    history_path = Path(args.history_json).expanduser().resolve()
    output_path = Path(args.output_tex).expanduser().resolve()

    items = load_history(history_path)
    tex = build_latex(items, max_rows=args.max_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

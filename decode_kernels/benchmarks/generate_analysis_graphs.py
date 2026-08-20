"""Generate reproducible benchmark and Nsight Systems figures.

Inputs are checked-in machine-readable artifacts. No benchmark or profiler
numbers are embedded in this script.
"""

from __future__ import annotations

import json
import math
import os
import hashlib
import sqlite3
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cuda-ml-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
NSYS = ARTIFACTS / "nsys"
OUTPUT = ARTIFACTS / "graphs"

BG = "#07131c"
PANEL = "#0d1d29"
GRID = "#28404f"
TEXT = "#e8f2f6"
MUTED = "#8da7b5"
TEAL = "#23d5ab"
CYAN = "#38bdf8"
GOLD = "#f6c85f"
CORAL = "#ff7b72"
PURPLE = "#b69cff"

LABELS = {
    "residual_rms_decode_b1": "Residual RMSNorm\nB=1",
    "residual_rms_decode_b32": "Residual RMSNorm\nB=32",
    "rms_quant_decode_b32": "RMSNorm -> INT8\nB=32",
    "rope_gqa_decode_b32": "QK norm + RoPE\nGQA B=32",
    "kv_append_decode_b32": "KV append\nB=32",
    "swiglu_decode_b1": "SwiGLU\nB=1",
    "swiglu_decode_b32": "SwiGLU\nB=32",
    "sampling_decode_b32": "Sampling\nB=32",
    "small_n_linear_b1": "Linear\nB=1",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def title(fig, heading: str, subheading: str) -> None:
    fig.text(0.055, 0.965, heading, ha="left", va="top", fontsize=17, fontweight="bold", color=TEXT)
    fig.text(0.055, 0.925, subheading, ha="left", va="top", fontsize=9.5, color=MUTED)


def save(fig, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def cloud_records() -> list[dict]:
    payload = json.loads((ARTIFACTS / "modal_a10g_decode_validation.json").read_text())
    return payload["benchmark"]["results"]


def cloud_graphs(records: list[dict]) -> None:
    labels = [LABELS[row["trace"]["name"]] for row in records]
    speedups = [row["speedup_p50"] for row in records]
    colors = [TEAL if value > 1.05 else MUTED for value in speedups]

    fig, ax = plt.subplots(figsize=(13.2, 6.3))
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.20)
    title(fig, "Cloud A10 decode speedups", "Device-only p50 CUDA Event time; custom path versus independent PyTorch reference; 50 samples")
    x = np.arange(len(labels))
    bars = ax.bar(x, speedups, color=colors, width=0.68)
    ax.axhline(1.0, color=GOLD, linewidth=1.2, linestyle="--")
    ax.set_ylabel("Speedup (x)")
    ax.set_xticks(x, labels, rotation=0, fontsize=8.5)
    ax.set_ylim(0, max(speedups) * 1.13)
    ax.grid(axis="x", visible=False)
    for bar, value in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(speedups) * 0.025, f"{value:.2f}x", ha="center", fontsize=9, color=TEXT)
    ax.text(8.35, 1.45, "ATen/cuBLAS baselines retained", ha="right", color=GOLD, fontsize=9)
    save(fig, "01_cloud_a10_speedups")

    hot = np.array([row["custom_warm"]["p50_us"] for row in records])
    cold = np.array([row["custom_cold_cache"]["p50_us"] for row in records])
    reference = np.array([row["reference_warm"]["p50_us"] for row in records])
    fig, ax = plt.subplots(figsize=(13.2, 6.5))
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.20)
    title(fig, "A10 latency anatomy", "Hot-cache custom, L2-flushed custom, and hot-cache reference; logarithmic microsecond scale")
    width = 0.24
    ax.bar(x - width, hot, width, color=TEAL, label="Custom / hot")
    ax.bar(x, cold, width, color=CYAN, label="Custom / L2 flushed")
    ax.bar(x + width, reference, width, color=PURPLE, label="PyTorch reference / hot")
    ax.set_yscale("log")
    ax.set_ylabel("Latency (us, log scale)")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "02_cloud_a10_hot_cold_latency")

    p20 = np.array([row["custom_warm"]["p20_us"] for row in records])
    p50 = np.array([row["custom_warm"]["p50_us"] for row in records])
    p80 = np.array([row["custom_warm"]["p80_us"] for row in records])
    fig, ax = plt.subplots(figsize=(13.2, 6.3))
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.20)
    title(fig, "A10 steady-state variability", "P20-P80 CUDA Event interval around the p50; narrow bands indicate stable device execution")
    ax.errorbar(x, p50, yerr=(p50 - p20, p80 - p50), fmt="o", color=TEAL, ecolor=CYAN, capsize=5, linewidth=2)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (us, log scale)")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, "03_cloud_a10_latency_variability")


def pascal_ranges() -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(NSYS / "decode_gtx1050ti_stats_nvtxgpuproj.csv")
    iteration_rows = frame[frame["Name"].str.startswith("custom_iteration::", na=False)]
    iterations = int(iteration_rows["Name"].str.rsplit("::", n=1).str[-1].astype(int).max() + 1)
    outer = frame[frame["Name"].str.match(r"^(custom|reference)::", na=False)].copy()
    outer[["kind", "trace"]] = outer["Name"].str.split("::", n=1, expand=True)
    outer["gpu_span_us_per_iteration"] = outer["Projected Duration (ns)"] / iterations / 1_000.0
    outer["host_enqueue_us_per_iteration"] = outer["Orig Duration (ns)"] / iterations / 1_000.0
    outer["gpu_ops_per_iteration"] = outer["NumGPUOps"] / iterations
    return outer, iterations


def pascal_range_graphs(outer: pd.DataFrame, iterations: int) -> list[dict]:
    custom = outer[outer.kind == "custom"].set_index("trace")
    reference = outer[outer.kind == "reference"].set_index("trace")
    names = list(LABELS)
    custom_span = np.array([custom.loc[name, "gpu_span_us_per_iteration"] for name in names])
    reference_span = np.array([reference.loc[name, "gpu_span_us_per_iteration"] for name in names])
    speedups = reference_span / custom_span
    labels = [LABELS[name] for name in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(13.2, 6.3))
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.20)
    title(fig, "Pascal Nsight GPU-range speedups", f"GTX 1050 Ti projected GPU span per NVTX iteration; {iterations} profiled iterations per path")
    bars = ax.bar(x, speedups, color=[TEAL if value > 1.05 else MUTED for value in speedups], width=0.68)
    ax.axhline(1.0, color=GOLD, linestyle="--", linewidth=1.2)
    ax.set_ylabel("Projected GPU span speedup (x)")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, max(speedups) * 1.18)
    ax.grid(axis="x", visible=False)
    for bar, value in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(speedups) * 0.025, f"{value:.2f}x", ha="center", fontsize=9)
    save(fig, "04_pascal_nsys_range_speedups")

    custom_ops = np.array([custom.loc[name, "gpu_ops_per_iteration"] for name in names])
    reference_ops = np.array([reference.loc[name, "gpu_ops_per_iteration"] for name in names])
    fig, ax = plt.subplots(figsize=(13.2, 6.5))
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.20)
    title(fig, "Fusion removes launch multiplicity", "GPU operations projected into each NVTX iteration by Nsight Systems")
    width = 0.34
    ax.bar(x - width / 2, custom_ops, width, color=TEAL, label="Custom path")
    ax.bar(x + width / 2, reference_ops, width, color=PURPLE, label="PyTorch reference")
    ax.set_ylabel("GPU operations per iteration")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.legend(ncol=2, loc="upper left")
    ax.grid(axis="x", visible=False)
    for index, (custom_value, reference_value) in enumerate(zip(custom_ops, reference_ops)):
        ax.text(index - width / 2, custom_value + 0.7, f"{custom_value:.0f}", ha="center", fontsize=8)
        ax.text(index + width / 2, reference_value + 0.7, f"{reference_value:.0f}", ha="center", fontsize=8)
    save(fig, "05_pascal_nsys_gpu_ops_per_iteration")

    return [
        {
            "trace": name,
            "custom_gpu_span_us": float(custom.loc[name, "gpu_span_us_per_iteration"]),
            "reference_gpu_span_us": float(reference.loc[name, "gpu_span_us_per_iteration"]),
            "nsys_speedup": float(speedups[index]),
            "custom_gpu_ops": float(custom_ops[index]),
            "reference_gpu_ops": float(reference_ops[index]),
            "custom_host_enqueue_us": float(custom.loc[name, "host_enqueue_us_per_iteration"]),
            "reference_host_enqueue_us": float(reference.loc[name, "host_enqueue_us_per_iteration"]),
        }
        for index, name in enumerate(names)
    ]


def category(name: str) -> str:
    if "residual_rms_norm_kernel" in name:
        return "Custom residual RMSNorm"
    if "rms_norm_quantize_kernel" in name:
        return "Custom RMSNorm -> INT8"
    if "rope_qk_norm_kernel" in name:
        return "Custom QK norm + RoPE"
    if "kv_cache_append_kernel" in name:
        return "Custom KV append"
    if "bias_swiglu_kernel" in name:
        return "Custom SwiGLU"
    if "gemv2T" in name or "gemm" in name.lower():
        return "cuBLAS GEMV/GEMM"
    if "mbtopk" in name or "radixSort" in name or "topk" in name.lower():
        return "Top-k / radix selection"
    if "softmax" in name.lower():
        return "Softmax"
    if "reduce" in name.lower() or "scan" in name.lower():
        return "Reductions / scans"
    if "copy" in name.lower() or "cast" in name.lower():
        return "Copies / casts"
    if "distribution" in name.lower() or "curand" in name.lower():
        return "Random generation"
    if "elementwise" in name.lower() or "kernel_impl" in name.lower():
        return "ATen elementwise"
    return "Other CUDA"


def pascal_kernel_graphs() -> list[dict]:
    kernels = pd.read_csv(NSYS / "decode_gtx1050ti_stats_gpukernsum.csv")
    kernels["Category"] = kernels["Name"].map(category)
    grouped = kernels.groupby("Category", as_index=False).agg({"Total Time (ns)": "sum", "Instances": "sum"})
    grouped = grouped.sort_values("Total Time (ns)", ascending=True)

    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    fig.subplots_adjust(top=0.84, left=0.26, right=0.96, bottom=0.12)
    title(fig, "Pascal GPU-time composition", "All captured kernels, including warmup and profiled custom/reference ranges; Nsight kernel activities")
    colors = [TEAL if value.startswith("Custom") else PURPLE if value.startswith("cuBLAS") else CYAN for value in grouped.Category]
    bars = ax.barh(grouped.Category, grouped["Total Time (ns)"] / 1e6, color=colors)
    ax.set_xlabel("Total GPU kernel time (ms)")
    ax.grid(axis="y", visible=False)
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2, f"{row['Instances']:.0f} launches", va="center", fontsize=8.5, color=MUTED)
    save(fig, "06_pascal_nsys_kernel_time_mix")

    execution = pd.read_csv(NSYS / "decode_gtx1050ti_stats_kernexecsum.csv")
    selections = [
        ("Residual RMSNorm", "residual_rms_norm_kernel"),
        ("RMSNorm -> INT8", "rms_norm_quantize_kernel"),
        ("QK norm + RoPE", "rope_qk_norm_kernel"),
        ("KV append", "kv_cache_append_kernel"),
        ("SwiGLU", "bias_swiglu_kernel"),
        ("cuBLAS GEMV", "gemv2T"),
        ("Top-k radix", "radixFindKthValues"),
    ]
    rows = []
    for label, pattern in selections:
        matches = execution[execution["Kernel Name"].str.contains(pattern, regex=False, na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            rows.append((label, row["AAvg (ns)"] / 1e3, row["QAvg (ns)"] / 1e3, row["KAvg (ns)"] / 1e3))

    fig, ax = plt.subplots(figsize=(12.4, 6.8))
    fig.subplots_adjust(top=0.82, left=0.11, right=0.97, bottom=0.18)
    x = np.arange(len(rows))
    width = 0.24
    ax.bar(x - width, [row[1] for row in rows], width, color=GOLD, label="CUDA API")
    ax.bar(x, [max(row[2], 0.1) for row in rows], width, color=CORAL, label="Queue wait")
    ax.bar(x + width, [row[3] for row in rows], width, color=TEAL, label="Kernel execution")
    ax.set_yscale("log")
    ax.set_ylabel("Average time (us, log scale)")
    ax.set_xticks(x, [row[0] for row in rows], rotation=18, ha="right", fontsize=9)
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    title(fig, "Launch-to-execution pipeline under queued load", "Nsight kernel-execution correlation; queue wait reflects the intentionally asynchronous profiling workload")
    save(fig, "07_pascal_nsys_launch_queue_anatomy")

    return [
        {
            "category": row.Category,
            "total_gpu_ms": float(row["Total Time (ns)"] / 1e6),
            "instances": int(row.Instances),
        }
        for _, row in grouped.sort_values("Total Time (ns)", ascending=False).iterrows()
    ]


def timeline_graph() -> None:
    frame = pd.read_csv(NSYS / "decode_gtx1050ti_stats_nvtxgpuproj.csv")
    frame = frame[frame["Name"].str.startswith("custom_iteration::", na=False)].copy()
    pieces = frame["Name"].str.split("::", expand=True)
    frame["trace"] = pieces[1]
    frame["iteration"] = pieces[2].astype(int)
    origin = frame["Projected Start (ns)"].min()
    frame["start_ms"] = (frame["Projected Start (ns)"] - origin) / 1e6
    frame["duration_ms"] = frame["Projected Duration (ns)"] / 1e6
    names = list(LABELS)
    y_for = {name: len(names) - 1 - index for index, name in enumerate(names)}
    palette = [TEAL, CYAN, GOLD, PURPLE, CORAL, "#70e1f5", "#a8e063", "#fca5a5", "#c4b5fd"]

    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    fig.subplots_adjust(top=0.83, left=0.18, right=0.97, bottom=0.12)
    title(fig, "Nsight projected GPU timeline", "GTX 1050 Ti custom paths; each bar is one NVTX iteration projected over its GPU operations")
    for color, name in zip(palette, names):
        subset = frame[frame.trace == name]
        ax.broken_barh(list(zip(subset.start_ms, subset.duration_ms)), (y_for[name] - 0.33, 0.66), facecolors=color)
    ax.set_yticks([y_for[name] for name in names], [LABELS[name].replace("\n", " ") for name in names])
    ax.set_xlabel("Time from first profiled custom GPU operation (ms)")
    ax.grid(axis="y", visible=False)
    save(fig, "08_pascal_nsys_projected_timeline")


def modal_nsys_graph() -> dict:
    api = pd.read_csv(NSYS / "decode_a10_nsys_stats.csv_cuda_api_sum.csv")
    api["Total ms"] = api["Total Time (ns)"] / 1e6
    top = api.nlargest(8, "Total Time (ns)").sort_values("Total Time (ns)")
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    fig.subplots_adjust(top=0.82, left=0.23, right=0.96, bottom=0.13)
    title(fig, "Modal A10 Nsight CUDA API view", "Real CUDA API trace; Modal/gVisor exposed no GPU workload activities, so no kernel durations are inferred here")
    bars = ax.barh(top.Name, top["Total ms"], color=[CORAL if name == "cudaLaunchKernel" else CYAN for name in top.Name])
    ax.set_xlabel("Total traced host API time (ms)")
    ax.grid(axis="y", visible=False)
    for bar, calls in zip(bars, top["Num Calls"]):
        ax.text(bar.get_width() + max(top["Total ms"]) * 0.015, bar.get_y() + bar.get_height() / 2, f"{calls:.0f} calls", va="center", fontsize=8.5, color=MUTED)
    save(fig, "09_modal_a10_nsys_cuda_api_mix")

    diagnostics = []
    with sqlite3.connect(NSYS / "decode_a10_nsys.sqlite") as connection:
        if connection.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='DIAGNOSTIC_EVENT'").fetchone()[0]:
            diagnostics = [row[0] for row in connection.execute("SELECT text FROM DIAGNOSTIC_EVENT")]
    return {
        "cuda_api_calls": int(api["Num Calls"].sum()),
        "cuda_launch_calls": int(api.loc[api.Name == "cudaLaunchKernel", "Num Calls"].sum()),
        "gpu_kernel_table_rows": 0,
        "diagnostics": diagnostics,
    }


def write_summary(cloud: list[dict], pascal_ranges_summary: list[dict], kernel_categories: list[dict], modal_summary: dict, iterations: int) -> None:
    summary = {
        "schema_version": 1,
        "sources": {
            "cloud_benchmark": "artifacts/modal_a10g_decode_validation.json",
            "modal_nsys": "artifacts/nsys/decode_a10_nsys.nsys-rep",
            "pascal_nsys": "artifacts/nsys/decode_gtx1050ti_nsys.nsys-rep",
        },
        "cloud_a10": [
            {
                "trace": row["trace"]["name"],
                "custom_hot_p50_us": row["custom_warm"]["p50_us"],
                "custom_cold_p50_us": row["custom_cold_cache"]["p50_us"],
                "reference_hot_p50_us": row["reference_warm"]["p50_us"],
                "speedup": row["speedup_p50"],
            }
            for row in cloud
        ],
        "pascal_nsys": {
            "iterations_per_range": iterations,
            "ranges": pascal_ranges_summary,
            "kernel_categories": kernel_categories,
        },
        "modal_nsys": modal_summary,
        "caveats": [
            "A10 speedups use queued device-only CUDA Events from a separate Modal run.",
            "Modal/gVisor provided CUDA API and NVTX traces but no GPU workload activity table to Nsight Systems.",
            "Kernel-level Nsight evidence is therefore reported separately on the physical GTX 1050 Ti.",
            "Synthetic representative shapes are not captured production traffic.",
        ],
    }
    (ARTIFACTS / "decode_performance_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")


def write_nsys_manifest() -> None:
    files = []
    for path in sorted(NSYS.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})

    table_counts = {}
    for label, database in {
        "modal_a10": NSYS / "decode_a10_nsys.sqlite",
        "gtx1050ti": NSYS / "decode_gtx1050ti_nsys.sqlite",
    }.items():
        with sqlite3.connect(database) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            table_counts[label] = {
                "cuda_kernel_activities": connection.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
                if "CUPTI_ACTIVITY_KIND_KERNEL" in tables
                else 0,
                "cuda_runtime_activities": connection.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
                if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables
                else 0,
                "nvtx_events": connection.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()[0]
                if "NVTX_EVENTS" in tables
                else 0,
            }
    manifest = {
        "schema_version": 1,
        "profilers": {
            "modal_a10": "NVIDIA Nsight Systems 2026.4.1.191; explicit cuda-sw mode",
            "gtx1050ti": "NVIDIA Nsight Systems 2022.4.2.50; CUDA 11.8 container",
        },
        "table_counts": table_counts,
        "files": files,
    }
    (NSYS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    configure()
    cloud = cloud_records()
    cloud_graphs(cloud)
    outer, iterations = pascal_ranges()
    range_summary = pascal_range_graphs(outer, iterations)
    kernel_categories = pascal_kernel_graphs()
    timeline_graph()
    modal_summary = modal_nsys_graph()
    write_summary(cloud, range_summary, kernel_categories, modal_summary, iterations)
    write_nsys_manifest()
    print(f"Generated graphs and analysis under {OUTPUT} and {ARTIFACTS}")


if __name__ == "__main__":
    main()

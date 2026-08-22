"""Render editable Rank 3/4 benchmark graphs from the checked-in JSON evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts" / "gtx1050ti_rank34_benchmarks.json"
OUTPUT = ROOT / "artifacts" / "graphs"
COLORS = {"paged_fp16": "#46c2b3", "paged_int8": "#f2a65a", "dense": "#66778a"}


def style(axis, title, subtitle):
    axis.set_facecolor("#101820")
    axis.figure.set_facecolor("#101820")
    axis.set_title(title, color="#f4f7f8", loc="left", fontsize=15, weight="bold", pad=18)
    axis.text(0, 1.02, subtitle, transform=axis.transAxes, color="#9aabb7", fontsize=9)
    axis.tick_params(colors="#c4d0d6")
    for spine in axis.spines.values(): spine.set_color("#33434d")
    axis.grid(axis="y", color="#2a3942", alpha=.65)


def save(figure, name):
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(OUTPUT / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA.read_text())
    paged = [row for row in data["records"] if row["family"] == "rank3_paged_decode"]
    sequences = sorted({row["sequence"] for row in paged})
    x = np.arange(len(sequences)); width = .34

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for offset, variant in ((-.17, "paged_fp16"), (.17, "paged_int8")):
        rows = [next(row for row in paged if row["sequence"] == seq and row["variant"] == variant) for seq in sequences]
        bars = ax.bar(x + offset, [row["speedup_vs_dense_sdpa"] for row in rows], width, color=COLORS[variant], label=variant.replace("paged_", "").upper())
        ax.bar_label(bars, fmt="%.2f×", color="#dce5e9", fontsize=9, padding=3)
    ax.axhline(1, color="#eef3f5", linewidth=1)
    ax.set_xticks(x, [str(value) for value in sequences]); ax.set_xlabel("Ragged sequence length", color="#c4d0d6")
    ax.set_ylabel("Speedup vs dense PyTorch SDPA", color="#c4d0d6"); ax.legend(frameon=False, labelcolor="#dce5e9")
    style(ax, "Rank 3 paged-decode speedups", "GTX 1050 Ti · B1 · 8 Q heads / 2 KV heads · D64 · CUDA Events")
    save(fig, "10_rank3_paged_speedups")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    dense = [next(row for row in paged if row["sequence"] == seq)["dense_sdpa_latency_us"] for seq in sequences]
    ax.plot(sequences, dense, "o-", color=COLORS["dense"], linewidth=2.2, label="Dense SDPA")
    for variant in ("paged_fp16", "paged_int8"):
        rows = [next(row for row in paged if row["sequence"] == seq and row["variant"] == variant) for seq in sequences]
        ax.plot(sequences, [row["latency_us"] for row in rows], "o-", color=COLORS[variant], linewidth=2.2, label=variant.replace("paged_", "Paged ").upper())
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(sequences, [str(value) for value in sequences]); ax.set_xlabel("Sequence length", color="#c4d0d6")
    ax.set_ylabel("Latency (µs, log₂)", color="#c4d0d6"); ax.legend(frameon=False, labelcolor="#dce5e9")
    style(ax, "Rank 3 latency scaling", "Split-KV increases 1 → 2 → 8; workspace and page table are preallocated")
    save(fig, "11_rank3_latency_scaling")

    dspark = next(row for row in data["records"] if row["family"] == "rank4_dspark_block")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(["Eager dispatch", "CUDA Graph replay"], [dspark["eager_latency_us"], dspark["latency_us"]], color=[COLORS["dense"], COLORS["paged_fp16"]], width=.58)
    ax.bar_label(bars, fmt="%.1f µs", color="#dce5e9", fontsize=10, padding=4)
    ax.text(.5, .88, f"{dspark['speedup_vs_eager']:.3f}×", transform=ax.transAxes, ha="center", color="#f2a65a", fontsize=18, weight="bold")
    ax.set_ylabel("Seven-token draft-block latency", color="#c4d0d6")
    style(ax, "Rank 4 DSpark graph replay", "GTX 1050 Ti · B1 · K7 · V4096 · rank64 · fused Pascal correction+argmax")
    save(fig, "12_rank4_graph_replay")

    projection = pd.read_csv(ROOT / "artifacts" / "nsys" / "rank34_gtx1050ti_stats_nvtxgpuproj.csv")
    names = ["rank3/dense_sdpa/S2048", "rank3/paged_fp16/S2048/split8", "rank3/paged_int8/S2048/split8", "rank4/dspark_eager/B1_K7_V4096_R64"]
    labels = ["Dense SDPA", "Paged FP16", "Paged INT8", "DSpark eager K7"]
    medians = [projection.loc[projection.Name == name, "Projected Duration (ns)"].median() / 1000 for name in names]
    fig, ax = plt.subplots(figsize=(8.7, 4.9))
    bars = ax.bar(labels, medians, color=[COLORS["dense"], COLORS["paged_fp16"], COLORS["paged_int8"], "#b58ad7"], width=.64)
    ax.bar_label(bars, labels=[f"{value:,.0f} µs" for value in medians], color="#dce5e9", fontsize=9, padding=4)
    ax.set_yscale("log"); ax.set_ylabel("Median NVTX-projected GPU span (µs, log)", color="#c4d0d6")
    style(ax, "Rank 3/4 Nsight GPU projection", "12 physical traces · S2048 split8 · DSpark B1 K7 · graph replay excluded: Nsight 2022.4 does not project captured child kernels")
    save(fig, "13_rank34_nsys_gpu_projection")

    kernels = pd.read_csv(ROOT / "artifacts" / "nsys" / "rank34_gtx1050ti_stats_gpukernsum.csv")
    categories = {"Dense SDPA": 0.0, "Paged FP16 split": 0.0, "Paged INT8 split": 0.0, "Split reduction": 0.0, "DSpark fused greedy": 0.0, "Other": 0.0}
    for _, row in kernels.iterrows():
        name = row["Name"]; value = row["Total Time (ns)"] / 1e6
        if "fmha_cutlass" in name: category = "Dense SDPA"
        elif "paged_attention_split" in name and "signed char" in name: category = "Paged INT8 split"
        elif "paged_attention_split" in name: category = "Paged FP16 split"
        elif "paged_attention_reduce" in name: category = "Split reduction"
        elif "markov_greedy" in name: category = "DSpark fused greedy"
        else: category = "Other"
        categories[category] += value
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ordered = list(categories); values = [categories[key] for key in ordered]
    colors = [COLORS["dense"], COLORS["paged_fp16"], COLORS["paged_int8"], "#d7cb75", "#b58ad7", "#4a5963"]
    bars = ax.barh(ordered[::-1], values[::-1], color=colors[::-1])
    ax.bar_label(bars, labels=[f"{v:.2f} ms" for v in values[::-1]], color="#dce5e9", fontsize=9, padding=4)
    ax.set_xscale("log"); ax.set_xlabel("Aggregate kernel time across trace (ms, log)", color="#c4d0d6")
    style(ax, "Nsight kernel-time composition", "Equal 12-range workload; setup kernels grouped as Other · stage-1 attention dominates both paged paths")
    save(fig, "14_rank34_nsys_kernel_composition")

    modal = json.loads((ROOT / "artifacts" / "modal_a10g_rank1_to_rank4_validation.json").read_text())["rank34_benchmark"]
    a10_paged = [row for row in modal["records"] if row["family"] == "rank3_paged_decode" and row["variant"] == "paged_fp16"]
    pascal_paged = [row for row in paged if row["variant"] == "paged_fp16"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars_pascal = ax.bar(x - .18, [row["speedup_vs_dense_sdpa"] for row in pascal_paged], .36, color=COLORS["paged_fp16"], label="GTX 1050 Ti")
    bars_a10 = ax.bar(x + .18, [row["speedup_vs_dense_sdpa"] for row in a10_paged], .36, color="#b58ad7", label="A10")
    ax.bar_label(bars_pascal, fmt="%.2f×", color="#dce5e9", fontsize=9, padding=3)
    ax.bar_label(bars_a10, fmt="%.2f×", color="#dce5e9", fontsize=9, padding=3)
    ax.axhline(1, color="#eef3f5", linewidth=1); ax.set_yscale("log")
    ax.set_xticks(x, [str(value) for value in sequences]); ax.set_xlabel("Sequence length", color="#c4d0d6")
    ax.set_ylabel("Paged FP16 speedup vs contiguous SDPA (log)", color="#c4d0d6"); ax.legend(frameon=False, labelcolor="#dce5e9")
    style(ax, "Rank 3 architecture crossover", "Same B1 Hq8 Hkv2 D64 contract · direct paged access wins on Pascal; tuned contiguous SDPA wins on A10")
    save(fig, "15_rank3_cross_gpu_crossover")

    a10_dspark = next(row for row in modal["records"] if row["family"] == "rank4_dspark_block")
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    labels = ["GTX 1050 Ti", "A10"]
    eager_values = [dspark["eager_latency_us"], a10_dspark["eager_latency_us"]]
    graph_values = [dspark["latency_us"], a10_dspark["latency_us"]]
    x2 = np.arange(2)
    ebar = ax.bar(x2 - .18, eager_values, .36, color=COLORS["dense"], label="Eager")
    gbar = ax.bar(x2 + .18, graph_values, .36, color=COLORS["paged_fp16"], label="CUDA Graph")
    ax.bar_label(ebar, fmt="%.1f µs", color="#dce5e9", fontsize=9, padding=3)
    ax.bar_label(gbar, fmt="%.1f µs", color="#dce5e9", fontsize=9, padding=3)
    ax.set_xticks(x2, labels); ax.set_ylabel("Seven-token DSpark block latency", color="#c4d0d6")
    ax.legend(frameon=False, labelcolor="#dce5e9")
    style(ax, "Rank 4 graph replay across GPUs", f"Pascal {dspark['speedup_vs_eager']:.3f}× · A10 {a10_dspark['speedup_vs_eager']:.2f}× · B1 K7 V4096 rank64")
    save(fig, "16_rank4_cross_gpu_graph_replay")


if __name__ == "__main__": main()

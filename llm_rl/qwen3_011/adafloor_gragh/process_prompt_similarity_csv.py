#!/usr/bin/env python3
"""Filter comparable prompt similarity rows and generate summary plots."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _default_filtered_path(input_csv: Path) -> Path:
    stem = input_csv.stem
    return input_csv.with_name(f"{stem}_comparable_only.csv")


def _default_position_plot_path(input_csv: Path) -> Path:
    stem = input_csv.stem
    return input_csv.with_name(f"{stem}_position_mean.png")


def _default_layer_plot_path(input_csv: Path) -> Path:
    stem = input_csv.stem
    return input_csv.with_name(f"{stem}_random5layers.png")


def _filter_comparable(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["route_rows_a"] > 0) & (df["route_rows_b"] > 0)].copy()


def _plot_position_mean(df: pd.DataFrame, output_path: Path) -> None:
    grouped = (
        df.groupby("token_position", as_index=False)[
            ["cosine_similarity", "jaccard_selected_experts"]
        ]
        .mean()
        .sort_values("token_position")
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(grouped["token_position"],
                 grouped["cosine_similarity"],
                 color="#1f77b4",
                 linewidth=1.2)
    axes[0].set_ylabel("Mean Cosine")
    axes[0].set_title("Mean Cosine Similarity Across 48 Layers by Token Position")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(grouped["token_position"],
                 grouped["jaccard_selected_experts"],
                 color="#d62728",
                 linewidth=1.2)
    axes[1].set_xlabel("Token Position")
    axes[1].set_ylabel("Mean Jaccard")
    axes[1].set_title("Mean Jaccard Across 48 Layers by Token Position")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pick_layers(df: pd.DataFrame, *, num_layers: int, seed: int) -> list[int]:
    layers = sorted(int(x) for x in df["layer_idx"].unique().tolist())
    rng = random.Random(seed)
    if len(layers) <= num_layers:
        return layers
    return sorted(rng.sample(layers, num_layers))


def _plot_random_layers(df: pd.DataFrame,
                        output_path: Path,
                        *,
                        selected_layers: list[int]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = plt.cm.tab10.colors

    for idx, layer in enumerate(selected_layers):
        layer_df = (
            df[df["layer_idx"] == layer]
            .sort_values("token_position")
        )
        color = colors[idx % len(colors)]
        label = f"layer {layer}"
        axes[0].plot(layer_df["token_position"],
                     layer_df["cosine_similarity"],
                     linewidth=1.1,
                     color=color,
                     label=label)
        axes[1].plot(layer_df["token_position"],
                     layer_df["jaccard_selected_experts"],
                     linewidth=1.1,
                     color=color,
                     label=label)

    axes[0].set_ylabel("Cosine")
    axes[0].set_title("Cosine Similarity by Token Position for 5 Random Layers")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].set_xlabel("Token Position")
    axes[1].set_ylabel("Jaccard")
    axes[1].set_title("Jaccard by Token Position for 5 Random Layers")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter prompt similarity CSV to comparable rows and generate "
            "position/layer plots."
        ))
    parser.add_argument("input_csv", help="Input similarity CSV path.")
    parser.add_argument("--filtered-csv",
                        default=None,
                        help="Output CSV path for comparable rows only.")
    parser.add_argument("--position-plot",
                        default=None,
                        help="Output PNG path for mean-across-layers plot.")
    parser.add_argument("--layers-plot",
                        default=None,
                        help="Output PNG path for random-5-layers plot.")
    parser.add_argument("--num-random-layers",
                        type=int,
                        default=5,
                        help="How many random layers to plot. Default: 5.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for choosing layers. Default: 42.")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    filtered_csv = Path(args.filtered_csv) if args.filtered_csv else _default_filtered_path(
        input_csv)
    position_plot = Path(args.position_plot) if args.position_plot else _default_position_plot_path(
        input_csv)
    layers_plot = Path(args.layers_plot) if args.layers_plot else _default_layer_plot_path(
        input_csv)

    df = pd.read_csv(input_csv)
    comparable_df = _filter_comparable(df)
    comparable_df = comparable_df.sort_values(["token_position", "layer_idx"])
    filtered_csv.parent.mkdir(parents=True, exist_ok=True)
    comparable_df.to_csv(filtered_csv, index=False)

    _plot_position_mean(comparable_df, position_plot)
    selected_layers = _pick_layers(comparable_df,
                                   num_layers=args.num_random_layers,
                                   seed=args.seed)
    _plot_random_layers(comparable_df,
                        layers_plot,
                        selected_layers=selected_layers)

    print(f"Input rows: {len(df)}")
    print(f"Comparable rows: {len(comparable_df)}")
    print(f"Filtered CSV: {filtered_csv}")
    print(f"Position-mean plot: {position_plot}")
    print(f"Random-layer plot: {layers_plot}")
    print(f"Selected layers (seed={args.seed}): {selected_layers}")


if __name__ == "__main__":
    main()

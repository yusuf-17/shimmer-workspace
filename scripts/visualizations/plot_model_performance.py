from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize

from model_performance_values import MODEL_POINTS


POINT_COLORS = ["#2b8cbe", "#f28e2b", "#59a14f", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#1f77b4"]
BASELINE_LABEL = "VAE"
TRAJECTORY_CMAP = plt.cm.viridis
TRAJECTORY_LABELS = {
    "GW base",
    "GW no cont",
    "cont 0.01",
    "cont 0.05",
    "cont 0.055",
    "cont 0.06",
    "cont 0.08",
    "cont 0.09",
}
NEUTRAL_POINT_COLOR = "#a8a8a8"


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(13, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfc")

    trajectory_points = sorted(
        [point for point in MODEL_POINTS if point["label"] in TRAJECTORY_LABELS],
        key=lambda point: (point["contrastive"], point["floor_color"], point["ball_vs_no_ball"], point["label"]),
    )
    trajectory_coords = np.array(
        [[point["floor_color"], point["contrastive"], point["ball_vs_no_ball"]] for point in trajectory_points],
        dtype=float,
    )
    trajectory_values = np.array([point["contrastive"] for point in trajectory_points], dtype=float)

    if len(trajectory_coords) > 1:
        segments = np.stack([trajectory_coords[:-1], trajectory_coords[1:]], axis=1)
        segment_values = 0.5 * (trajectory_values[:-1] + trajectory_values[1:])
        norm = Normalize(vmin=float(trajectory_values.min()), vmax=float(trajectory_values.max()))
        line_collection = Line3DCollection(
            segments,
            cmap=TRAJECTORY_CMAP,
            norm=norm,
            linewidth=2.8,
            alpha=0.95,
        )
        line_collection.set_array(segment_values)
        ax.add_collection3d(line_collection)

    baseline_point = next(point for point in MODEL_POINTS if point["label"] == BASELINE_LABEL)

    for point_idx, point in enumerate(MODEL_POINTS):
        is_baseline = point["label"] == BASELINE_LABEL
        in_trajectory = point["label"] in TRAJECTORY_LABELS
        color = TRAJECTORY_CMAP(point["contrastive"]) if in_trajectory else NEUTRAL_POINT_COLOR
        edge_color = "white" if in_trajectory else "#ececec"
        label_x_offset = 0.008 + 0.002 * (point_idx % 3)
        label_y_offset = 0.004 + 0.0015 * (point_idx % 4)
        label_z_offset = 0.004 + 0.001 * (point_idx % 5)
        ax.scatter(
            point["floor_color"],
            point["contrastive"],
            point["ball_vs_no_ball"],
            s=170 if is_baseline else 95,
            color=color,
            edgecolor=edge_color,
            linewidth=1.2,
            marker="*" if is_baseline else "o",
            depthshade=in_trajectory,
        )
        ax.text(
            point["floor_color"] + label_x_offset,
            point["contrastive"] + label_y_offset,
            point["ball_vs_no_ball"] + label_z_offset,
            point["label"],
            fontsize=8.5,
            color="#1f1f1f" if in_trajectory or is_baseline else "#666666",
        )

    ax.scatter(
        baseline_point["floor_color"],
        baseline_point["contrastive"],
        baseline_point["ball_vs_no_ball"],
        s=280,
        facecolors="none",
        edgecolors="#1f1f1f",
        linewidths=1.6,
        marker="o",
    )

    ax.text(
        baseline_point["floor_color"] + 0.02,
        baseline_point["contrastive"] + 0.01,
        baseline_point["ball_vs_no_ball"] + 0.01,
        "VAE baseline",
        fontsize=9,
        weight="semibold",
        color="#1f1f1f",
    )

    ax.set_xlabel("Floor color")
    ax.set_ylabel("Contrastive")
    ax.set_zlabel("Ball vs no ball")
    ax.set_title("Model Landscape on Floor Color, Contrastive Depth, and Ball Performance", pad=16, weight="semibold")

    ax.set_box_aspect((1.4, 1.0, 1.0))

    x_values = [point["floor_color"] for point in MODEL_POINTS]
    y_values = [point["contrastive"] for point in MODEL_POINTS]
    z_values = [point["ball_vs_no_ball"] for point in MODEL_POINTS]
    x_padding = max(0.03, (max(x_values) - min(x_values)) * 0.12)
    y_padding = max(0.03, (max(y_values) - min(y_values)) * 0.12)
    z_padding = max(0.02, (max(z_values) - min(z_values)) * 0.2)
    ax.set_xlim(min(x_values) - x_padding, max(x_values) + x_padding)
    ax.set_ylim(min(y_values) - y_padding, max(y_values) + y_padding)
    ax.set_zlim(min(z_values) - z_padding, max(z_values) + z_padding)
    ax.view_init(elev=28, azim=-70)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

    fig.tight_layout()
    return fig


def main() -> None:
    fig = build_figure()
    output_dir = Path("graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "model_performance_comparison.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
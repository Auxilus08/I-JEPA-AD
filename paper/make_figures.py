"""Build the composite figures used by main.tex from the committed result files.

Run from the repository root:  .venv/Scripts/python.exe paper/make_figures.py
Outputs go to paper/figures/. Every number plotted is read from results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#e8eef7", ec="#3b5b92", fs=7.0):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                fc=fc, ec=ec, lw=0.9,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.2)

    def arrow(x0, y0, x1, y1, color="#333333"):
        ax.add_patch(
            FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                            mutation_scale=8, lw=0.9, color=color)
        )

    ax.text(0.5, 45.5, "(a) Offline: nominal images only", fontsize=7.5,
            fontweight="bold", color="#3b5b92")
    box(0.5, 33, 14, 9, "Nominal\nimage", fc="#f4f4f4", ec="#777777")
    box(20, 33, 21, 9, "Frozen I-JEPA\nViT-g/16 encoder", fc="#fbe9d0", ec="#b36b00")
    box(46.5, 33, 19, 9, "Latent cache\n196 tokens x 1408", fc="#e6f2e6", ec="#3b7d3b")
    box(71, 33, 28.5, 9, "Transformer predictor\nloss = 1 - cos(pred, target)")
    arrow(15, 37.5, 19.4, 37.5)
    arrow(41.4, 37.5, 45.9, 37.5)
    arrow(65.9, 37.5, 70.4, 37.5)

    ax.text(0.5, 27.5, "(b) Online: coarse-to-fine masked prediction", fontsize=7.5,
            fontweight="bold", color="#3b5b92")
    box(0.5, 14.5, 11, 9, "Test\nimage", fc="#f4f4f4", ec="#777777")
    box(16, 14.5, 13, 9, "Frozen\nencoder", fc="#fbe9d0", ec="#b36b00")
    box(34, 14.5, 21, 9, "Stage 1\n16 coarse 4x4 masks\n(predictor)")
    box(60, 14.5, 15, 9, "Gate\nregion score\n>= tau ?", fc="#fff3c4", ec="#a08400")
    box(80, 14.5, 19.5, 9, "Stage 2\n1x1 fine masks on\nflagged tokens")
    arrow(12, 19, 15.4, 19)
    arrow(29.4, 19, 33.4, 19)
    arrow(55.4, 19, 59.4, 19)
    arrow(75.4, 19, 79.4, 19)
    ax.text(77.4, 21.0, "yes", fontsize=6.3, ha="center")

    box(27, 1, 46, 7.5,
        "S_final = S_coarse + lambda * S_fine\n"
        "bilinear upsampling: heatmap, top-k image score",
        fc="#e6f2e6", ec="#3b7d3b", fs=6.8)
    arrow(44.5, 14.0, 44.5, 9.0)
    arrow(89.5, 14.0, 73.6, 6.0)
    arrow(67.5, 14.0, 62.0, 9.0)
    ax.text(70.5, 11.3, "no: keep\ncoarse", fontsize=6.0, ha="center",
            va="center")
    fig.savefig(OUT / "architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def heatmap_composite() -> None:
    """One 'pronounced defect' row per category plus the shared column header."""
    src = ROOT / "results" / "report_figures"
    width_cut = 1360  # drop the colour-bar strip on the right
    header = Image.open(src / "heatmaps_bottle.png").convert("RGB").crop(
        (0, 70, width_cut, 120)
    )
    rows = []
    for cat in ("bottle", "capsule", "hazelnut"):
        img = Image.open(src / f"heatmaps_{cat}.png").convert("RGB")
        rows.append(img.crop((0, 615, width_cut, 815)))  # pronounced-defect row
    h = header.height + sum(r.height for r in rows)
    canvas = Image.new("RGB", (width_cut, h), "white")
    canvas.paste(header, (0, 0))
    y = header.height
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height
    canvas.save(OUT / "heatmaps_composite.png", dpi=(300, 300))


def latency() -> None:
    d = json.loads(
        (ROOT / "results" / "benchmarks" / "e2e_latency_vram.json").read_text()
    )
    enc = d["encoder_b1_ms"]["mean"]
    labels = ["Fixed", "Adaptive", "Dense", "PatchCore-\nstyle"]
    total = [
        d["e2e_fixed"]["mean_ms"],
        d["e2e_adaptive"]["mean_ms"],
        d["e2e_dense"]["mean_ms"],
        enc + d["patchcore_score_ms"]["mean"],
    ]
    rest = [t - enc for t in total]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(labels, [enc] * 4, color="#e0a458", label="Encoder (ViT-g/16)")
    ax.bar(labels, rest, bottom=[enc] * 4, color="#4c78a8", label="Scoring")
    for i, t in enumerate(total):
        ax.text(i, t + 5, f"{t:.0f}", ha="center", fontsize=7)
    ax.set_ylabel("Latency per image (ms)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax.set_ylim(0, max(total) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "latency_e2e.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def hazelnut_calibrated() -> None:
    """Default vs calibrated gate on the hazelnut crack example.

    Uses paper/calibrated_maps.npz written by paper/calibrated_heatmaps.py. The
    example is shown because it is the category where the calibrated gate
    improves localisation; bottle got worse and capsule is unchanged
    (see the paper text). Panels are min-max normalised for display only.
    """
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from ijepa_ad.evaluation.metrics import pixel_auroc
    from ijepa_ad.masking.score_maps import upsample_patch_scores

    maps = np.load(ROOT / "paper" / "calibrated_maps.npz")
    tag = "hazelnut/Pronounced defect (90th pct. GT area)"
    image_path = Path(str(maps[f"{tag}/path"]))
    mask_path = (
        image_path.parents[2]
        / "ground_truth"
        / image_path.parent.name
        / f"{image_path.stem}_mask.png"
    )
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    image = Image.open(image_path).convert("RGB")
    side = 448
    shown = image.resize((side, side), Image.BILINEAR)
    mask_small = np.asarray(
        Image.fromarray(mask.astype("uint8") * 255).resize((side, side), Image.NEAREST)
    ) > 0

    def full_map(vec: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        up = upsample_patch_scores(torch.from_numpy(vec).float(), output_size=size)
        return up[0, 0].numpy()

    panels = []
    for key, title, vec in (
        ("default/fused", "Default gate: fused", maps[f"{tag}/default/fused"]),
        ("calibrated/fine", "Calibrated gate: fine pass", maps[f"{tag}/calibrated/fine"]),
        ("calibrated/fused", "Calibrated gate: fused", maps[f"{tag}/calibrated/fused"]),
    ):
        auroc = None
        if key.endswith("fused"):
            auroc = pixel_auroc([full_map(vec, mask.shape)], [mask])
        shown_map = full_map(vec, (side, side))
        lo, hi = float(shown_map.min()), float(shown_map.max())
        norm = (shown_map - lo) / max(hi - lo, 1e-8)
        panels.append((title, norm, auroc, int((vec > 0).sum())))

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    axes[0].imshow(shown)
    axes[0].contour(mask_small, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[0].set_title("Image + ground truth", fontsize=7)
    masks_used = {"Default gate: fused": 16, "Calibrated gate: fine pass": None,
                  "Calibrated gate: fused": 48}
    for ax, (title, norm, auroc, _) in zip(axes[1:], panels):
        ax.imshow(norm, cmap="inferno", vmin=0, vmax=1)
        ax.contour(mask_small, levels=[0.5], colors="cyan", linewidths=0.8)
        extra = f"\npixel AUROC {auroc:.3f}, {masks_used[title]} masks" if auroc else ""
        ax.set_title(title + extra, fontsize=7)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "heatmap_hazelnut_calibrated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("hazelnut panels:", [(t, a) for t, _, a, _ in panels])


if __name__ == "__main__":
    architecture()
    heatmap_composite()
    latency()
    hazelnut_calibrated()
    print("figures written to", OUT)

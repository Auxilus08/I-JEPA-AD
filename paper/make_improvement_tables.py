"""Write paper/tab_improvements.tex and paper/tab_improvements_boot.tex from saved result CSVs (no hand-typed numbers)."""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "improvements"
OUT = Path(__file__).resolve().parent
CATS = ("bottle", "capsule", "hazelnut")
ROWEND = r"\\"

l11 = pd.read_csv(R / "layer_predictor" / "layer_predictor_metrics.csv")
l13 = pd.read_csv(R / "layer32_confirmation" / "layer32_confirmation_metrics.csv")
knn = pd.read_csv(R / "layer_confirmation" / "layer_vs_final.csv")


def pick(df, col, value, cat):
    return df[(df.category == cat) & (df[col] == value)].iloc[0]


ROWS = [
    (r"Original baseline (final layer, $1-\cos$)", lambda c: pick(l11, "row", "final / baseline score", c), True),
    (r"+ position z-score (final layer)", lambda c: pick(l11, "row", "final / selected", c), True),
    (r"Layer 32, $1-\cos$", lambda c: pick(l11, "row", "layer 32 / baseline score", c), True),
    (r"\textbf{Layer 32 + position z-score}", lambda c: pick(l11, "row", "layer 32 / selected", c), True),
    (r"Layer 32 + Mahalanobis $d{=}128$\textsuperscript{a}", lambda c: pick(l13, "row", "selected", c), True),
    (r"kNN reference, layer 32\textsuperscript{b}", lambda c: pick(knn, "features", "layer 32 (layer norm)", c), False),
]


def cells(r, has_masks):
    masks = f"{r['masks_per_image']:.1f}" if has_masks else "--"
    return f"{r['image_auroc_max']:.3f} & {r['image_auroc_top3']:.3f} & {r['pixel_auroc']:.3f} & {r['aupro']:.3f} & {masks}"


CAPTION1 = (
    r"\caption{Test results after the validation-selected changes (official test split, one scoring pass per row). "
    r"All layer-32 rows use the same predictor architecture, training recipe and masks as the baseline; only the cached tokens change. "
    r"Image AUROC is given for the max score and for the mean of the top-3 token scores. "
    r"The z-score settings (position z-score, $k{=}3$) were chosen on synthetic validation defects for the final-layer predictor and carried over unchanged; "
    r"layer 32 was chosen on the same synthetic data. "
    r"\textsuperscript{a}Re-selected on layer-32 synthetic validation data and scored once on test; its score is the max (top-3 shown for comparison only). "
    r"\textsuperscript{b}Nearest-neighbour reference on layer-32 tokens (full memory bank), context only.}"
)
lines = [
    r"\begin{table*}[t]", CAPTION1, r"\label{tab:improved}", r"\centering", r"\footnotesize", r"\begin{tabular}{llccccc}", r"\hline",
    r"\textbf{Category} & \textbf{Method} & \textbf{Img AUROC (max)} & \textbf{Img AUROC (top-3)} & \textbf{Pixel AUROC} & \textbf{AUPRO} & \textbf{Masks/img} " + ROWEND,
    r"\hline",
]
for cat in (*CATS, "mean"):
    lines.append(r"\multirow{" + str(len(ROWS)) + r"}{*}{" + cat.capitalize() + "}")
    for label, getter, has_masks in ROWS:
        if cat == "mean":
            parts = [getter(c) for c in CATS]
            r = {k: sum(round(float(p[k]), 3) for p in parts) / 3 for k in ("image_auroc_max", "image_auroc_top3", "pixel_auroc", "aupro")}   # mean of the rounded values, as in Table I
            r["masks_per_image"] = sum(round(float(p["masks_per_image"]), 1) for p in parts) / 3 if has_masks else 0
        else:
            r = getter(cat)
        lines.append(f" & {label} & {cells(r, has_masks)} {ROWEND}")
    lines.append(r"\hline")
lines += [r"\end{tabular}", r"\end{table*}", ""]
(OUT / "tab_improvements.tex").write_text("\n".join(lines), encoding="utf-8")

# ---- paired bootstrap table ----
b11 = pd.read_csv(R / "layer_predictor" / "bootstrap_deltas.csv")
b07 = pd.read_csv(R / "confirmation" / "bootstrap_deltas.csv")
b13 = pd.read_csv(R / "layer32_confirmation" / "bootstrap_deltas.csv")


def fmt(row):
    star = r"\textsuperscript{*}" if (row.ci95_low > 0 or row.ci95_high < 0) else ""

    def f(v):
        return f"{v:+.3f}".replace("-", "$-$")

    return f"{f(row.delta_image_auroc)} [{f(row.ci95_low)}, {f(row.ci95_high)}]{star}"


SPECS = [
    (r"Final layer + z-score (top-3) vs original (max)", b07, lambda d: d.reference.str.contains("requested")),
    (r"Layer 32 + z-score (top-3) vs original (max)", b11, lambda d: d.comparison.str.startswith("overall")),
    (r"Layer 32 + z-score vs final layer + z-score (top-3)", b11, lambda d: d.comparison.str.startswith("HEADLINE")),
    (r"Layer 32, $1-\cos$ vs original (max)", b11, lambda d: d.comparison.str.startswith("layer 32 vs final, baseline score, max")),
    (r"Layer 32 Mahalanobis vs layer 32 $1-\cos$ (max)", b13, lambda d: d.comparison.str.startswith("1 PRIMARY")),
    (r"Layer 32 Mahalanobis (max) vs layer 32 z-score (top-3)", b13, lambda d: d.comparison.str.startswith("2a ")),
    (r"Layer 32 Mahalanobis (max) vs original (max)", b13, lambda d: d.comparison.str.startswith("3 HISTORICAL")),
]
CAPTION2 = (
    r"\caption{Paired-bootstrap change in image AUROC on the test split (1,000 resamples, same resamples for all comparisons within a category, 95\,\% interval). "
    r"Asterisk: interval excludes zero. Rows 1--4 use configurations fixed before their test evaluation; rows 5--7 evaluate the layer-32 re-selection (Mahalanobis).}"
)
lines = [
    r"\begin{table*}[t]", CAPTION2, r"\label{tab:boot2}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lccc}", r"\hline",
    r"\textbf{Comparison (new vs reference)} & \textbf{Bottle} & \textbf{Capsule} & \textbf{Hazelnut} " + ROWEND, r"\hline",
]
for label, df, mask in SPECS:
    sel = df[mask(df)]
    assert all(len(sel[sel.category == c]) == 1 for c in CATS), label
    lines.append(f"{label} & " + " & ".join(fmt(sel[sel.category == c].iloc[0]) for c in CATS) + f" {ROWEND}")
lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
(OUT / "tab_improvements_boot.tex").write_text("\n".join(lines), encoding="utf-8")
print("written")

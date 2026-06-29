"""
================================================================
  MAGOGO — BSF Farming Optimisation  |  Static Chart Generator
================================================================
  Generates all 7 publication-ready PNG charts and saves them
  to ./visualizations/.

  This is the offline / standalone version. The full interactive
  web app is in app.py (run with: streamlit run app.py).

  Usage:
    python generate_charts.py
================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from bsf_model import load_and_clean, train_models, FEATURE_LABELS

warnings.filterwarnings("ignore")

# ── Colour palette ────────────────────────────────────────
P = {
    "dark":  "#1B4332",
    "main":  "#2D6A4F",
    "mid":   "#52B788",
    "light": "#A8D8B9",
    "pale":  "#D8F3DC",
    "amber": "#F9A825",
    "red":   "#E63946",
    "blue":  "#457B9D",
    "grey":  "#6C757D",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.edgecolor":   "#DDDDDD",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelcolor":  P["dark"],
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.titlecolor":  P["dark"],
    "xtick.color":      P["dark"],
    "ytick.color":      P["dark"],
    "font.family":      "DejaVu Sans",
    "grid.color":       "#EEEEEE",
    "grid.linewidth":   0.7,
    "legend.frameon":   False,
})

VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# ── Load data & train ─────────────────────────────────────
DATA = "main_dataset.csv"
FALLBACK = "/mnt/user-data/uploads/main_dataset.csv"
if not os.path.exists(DATA) and os.path.exists(FALLBACK):
    DATA = FALLBACK

print("\n🦟  MAGOGO — Generating charts …\n")
df     = load_and_clean(DATA)
bundle = train_models(df)
hdf    = bundle["harvest_df"]
fi     = bundle["feature_imp"]
comp   = bundle["comparison"]
bname  = bundle["best_name"]

def save(fig, name):
    path = os.path.join(VIS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {name}")


# ── 1. Model comparison bar chart ────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
colors  = [P["main"] if r["Model"] == bname else P["light"]
           for _, r in comp.iterrows()]
bars = ax.bar(comp["Model"], comp["mae_mean"],
              yerr=comp["mae_std"], color=colors,
              edgecolor=P["dark"], linewidth=0.6,
              capsize=5, width=0.45)
for bar, val in zip(bars, comp["mae_mean"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + comp["mae_std"].max()*0.15,
            f"{val:.2f} kg", ha="center", fontsize=9, color=P["dark"])
ax.set_ylabel("Mean Absolute Error (kg)  ↓", labelpad=8)
ax.set_title(f"Model Comparison — 5-Fold CV   |   Best: {bname}")
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
patches = [mpatches.Patch(color=P["main"], label="Best model"),
           mpatches.Patch(color=P["light"], label="Other models")]
ax.legend(handles=patches, fontsize=8)
fig.tight_layout()
save(fig, "1_model_comparison.png")


# ── 2. Harvest time-series ────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(hdf["Date"], hdf["Harvest_Mass"],
                alpha=0.18, color=P["main"])
ax.plot(hdf["Date"], hdf["Harvest_Mass"],
        color=P["main"], linewidth=2.2, marker="o",
        markersize=5, markeredgecolor=P["dark"], markeredgewidth=0.5,
        label="Harvest Mass (kg)")
ma = hdf["Harvest_Mass"].rolling(5, center=True).mean()
ax.plot(hdf["Date"], ma, color=P["amber"], linewidth=2,
        linestyle="--", label="5-cycle moving avg")
ax.set_xlabel("Date"); ax.set_ylabel("Harvest Mass (kg)")
ax.set_title("BSF Harvest Mass Over Time")
ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
fig.autofmt_xdate()
fig.tight_layout()
save(fig, "2_harvest_timeseries.png")


# ── 3. Cumulative harvest ─────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
cum = hdf["Harvest_Mass"].cumsum()
ax.fill_between(hdf["Date"], cum, alpha=0.18, color=P["dark"])
ax.plot(hdf["Date"], cum, color=P["dark"], linewidth=2.5)
ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Harvest Mass (kg)")
ax.set_title(f"Cumulative BSF Production  (Total: {cum.iloc[-1]:.0f} kg)")
ax.grid(True, linestyle="--", alpha=0.4)
fig.autofmt_xdate()
fig.tight_layout()
save(fig, "3_cumulative_harvest.png")


# ── 4. Environmental trends ───────────────────────────────
env_df = df.dropna(subset=["Date"])
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
env_specs = [
    ("Temperature", "Ambient Temperature (°C)", P["red"]),
    ("pH",          "pH Level",                 P["blue"]),
    ("Water_Added", "Water Added (L)",           P["mid"]),
]
for ax, (col, label, colour) in zip(axes, env_specs):
    sub = env_df.dropna(subset=[col])
    ax.plot(sub["Date"], sub[col], color=colour, linewidth=1.8,
            marker="o", markersize=3.5, alpha=0.85)
    ax.fill_between(sub["Date"], sub[col], alpha=0.12, color=colour)
    ax.set_title(label); ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
fig.suptitle("Environmental Conditions Over Time", fontsize=12,
             fontweight="bold", color=P["dark"], y=1.01)
fig.tight_layout()
save(fig, "4_environmental_trends.png")


# ── 5. Scatter: Feed vs Harvest ───────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(hdf["Feed_Amount"], hdf["Harvest_Mass"],
                c=hdf["Cycle_Days"], cmap="Greens",
                s=65, edgecolors=P["dark"], linewidths=0.5, zorder=3)
cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("Cycle Day", fontsize=9)
z = np.polyfit(hdf["Feed_Amount"].dropna(), hdf["Harvest_Mass"].dropna(), 1)
xr = np.linspace(hdf["Feed_Amount"].min(), hdf["Feed_Amount"].max(), 200)
ax.plot(xr, np.poly1d(z)(xr), color=P["amber"], linewidth=2,
        linestyle="--", label="Linear trend")
ax.set_xlabel("Feed Amount (kg)"); ax.set_ylabel("Harvest Mass (kg)")
ax.set_title("Feed Amount vs BSF Harvest Mass")
ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
save(fig, "5_feed_vs_harvest.png")


# ── 6. Actual vs Predicted ────────────────────────────────
X_all  = bundle["X"]; y_all = bundle["y"]
y_pred = bundle["best_model"].predict(X_all)
lo = min(y_all.min(), y_pred.min()) * 0.9
hi = max(y_all.max(), y_pred.max()) * 1.08

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([lo, hi], [lo, hi], color=P["grey"], linewidth=1.5,
        linestyle="--", label="Perfect prediction")
for a, p in zip(y_all, y_pred):
    ax.plot([a, a], [a, p], color=P["main"], alpha=0.15, linewidth=1)
ax.scatter(y_all, y_pred, color=P["mid"], edgecolors=P["dark"],
           linewidths=0.5, s=65, zorder=3)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Actual Harvest Mass (kg)")
ax.set_ylabel("Predicted Harvest Mass (kg)")
best_row = comp[comp["Model"] == bname].iloc[0]
ax.set_title(f"Actual vs Predicted  |  MAE={best_row['mae_mean']:.2f} kg  R²={best_row['r2_mean']:.3f}")
ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
save(fig, "6_actual_vs_predicted.png")


# ── 7. Feature importance ─────────────────────────────────
q33 = fi["Importance"].quantile(0.33)
q66 = fi["Importance"].quantile(0.66)
bar_cols = [P["amber"] if v >= q66 else
            (P["mid"] if v >= q33 else P["light"])
            for v in fi["Importance"]]

fig, ax = plt.subplots(figsize=(8, 5))
fi_sorted = fi.sort_values("Importance", ascending=True)
bar_cols_s = bar_cols[::-1]   # match sorted order

bars = ax.barh(fi_sorted["Label"], fi_sorted["Importance"],
               color=bar_cols_s, edgecolor=P["dark"],
               linewidth=0.5, height=0.55)
for bar, val in zip(bars, fi_sorted["Importance"]):
    ax.text(bar.get_width() + fi["Importance"].max()*0.015,
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8.5, color=P["dark"])

ax.set_xlabel("Mean Decrease in Impurity")
ax.set_title("Feature Importance — What Drives BSF Yield?")
ax.set_xlim(0, fi["Importance"].max()*1.22)
ax.grid(True, axis="x", linestyle="--", alpha=0.4)
patches = [mpatches.Patch(color=P["amber"],  label="High"),
           mpatches.Patch(color=P["mid"],    label="Medium"),
           mpatches.Patch(color=P["light"],  label="Lower")]
ax.legend(handles=patches, fontsize=8, loc="lower right")
fig.tight_layout()
save(fig, "7_feature_importance.png")

print(f"\n  All charts saved to ./{VIS_DIR}/\n")

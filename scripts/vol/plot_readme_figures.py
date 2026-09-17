"""Generate the two README figures from existing vol-S artifacts (no number is hardcoded).

  docs/assets/qlike_comparison.png  <- results/vols/qlike_report_{interval}_test_{arch}.json
      (written by scripts/vol/dev_vols_qlike.py): naive, HAR-RV, HAR-C, NN test QLIKE.
  docs/assets/rv_pred_vs_actual.png <- results/vols/vols_predictions_{interval}_test_{arch}.parquet
      (written by scripts/vol/export_vols_predictions.py): realized vs predicted RV with the
      q10-q90 band, on the last `--window` bars of the test split (fixed rule, not a chosen window).

Fail-fast if the report's scaler provenance is not verified (`provenance.matches` must be true):
an unverifiable model/npz pair must not become a README figure.

Usage (from the project root):
    python scripts/vol/plot_readme_figures.py [--arch canonical_1h_vols] [--window 500]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import load_config  # noqa: E402

# Explicit white background so the PNGs stay readable in GitHub dark mode.
BG = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"
HIGHLIGHT = "#2a78d6"      # NN
NEUTRAL = "#b4b3ad"        # baselines
BAND = "#2a78d6"
OUT_DIR = Path("docs/assets")


def _style(ax):
    # Recessive axes: no top/right spines, light grid behind the marks.
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_qlike(report: dict, n_members: int, out: Path):
    # Values read from the judge's report: HAR-RV is `metrics.har`, HAR-C lives in the HAR-CJ block.
    rows = [
        ("Naive (last 30h RV)", report["metrics"]["naive"]["qlike"], False),
        ("HAR-RV", report["metrics"]["har"]["qlike"], False),
        ("HAR-C", report["har_cj"]["har_c"]["qlike_har_c"], False),
        (f"NN (iTransformer, {n_members} members)" if n_members else "NN (iTransformer)",
         report["metrics"]["nn"]["qlike"], True),
    ]
    # Worst (highest) on top, best at the bottom.
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [HIGHLIGHT if r[2] else NEUTRAL for r in rows]

    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150, facecolor=BG)
    _style(ax)
    ax.grid(axis="y", visible=False)
    bars = ax.barh(labels, vals, color=colors, height=0.6)
    xmax = max(vals)
    for b, v, is_nn in zip(bars, vals, [r[2] for r in rows]):
        ax.text(v + xmax * 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", ha="left", fontsize=10, color=INK,
                fontweight="bold" if is_nn else "normal")
    ax.set_xlim(0, xmax * 1.12)
    ax.set_xlabel("QLIKE on the test split (lower is better)", color=INK_2)
    ax.set_title(f"Volatility forecast loss, BTC/USDT 1h (out-of-sample, "
                 f"n = {report['gate']['n_obs']:,})",
                 color=INK, fontsize=11, loc="left")
    ax.tick_params(axis="y", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG, transparent=False)
    plt.close(fig)


def plot_rv(preds: pd.DataFrame, window: int, horizon: int, out: Path):
    df = preds.sort_values("open_time").tail(window)
    t = pd.to_datetime(df["open_time"])

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150, facecolor=BG)
    _style(ax)
    ax.fill_between(t, df["rv_nn_q10"], df["rv_nn_q90"], color=BAND, alpha=0.18, linewidth=0,
                    label="NN 10%-90% predicted range")
    ax.plot(t, df["rv_true"], color=INK, linewidth=1.6, label="Realized")
    ax.plot(t, df["rv_nn_q50"], color=HIGHLIGHT, linewidth=2.0, label="NN forecast (median)")
    ax.set_yscale("log")
    ax.set_ylabel(f"Realized variance, next {horizon}h (log scale)", color=INK_2)
    ax.set_title(f"Forecast vs realized variance, last {len(df)} hourly forecasts of the test split",
                 color=INK, fontsize=11, loc="left")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(INK)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, facecolor=BG, transparent=False)
    plt.close(fig)


def main():
    # Windows console: force UTF-8 so unicode in messages cannot crash on cp1252.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Generate README figures from vol-S artifacts")
    ap.add_argument("--arch", default="canonical_1h_vols")
    ap.add_argument("--window", type=int, default=500, help="last N test bars in the RV figure")
    args = ap.parse_args()

    cfg = load_config()
    interval = cfg["data"]["interval"]
    horizon = int(cfg["features"]["forecast_horizon"])
    report_path = Path("results/vols") / f"qlike_report_{interval}_test_{args.arch}.json"
    preds_path = Path("results/vols") / f"vols_predictions_{interval}_test_{args.arch}.parquet"

    missing = [p for p in (report_path, preds_path) if not p.exists()]
    if missing:
        sys.exit("missing artifacts: " + ", ".join(map(str, missing)) +
                 f"\n  report: python scripts/vol/dev_vols_qlike.py --arch {args.arch}"
                 f"  (with QUANTSYS_VOLS_SPLIT=test)"
                 f"\n  preds:  python scripts/vol/export_vols_predictions.py --arch {args.arch} --split test")

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    if report.get("provenance", {}).get("matches") is not True:
        sys.exit(f"{report_path}: scaler provenance not verified (provenance.matches != true)")
    model_dir = Path(report["provenance"].get("model_dir", ""))
    n_members = len(list(model_dir.glob("best_model_[0-9]*.pt"))) if model_dir.is_dir() else 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_qlike(report, n_members, OUT_DIR / "qlike_comparison.png")
    plot_rv(pd.read_parquet(preds_path), args.window, horizon, OUT_DIR / "rv_pred_vs_actual.png")
    print(f"written {OUT_DIR / 'qlike_comparison.png'} and {OUT_DIR / 'rv_pred_vs_actual.png'}")


if __name__ == "__main__":
    main()

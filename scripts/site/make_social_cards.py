"""Generate the social preview cards (link previews on LinkedIn/Reddit/X, GitHub social preview).

  docs/assets/og_card.png        1200x630  <- og:image of docs/index.html     (English)
  docs/assets/og_card.it.png     1200x630  <- og:image of docs/index.it.html  (Italian)
  docs/assets/social_preview.png 1280x640  <- GitHub repo Settings > Social preview (manual upload)

No number is hardcoded: QLIKE values, the claim percentage and n come from the judge's report
`results/vols/qlike_report_{interval}_{split}_{arch}.json` (written by scripts/vol/dev_vols_qlike.py),
the same artifact the README figures and the project site read. A card is a public claim, so the
same rule applies: a machine-derivable number must be DERIVED.

Fail-fast if the report's scaler provenance is not verified (`provenance.matches` must be true):
an unverifiable model/npz pair must not become a shared preview image.

Usage (from the project root):
    python scripts/site/make_social_cards.py [--arch canonical_1h_vols] [--split test]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import load_config  # noqa: E402

# Site palette (docs/index.html light theme): the card and the landing page must look like one
# artifact, since the card is what a reader sees immediately before the page.
BG = "#fbfaf7"
INK = "#1b1a17"
MUT = "#6b6862"
LINE = "#e2ded5"
ACCENT = "#8a5a2b"
HIGHLIGHT = "#2a78d6"   # the model, same blue as the README figures
NEUTRAL = "#b4b3ad"     # baselines
OUT_DIR = Path("docs/assets")
REPO_URL = "github.com/luca-feleppa/quantsys"

# Per-language card copy. English is canonical, as for every doc in the repo.
COPY = {
    "en": {
        "sub": "Probabilistic volatility forecasting on BTC/USDT\nand the full record of what failed.",
        "chart_title": "Forecast loss, out-of-sample test split (lower is better)",
        "naive": "Naive",
        "members": "NN ({n} seeds)",
        "claim_label": "QLIKE vs HAR-C\nout-of-sample",
        "claim_note": "n = {n:,} hourly forecasts, 30h horizon\nval and test coherent",
        "foot": "pre-registered gates · negative results published in full",
    },
    "it": {
        "sub": "Previsione probabilistica di volatilità su BTC/USDT\ne il registro completo di ciò che ha fallito.",
        "chart_title": "Errore di previsione, test split out-of-sample (più basso è meglio)",
        "naive": "Naive",
        "members": "NN ({n} seed)",
        "claim_label": "QLIKE contro HAR-C\nout-of-sample",
        "claim_note": "n = {n:,} previsioni orarie, orizzonte 30h\nval e test coerenti",
        "foot": "gate pre-registrati · risultati negativi pubblicati per intero",
    },
}


def read_numbers(report: dict) -> dict:
    # The claim denominator is HAR-C (jump-robust continuous component), not HAR-RV: the
    # pre-registered gate used HAR-RV, the published claim uses HAR-C. The card states the claim.
    har_c = report["har_cj"]["har_c"]
    return {
        "nn": report["metrics"]["nn"]["qlike"],
        "har_c": har_c["qlike_har_c"],
        "naive": report["metrics"]["naive"]["qlike"],
        "delta_pct": (har_c["ratio_nn_over_har_c"] - 1.0) * 100.0,
        "n": int(report["gate"]["n_obs"]),
    }


def draw_card(nums: dict, lang: str, size_px: tuple, n_members: int, out: Path) -> None:
    # Fractional layout, so the same drawing code serves both 1200x630 and 1280x640.
    c = COPY[lang]
    w, h = size_px
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=BG)

    # Accent rule along the top edge: identifies the card at thumbnail size.
    fig.add_artist(Rectangle((0, 0.977), 1, 0.023, color=ACCENT, zorder=3))

    fig.text(0.055, 0.88, "QUANTSYS", color=INK, fontsize=52, fontweight="bold", va="top")
    fig.text(0.055, 0.735, c["sub"], color=MUT, fontsize=20, va="top", linespacing=1.45)

    # Left half: the three QLIKE values as a bar chart, so the claim is shown and not just asserted.
    ax = fig.add_axes([0.175, 0.17, 0.36, 0.32])
    ax.set_facecolor(BG)
    nn_label = c["members"].format(n=n_members) if n_members else "NN"
    rows = [(c["naive"], nums["naive"], False),
            ("HAR-C", nums["har_c"], False),
            (nn_label, nums["nn"], True)]
    rows.sort(key=lambda r: r[1])
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows],
                   color=[HIGHLIGHT if r[2] else NEUTRAL for r in rows], height=0.6)
    xmax = max(r[1] for r in rows)
    for bar, (_, val, is_nn) in zip(bars, rows):
        ax.text(val + xmax * 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", ha="left", fontsize=15, color=INK,
                fontweight="bold" if is_nn else "normal")
    ax.set_xlim(0, xmax * 1.22)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, labelsize=15, labelcolor=INK)
    # Title anchored to the card margin, not to the axes, so it lines up with the heading above.
    ax.set_title(c["chart_title"], color=MUT, fontsize=13, loc="left", pad=10, x=-0.33)

    # Right half: the headline number, with its qualifier attached rather than in a footnote.
    # U+2212 (minus sign) instead of the hyphen a format spec emits: it reads as a number, not a dash.
    headline = f"{nums['delta_pct']:+.2f}%".replace("-", "−")
    fig.text(0.62, 0.50, headline, color=HIGHLIGHT, fontsize=56, fontweight="bold",
             va="center", ha="left")
    fig.text(0.62, 0.385, c["claim_label"], color=INK, fontsize=18, va="top", linespacing=1.35)
    fig.text(0.62, 0.275, c["claim_note"].format(n=nums["n"]), color=MUT, fontsize=13.5,
             va="top", linespacing=1.45)

    fig.add_artist(Rectangle((0.055, 0.115), 0.89, 0.002, color=LINE))
    fig.text(0.055, 0.058, c["foot"], color=MUT, fontsize=12.5, va="center")
    fig.text(0.945, 0.058, REPO_URL, color=ACCENT, fontsize=12.5, va="center", ha="right")

    fig.savefig(out, facecolor=BG, transparent=False)
    plt.close(fig)


def main() -> int:
    # Windows console: force UTF-8 so unicode in messages cannot crash on cp1252.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Generate the social preview cards in docs/assets")
    ap.add_argument("--arch", default="canonical_1h_vols")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    cfg = load_config()
    interval = cfg["data"]["interval"]
    report_path = Path("results/vols") / f"qlike_report_{interval}_{args.split}_{args.arch}.json"
    if not report_path.exists():
        sys.exit(f"missing artifact: {report_path}"
                 f"\n  report: python scripts/vol/dev_vols_qlike.py --arch {args.arch}"
                 f"  (with QUANTSYS_VOLS_SPLIT={args.split})")

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    if report.get("provenance", {}).get("matches") is not True:
        sys.exit(f"{report_path}: scaler provenance not verified (provenance.matches != true)")
    model_dir = Path(report["provenance"].get("model_dir", ""))
    n_members = len(list(model_dir.glob("best_model_[0-9]*.pt"))) if model_dir.is_dir() else 0

    nums = read_numbers(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 1200x630 is the Open Graph reference size (LinkedIn/Reddit/X); GitHub serves its social
    # preview at 1280x640, hence the second aspect ratio.
    targets = [("en", (1200, 630), OUT_DIR / "og_card.png"),
               ("it", (1200, 630), OUT_DIR / "og_card.it.png"),
               ("en", (1280, 640), OUT_DIR / "social_preview.png")]
    for lang, size, out in targets:
        draw_card(nums, lang, size, n_members, out)
        print(f"written {out}  {size[0]}x{size[1]}")
    print(f"  numbers from {report_path}: NN {nums['nn']:.4f} - HAR-C {nums['har_c']:.4f} "
          f"- naive {nums['naive']:.4f} - claim {nums['delta_pct']:+.2f}% - n = {nums['n']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Script 05 — Analisi dei segnali live registrati.
Legge results/live_signals.jsonl e produce statistiche + aggiorna
dashboard_results.json con i dati della sessione live.

Run configuration PyCharm:
  Script: scripts/05_analyze_signals.py
  Working dir: <root del progetto>
"""
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from quantsys.utils import setup_logging
setup_logging()
log = logging.getLogger("quantsys.script.05")


# IT: legge architettura corrente da config (fallback su lstm)
# EN: read current architecture from config (fallback to lstm)
def _default_arch() -> str:
    try:
        import re
        root = Path(__file__).parent.parent
        txt = (root / "config" / "default.yaml").read_text(encoding="utf-8")
        m = re.search(r'architecture:\s*["\']?(\w+)["\']?', txt)
        if m and m.group(1) in ("lstm", "itransformer", "tft", "tcnmamba", "nhits"):
            return m.group(1)
    except Exception:
        pass
    return "lstm"


# IT: analizza live_signals.jsonl (stats+equity) e aggiorna dashboard_results.json
# EN: analyze live_signals.jsonl (stats+equity) and update dashboard_results.json
def main():
    # IT: env QUANTSYS_ARCH ha precedenza per propagazione da pipeline
    # EN: QUANTSYS_ARCH env wins to propagate from pipeline parent
    import os as _os
    arch = _os.environ.get("QUANTSYS_ARCH") or _default_arch()
    results_dir = Path("results") / arch
    results_dir.mkdir(parents=True, exist_ok=True)
    LIVE_FILE    = results_dir / "live_signals.jsonl"
    RESULTS_FILE = results_dir / "dashboard_results.json"

    log_path = LIVE_FILE
    if not log_path.exists():
        print(f"  File non trovato: {log_path}")
        print("  Esegui prima: python scripts/04_live_signals.py")
        return

    # IT: carica JSONL ignorando righe corrotte (resilienza a write parziali)
    # EN: load JSONL ignoring corrupt lines (resilient to partial writes)
    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print("Nessun segnale trovato nel file.")
        return

    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    print(f"""
{'═'*60}
  ANALISI SEGNALI LIVE
  File    : {log_path}
  Periodo : {df['ts'].iloc[0].strftime('%Y-%m-%d %H:%M')} → {df['ts'].iloc[-1].strftime('%Y-%m-%d %H:%M')}
  Record  : {len(df):,}
{'═'*60}
""")

    # IT: distribuzione BUY/SELL/HOLD con bar chart ASCII
    # EN: BUY/SELL/HOLD distribution with ASCII bar chart
    sig_counts = df["signal"].value_counts()
    total      = len(df)
    print("  DISTRIBUZIONE SEGNALI:")
    for sig, cnt in sig_counts.items():
        bar = "█" * int(cnt / total * 40)
        print(f"    {sig:<6} {cnt:>5}  ({cnt/total:.1%})  {bar}")

    # IT: stats medie parametri t-Student (sanity check su drift di mu/sigma/nu)
    # EN: mean t-Student params (sanity check on mu/sigma/nu drift)
    print(f"""
  PARAMETRI DISTRIBUZIONE (media):
    μ medio  : {df['mu'].mean():+.6f}  (std: {df['mu'].std():.6f})
    σ medio  : {df['sigma'].mean():.6f}  (vol per candela)
    ν medio  : {df['nu'].mean():.2f}   (gradi di libertà t-Student)
    P↑ medio : {df['prob_up'].mean():.3f}""")

    # IT: equity curve di sessione + max drawdown via cummax
    # EN: session equity curve + max drawdown via cummax
    if "equity" in df.columns and df["equity"].notna().any():
        eq_start = df["equity"].iloc[0]
        eq_end   = df["equity"].iloc[-1]
        eq_min   = df["equity"].min()
        eq_max   = df["equity"].max()
        ret      = (eq_end - eq_start) / eq_start

        roll_max = df["equity"].cummax()
        dd       = (roll_max - df["equity"]) / roll_max
        max_dd   = dd.max()

        print(f"""
  EQUITY SESSIONE:
    Inizio    : ${eq_start:,.2f}
    Fine      : ${eq_end:,.2f}
    Rendimento: {ret:+.2%}
    Min       : ${eq_min:,.2f}
    Max       : ${eq_max:,.2f}
    Max DD    : {max_dd:.2%}""")

    # IT: ultimo session_*.json (scritto da 04 al shutdown) per metriche reali
    # EN: last session_*.json (written by 04 on shutdown) for real metrics
    session_files = sorted(results_dir.glob("session_*.json"))
    if session_files:
        latest = session_files[-1]
        with open(latest, encoding="utf-8") as f:
            sess = json.load(f)
        print(f"""
  ULTIMA SESSIONE ({latest.name}):
    N° trade    : {sess.get('n_trades', 0)}
    Win rate    : {sess.get('win_rate', 0):.1%}
    Profit fact : {sess.get('profit_factor', 0):.2f}
    Sharpe      : {sess.get('sharpe', 0):.2f}
    Max DD      : {sess.get('max_drawdown', 0):.2%}""")

    # IT: merge dati live in dashboard_results.json per la tab BACKTEST
    # EN: merge live data into dashboard_results.json for BACKTEST tab
    dashboard_path = RESULTS_FILE
    if dashboard_path.exists():
        with open(dashboard_path, encoding="utf-8") as f:
            dashboard = json.load(f)
    else:
        dashboard = {"metrics": {}, "equity_curve": [], "trades": []}

    # IT: downsample lineare a 300 punti per ridurre payload dashboard
    # EN: linear downsample to 300 points to shrink dashboard payload
    def downsample(arr, n=300):
        if len(arr) <= n: return arr
        idx = np.linspace(0, len(arr)-1, n, dtype=int)
        return [arr[i] for i in idx]

    # IT: JSON non supporta NaN/Inf -> sanifica a None
    # EN: JSON does not support NaN/Inf -> sanitize to None
    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
        return v

    live_eq = df["equity"].dropna().tolist()
    dashboard["live_equity_curve"] = downsample([float(v) for v in live_eq])
    dashboard["live_signal_distribution"] = {
        k: int(v) for k, v in sig_counts.items()
    }
    dashboard["live_mu_series"] = downsample([clean(float(v)) for v in df["mu"]])
    dashboard["live_period"] = {
        "start": df["ts"].iloc[0].isoformat(),
        "end":   df["ts"].iloc[-1].isoformat(),
        "n_records": len(df),
    }

    with open(dashboard_path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, separators=(",", ":"))

    print(f"""
  Dashboard aggiornata → {dashboard_path}
  Ricarica la tab BACKTEST nella dashboard React.
{'═'*60}
""")


if __name__ == "__main__":
    main()

"""
Script 03 — Backtest + export risultati per la dashboard React.

Run configuration PyCharm:
  Script: scripts/03_backtest.py
  Working dir: <root del progetto>
"""
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# IT: Forza UTF-8 su stdout/stderr — Windows cp1252 crasha sui box-drawing.
# EN: Force UTF-8 on stdout/stderr — Windows cp1252 crashes on box-drawing chars.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml as _yaml
with open(Path(__file__).resolve().parent.parent / "config" / "default.yaml", encoding="utf-8") as _f:
    _cpu_frac = _yaml.safe_load(_f).get("hardware", {}).get("cpu_fraction", 0.5)
_cpu_limit = str(max(1, int(os.cpu_count() * _cpu_frac)))
os.environ.setdefault("OMP_NUM_THREADS", _cpu_limit)
os.environ.setdefault("MKL_NUM_THREADS", _cpu_limit)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

torch.set_num_threads(int(_cpu_limit))

from quantsys.utils import load_config, setup_logging, setup_device, ensure_dirs, PipelineState
from quantsys.model import load_model
from quantsys.model.ensemble import EnsembleModel
from quantsys.trading import RiskManager, SignalGenerator, Side, CloseReason

setup_logging()
log = logging.getLogger("quantsys.script.03")


# IT: Sottocampiona un array a max n punti (per payload dashboard leggero).
# EN: Downsample an array to at most n points (keeps the dashboard payload light).
def _downsample(arr, n=500):
    if len(arr) <= n: return arr
    idx = np.linspace(0, len(arr)-1, n, dtype=int)
    return [arr[i] for i in idx]


# IT: Intervalli bootstrap su Sharpe/Sortino — annualizzati a 1-minuto (525600).
# EN: Bootstrap CI for Sharpe/Sortino — annualised at 1-minute basis (525600).
def bootstrap_sharpe_ci(pnl_list, n_boot=5000, confidence=0.95, annualize=525600):
    """Bootstrap confidence interval per Sharpe e Sortino (annualizzati 1m)."""
    if len(pnl_list) < 30:
        return {"sharpe_ci_low": None, "sharpe_ci_high": None,
                "sortino_ci_low": None, "sortino_ci_high": None}
    arr = np.array(pnl_list, dtype=np.float64)
    rng = np.random.default_rng(42)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    means = samples.mean(axis=1)
    stds = samples.std(axis=1) + 1e-10
    scale = np.sqrt(annualize / len(arr))
    sharpes = (means / stds) * scale
    neg_mask = samples < 0
    neg_counts = neg_mask.sum(axis=1)
    neg_sums = np.where(neg_mask, samples, 0.0)
    neg_means = np.where(neg_counts > 0, neg_sums.sum(axis=1) / np.maximum(neg_counts, 1), 0.0)
    neg_sq_sums = np.where(neg_mask, samples**2, 0.0).sum(axis=1)
    neg_var = np.where(neg_counts > 1,
                       neg_sq_sums / neg_counts - neg_means**2,
                       0.0)
    neg_std = np.sqrt(np.maximum(neg_var, 0.0)) + 1e-10
    dstds = np.where(neg_counts > 1, neg_std, stds)
    sortinos = (means / dstds) * scale
    alpha = (1 - confidence) / 2
    return {
        "sharpe_ci_low":  float(np.percentile(sharpes, alpha*100)),
        "sharpe_ci_high": float(np.percentile(sharpes, (1-alpha)*100)),
        "sortino_ci_low":  float(np.percentile(sortinos, alpha*100)),
        "sortino_ci_high": float(np.percentile(sortinos, (1-alpha)*100)),
    }


# IT: Durata e recovery time del max drawdown sull'equity curve.
# EN: Duration and recovery time of the max drawdown along the equity curve.
def mdd_stats(equity_arr):
    """Calcola durata e recovery time del max drawdown."""
    eq = np.array(equity_arr, dtype=np.float64)
    peak_idx, trough_idx, recovery_idx = 0, 0, 0
    max_dd_val = 0.0
    running_peak = eq[0]; running_peak_idx = 0
    for i in range(1, len(eq)):
        if eq[i] >= running_peak:
            running_peak = eq[i]; running_peak_idx = i
        dd = (running_peak - eq[i]) / running_peak if running_peak > 0 else 0
        if dd > max_dd_val:
            max_dd_val = dd
            peak_idx = running_peak_idx; trough_idx = i
    # IT: Recovery = prima candela post-trough che riallinea il peak precedente.
    # EN: Recovery = first post-trough bar that recovers the previous peak.
    recovery_idx = len(eq) - 1  # IT: default non recuperato | EN: default not recovered
    for i in range(trough_idx, len(eq)):
        if eq[i] >= eq[peak_idx]:
            recovery_idx = i; break
    recovered = eq[recovery_idx] >= eq[peak_idx]
    return {
        "mdd_duration_candles": int(trough_idx - peak_idx),
        "mdd_recovery_candles": int(recovery_idx - trough_idx) if recovered else None,
        "mdd_recovered": bool(recovered),
    }


# IT: Riesegue il backtest con fee/slippage stressati su segnali pre-calcolati.
# EN: Re-runs the backtest with stressed fee/slippage on pre-computed signals.
def run_stress_scenario(scenario_name, fee_mult, slip_mult, cfg_backtest, cfg_risk,
                        ohlcv, atr, pre_signals, adv_1m=None, seed_rm=None):
    """Replica il backtest con parametri di stress usando segnali pre-calcolati.

    pre_signals: lista di (side, dist) gia' calcolati dal loop principale.
    adv_1m: array di average daily volume 1m in USD (per slippage sqrt).
    seed_rm: RiskManager principale di cui ereditare lo storico (fix #16). Senza
      questo, autocorr_kelly_factor parte da 1.0 sottostimando rischio direzionale.
    """
    # IT: Stress scenario riusa i segnali già emessi dal loop principale; cambia
    #     solo fee/slippage e si reinizializza un RiskManager dedicato.
    # EN: Stress scenarios reuse signals emitted by the main loop and only swap
    #     fee/slippage on a fresh RiskManager.
    rm_s = RiskManager(
        initial_capital    = cfg_risk["initial_capital"],
        max_risk_per_trade = cfg_risk["max_risk_per_trade"],
        sl_atr_mult        = cfg_risk["sl_atr_mult"],
        tp_rr_ratio        = cfg_risk["tp_rr_ratio"],
        max_position_pct   = cfg_risk["max_position_pct"],
        max_drawdown_stop  = cfg_risk["max_drawdown_stop"],
        max_hold_candles   = cfg_risk["max_hold_candles"],
        use_trailing_stop  = cfg_risk["use_trailing_stop"],
        trailing_atr_mult  = cfg_risk["trailing_atr_mult"],
        fee_rate           = cfg_backtest["fee_rate"] * fee_mult,
        slippage_rate      = cfg_backtest["slippage_rate"] * slip_mult,
        slippage_model     = cfg_backtest.get("slippage_model", "fixed"),
        correlation_window = cfg_risk.get("correlation_window", 10),
        max_directional_exposure = cfg_risk.get("max_directional_exposure", 0.6),
    )
    # IT: Fix #16 — eredita storico direzionale/return dal RM principale, altrimenti
    #     autocorr_kelly_factor parte a 1.0 e sottostima il rischio direzionale.
    # EN: Fix #16 — inherit directional/return history from the main RM, otherwise
    #     autocorr_kelly_factor resets to 1.0 and underestimates directional risk.
    if seed_rm is not None:
        import copy
        rm_s._recent_sides         = copy.deepcopy(seed_rm._recent_sides)
        rm_s._recent_trade_returns = copy.deepcopy(seed_rm._recent_trade_returns)
    n = len(ohlcv)
    for i in range(n - 1):
        if rm_s.circuit_breaker: break
        o_c, h_c, l_c, c_c = ohlcv[i]
        o_n, h_n, l_n, c_n = ohlcv[i+1]
        atr_i = max(atr[i], c_c * 0.0005)
        adv_i = float(adv_1m[i]) if adv_1m is not None else 0.0
        side, dist = pre_signals[i]
        if rm_s.position: rm_s.update_trailing(c_c, atr_i)
        if rm_s.position:
            reason = rm_s.check_exit(h_n, l_n, c_n, i+1, side)
            if reason:
                ep = rm_s.position.stop_loss if reason == CloseReason.STOP_LOSS else \
                     rm_s.position.take_profit if reason == CloseReason.TAKE_PROFIT else c_n
                rm_s.close_position(reason, ep, i+1, adv_1m=adv_i)
        if side != Side.NONE and not rm_s.position:
            rm_s.open_position(side, o_n, i+1, atr_i, dist, adv_1m=adv_i)
    if rm_s.position: rm_s.close_position(CloseReason.END_OF_DATA, ohlcv[-1][3], n-1)
    ms = rm_s.metrics()

    # IT: NaN/inf → None così il JSON resta valido. | EN: NaN/inf → None to keep valid JSON.
    def _clean(v): return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v

    return {
        "scenario": scenario_name,
        "fee_mult": fee_mult, "slip_mult": slip_mult,
        "sharpe": _clean(ms.get("sharpe")),
        "max_drawdown": _clean(ms.get("max_drawdown")),
        "total_return": _clean(ms.get("total_return")),
        "n_trades": ms.get("n_trades", 0),
        "final_equity": _clean(ms.get("final_equity")),
    }


# IT: Entry point — inferenza batch, loop backtest, stress test ed export dashboard.
# EN: Entry point — batch inference, backtest loop, stress tests and dashboard export.
def main():
    cfg  = load_config("config/default.yaml")
    bcfg = cfg["backtest"]
    rcfg = cfg["risk"]
    mcfg = cfg["model"]
    # IT: La policy Spectral Norm deve matchare il training per caricare i checkpoint.
    # EN: Spectral-norm policy must match training to load checkpoints correctly.
    from quantsys.model import set_sn_on_mu_only
    set_sn_on_mu_only(bool(cfg.get("training", {}).get("sn_on_mu_only", False)))

    device = setup_device(cfg)
    out = Path(cfg["training"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    results_out = Path(bcfg["output_dir"])
    ensure_dirs(bcfg["output_dir"])
    results_out.mkdir(parents=True, exist_ok=True)

    # IT: Split di valutazione — QUANTSYS_BACKTEST_SPLIT=val|test (default test).
    #     'val' valida una config (es. Quiet rank-entry) su un periodo HELD-OUT senza
    #     tararla sul test set; 'test' è lo stato production. Tutti i tensori (X/y/t e
    #     macro) e l'allineamento OHLCV/regimi seguono lo split scelto.
    # EN: Evaluation split — QUANTSYS_BACKTEST_SPLIT=val|test (default test).
    #     'val' validates a config (e.g. Quiet rank-entry) on a HELD-OUT period without
    #     tuning on the test set; 'test' is the production state. All tensors (X/y/t and
    #     macro) and the OHLCV/regime alignment follow the chosen split.
    _split = os.environ.get("QUANTSYS_BACKTEST_SPLIT", "test").strip().lower()
    if _split not in ("val", "test"):
        raise ValueError(
            f"QUANTSYS_BACKTEST_SPLIT='{_split}' non valido — usa 'val' o 'test'."
        )
    # IT: Suffisso output — solo 'test' scrive i file production-clean
    #     (metrics.json/dashboard_results.json); 'val' va su *_val per non clobberarli.
    # EN: Output suffix — only 'test' writes the production-clean files; 'val' goes to
    #     *_val so a validation run never clobbers the production state.
    _out_suffix = "" if _split == "test" else f"_{_split}"

    # IT: Dataset serializzato (split scelto) — input al backtest e all'inferenza batch.
    # EN: Serialized dataset (chosen split) — feeds both backtest loop and batch inference.
    data = np.load("data/lstm_dataset.npz", allow_pickle=True)
    X, y = data[f"X_{_split}"], data[f"y_{_split}"]
    if _split != "test":
        log.warning(
            f"BACKTEST SPLIT = '{_split.upper()}' (held-out validation, NON production). "
            "I file metrics.json/dashboard_results.json riflettono il VAL, non il test."
        )

    # IT: ETA pessimistica per-sample; il batch reale è ~10-20x più veloce.
    # EN: Pessimistic per-sample ETA; actual batch path is ~10-20x faster.
    n_samples = len(X)
    dev_name   = str(device)
    ms_per_sam = 0.15 if device.type == "cuda" else 2.0
    eta_sec    = n_samples * ms_per_sam / 1000
    log.info(
        f"{_split.upper()} set: {n_samples:,} campioni  |  {X.shape[1]}w × {X.shape[2]}f  |  "
        f"device={dev_name}  |  ETA stimata ≤ {eta_sec:.0f}s "
        f"({'GPU batch' if device.type == 'cuda' else 'CPU batch'})"
    )

    # IT: OHLCV reali da raw_candles.parquet (USD raw). Mai usare features.parquet:
    #     contiene OHLCV scalati dal RobustScaler e SL/TP scatterebbero su prezzi inventati.
    # EN: Use raw_candles.parquet (raw USD) for OHLCV. Never features.parquet: those
    #     prices are RobustScaler-scaled and SL/TP would trigger on fictional bars.
    parquet_path = Path("data/raw_candles.parquet")
    t_eval       = pd.to_datetime(data[f"t_{_split}"])   # IT: timestamp split | EN: split timestamps

    if parquet_path.exists():
        log.info("Caricamento OHLCV reali da raw_candles.parquet ...")
        df_feat  = pd.read_parquet(parquet_path,
                                   columns=["open_time","open","high","low","close","volume"])
        df_feat["open_time"] = pd.to_datetime(df_feat["open_time"], utc=True)

        # IT: merge_asof tollera piccoli scarti (tz, arrotondamenti) entro 2 min.
        # EN: merge_asof absorbs small skew (tz, rounding) up to a 2-min tolerance.
        t_eval_df = pd.DataFrame({"open_time": pd.to_datetime(t_eval, utc=True)})
        t_eval_df = t_eval_df.sort_values("open_time")
        n_eval_orig = len(t_eval_df)   # IT: lunghezza split PRIMA del merge | EN: split length BEFORE the merge
        merged    = pd.merge_asof(
            t_eval_df,
            df_feat.sort_values("open_time"),
            on="open_time",
            direction="nearest",
            tolerance=pd.Timedelta("2min"),   # IT: scarto max | EN: max skew
        )

        # IT: Fix #1 — i gap Binance possono far duplicare/saltare righe in merge_asof
        #     e disallineare gli indici posizionali del loop. RuntimeError (no assert).
        # EN: Fix #1 — Binance gaps can dupe/skip rows in merge_asof and break the
        #     positional indexing of the main loop. RuntimeError (not assert).
        if len(merged) != n_eval_orig:
            raise RuntimeError(
                f"merge_asof ha alterato il numero di righe ({len(merged)} != {n_eval_orig}). "
                f"Possibile gap Binance nello split '{_split}' — verifica raw_candles.parquet."
            )

        # IT: Verifica copertura — sotto 80% fallback su prezzi ricostruiti.
        # EN: Coverage check — below 80% fall back to reconstructed prices.
        n_matched = merged[["open","high","low","close"]].notna().all(axis=1).sum()
        n_total   = len(merged)
        coverage  = n_matched / n_total
        log.info(f"OHLCV reali: {n_matched}/{n_total} candele ({coverage:.1%}) allineate")

        if coverage < 0.80:
            log.warning(
                f"Copertura OHLCV bassa ({coverage:.1%}). "
                f"Verifica che raw_candles.parquet copra il periodo dello split '{_split}'. "
                "Fallback: prezzi ricostruiti dai log-return."
            )

        # IT: Reimposta l'indice posizionale per allinearsi al test set originale.
        # EN: Reset positional index to align with the original test set ordering.
        merged = merged.set_index(pd.RangeIndex(n_total))

        opens   = merged["open"].ffill().values
        highs   = merged["high"].ffill().values
        lows    = merged["low"].ffill().values
        closes  = merged["close"].ffill().values
        volumes = merged["volume"].ffill().fillna(0).values

        log.info(
            f"Prezzi reali: close [{closes.min():,.0f} – {closes.max():,.0f}]  "
            f"ATR medio {(highs-lows).mean():.0f}"
        )
    else:
        log.warning(
            "data/raw_candles.parquet non trovato — uso prezzi ricostruiti dai log-return. "
            "Esegui prima 01_download_data.py per avere prezzi reali."
        )
        # IT: Fallback meno accurato — prezzi ricostruiti dai log-return.
        # EN: Less accurate fallback — prices reconstructed from log-returns.
        start_price = 67_000.0
        closes_list = [start_price]
        for r in y:
            closes_list.append(closes_list[-1] * np.exp(r))
        closes = np.array(closes_list[1:])
        np.random.seed(42)
        noise  = np.abs(np.random.normal(0, 0.0008, len(closes)))
        highs  = closes * (1 + noise)
        lows   = closes * (1 - noise)
        opens  = np.roll(closes, 1); opens[0] = start_price
        volumes = np.zeros(len(closes))  # IT: no volume nel fallback | EN: no volume in fallback

    ohlcv = np.stack([opens, highs, lows, closes], axis=1)

    # IT: ATR(14) tramite True Range standard (Wilder).
    # EN: ATR(14) computed from the standard Wilder True Range.
    c_prev = np.roll(closes, 1); c_prev[0] = closes[0]
    tr  = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - c_prev), np.abs(lows - c_prev))
    )
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    # IT: ADV_1m — MA20 del volume USD; input al modello di slippage Almgren-Chriss.
    # EN: ADV_1m — 20-bar MA of USD volume; feeds the Almgren-Chriss slippage model.
    vol_usd = volumes * closes
    adv_1m  = pd.Series(vol_usd).rolling(20, min_periods=1).mean().values

    # ── Modello / Model ───────────────────────────────────────────────────────
    # IT: Carica ensemble (eterogeneo o omogeneo) + config; fallback EWM se assente.
    # EN: Load ensemble (heterogeneous or homogeneous) + config; EWM fallback if absent.
    model_cfg  = {}
    has_macro  = False
    use_model  = False
    Xm_test    = None

    models_dir = Path(cfg["training"]["output_dir"])
    try:
        # IT: Preferisci ensemble eterogeneo (≥2 archs in distillation.archs);
        #     fallback su ensemble omogeneo se i checkpoint mancano.
        #     L'env var QUANTSYS_BACKTEST_SINGLE_ARCH=1 forza il path omogeneo
        #     anche se più archs sono disponibili — usato da run_all.py per
        #     produrre backtest per-arch comparabili dopo distillation.
        # EN: Prefer heterogeneous ensemble (>=2 archs in distillation.archs);
        #     fall back to homogeneous ensemble when checkpoints are missing.
        #     QUANTSYS_BACKTEST_SINGLE_ARCH=1 env var forces the homogeneous
        #     path even when multiple archs exist — used by run_all.py to
        #     produce per-arch comparable backtests after distillation.
        from quantsys.model.ensemble import get_distillation_archs
        _archs = get_distillation_archs(cfg)
        _het_available = sum(1 for a in _archs
                            if (Path("models") / a / "best_model.pt").exists())
        _single_arch_mode = os.environ.get("QUANTSYS_BACKTEST_SINGLE_ARCH") == "1"
        if _het_available >= 2 and not _single_arch_mode:
            model = EnsembleModel.load_heterogeneous(device, cfg=cfg)
            log.info(f"Ensemble ETEROGENEO: {model.n_members} architetture "
                     f"[{', '.join(model.arch_names)}]")
        else:
            model = EnsembleModel.load(str(models_dir), device)
            _mode_tag = "SINGLE-ARCH" if _single_arch_mode else "omogeneo"
            log.info(f"Ensemble {_mode_tag}: {model.n_members} membri da "
                     f"{models_dir} caricati")
        import json as _json
        with open(models_dir / "config.json", encoding="utf-8") as _f:
            model_cfg = _json.load(_f)
        has_macro = model_cfg.get("has_macro", False)
        use_model = True
        log.info(f"Modello caricato: {model_cfg.get('model_type','?')}  has_macro={has_macro}")

        # IT: Carica X_macro_{split} se il modello ha branch macro; altrimenti zeros.
        # EN: Load X_macro_{split} when the model has a macro branch; zeros otherwise.
        if has_macro and f"X_macro_{_split}" in data.files:
            Xm_test = torch.tensor(data[f"X_macro_{_split}"], dtype=torch.float32)
            log.info(f"X_macro_{_split} caricato: {tuple(Xm_test.shape)}")
        elif has_macro:
            log.warning(f"Modello macro ma X_macro_{_split} non trovato nel dataset — "
                        "usa zeros come fallback.")
            n_macro = model_cfg.get("n_macro", 1)
            Xm_test = torch.zeros(len(X), n_macro, dtype=torch.float32)

    except Exception as e:
        log.warning(f"Modello non trovato ({e}) — uso SimpleSignalModel (rolling EWM).")

    # IT: Senza modello, precalcoliamo rolling stats una sola volta (fallback EWM).
    # EN: Without a model, precompute rolling stats once (EWM fallback path).
    if not use_model:
        # IT: feature 0 = log_ret (ordine definito da feature_engineering.py).
        # EN: feature 0 = log_ret (ordering fixed by feature_engineering.py).
        all_rets = X[:, :, 0]

        rets_last = all_rets[:, -1]

        # IT: EWM span 5 (fast) / 20 (slow) sull'ultimo log-return — proxy di trend.
        # EN: EWM span 5 (fast) / 20 (slow) on the latest log-return — trend proxy.
        s = pd.Series(rets_last)
        fast_ema = s.ewm(span=5,  adjust=False, min_periods=1).mean().values
        slow_ema = s.ewm(span=20, adjust=False, min_periods=1).mean().values

        # IT: σ stimata dalla coda 20-bar di ciascuna finestra (vettorizzato).
        # EN: σ estimated from the trailing 20-bar tail of every window (vectorised).
        _tail = all_rets[:, -20:]
        rolling_std = np.maximum(_tail.std(axis=1, ddof=1), 1e-5)

        log.info(f"SimpleSignalModel pre-calcolato: "
                 f"fast_ema range [{fast_ema.min():.5f}, {fast_ema.max():.5f}]  "
                 f"vol medio={rolling_std.mean():.5f}")

    # ── Batch inference (Fix 1 — 200k campioni) ──────────────────────────────
    # IT: Pre-calcola TUTTE le predizioni in batch (poi lookup O(1) nel loop):
    #     200k forward pass sample-per-sample sono lenti anche su GPU per
    #     l'overhead di scheduling dei kernel CUDA.
    # EN: Pre-compute ALL predictions in batches (then O(1) lookup in the loop):
    #     200k per-sample forward passes are slow even on GPU due to CUDA kernel
    #     scheduling overhead.
    #
    if use_model:
        # IT: 2026-05-16: ridotto da 4096 a 256 per OOM su tcnmamba (Mamba SSM alloca
        #     tensori (B,T,d_inner,d_state) ~17MB/sample → 4096 = 70GB working set);
        #     256 mantiene la VRAM <3 GiB anche col branch Mamba attivo.
        # EN: 2026-05-16: lowered 4096→256 to avoid tcnmamba OOM (Mamba SSM allocates
        #     (B,T,d_inner,d_state) ~17MB/sample → 4096 = 70GB working set); 256 keeps
        #     VRAM <3 GiB even with the Mamba branch on.
        BATCH_SIZE = 256
        all_mu    = np.zeros(n_samples, dtype=np.float32)
        all_sigma = np.zeros(n_samples, dtype=np.float32)
        all_nu    = np.zeros(n_samples, dtype=np.float32)

        log.info(
            f"Batch inference: {n_samples:,} campioni  |  "
            f"batch_size={BATCH_SIZE}  |  "
            f"n_batch={math.ceil(n_samples/BATCH_SIZE)}"
        )
        model.eval()
        with torch.no_grad():
            for b_start in tqdm(range(0, n_samples, BATCH_SIZE),
                                desc="Batch inference", ncols=72):
                b_end = min(b_start + BATCH_SIZE, n_samples)
                xb    = torch.tensor(X[b_start:b_end], dtype=torch.float32).to(device)

                if has_macro and Xm_test is not None:
                    xm       = Xm_test[b_start:b_end].to(device)
                    mu_b, sigma_b, nu_b = model(xb, xm)
                else:
                    mu_b, sigma_b, nu_b = model(xb)

                all_mu[b_start:b_end]    = mu_b.squeeze(-1).cpu().numpy()
                all_sigma[b_start:b_end] = sigma_b.squeeze(-1).cpu().numpy()
                all_nu[b_start:b_end]    = nu_b.squeeze(-1).cpu().numpy()

        log.info("Batch inference completata — predizioni pre-calcolate per tutti i campioni")

        # ── Denormalizzazione μ/σ via PipelineState (fix bug z-score 2026-05-23) ──
        # IT: Il modello predice in spazio z-score; il trading layer opera in raw.
        #     La conversione centralizzata in PipelineState.denormalize_predictions
        #     previene drift tra script (vedi anche 04_live_signals.py).
        # EN: The model predicts in z-score space; the trading layer works in raw.
        #     Centralising the conversion in PipelineState.denormalize_predictions
        #     prevents drift across scripts (see also 04_live_signals.py).
        _ps_path = Path("models") / os.environ.get("QUANTSYS_ARCH", "lstm") / "pipeline_state.pkl"
        if not _ps_path.exists():
            _ps_path = Path("models/lstm/pipeline_state.pkl")
        if _ps_path.exists():
            _state = PipelineState.load(str(_ps_path))
            # IT: Fix #22 — blocca l'errore utente se forecast_horizon è cambiato tra
            #     training e backtest. La chiave è features.forecast_horizon (non data.)
            #     — fallback su data. per i pipeline state legacy.
            # EN: Fix #22 — guard against user error if forecast_horizon changed between
            #     training and backtest. Key is features.forecast_horizon (not data.)
            #     — fall back to data. for legacy pipeline states.
            _cfg_h = cfg.get("features", {}).get("forecast_horizon",
                       cfg.get("data", {}).get("forecast_horizon", 15))
            _state_h = _state.forecast_horizon
            if _cfg_h != _state_h:
                raise RuntimeError(
                    f"forecast_horizon mismatch: config={_cfg_h}, training={_state_h}. "
                    f"Il modello è stato addestrato per orizzonte {_state_h}; backtest a {_cfg_h} "
                    f"produce metriche invalide. Allinea config/default.yaml o rigenera il modello."
                )
            # IT: max_hold_candles ≥ forecast_horizon (non chiudere prima dell'orizzonte).
            # EN: max_hold_candles ≥ forecast_horizon (don't close before the horizon is covered).
            _max_hold = rcfg.get("max_hold_candles", 0)
            if _max_hold < _state_h:
                log.warning(
                    f"max_hold_candles ({_max_hold}) < forecast_horizon ({_state_h}). "
                    f"Il TP/SL potrebbe non avere tempo di triggerare prima del MAX_HOLD."
                )
            all_mu, all_sigma = _state.denormalize_predictions(all_mu, all_sigma)
            # IT: Step 0.5 — ricalibrazione σ sperimentale (env, default 1.0 = INERTE). Lo Step 0
            #     (scripts/dev_step0_regime_sigma.py) ha mostrato σ ~3× troppo grande (std(z)≈0.4)
            #     → SL/TP (σ·price·1.5) ~3× troppo larghi. Scala σ post-denorm per misurare l'impatto
            #     su SL/TP, Kelly (f*=μ/σ²) e gate SNR (|μ|/σ). Reversibile; se promosso va bakato in
            #     PipelineState.denormalize_predictions (parity-safe: identico backtest↔live).
            # EN: Step 0.5 — experimental σ recalibration (env, default 1.0 = INERT). Step 0 showed
            #     σ is ~3× too large → SL/TP too wide. Scales σ post-denorm to measure trading impact.
            _sigma_scale = float(os.environ.get("QUANTSYS_SIGMA_SCALE", "1") or "1")
            if _sigma_scale != 1.0:
                all_sigma = all_sigma * _sigma_scale
                log.info(f"σ RICALIBRATA sperimentale: ×{_sigma_scale} (QUANTSYS_SIGMA_SCALE)")
            log.info(
                f"μ/σ denormalizzate (target_scale={_state.target_scale:.6f}) → "
                f"μ range [{all_mu.min():.5f}, {all_mu.max():.5f}], "
                f"σ range [{all_sigma.min():.5f}, {all_sigma.max():.5f}]"
            )
            # IT: Safety check — σ raw > 5%/min su BTC è fisicamente impossibile: se scatta
            #     la denormalizzazione è mancata o lo scaler è rotto. RuntimeError (non
            #     assert) perché `python -O` rimuove gli assert anche in produzione.
            # EN: Safety check — raw σ > 5%/min on BTC is physically impossible: if it fires
            #     denormalization was skipped or the scaler is broken. RuntimeError (not
            #     assert) since `python -O` strips asserts even in production builds.
            if all_sigma.max() >= 0.05:
                raise RuntimeError(
                    f"σ post-denorm = {all_sigma.max():.4f} >= 0.05. "
                    "Probabile mancata denormalizzazione (target_scale = 1.0?) "
                    "o scaler corrotto."
                )
        else:
            log.warning(f"pipeline_state.pkl non trovato in {_ps_path} — μ/σ in spazio z-score!")
    else:
        # IT: Array placeholder per lo stress test quando non c'è modello (use_model=False).
        # EN: Placeholder arrays for the stress test when no model is loaded (use_model=False).
        all_mu    = np.zeros(n_samples, dtype=np.float32)
        all_sigma = np.full(n_samples, 0.001, dtype=np.float32)
        all_nu    = np.full(n_samples, 5.0,   dtype=np.float32)

    # IT: Restituisce (mu, sigma, nu) per il sample step_idx: lookup o fallback EWM.
    # EN: Returns (mu, sigma, nu) for sample step_idx: array lookup or EWM fallback.
    def predict(window: np.ndarray, step_idx: int):
        if use_model:
            # IT: Lookup O(1) negli array pre-calcolati — forward pass già fatto.
            # EN: O(1) lookup into the pre-computed arrays — forward pass already done.
            return float(all_mu[step_idx]), float(all_sigma[step_idx]), float(all_nu[step_idx])

        # IT: Fallback — usa le rolling stats EWM pre-calcolate.
        # EN: Fallback — use the pre-computed EWM rolling stats.
        mu  = float(fast_ema[step_idx] * 0.6 + slow_ema[step_idx] * 0.4)
        sig = float(rolling_std[step_idx])
        nu  = float(np.clip(5.0 + (0.002 - sig) / 0.0005, 3.0, 12.0))
        return mu, sig, nu

    # ── Setup risk + signal ──────────────────────────────────────────────────
    rm = RiskManager(
        initial_capital    = rcfg["initial_capital"],
        max_risk_per_trade = rcfg["max_risk_per_trade"],
        sl_atr_mult        = rcfg["sl_atr_mult"],
        tp_rr_ratio        = rcfg["tp_rr_ratio"],
        max_position_pct   = rcfg["max_position_pct"],
        max_drawdown_stop  = rcfg["max_drawdown_stop"],
        max_hold_candles   = rcfg["max_hold_candles"],
        use_trailing_stop  = rcfg["use_trailing_stop"],
        trailing_atr_mult  = rcfg["trailing_atr_mult"],
        fee_rate           = bcfg["fee_rate"],
        slippage_rate      = bcfg["slippage_rate"],
        slippage_model     = bcfg.get("slippage_model", "fixed"),
        correlation_window       = rcfg.get("correlation_window", 10),
        max_directional_exposure  = rcfg.get("max_directional_exposure", 0.6),
    )
    sig_gen = SignalGenerator(
        prob_threshold   = bcfg["prob_threshold"],
        min_expected_ret = bcfg["min_expected_ret"],
        max_sigma        = bcfg["max_sigma"],
        conviction_alpha = bcfg.get("conviction_alpha", 0.5),
        # IT: SNR gate da config (default 0.0 = disattivato) — quick-win #2
        # EN: SNR gate from config (default 0.0 = disabled) — quick-win #2
        min_snr          = bcfg.get("min_snr", 0.0),
    )

    # ── Backtest loop ────────────────────────────────────────────────────────
    n = len(X)
    equity_ts, dd_ts = [rcfg["initial_capital"]], [0.0]
    peak = rcfg["initial_capital"]
    pre_signals = []

    # IT: Tracking per regime — separa le metriche per alta/bassa volatilità;
    #     "alta vol" = ATR > mediana ATR del test set (proxy di regime difficile).
    # EN: Regime-conditioned tracking — split metrics by high/low volatility;
    #     "high vol" = ATR > test-set median ATR (proxy for a hard regime).
    atr_median = np.median(atr)
    regime_trades = {"high_vol": [], "low_vol": []}
    log.info(f"ATR mediana split '{_split}': {atr_median:.0f}  (soglia high/low vol)")

    # IT: Carica i regimi BTC `RegimeMarkovBTC` (Quiet/Trending/Stress) allineati a t_eval.
    #     Sostituisce il vecchio proxy ATR-based usato nel loop per `rm.set_regime(...)`.
    #     Se il file non esiste o l'allineamento fallisce, fallback al proxy ATR storico.
    # EN: Loads BTC `RegimeMarkovBTC` regimes (Quiet/Trending/Stress) aligned to t_eval.
    #     Replaces the previous ATR-proxy used in the loop for `rm.set_regime(...)`.
    #     Falls back to the historical ATR proxy if the file is missing or alignment fails.
    btc_regime_per_step: "np.ndarray | None" = None
    try:
        reg_path = Path("data") / "regime_probs.parquet"
        if reg_path.exists():
            df_reg = pd.read_parquet(reg_path)
            if "regime_dominant" in df_reg.columns:
                # IT: normalizza indice/timestamp a UTC ns-naive per merge_asof.
                # EN: normalize index/timestamp to UTC ns-naive for merge_asof.
                if not isinstance(df_reg.index, pd.DatetimeIndex):
                    tcol = next((c for c in ("open_time", "timestamp", "date")
                                 if c in df_reg.columns), None)
                    if tcol is not None:
                        df_reg = df_reg.set_index(pd.to_datetime(df_reg[tcol])).sort_index()
                df_reg = df_reg.sort_index()
                def _to_ns_naive(s):
                    s = pd.to_datetime(s)
                    if getattr(s, "tz", None) is not None:
                        s = s.tz_convert("UTC").tz_localize(None)
                    return s.astype("datetime64[ns]")
                df_reg.index = _to_ns_naive(df_reg.index)
                t_step = _to_ns_naive(pd.Index(pd.to_datetime(t_eval)))
                df_step = pd.DataFrame({"_t": t_step})
                merged = pd.merge_asof(
                    df_step.sort_values("_t"),
                    df_reg[["regime_dominant"]].reset_index().rename(
                        columns={df_reg.index.name or "index": "_t"}
                    ),
                    on="_t", direction="backward",
                )
                regimes = merged["regime_dominant"].to_numpy()
                order = np.argsort(np.argsort(t_step.values))
                btc_regime_per_step = regimes[order].astype(np.int64)
                _counts = {int(k): int(v) for k, v in
                           pd.Series(btc_regime_per_step).value_counts().sort_index().items()}
                log.info(f"RegimeMarkovBTC allineato a t_{_split}: distribuzione {_counts}")
    except Exception as _e:
        log.warning(f"RegimeMarkovBTC alignment fallito ({_e}) — fallback proxy ATR")
        btc_regime_per_step = None

    # IT: Regime-gating sperimentale (2026-06-04) — env-controlled, reversibile.
    #     QUANTSYS_REGIME_ALLOW="0,2"  → entra SOLO in questi regimi (altrove side=NONE).
    #     QUANTSYS_REGIME_INVERT="1"   → inverte il side in questi regimi (anti-edge → edge).
    #     Basato su edge per-regime: Quiet(0)=+0.13, Trending(1)=-0.13, Stress(2)=+0.04.
    # EN: Experimental regime-gating (2026-06-04) — env-controlled, reversible.
    #     QUANTSYS_REGIME_ALLOW="0,2"  → enter ONLY in these regimes (NONE elsewhere).
    #     QUANTSYS_REGIME_INVERT="1"   → flip the side in these regimes (anti-edge → edge).
    _rg_allow  = os.environ.get("QUANTSYS_REGIME_ALLOW")
    _rg_invert = os.environ.get("QUANTSYS_REGIME_INVERT")
    _regime_allow  = {int(x) for x in _rg_allow.split(",")  if x.strip() != ""} if _rg_allow  else None
    _regime_invert = {int(x) for x in _rg_invert.split(",") if x.strip() != ""} if _rg_invert else set()
    if (_regime_allow is not None or _regime_invert) and btc_regime_per_step is not None:
        log.info(f"Regime-gating ATTIVO: allow={_regime_allow}, invert={_regime_invert}")

    # IT: Entry RANK-based per regime Quiet (2026-06-04) — sfrutta l'edge di RANGO
    #     (Spearman +0.13÷0.19, stabile in tutti i sotto-periodi OOS) che l'entry a
    #     soglia |μ| non cattura (in Quiet μ piccole → 0 trade). Gate a quantile
    #     causale: LONG se μ nel top-q / SHORT se bottom-q della distribuzione recente
    #     di μ osservata in Quiet; NONE negli altri regimi (isola l'edge robusto).
    # EN: RANK-based entry for the Quiet regime — harvests the RANK edge (Spearman
    #     +0.13÷0.19, stable across all OOS sub-periods) that |μ|-threshold entry misses.
    #     Causal rolling-quantile gate; trades only the Quiet regime.
    _quiet_q       = float(os.environ.get("QUANTSYS_QUIET_RANK_Q", "0") or "0")
    _quiet_reg     = int(os.environ.get("QUANTSYS_QUIET_REGIME", "0"))
    _quiet_conv    = float(os.environ.get("QUANTSYS_QUIET_CONVICTION", "0.5") or "0.5")
    # IT: Floor di σ (raw) — salta i trade Quiet con movimento atteso troppo piccolo per
    #     coprire le fee (~26bps round-trip). 0 = off. Cost-derived, NON tuned sul test.
    # EN: σ floor (raw) — skips Quiet trades whose expected move is too small to cover
    #     fees (~26bps round-trip). 0 = off. Cost-derived, NOT tuned on test.
    _quiet_min_sig = float(os.environ.get("QUANTSYS_QUIET_MIN_SIGMA", "0") or "0")
    _quiet_buf_max = 1000   # IT: finestra causale μ-Quiet | EN: causal Quiet-μ window
    _quiet_min_buf = 200    # IT: min sample prima di tradare | EN: min samples before trading
    _quiet_mu_buf: "list[float]" = []
    _quiet_active  = _quiet_q > 0 and btc_regime_per_step is not None
    if _quiet_active:
        log.info(f"Quiet rank-entry ATTIVO: regime={_quiet_reg}, q={_quiet_q} "
                 f"(LONG top-{_quiet_q:.0%} / SHORT bottom-{_quiet_q:.0%}), "
                 f"buffer={_quiet_buf_max}, conviction={_quiet_conv}")

    # IT: ── Fix ① — CADENZA DECISIONALE = ORIZZONTE (sperimentale, reversibile, default off) ──
    #     Un segnale a orizzonte h tradato ogni candela genera h bet sovrapposti e
    #     autocorrelati: stesso costo fee, informazione incrementale ≈0 (breadth effettiva
    #     ≪ nominale, legge fondamentale IR≈IC·√breadth). Gate causale: una NUOVA entry apre
    #     solo se sono passate ≥cadence candele dall'ultima → bet quasi-indipendenti, fee drag
    #     tagliato. Gli EXIT (SL/TP/trailing/circuit-breaker) restano ogni candela.
    #     0=off (ogni candela); "h"=usa forecast_horizon (cadenza allineata all'orizzonte).
    # EN: Fix ① — decision cadence = horizon (experimental, reversible, default off). A horizon-h
    #     signal traded every candle yields h overlapping, autocorrelated bets: same fee cost,
    #     ~0 incremental info (effective breadth ≪ nominal). Causal gate: a NEW entry opens only
    #     if ≥cadence candles passed since the last one. Exits run every candle. 0=off; "h"=horizon.
    _cad_raw = os.environ.get("QUANTSYS_DECISION_CADENCE", "0").strip().lower()
    _fh = int(cfg.get("features", {}).get("forecast_horizon",
              cfg.get("data", {}).get("forecast_horizon", 30)))
    _decision_cadence = _fh if _cad_raw == "h" else int(float(_cad_raw or "0"))
    _last_entry_i = -10**9   # IT: indice ultima entry (causale) | EN: last-entry index (causal)
    if _decision_cadence > 0:
        log.info(f"Decision-cadence ATTIVA: nuove entry ogni ≥{_decision_cadence} candele (h={_fh})")

    # IT: ── Fix ② — ESPOSIZIONE CONTINUA RANK-BASED, REGIME-GATED (sperimentale, reversibile) ──
    #     L'edge reale è ordinale (Spearman Quiet +0.13÷0.19), non μ calibrato: l'entry a soglia
    #     |μ| e il rank-entry DISCRETO (QUANTSYS_QUIET_RANK_Q) lo distruggono. Qui il segnale è
    #     CONTINUO: r = percentile causale di μ nel buffer ∈[0,1]; s = 2r−1 ∈[−1,+1] (segno=
    #     direzione, |s|=forza). No-trade band |s|<band = deadzone/isteresi (niente flip su rank
    #     debole → throttling naturale dei flip-flop). conviction = (|s|−band)/(1−band) ∈(0,1]
    #     scala il Kelly con continuità (dist.conviction → RiskManager._size). Attivo SOLO nel
    #     regime target (Quiet di default), NONE altrove → isola l'unico edge stabile OOS.
    # EN: Fix ② — continuous rank-proportional, regime-gated exposure (experimental, reversible).
    #     The real edge is ordinal (Quiet Spearman), not calibrated μ: |μ|-threshold and DISCRETE
    #     rank-entry destroy it. Continuous signal: r=causal percentile of μ ∈[0,1]; s=2r−1
    #     ∈[−1,+1] (sign=direction, |s|=strength). No-trade band |s|<band = deadzone/hysteresis.
    #     conviction=(|s|−band)/(1−band) scales Kelly continuously. Active ONLY in target regime.
    _rank_active  = os.environ.get("QUANTSYS_RANK_EXPOSURE", "0").strip() == "1" and btc_regime_per_step is not None
    _rank_reg     = int(os.environ.get("QUANTSYS_RANK_REGIME", "0"))
    _rank_band    = float(os.environ.get("QUANTSYS_RANK_BAND", "0.5") or "0.5")
    # IT: Floor di σ (raw) — salta i trade con movimento atteso < fee (~26bps round-trip). 0=off.
    # EN: σ floor (raw) — skips trades whose expected move can't cover fees. 0=off.
    _rank_min_sig = float(os.environ.get("QUANTSYS_RANK_MIN_SIGMA", "0") or "0")
    _rank_buf_max = int(os.environ.get("QUANTSYS_RANK_WIN", "1000") or "1000")
    _rank_min_buf = 200      # IT: min sample prima di tradare | EN: min samples before trading
    _rank_mu_buf: "list[float]" = []
    if _rank_active:
        log.info(f"Rank-exposure ATTIVO: regime={_rank_reg}, band={_rank_band} "
                 f"(trade solo |percentile−0.5|≥{_rank_band/2:.2f}), σ_floor={_rank_min_sig}, "
                 f"buffer={_rank_buf_max}")

    # IT: ── EXIT ORIZZONTE-LOCKED (test di isolamento, sperimentale, reversibile, default off) ──
    #     Chiusura puramente TEMPORALE a esattamente h candele, bypassando SL/TP/SIGNAL/trailing.
    #     Scopo: isolare l'edge di RANGO dal path di realizzazione del trade — così la PnL del
    #     trade coincide col rendimento cumulato a orizzonte-h su cui è misurato lo Spearman
    #     (altrimenti SL/TP/flip dominano la PnL e mascherano l'edge ordinale, cfr. esito 2026-06-05).
    #     0=off (exit normale SL/TP/SIGNAL); "h"=usa forecast_horizon. Il circuit-breaker resta
    #     attivo (DD realizzato in close_position). Diagnostico, NON una regola di produzione.
    # EN: ── HORIZON-LOCKED EXIT (isolation test, experimental, reversible, default off) ──
    #     Pure TIME close at exactly h candles, bypassing SL/TP/SIGNAL/trailing, so a trade's PnL
    #     equals the horizon-h cumulative return the Spearman is measured on (otherwise SL/TP/flip
    #     dominate PnL and mask the ordinal edge). 0=off; "h"=forecast_horizon. Circuit-breaker
    #     stays active. Diagnostic, NOT a production rule.
    _hx_raw = os.environ.get("QUANTSYS_HORIZON_EXIT", "0").strip().lower()
    _horizon_exit = _fh if _hx_raw == "h" else int(float(_hx_raw or "0"))
    if _horizon_exit > 0:
        log.info(f"Horizon-locked exit ATTIVO: chiusura temporale a {_horizon_exit} candele "
                 f"(bypassa SL/TP/SIGNAL/trailing; h={_fh})")

    t0 = time.time()
    for i in tqdm(range(n-1), desc="Backtest", ncols=72):
        if rm.circuit_breaker: break

        o_c, h_c, l_c, c_c = ohlcv[i]
        o_n, h_n, l_n, c_n = ohlcv[i+1]
        atr_i = max(atr[i], c_c*0.0005)

        mu, sigma, nu    = predict(X[i], i)
        # IT: Rischio condizionato al regime — preferisce RegimeMarkovBTC se disponibile,
        #     fallback al proxy ATR storico (alta/media/bassa vol).
        # EN: Regime-conditioned risk — prefers RegimeMarkovBTC when available, falls back
        #     to the historical ATR proxy (high/mid/low vol).
        if btc_regime_per_step is not None:
            rm.set_regime(int(btc_regime_per_step[i]))
            # IT: regime threshold rimosso 2026-06-03 — calibrazione da rifare post-paper-trading
            # EN: removed — re-calibrate post paper-trading
        elif atr_i > atr_median * 1.5:
            rm.set_regime("stagflation")
        elif atr_i > atr_median:
            rm.set_regime("overheating")
        else:
            rm.set_regime("expansion")
        side, dist       = sig_gen.generate(mu, sigma, nu)
        # IT: Fix ② — esposizione continua rank-based (priorità) — direzione+conviction dal
        #     percentile causale di μ; no-trade band = deadzone/isteresi; NONE fuori regime.
        # EN: Fix ② — continuous rank exposure (priority) — direction+conviction from the causal
        #     μ-percentile; no-trade band = deadzone/hysteresis; NONE outside the target regime.
        if _rank_active:
            _reg = int(btc_regime_per_step[i])
            _new_side = Side.NONE
            if _reg == _rank_reg:
                if len(_rank_mu_buf) >= _rank_min_buf and sigma <= sig_gen.max_sigma and sigma >= _rank_min_sig:
                    _arr = np.asarray(_rank_mu_buf)
                    _r   = float(np.mean(_arr < mu))        # IT: percentile causale di μ ∈[0,1]
                    _s   = 2.0 * _r - 1.0                    # IT: rango centrato ∈[−1,+1]
                    if abs(_s) > _rank_band:                 # IT: fuori dalla deadzone (= isteresi)
                        _new_side = Side.LONG if _s > 0 else Side.SHORT
                        # IT: conviction continua → scala il Kelly proporzionalmente al rango
                        # EN: continuous conviction → scales Kelly proportionally to rank distance
                        dist.conviction = float(np.clip((abs(_s) - _rank_band) / (1.0 - _rank_band), 0.0, 1.0))
                _rank_mu_buf.append(float(mu))               # IT: update DOPO la decisione (causale)
                if len(_rank_mu_buf) > _rank_buf_max:
                    _rank_mu_buf.pop(0)
            side = _new_side                                 # IT: NONE fuori regime → isola l'edge
        # IT: Quiet rank-entry (priorità) — override il side via quantile causale di μ.
        # EN: Quiet rank-entry (priority) — override side via causal μ-quantile.
        elif _quiet_active:
            _reg = int(btc_regime_per_step[i])
            if _reg == _quiet_reg:
                _new_side = Side.NONE
                if len(_quiet_mu_buf) >= _quiet_min_buf and sigma <= sig_gen.max_sigma and sigma >= _quiet_min_sig:
                    _arr = np.asarray(_quiet_mu_buf)
                    _hi  = float(np.quantile(_arr, 1.0 - _quiet_q))
                    _lo  = float(np.quantile(_arr, _quiet_q))
                    if   mu >= _hi: _new_side = Side.LONG
                    elif mu <= _lo: _new_side = Side.SHORT
                _quiet_mu_buf.append(float(mu))             # IT: update DOPO la decisione (causale)
                if len(_quiet_mu_buf) > _quiet_buf_max:
                    _quiet_mu_buf.pop(0)
                if _new_side != Side.NONE:
                    dist.conviction = _quiet_conv           # IT: size fissa (prob_up non affidabile in Quiet)
                side = _new_side
            else:
                side = Side.NONE                            # IT: isola l'edge Quiet — niente trade fuori regime
        # IT: Regime-gating sperimentale — inverte o azzera il side per regime.
        # EN: Experimental regime-gating — invert or null the side per regime.
        elif side != Side.NONE and btc_regime_per_step is not None and (_regime_allow is not None or _regime_invert):
            _reg = int(btc_regime_per_step[i])
            if _reg in _regime_invert:
                side = Side.SHORT if side == Side.LONG else Side.LONG
            elif _regime_allow is not None and _reg not in _regime_allow:
                side = Side.NONE
        pre_signals.append((side, dist))

        if rm.position:
            rm.update_trailing(c_c, atr_i)

        if rm.position:
            if _horizon_exit > 0:
                # IT: exit orizzonte-locked — chiusura TEMPORALE pura al close, bypassa SL/TP/SIGNAL.
                # EN: horizon-locked exit — pure TIME close at the close price, bypasses SL/TP/SIGNAL.
                trade = None
                if (i + 1) - rm.position.entry_candle >= _horizon_exit:
                    trade = rm.close_position(CloseReason.MAX_HOLD, c_n, i+1, adv_1m=adv_1m[i])
            else:
                reason = rm.check_exit(h_n, l_n, c_n, i+1, side)
                trade = None
                if reason:
                    ep = rm.position.stop_loss if reason==CloseReason.STOP_LOSS else \
                         rm.position.take_profit if reason==CloseReason.TAKE_PROFIT else c_n
                    trade = rm.close_position(reason, ep, i+1, adv_1m=adv_1m[i])
            if trade:
                # IT: Assegna il trade al regime corrente. | EN: Assign the trade to the current regime.
                regime_key = "high_vol" if atr_i > atr_median else "low_vol"
                regime_trades[regime_key].append(trade.net_pnl)

        # IT: Fix ① — gate cadenza: la NUOVA entry apre solo se sono passate ≥cadence candele
        #     dall'ultima (bet quasi-indipendenti). cadence=0 → condizione sempre vera (baseline).
        # EN: Fix ① — cadence gate: a NEW entry opens only if ≥cadence candles passed since the
        #     last one (quasi-independent bets). cadence=0 → always true (baseline, inert).
        if side != Side.NONE and not rm.position and (i - _last_entry_i) >= _decision_cadence:
            rm.open_position(side, o_n, i+1, atr_i, dist, adv_1m=adv_1m[i])
            _last_entry_i = i

        mtm = rm.portfolio.cash + (rm.position.unrealized_pnl(c_n)+rm.position.size_usd if rm.position else 0)
        equity_ts.append(mtm)
        peak = max(peak, mtm); dd_ts.append((peak-mtm)/peak if peak>0 else 0)

    if rm.position:
        rm.close_position(CloseReason.END_OF_DATA, ohlcv[-1][3], n-1)

    elapsed = time.time() - t0
    m = rm.metrics()

    # ── MDD stats ────────────────────────────────────────────────────────────
    mdd_info = mdd_stats(equity_ts)

    # ── Bootstrap CI su Sharpe e Sortino ─────────────────────────────────────
    pnl_arr = [t.net_pnl for t in rm.trades]
    boot_ci = bootstrap_sharpe_ci(pnl_arr)

    # ── Stress testing ────────────────────────────────────────────────────────
    log.info("Avvio stress test scenari ...")
    stress_results = []
    for scenario_name, fee_mult, slip_mult in [
        ("pessimistic_fee",  2.0, 3.0),
        ("flash_crash_vol",  1.5, 5.0),
    ]:
        log.info(f"  Stress scenario: {scenario_name} (fee×{fee_mult}, slip×{slip_mult})")
        sr = run_stress_scenario(
            scenario_name=scenario_name,
            fee_mult=fee_mult,
            slip_mult=slip_mult,
            cfg_backtest=bcfg,
            cfg_risk=rcfg,
            ohlcv=ohlcv,
            atr=atr,
            pre_signals=pre_signals,
            adv_1m=adv_1m,
            seed_rm=rm,
        )
        stress_results.append(sr)
        log.info(f"    Sharpe={sr['sharpe']}  MDD={sr['max_drawdown']}  "
                 f"Return={sr['total_return']}  N={sr['n_trades']}")

    # ── Salva risultati / Save results ─────────────────────────────────────────
    # IT: metrics JSON — scarta le liste lunghe (vanno in file separati).
    # EN: metrics JSON — drop long lists (they go to separate files).
    m_save = {k: v for k,v in m.items() if not isinstance(v, list)}
    m_save["close_reasons"] = m.get("close_reasons", {})

    # IT: Breakdown per motivo di chiusura con P&L dettagliato.
    # EN: Close-reason breakdown with detailed P&L.
    close_reason_breakdown = {}
    for reason_str, count in m.get("close_reasons", {}).items():
        trades_for_reason = [t for t in rm.trades if t.close_reason.value == reason_str]
        if trades_for_reason:
            pnls = [t.net_pnl for t in trades_for_reason]
            close_reason_breakdown[reason_str] = {
                "count": count,
                "avg_pnl": float(np.mean(pnls)),
                "total_pnl": float(np.sum(pnls)),
                "win_rate": float(sum(1 for p in pnls if p > 0) / len(pnls)),
            }
    m_save["close_reason_breakdown"] = close_reason_breakdown

    # IT: Analisi condizionata al regime (high/low vol).
    # EN: Regime-conditioned analysis (high/low vol).
    regime_report = {}
    for regime, pnls in regime_trades.items():
        if pnls:
            arr = np.array(pnls)
            wins = arr[arr > 0]
            regime_report[regime] = {
                "n_trades":     len(arr),
                "win_rate":     float(len(wins) / len(arr)),
                "net_pnl":      float(arr.sum()),
                "avg_pnl":      float(arr.mean()),
            }
    m_save["regime_analysis"] = regime_report

    # IT: Bootstrap CI. | EN: Bootstrap CI.
    m_save["bootstrap_ci"] = boot_ci

    # IT: Statistiche MDD. | EN: MDD stats.
    m_save["mdd_stats"] = mdd_info

    # IT: Risultati stress test. | EN: Stress-test results.
    m_save["stress_scenarios"] = stress_results

    with open(out/f"metrics{_out_suffix}.json","w", encoding="utf-8") as f: json.dump(m_save,f,indent=2)

    np.savez_compressed(out/f"equity_curve{_out_suffix}.npz",
                        equity=np.array(equity_ts), drawdown=np.array(dd_ts))

    # IT: CSV dei trade. | EN: trades CSV.
    rows = [{"side":t.side.value,"entry_price":t.entry_price,"exit_price":t.exit_price,
             "size_usd":t.size_usd,"hold_candles":t.hold_candles,
             "close_reason":t.close_reason.value,"gross_pnl":t.gross_pnl,
             "fees":t.fees,"net_pnl":t.net_pnl,"pnl_pct":t.pnl_pct} for t in rm.trades]
    pd.DataFrame(rows).to_csv(out/f"trades{_out_suffix}.csv",index=False)

    # IT: JSON dashboard (pronto per la React app).
    # EN: dashboard JSON (ready for the React app).
    # IT: NaN/inf → None per un JSON valido. | EN: NaN/inf → None for valid JSON.
    def clean(v): return None if isinstance(v,float) and (math.isnan(v) or math.isinf(v)) else v
    cum_pnl = list(np.cumsum(pnl_arr)) if pnl_arr else []
    dashboard = {
        "metrics": {k: clean(v) for k,v in m_save.items() if not isinstance(v,dict)} | {
            "close_reasons": m_save.get("close_reasons", {}),
            "close_reason_breakdown": close_reason_breakdown,
        },
        "equity_curve":   _downsample([float(v) for v in equity_ts]),
        "drawdown_curve": _downsample([float(v) for v in dd_ts]),
        "pnl_series":     _downsample([float(v) for v in cum_pnl]),
        "pnl_per_trade":  _downsample([float(v) for v in pnl_arr]),
        "trades": [{"side":t.side.value,"entry_price":t.entry_price,"exit_price":t.exit_price,
                    "size_usd":t.size_usd,"hold_candles":t.hold_candles,
                    "close_reason":t.close_reason.value,"net_pnl":t.net_pnl,"pnl_pct":t.pnl_pct}
                   for t in rm.trades[-200:]],
        "bootstrap_ci": boot_ci,
        "mdd_stats": mdd_info,
        "stress_scenarios": stress_results,
        "close_reason_breakdown": close_reason_breakdown,
    }
    with open(results_out/f"dashboard_results{_out_suffix}.json","w", encoding="utf-8") as f:
        json.dump(dashboard, f, separators=(",",":"))

    # ── Print finale / Final print ─────────────────────────────────────────────
    # IT: Riepilogo a console di metriche, regime, stress test, CI e MDD.
    # EN: Console summary of metrics, regime, stress tests, CI and MDD.
    print(f"""
═══════════════════════════════════════════
  03 · BACKTEST · COMPLETATO  ({elapsed:.1f}s)
═══════════════════════════════════════════
  N° Trade        : {m.get('n_trades',0)}
  Win Rate        : {m.get('win_rate',0):.1%}
  Profit Factor   : {m.get('profit_factor',0):.2f}
  Sharpe          : {m.get('sharpe',0):.2f}
  Sortino         : {m.get('sortino',0):.2f}
  Calmar          : {m.get('calmar',0):.2f}
  Max Drawdown    : {m.get('max_drawdown',0):.1%}
  Rendimento      : {m.get('total_return',0):+.2%}
  Eq finale       : ${m.get('final_equity',0):,.2f}
  Fee totali      : ${m.get('total_fees',0):,.2f}

  ── Analisi per regime (ATR mediana={atr_median:.0f}) ───────""")
    for regime, stats in regime_report.items():
        label = "Alta vol " if regime == "high_vol" else "Bassa vol"
        print(f"  {label}: {stats['n_trades']} trade | "
              f"WR={stats['win_rate']:.1%} | "
              f"P&L={stats['net_pnl']:+.0f}$")
    if len(regime_report) == 2:
        hv = regime_report.get("high_vol", {})
        lv = regime_report.get("low_vol", {})
        if hv and lv:
            diff = hv.get("win_rate", 0) - lv.get("win_rate", 0)
            if abs(diff) > 0.10:
                regime_label = "alta vol" if diff > 0 else "bassa vol"
                print(f"\n  Forte dipendenza dal regime: WR migliore in {regime_label} "
                      f"(Delta={diff:+.1%}) -> generalizzazione limitata.")

    # IT: Print degli scenari di stress. | EN: Stress-test print.
    pess = next((s for s in stress_results if s["scenario"] == "pessimistic_fee"), {})
    flash = next((s for s in stress_results if s["scenario"] == "flash_crash_vol"), {})

    # IT: Formatter None-safe per %, return e Sharpe. | EN: None-safe formatters for %, return, Sharpe.
    def _fmt_pct(v): return f"{v:.1%}" if v is not None else "N/A"
    def _fmt_ret(v): return f"{v:.2%}" if v is not None else "N/A"
    def _fmt_sh(v):  return f"{v:.2f}" if v is not None else "N/A"

    print(f"""
  ── Stress Test ─────────────────────────────────
  Pessimistic (fee×2, slip×3):
    Sharpe={_fmt_sh(pess.get('sharpe'))}  MDD={_fmt_pct(pess.get('max_drawdown'))}  Return={_fmt_ret(pess.get('total_return'))}
  Flash Crash (fee×1.5, slip×5):
    Sharpe={_fmt_sh(flash.get('sharpe'))}  MDD={_fmt_pct(flash.get('max_drawdown'))}  Return={_fmt_ret(flash.get('total_return'))}""")

    # IT: Print del bootstrap CI. | EN: Bootstrap CI print.
    sh_ci_low  = boot_ci.get("sharpe_ci_low")
    sh_ci_high = boot_ci.get("sharpe_ci_high")
    so_ci_low  = boot_ci.get("sortino_ci_low")
    so_ci_high = boot_ci.get("sortino_ci_high")
    sh_val     = m.get("sharpe", 0) or 0.0
    so_val     = m.get("sortino", 0) or 0.0

    if sh_ci_low is not None:
        print(f"""
  ── Bootstrap CI (95%, 5000 campioni) ────────────
  Sharpe:  {sh_val:.2f}  [{sh_ci_low:.2f}, {sh_ci_high:.2f}]
  Sortino: {so_val:.2f}  [{so_ci_low:.2f}, {so_ci_high:.2f}]""")
    else:
        print(f"""
  ── Bootstrap CI (95%, 5000 campioni) ────────────
  Trade insufficienti per bootstrap CI (< 30 trade)""")

    # IT: Print delle statistiche MDD. | EN: MDD stats print.
    mdd_dur  = mdd_info.get("mdd_duration_candles", 0)
    mdd_rec  = mdd_info.get("mdd_recovery_candles")
    mdd_recov = mdd_info.get("mdd_recovered", False)
    mdd_rec_str = str(mdd_rec) if mdd_rec is not None else "non recuperato"

    print(f"""
  ── MDD ──────────────────────────────────────────
  Durata: {mdd_dur} candele  Recovery: {mdd_rec_str} candele (recuperato: {mdd_recov})""")

    print(f"""
  Split valutato  -> {_split.upper()}
  Output modello -> {out}/
    metrics{_out_suffix}.json
    equity_curve{_out_suffix}.npz
    trades{_out_suffix}.csv
  Output dashboard -> {results_out}/
    dashboard_results{_out_suffix}.json
""")


# IT: Esecuzione diretta dello script. | EN: Direct script execution.
if __name__ == "__main__":
    main()

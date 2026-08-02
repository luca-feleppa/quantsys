"""Script 07 — Verifica quale architettura dovrebbe fare da teacher.

Analizza le 3 architetture (iTransformer, LSTM, TCNMamba) confrontando:
  1. Numero di parametri (capacità del modello)
  2. Complessità computazionale (tempo forward pass)
  3. Metriche di backtest (se disponibili da checkpoint esistenti)
  4. Ricchezza delle rappresentazioni interne

Raccomanda l'architettura migliore come teacher per knowledge distillation.

Usage:
  python scripts/07_verify_teacher.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from quantsys.utils import load_config, setup_device


# IT: conteggio parametri totali e trainable (capacita' del modello)
# EN: count total and trainable parameters (model capacity)
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# IT: timing forward pass medio (warm-up + N run, ms) con sync CUDA per accuratezza
# EN: average forward-pass timing (warm-up + N runs, ms) with CUDA sync for accuracy
def measure_forward_time(model, x_dummy, device, n_warmup=3, n_runs=10):
    model.eval()
    x = x_dummy.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_runs
    return elapsed * 1000  # IT: secondi -> ms | EN: seconds -> ms


# IT: parametri output heads (trasferibili allo student via distillation)
# EN: output-head parameters (transferable to student via distillation)
def count_output_head_params(model):
    """Conta i parametri delle output heads (mu, sigma, nu) — quelli trasferiti al student."""
    head_names = {
        "out_mu", "out_logsig2", "out_lognu",  # IT: LSTM, iTransformer | EN: LSTM, iTransformer
        "mu_head", "ls2_head", "lnu_head",      # IT: TCNMamba | EN: TCNMamba
    }
    total = 0
    for name, param in model.named_parameters():
        for hname in head_names:
            if hname in name:
                total += param.numel()
                break
    return total


# IT: helper — carica metrics.json di backtest se presente
# EN: helper — load backtest metrics.json if present
def load_backtest_metrics(arch_dir):
    metrics_path = Path(arch_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)
    return None


# IT: helper — carica predizioni di test salvate da 03_backtest
# EN: helper — load test predictions saved by 03_backtest
def load_test_predictions(arch_dir):
    pred_path = Path(arch_dir) / "test_predictions.npz"
    if pred_path.exists():
        data = np.load(pred_path)
        return {k: data[k] for k in data.files}
    return None


# IT: benchmark 3 architetture (params/latenza/backtest) -> raccomanda il teacher
# EN: benchmark 3 architectures (params/latency/backtest) -> recommend the teacher
def main():
    # IT: console Windows default cp1252 — qualsiasi unicode nei banner/report crasha
    #     il print con UnicodeEncodeError. Reconfigure UTF-8 come 01/02/04.
    # EN: Windows console defaults to cp1252 — any unicode in banners/reports crashes
    #     the print with UnicodeEncodeError. Reconfigure UTF-8 like 01/02/04.
    import sys as _sys
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    cfg = load_config("config/default.yaml")
    device = setup_device(cfg)
    mcfg = cfg["model"]

    # IT: legge dimensioni reali dal dataset (mmap per non caricare X_train in RAM)
    # EN: read real dimensions from dataset (mmap to avoid loading X_train in RAM)
    dataset_path = Path("data/lstm_dataset.npz")
    if dataset_path.exists():
        _ds = np.load(dataset_path, mmap_mode='r', allow_pickle=True)
        n_feat = int(_ds["X_train"].shape[2])
        n_dynamic = int(_ds["n_dynamic_features"][0]) if "n_dynamic_features" in _ds.files else n_feat
        del _ds
    else:
        # IT: senza dataset il feature space reale non è noto (cambia col set C-funding,
        #     vedi LIVE_DROP_FEATURES) → niente magic number stale, errore esplicito e azionabile.
        # EN: without the dataset the real feature space is unknown (it changes with the
        #     C-funding set, see LIVE_DROP_FEATURES) → no stale magic number, explicit actionable error.
        raise FileNotFoundError(
            f"Dataset non trovato: {dataset_path}. Esegui prima la pipeline dati "
            "(es. `python run_all.py --arch nhits` o `python scripts/01_download_data.py`) "
            "per definire il feature space reale prima di verificare i teacher."
        )

    T = mcfg.get("window_size", 120)
    B = 32

    x_dummy = torch.randn(B, T, n_feat)

    print(f"""
{'═'*70}
  07 · VERIFICA TEACHER MODEL
{'═'*70}
  Device: {device}
  Input:  (B={B}, T={T}, F={n_feat})  n_dynamic={n_dynamic}
{'═'*70}
""")

    results = {}

    # IT: 1) iTransformer — attention O(F^2) sulle feature
    # EN: 1) iTransformer — O(F^2) attention over features
    from quantsys.model import QuantiTransformer
    itrans = QuantiTransformer(
        n_features=n_feat, T=T, n_dynamic=n_dynamic, n_macro=0,
        d_model=mcfg.get("tft_d_model", 128),
        n_heads=mcfg.get("tft_n_heads", 4),
        n_layers=mcfg.get("tft_n_layers", 3),
        dropout=mcfg.get("tft_dropout", 0.1),
        patch_size=mcfg.get("patch_size", 1),
        drop_path_rate=mcfg.get("drop_path_rate", 0.0),
        loss_type=mcfg.get("loss_type", "t_student"),
        use_multitask=mcfg.get("use_multitask", False),
        n_output_experts=mcfg.get("n_output_experts", 1),
    ).to(device)
    total_p, train_p = count_parameters(itrans)
    head_p = count_output_head_params(itrans)
    fwd_ms = measure_forward_time(itrans, x_dummy, device)
    results["itransformer"] = {
        "total_params": total_p, "trainable_params": train_p,
        "head_params": head_p, "forward_ms": fwd_ms,
    }
    # IT: libera VRAM prima del prossimo modello (test secventiale)
    # EN: free VRAM before next model (sequential benchmark)
    del itrans
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # IT: 2) LSTM+GRU — baseline ricorrente veloce
    # EN: 2) LSTM+GRU — fast recurrent baseline
    from quantsys.model import QuantLSTM
    lstm = QuantLSTM(
        n_features=n_feat,
        lstm_hidden=mcfg.get("lstm_hidden", 256),
        gru_hidden=mcfg.get("gru_hidden", 128),
        mlp_hidden=mcfg.get("mlp_hidden", 64),
        n_lstm_layers=mcfg.get("lstm_layers", 2),
        dropout=mcfg.get("dropout", 0.2),
        n_dynamic_features=n_dynamic,
        loss_type=mcfg.get("loss_type", "t_student"),
        use_multitask=mcfg.get("use_multitask", False),
        n_output_experts=mcfg.get("n_output_experts", 1),
    ).to(device)
    total_p, train_p = count_parameters(lstm)
    head_p = count_output_head_params(lstm)
    fwd_ms = measure_forward_time(lstm, x_dummy, device)
    results["lstm"] = {
        "total_params": total_p, "trainable_params": train_p,
        "head_params": head_p, "forward_ms": fwd_ms,
    }
    del lstm
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # IT: 3) TCN+Mamba — pattern locali TCN + contesto lungo via SSM
    # EN: 3) TCN+Mamba — local TCN patterns + long context via SSM
    from quantsys.model import QuantTCNMamba
    tcnmamba = QuantTCNMamba(
        n_features=n_feat,
        d_model=mcfg.get("d_model", 128),
        tcn_layers=mcfg.get("tcn_layers", 4),
        tcn_kernel=mcfg.get("tcn_kernel", 3),
        mamba_layers=mcfg.get("mamba_layers", 3),
        mamba_d_state=mcfg.get("mamba_d_state", 16),
        mamba_expand=mcfg.get("mamba_expand", 2),
        dropout=mcfg.get("dropout", 0.1),
        drop_path_rate=mcfg.get("drop_path_rate", 0.0),
        n_dynamic_features=n_dynamic,
        use_multitask=mcfg.get("use_multitask", False),
    ).to(device)
    total_p, train_p = count_parameters(tcnmamba)
    head_p = count_output_head_params(tcnmamba)
    fwd_ms = measure_forward_time(tcnmamba, x_dummy, device)
    results["tcnmamba"] = {
        "total_params": total_p, "trainable_params": train_p,
        "head_params": head_p, "forward_ms": fwd_ms,
    }
    del tcnmamba
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # IT: tabella comparativa parametri vs latenza forward
    # EN: comparative table — parameters vs forward latency
    print(f"  {'Architettura':<18} {'Parametri':>12} {'Head params':>12} {'Forward (ms)':>14}")
    print(f"  {'─'*18} {'─'*12} {'─'*12} {'─'*14}")
    for arch, r in results.items():
        print(f"  {arch:<18} {r['trainable_params']:>12,} {r['head_params']:>12,} {r['forward_ms']:>13.2f}")

    # IT: aggancia metriche backtest se i checkpoint sono gia' stati valutati
    # EN: attach backtest metrics if checkpoints have already been evaluated
    print(f"\n  {'─'*58}")
    print("  Metriche backtest da checkpoint esistenti:")
    print(f"  {'─'*58}")

    has_backtest = False
    for arch in ["itransformer", "lstm", "tcnmamba", "nhits"]:
        arch_dir = Path("models") / arch
        metrics = load_backtest_metrics(arch_dir)
        pred_path = Path(arch_dir) / "test_predictions.npz"
        if metrics:
            has_backtest = True
            results[arch]["backtest"] = metrics
            print(f"\n  {arch}:")
            print(f"    Sharpe:       {metrics.get('sharpe', 'N/A')}")
            print(f"    Win Rate:     {metrics.get('win_rate', 'N/A')}")
            print(f"    N Trade:      {metrics.get('n_trades', 'N/A')}")
            print(f"    Max Drawdown: {metrics.get('max_drawdown', 'N/A')}")
            print(f"    Total Return: {metrics.get('total_return', 'N/A')}")
        if pred_path.exists():
            results[arch]["has_predictions"] = True
        if not metrics and not pred_path.exists():
            print(f"\n  {arch}: nessun checkpoint/metrica trovata")

    # IT: scoring composito — performance domina su capacita' (~75/25)
    # EN: composite scoring — performance dominates over capacity (~75/25)
    print(f"\n{'═'*70}")
    print("  ANALISI E RACCOMANDAZIONE")
    print(f"{'═'*70}")

    scores = {}
    for arch, r in results.items():
        score = 0
        bt = r.get("backtest", {})
        # IT: Sharpe dominante (max 100 pts), saturazione a 50 per evitare outlier
        # EN: Sharpe dominant (max 100 pts), capped at 50 to avoid outliers
        if bt.get("sharpe", 0) > 0:
            score += min(bt["sharpe"], 50) * 2.0
        # IT: bonus solo oltre soglia minima WR 40% (sotto e' rumore)
        # EN: bonus only above 40% WR threshold (below is noise)
        if bt.get("win_rate", 0) > 0.40:
            score += (bt["win_rate"] - 0.40) * 200
        # IT: capacita' come segnale secondario (2 pts per M params)
        # EN: capacity as secondary signal (2 pts per M params)
        score += r["trainable_params"] / 1_000_000 * 2
        # IT: piccola penalita' per modelli lenti (no game della latenza pura)
        # EN: small penalty for slow models (no pure-latency gaming)
        score -= r["forward_ms"] * 0.1
        scores[arch] = score

    best_arch = max(scores, key=scores.get)

    print("\n  Punteggi teacher (piu' alto = migliore):")
    for arch, s in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " ★ RACCOMANDATO" if arch == best_arch else ""
        print(f"    {arch:<18} {s:>8.2f}{marker}")

    print(f"""
  Criteri di valutazione (performance ~75%, capacity ~25%):
  - Performance backtest (dominante): Sharpe (x2.0, fino a 100 pts), Win Rate (x200)
  - Capacita' del modello (parametri): 2 pts per M params (segnale secondario)
  - Output heads compatibili: mu_head, ls2_head, lnu_head (trasferibili)

  NOTA: Il teacher ideale e' il modello con la migliore capacita' espressiva
  e le migliori performance predittive. Gli output heads devono avere la
  stessa dimensione di output (1 per mu, 1 per ls2, 1 per lnu) — condizione
  soddisfatta da tutte e 3 le architetture.

  ┌──────────────────────────────────────────────────────────┐
  │  RACCOMANDAZIONE: {best_arch:^18} come teacher     │
  └──────────────────────────────────────────────────────────┘

  Motivi:
""")

    r = results[best_arch]
    print(f"    1. {r['trainable_params']:,} parametri trainabili (maggiore capacita')")
    if r.get("backtest", {}).get("sharpe"):
        print(f"    2. Sharpe {r['backtest']['sharpe']:.2f} in backtest (migliore performance)")
    print(f"    3. {r['head_params']:,} parametri nelle output heads (trasferibili agli student)")
    print(f"    4. Forward pass: {r['forward_ms']:.2f} ms (usato solo per generare soft labels)")

    # IT: serializza analisi -> consumata dalla tab Ensemble della dashboard
    # EN: serialize analysis -> consumed by the dashboard Ensemble tab
    out_path = Path("models") / "teacher_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "scores": scores, "recommended_teacher": best_arch},
                  f, indent=2, default=str)
    print(f"\n  Risultati salvati in: {out_path}")


if __name__ == "__main__":
    main()

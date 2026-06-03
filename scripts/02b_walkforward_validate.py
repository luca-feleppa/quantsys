"""
Script 02b — Walk-Forward Validation con riaddestramento per fold.

Miglioramento 7 — Walk-forward che riaddestra:
  La versione precedente valutava il modello già addestrato su fold diversi.
  Questo NON è un vero walk-forward: il modello aveva già "visto" i dati
  futuri durante il suo training originale (overfitting temporale mascherato).

  Ora ogni fold:
  1. Riaddestra il modello da zero (o da un init) su [0, train_end)
  2. Valuta su [val_start, val_end) — dati mai visti in training
  3. Calcola le metriche di quel periodo specifico

  Questo è costoso (K × training_time) ma dà una stima molto più
  realistica: se le metriche sono stabili tra fold, il segnale è robusto.
  Se variano molto, il modello si overfita al regime recente.

  Ottimizzazione: usa un training abbreviato per ogni fold (early stopping
  aggressivo, max_epochs ridotto) per mantenere tempi ragionevoli.

Run:
  python scripts/02b_walkforward_validate.py            # riaddestra
  python scripts/02b_walkforward_validate.py --no-retrain  # solo valuta
"""
import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

# IT: cap thread BLAS/OMP prima di importare numpy/torch (cpu_fraction da config)
# EN: cap BLAS/OMP threads before importing numpy/torch (cpu_fraction from config)
import yaml as _yaml
with open(Path(__file__).resolve().parent.parent / "config" / "default.yaml", encoding="utf-8") as _f:
    _cpu_frac = _yaml.safe_load(_f).get("hardware", {}).get("cpu_fraction", 0.5)
_cpu_limit = str(max(1, int(os.cpu_count() * _cpu_frac)))
os.environ.setdefault("OMP_NUM_THREADS", _cpu_limit)
os.environ.setdefault("MKL_NUM_THREADS", _cpu_limit)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(int(_cpu_limit))

from quantsys.utils import load_config, setup_logging, setup_device, ensure_dirs
from quantsys.model import QuantLSTM, student_t_nll, quantile_loss, EarlyStopping, set_clip_bounds, direction_value_loss
from quantsys.features import walk_forward_folds

setup_logging()
log = logging.getLogger("quantsys.script.02b")


# IT: metriche per fold (DA, Spearman, WHR, coverage CI90)
# EN: per-fold metrics (DA, Spearman, WHR, CI90 coverage)

def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 sig_arr: np.ndarray, nu_arr: np.ndarray) -> dict:
    from scipy.stats import spearmanr, t as t_dist

    da = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    with np.errstate(invalid="ignore"):
        sp, pv = spearmanr(y_true, y_pred)
    sp = float(sp) if not np.isnan(sp) else 0.0

    # IT: WHR = pct di |y| catturato dai segni corretti (qualita' direzionale pesata)
    # EN: WHR = % of |y| captured by correct signs (weighted directional quality)
    correct = np.sign(y_true) == np.sign(y_pred)
    whr = float(np.abs(y_true[correct]).sum() / (np.abs(y_true).sum() + 1e-10))

    # IT: quantile t-Student a 95% per CI bilaterale 90% (df = nu medio del fold)
    # EN: 95% t-Student quantile for 90% two-sided CI (df = fold-mean nu)
    z90  = t_dist.ppf(0.95, df=nu_arr.mean())
    cov90= float(np.mean(
        (y_true >= y_pred - z90*sig_arr) & (y_true <= y_pred + z90*sig_arr)
    ))

    return {"da": da, "spearman": sp, "whr": whr, "ci90": cov90}


# IT: training per fold (riaddestra da zero per stima onesta)
# EN: per-fold training (retrain from scratch for an honest estimate)

# IT: warmup lineare + decay cosine fino a min_frac del lr iniziale
# EN: linear warmup + cosine decay down to min_frac of initial lr
class CosineWarmup(torch.optim.lr_scheduler.LambdaLR):
    # IT: salva i parametri warmup/total/min_frac e registra il lambda LR
    # EN: store warmup/total/min_frac params and register the LR lambda
    def __init__(self, opt, warmup, total, min_frac=0.05):
        self.w, self.t, self.m = warmup, total, min_frac
        super().__init__(opt, self._lr)
    # IT: moltiplicatore LR per step: warmup lineare poi decay cosine
    # EN: per-step LR multiplier: linear warmup then cosine decay
    def _lr(self, step):
        if step < self.w: return step / max(self.w, 1)
        p = (step - self.w) / max(self.t - self.w, 1)
        return self.m + (1 - self.m) * 0.5 * (1 + math.cos(math.pi * p))


# IT: Addestra il modello su un singolo fold walk-forward con early stopping, ritorna le metriche di validation.
# EN: Trains the model on a single walk-forward fold with early stopping, returns validation metrics.
def train_fold(
    X_tr, y_tr, X_vl, y_vl,
    X_macro_tr=None, X_macro_vl=None,
    cfg=None, device=None, fold_id=0,
    max_epochs=40, patience=10,
    n_dynamic=None,
) -> tuple:
    """
    Riaddestra il modello sul training set del fold e valuta sul validation.
    Usa la stessa loss asimmetrica del training principale (Miglioramento 2).
    """
    mcfg  = cfg["model"]; tcfg = cfg["training"]; mccfg = cfg.get("macro", {})
    hwcfg = cfg["hardware"]

    # IT: stessi iperparametri loss del training principale (coerenza fold/full)
    # EN: same loss hyperparams as main training (fold/full consistency)
    asym_alpha     = tcfg.get("asymmetry_alpha",     2.0)
    asym_threshold = tcfg.get("asymmetry_threshold", 0.002)
    dv_lambda      = tcfg.get("dv_lambda",           0.0)

    n_feat   = X_tr.shape[2]
    has_macro= X_macro_tr is not None
    n_macro  = X_macro_tr.shape[1] if has_macro else 0

    batch = tcfg["batch_size"]
    kw    = dict(pin_memory=hwcfg["pin_memory"],
                 num_workers=min(hwcfg["num_workers"], 4),
                 persistent_workers=min(hwcfg["num_workers"], 4) > 0)

    # IT: helper DataLoader — include il tensore macro solo se presente
    # EN: DataLoader helper — includes the macro tensor only when present
    def mk_dl(X, Xm, y, shuffle):
        if has_macro:
            return DataLoader(TensorDataset(
                torch.as_tensor(X, dtype=torch.float32),
                torch.as_tensor(Xm, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32),
            ), batch, shuffle=shuffle, **kw)
        return DataLoader(TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        ), batch, shuffle=shuffle, **kw)

    tr_dl = mk_dl(X_tr, X_macro_tr, y_tr, True)
    vl_dl = mk_dl(X_vl, X_macro_vl, y_vl, False)

    # IT: nuovo modello per ogni fold (no transfer learning -> stima pulita)
    # EN: fresh model per fold (no transfer learning -> clean estimate)
    architecture = mcfg.get("architecture", "lstm")
    _loss_type    = mcfg.get("loss_type", "t_student")
    _use_multitask= mcfg.get("use_multitask", False)
    if architecture == "itransformer":
        from quantsys.model import QuantiTransformer
        _n_dyn = n_dynamic if n_dynamic is not None else n_feat
        _T     = mcfg.get("window_size", 120)
        model = QuantiTransformer(
            n_features       = n_feat,
            T                = _T,
            n_dynamic        = _n_dyn,
            n_macro          = n_macro if has_macro else 0,
            d_model          = mcfg.get("tft_d_model", 128),
            n_heads          = mcfg.get("tft_n_heads", 4),
            n_layers         = mcfg.get("tft_n_layers", 3),
            dropout          = mcfg.get("tft_dropout", 0.1),
            patch_size       = mcfg.get("patch_size", 1),
            drop_path_rate   = mcfg.get("drop_path_rate", 0.0),
            loss_type        = _loss_type,
            use_multitask    = _use_multitask,
            n_output_experts = mcfg.get("n_output_experts", 1),
        ).to(device)
    elif architecture == "tft":
        from quantsys.model import QuantTFT
        _n_dyn  = n_dynamic if n_dynamic is not None else n_feat
        _n_str  = n_feat - _n_dyn
        model = QuantTFT(
            n_dynamic    = _n_dyn,
            n_structural = _n_str,
            n_macro      = n_macro if has_macro else 0,
            d_model      = mcfg.get("tft_d_model", 64),
            n_heads      = mcfg.get("tft_n_heads", 4),
            dropout      = mcfg.get("tft_dropout", 0.1),
        ).to(device)
    elif architecture == "tcnmamba":
        from quantsys.model import QuantTCNMamba
        _n_dyn = n_dynamic if n_dynamic is not None else n_feat
        model = QuantTCNMamba(
            n_features         = n_feat,
            d_model            = mcfg.get("d_model", 128),
            tcn_layers         = mcfg.get("tcn_layers", 4),
            tcn_kernel         = mcfg.get("tcn_kernel", 3),
            mamba_layers       = mcfg.get("mamba_layers", 3),
            mamba_d_state      = mcfg.get("mamba_d_state", 16),
            mamba_expand       = mcfg.get("mamba_expand", 2),
            dropout            = mcfg.get("dropout", 0.1),
            n_dynamic_features = _n_dyn,
            use_multitask      = mcfg.get("use_multitask", False),
            loss_type          = mcfg.get("loss_type", "t_student"),
            n_output_experts   = mcfg.get("n_output_experts", 1),
        ).to(device)
    elif has_macro:
        from quantsys.macro.regime import QuantLSTMWithMacro
        model = QuantLSTMWithMacro(
            n_price_features   = n_feat,
            n_macro_features   = n_macro,
            lstm_hidden        = mcfg["lstm_hidden"],
            gru_hidden         = mcfg["gru_hidden"],
            mlp_hidden         = mcfg["mlp_hidden"],
            macro_embed_dim    = mccfg.get("embed_dim", 16),
            n_lstm_layers      = mcfg["lstm_layers"],
            dropout            = mcfg["dropout"],
            n_dynamic_features = n_dynamic,
        ).to(device)
    else:
        model = QuantLSTM(
            n_features         = n_feat,
            lstm_hidden        = mcfg["lstm_hidden"],
            gru_hidden         = mcfg["gru_hidden"],
            mlp_hidden         = mcfg["mlp_hidden"],
            n_lstm_layers      = mcfg["lstm_layers"],
            dropout            = mcfg["dropout"],
            n_dynamic_features = n_dynamic,
        ).to(device)

    # IT: clip bounds fittati su X_tr del fold (evita data leakage dal val)
    # EN: clip bounds fit on fold X_tr only (avoids leakage from val)
    if hasattr(model, "clip_lo"):
        _Xf = X_tr.reshape(-1, n_feat)
        with np.errstate(all="ignore"):
            _clo = np.nanpercentile(_Xf, 0.1, axis=0).astype(np.float32)
            _chi = np.nanpercentile(_Xf, 99.9, axis=0).astype(np.float32)
        set_clip_bounds(model, _clo, _chi)

    opt      = torch.optim.AdamW(model.parameters(),
                                  lr=tcfg["learning_rate"],
                                  weight_decay=tcfg["weight_decay"])
    total_s  = max_epochs * len(tr_dl)
    sched    = CosineWarmup(opt, min(200, total_s // 10), total_s)
    use_amp  = tcfg["use_amp"] and device.type == "cuda"
    scaler   = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    ckpt_path = str(Path(cfg["training"]["output_dir"]) / f"wf_fold{fold_id}_best.pt")
    es        = EarlyStopping(patience=patience, path=ckpt_path)

    for epoch in range(1, max_epochs + 1):
        # IT: training step
        # EN: training step
        model.train(); tr_loss = 0.0
        for batch_data in tr_dl:
            if has_macro:
                Xb, Xmb, yb = [x.to(device, non_blocking=True) for x in batch_data]
            else:
                Xb, yb = [x.to(device, non_blocking=True) for x in batch_data]; Xmb = None
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out  = model(Xb, Xmb) if has_macro else model(Xb)
            # IT: loss SEMPRE in fp32 (quantile_loss non e' fp16-safe -> NaN)
            # EN: loss ALWAYS in fp32 (quantile_loss is not fp16-safe -> NaN)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                if model.loss_type == "quantile":
                    loss = quantile_loss(yb.float(), out[0].float())
                else:
                    loss = student_t_nll(yb, out[0], out[1], out[2],
                                         asymmetry_alpha=asym_alpha,
                                         large_move_threshold=asym_threshold)
                    if dv_lambda > 0:
                        loss = loss + direction_value_loss(yb.float(), out[0].float(), lambda_dv=dv_lambda)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tr_loss += float(loss.item())

        # IT: validation step
        # EN: validation step
        model.eval(); vl_loss = 0.0; vl_n = 0
        with torch.no_grad():
            for batch_data in vl_dl:
                if has_macro:
                    Xb, Xmb, yb = [x.to(device) for x in batch_data]
                    out = model(Xb, Xmb)
                else:
                    Xb, yb = [x.to(device) for x in batch_data]
                    out = model(Xb)
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    if model.loss_type == "quantile":
                        _l = quantile_loss(yb.float(), out[0].float()).item()
                    else:
                        _l = student_t_nll(yb, out[0], out[1], out[2]).item()
                if math.isfinite(_l):
                    vl_loss += _l; vl_n += 1

        vl_nll = vl_loss / max(vl_n, 1)
        if epoch % 5 == 0:
            log.info(f"  Fold {fold_id} Ep {epoch:3d}  "
                     f"train={tr_loss/len(tr_dl):.4f}  val={vl_nll:.4f}")
        if es(vl_nll, model):
            break

    es.restore(model)
    return model, es.best


# IT: inferenza sul fold di test + calcolo metriche (gestisce quantile e t-Student)
# EN: inference on the test fold + metric computation (handles quantile and t-Student)

def eval_model(model, X, y, X_macro=None, device=None,
               cfg=None, has_macro=False) -> dict:
    """Valuta il modello sul fold di test (mai visto in training)."""
    batch = cfg["training"]["batch_size"]
    if has_macro:
        dl = DataLoader(TensorDataset(
            torch.as_tensor(X,       dtype=torch.float32),
            torch.as_tensor(X_macro, dtype=torch.float32),
            torch.as_tensor(y,       dtype=torch.float32),
        ), batch, shuffle=False)
    else:
        dl = DataLoader(TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        ), batch, shuffle=False)

    model.eval()
    loss_type = getattr(model, "loss_type", "t_student")
    mus, sigs, nus, ys = [], [], [], []
    with torch.no_grad():
        for bd in dl:
            if has_macro:
                Xb, Xmb, yb = [x.to(device) for x in bd]
                out = model(Xb, Xmb)
            else:
                Xb, yb = [x.to(device) for x in bd]
                out = model(Xb)
            if loss_type == "quantile":
                qp = out[0]                                       # IT: (B, 5) | EN: (B, 5)
                mus.append(qp[:, 2].cpu().numpy())                # IT: q50 come stima puntuale | EN: q50 as point estimate
                # IT: sigma da q95-q5 diviso 2.56 (gaussiana approx) | EN: sigma from q95-q5 / 2.56 (gaussian approx)
                sigs.append(((qp[:, 4] - qp[:, 0]).clamp(min=1e-6) / 2.56).cpu().numpy())
                nus.append(np.full(len(qp), 5.0, dtype=np.float32))
            else:
                mus.append(out[0].cpu().numpy())
                sigs.append((F.softplus(out[1]) + 1e-6).sqrt().cpu().numpy())
                nus.append((F.softplus(out[2]) + 2.0 + 1e-6).cpu().numpy())
            ys.append(yb.cpu().numpy())

    mu_a  = np.concatenate(mus)
    sig_a = np.concatenate(sigs)
    nu_a  = np.concatenate(nus)
    y_a   = np.concatenate(ys)
    return fold_metrics(y_a, mu_a, sig_a, nu_a)


# IT: main — CLI + caricamento dataset + loop sui fold + aggregato
# EN: main — CLI + dataset load + per-fold loop + aggregate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-retrain", action="store_true",
                        help="Valuta solo il modello già addestrato (veloce, meno rigoroso)")
    parser.add_argument("--max-epochs", type=int, default=40,
                        help="Max epoche per fold (default 40)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience early stopping per fold (default 10)")
    args = parser.parse_args()

    cfg    = load_config("config/default.yaml")
    device = setup_device(cfg)
    ensure_dirs("results", "models")
    Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    # IT: ricompone il dataset completo (train+val+test) per ri-splittare in fold
    # EN: rebuild full dataset (train+val+test) to re-split into temporal folds
    data = np.load("data/lstm_dataset.npz", allow_pickle=True)
    X    = np.concatenate([data["X_train"], data["X_val"], data["X_test"]])
    y    = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    t    = np.concatenate([data["t_train"], data["t_val"], data["t_test"]])

    # IT: n_dynamic costante tra fold, letto una sola volta
    # EN: n_dynamic constant across folds, read once
    n_dynamic = int(data["n_dynamic_features"][0]) if "n_dynamic_features" in data.files else None

    has_macro = ("X_macro_train" in data.files and
                 cfg["model"].get("use_macro", True))
    Xm = None
    if has_macro:
        Xm = np.concatenate([
            data["X_macro_train"], data["X_macro_val"],
            data.get("X_macro_test", data["X_macro_val"][:len(data["X_test"])]),
        ])

    val_cfg = cfg.get("validation", {})
    n_folds       = val_cfg.get("n_folds", 3)
    embargo_steps = val_cfg.get("embargo_steps", 60)

    folds = walk_forward_folds(X, y, t,
                               n_folds=n_folds,
                               embargo_steps=embargo_steps,
                               val_frac=cfg["training"]["val_fraction"])

    print(f"""
{'═'*64}
  02b · WALK-FORWARD VALIDATION
  Modo     : {'Valuta modello esistente' if args.no_retrain else 'Riaddestra per fold'}
  Fold     : {len(folds)} (embargo={embargo_steps})
  Dataset  : {len(y):,} campioni  |  macro={'sì' if has_macro else 'no'}
  Max ep.  : {args.max_epochs}  |  patience: {args.patience}
{'═'*64}
  {'Fold':<5} {'NLL':>8} {'DA':>7} {'ρ':>9} {'WHR':>7} {'CI90':>7} {'N':>6} {'Tempo':>7}
  {'─'*60}""")

    fold_results = []
    t0_total     = time.time()

    for fold in folds:
        k = fold["fold"]
        t0_fold = time.time()

        X_tr = fold["X_train"];        y_tr = fold["y_train"]
        X_vi = fold["X_val_internal"]; y_vi = fold["y_val_internal"]
        X_vl = fold["X_val"];          y_vl = fold["y_val"]

        # IT: allinea Xm agli stessi indici di X per il fold corrente
        # EN: align Xm to the same indices of X for the current fold
        n_tr = len(y_tr)
        Xm_tr = Xm[:n_tr] if has_macro else None
        Xm_vi = Xm[n_tr : fold["train_end_idx"]] if has_macro else None
        Xm_vl = Xm[fold["val_start_idx"] : fold["val_end_idx"]] if has_macro else None

        if args.no_retrain:
            # IT: modalita' veloce — valuta il modello globale gia' addestrato
            # EN: fast path — evaluate the already-trained global model
            try:
                from quantsys.model import load_model
                _best_pt = str(Path(cfg["training"]["output_dir"]) / "best_model.pt")
                model = load_model(_best_pt).to(device)
                val_nll = float("nan")
            except FileNotFoundError:
                log.error(f"{_best_pt} non trovato. Esegui prima 02_train.py.")
                return
        else:
            # IT: walk-forward vero — riaddestra da zero sul fold
            # EN: true walk-forward — retrain from scratch on this fold
            model, val_nll = train_fold(
                X_tr, y_tr, X_vi, y_vi,
                X_macro_tr=Xm_tr, X_macro_vl=Xm_vi,
                cfg=cfg, device=device, fold_id=k,
                max_epochs=args.max_epochs, patience=args.patience,
                n_dynamic=n_dynamic,
            )

        # IT: valuta sul fold di test (out-of-sample puro)
        # EN: evaluate on the held-out fold (pure out-of-sample)
        m = eval_model(model, X_vl, y_vl, Xm_vl, device=device,
                       cfg=cfg, has_macro=has_macro)
        elapsed = time.time() - t0_fold

        fold_results.append({**m, "fold": k, "val_nll": val_nll,
                              "n": len(y_vl), "elapsed_s": elapsed})

        print(f"  {k:<5} {val_nll:>8.4f} {m['da']:>7.3f} {m['spearman']:>+9.4f} "
              f"{m['whr']:>7.3f} {m['ci90']:>7.3f} {len(y_vl):>6} {elapsed:>6.0f}s")

    # IT: aggregato cross-fold (media, std, range)
    # EN: cross-fold aggregate (mean, std, range)
    total_elapsed = time.time() - t0_total
    keys = ["da", "spearman", "whr", "ci90"]
    agg  = {}
    print(f"  {'─'*60}")
    for k in keys:
        vals = [r[k] for r in fold_results]
        mean = float(np.mean(vals)); std = float(np.std(vals))
        agg[k] = {"mean": mean, "std": std,
                  "min": float(np.min(vals)), "max": float(np.max(vals))}
        print(f"  {k:<22} {mean:>+8.4f} ± {std:.4f}  "
              f"[{agg[k]['min']:+.4f}, {agg[k]['max']:+.4f}]")

    # IT: diagnosi automatica (stabilita' Spearman, calibrazione CI90)
    # EN: auto-diagnosis (Spearman stability, CI90 calibration)
    sp_mean = agg["spearman"]["mean"]
    sp_std  = agg["spearman"]["std"]
    ci_mean = agg["ci90"]["mean"]
    print(f"""
  ── Diagnosi ────────────────────────────────────────────
  {'⚠ Spearman instabile (σ='+f'{sp_std:.3f}'+'>0.05) → dipende dal regime' 
    if sp_std > 0.05 else '✓ Spearman stabile (σ='+f'{sp_std:.3f}'+')'}
  {'⚠ Spearman medio debole ('+f'{sp_mean:.4f}'+' < 0.02)' 
    if sp_mean < 0.02 else '✓ Spearman medio '+f'{sp_mean:.4f}'}
  {'⚠ CI90 lontano da 0.90: '+f'{ci_mean:.3f}' 
    if abs(ci_mean-0.90) > 0.08 else '✓ Calibrazione CI90 '+f'{ci_mean:.3f}'}
  Tempo totale: {total_elapsed:.0f}s""")

    # IT: salva risultati per dashboard /api/walkforward
    # EN: persist results for the dashboard /api/walkforward endpoint
    out_path = Path(cfg["backtest"]["output_dir"]) / "walkforward_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"fold_results": fold_results, "aggregate": agg,
                   "retrained": not args.no_retrain,
                   "n_folds": len(folds), "embargo_steps": embargo_steps}, f, indent=2)
    print(f"\n  Risultati → {out_path}")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()

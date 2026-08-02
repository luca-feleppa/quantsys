"""
Script 02c — Ricerca bayesiana degli iperparametri con Optuna.
Ogni trial allena QuantLSTM per max 20 epoche (patience=5) e restituisce
la validation NLL. Lo studio persiste su SQLite — può essere ripreso in qualunque
momento con gli stessi argomenti.

Run:
  python scripts/02c_optuna_search.py [--n-trials 50] [--study-name quantsys] [--timeout 3600]
"""
import argparse
import copy
import json
import logging
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# IT: cap thread BLAS/OMP prima di numpy/torch (cpu_fraction da config)
# EN: cap BLAS/OMP threads before numpy/torch (cpu_fraction from config)
import yaml as _yaml
with open(Path(__file__).resolve().parent.parent / "config" / "default.yaml", encoding="utf-8") as _f:
    _cpu_frac = _yaml.safe_load(_f).get("hardware", {}).get("cpu_fraction", 0.5)
_cpu_limit = str(max(1, int(os.cpu_count() * _cpu_frac)))
os.environ.setdefault("OMP_NUM_THREADS", _cpu_limit)
os.environ.setdefault("MKL_NUM_THREADS", _cpu_limit)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(int(_cpu_limit))

import optuna

from quantsys.utils import load_config, setup_device, setup_logging, ensure_dirs
from quantsys.model import QuantLSTM, student_t_nll

# IT: silenzia Optuna verbose — usiamo i nostri log applicativi
# EN: silence verbose Optuna logs — we use our own application logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

setup_logging()
log = logging.getLogger("quantsys.script.02c")


# IT: scheduler identico a 02_train.py per coerenza HPO/training
# EN: same scheduler as 02_train.py for HPO/training consistency

class CosineWarmup(torch.optim.lr_scheduler.LambdaLR):
    # IT: salva i parametri warmup/total/min_frac e registra il lambda LR
    # EN: store warmup/total/min_frac params and register the LR lambda
    def __init__(self, opt, warmup, total, min_frac=0.05):
        self.w, self.t, self.m = warmup, total, min_frac
        super().__init__(opt, self._lr)

    # IT: moltiplicatore LR per step: warmup lineare poi decay cosine
    # EN: per-step LR multiplier: linear warmup then cosine decay
    def _lr(self, step):
        if step < self.w:
            return step / max(self.w, 1)
        p = (step - self.w) / max(self.t - self.w, 1)
        return self.m + (1 - self.m) * 0.5 * (1 + math.cos(math.pi * p))


# IT: epoche train/val ridotte per HPO (fedeli alle equivalenti di 02_train)
# EN: lightweight train/val epochs for HPO (mirror 02_train counterparts)

def _run_epoch_train(model, loader, opt, scaler_amp, sched, device, use_amp):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            mu, ls2, lnu = model(Xb)
            # IT: NLL simmetrica in HPO — i pesi asimmetrici restano fissi
            # EN: symmetric NLL during HPO — asymmetric weights stay fixed
            loss = student_t_nll(yb, mu, ls2, lnu)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(opt)
        scaler_amp.update()
        sched.step()
        total += loss.item()
    return total / len(loader)


# IT: epoca di validation (no grad, NLL simmetrica) per il trial HPO
# EN: validation epoch (no grad, symmetric NLL) for the HPO trial

def _run_epoch_val(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            mu, ls2, lnu = model(Xb)
            total += student_t_nll(yb, mu, ls2, lnu).item()
    return total / len(loader)


# IT: objective Optuna (un trial = un training breve)
# EN: Optuna objective (one trial = one short training run)

MAX_EPOCHS = 20   # IT: budget per trial | EN: per-trial budget
PATIENCE   = 5    # IT: early stop trial | EN: per-trial early stop


def objective(trial, base_cfg, device,
              X_train, y_train, X_val, y_val,
              n_feat, n_dynamic):
    # IT: 1) campiona iperparametri dallo spazio di ricerca
    # EN: 1) sample hyperparameters from search space
    lstm_hidden      = trial.suggest_categorical("lstm_hidden",      [128, 256, 512])
    gru_hidden       = trial.suggest_categorical("gru_hidden",       [64, 128, 256])
    dropout          = trial.suggest_float("dropout",          0.1, 0.4)
    learning_rate    = trial.suggest_float("learning_rate",    1e-5, 1e-3, log=True)
    batch_size       = trial.suggest_categorical("batch_size",       [64, 128, 256])
    forecast_horizon = trial.suggest_categorical("forecast_horizon", [10, 15, 20])

    # IT: n_heads deve dividere lstm_hidden — scegli il massimo divisore valido
    # EN: n_heads must divide lstm_hidden — pick the largest valid divisor
    head_candidates = [h for h in [4, 8, 16] if lstm_hidden % h == 0]
    n_attention_heads = max(head_candidates)

    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["lstm_hidden"]      = lstm_hidden
    cfg["model"]["gru_hidden"]       = gru_hidden
    cfg["model"]["dropout"]          = dropout
    cfg["training"]["learning_rate"] = learning_rate
    cfg["training"]["batch_size"]    = batch_size
    cfg["features"]["forecast_horizon"] = forecast_horizon

    mcfg  = cfg["model"]
    tcfg  = cfg["training"]
    hwcfg = cfg["hardware"]

    # IT: 2) DataLoader trial — num_workers=0 evita deadlock Win con trial paralleli
    # EN: 2) per-trial DataLoader — num_workers=0 avoids Win deadlocks under parallelism
    kw = dict(pin_memory=(hwcfg["pin_memory"] and device.type == "cuda"),
              num_workers=0)
    train_dl = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=True, **kw)
    val_dl   = DataLoader(TensorDataset(X_val, y_val),
                          batch_size=batch_size, shuffle=False, **kw)

    # IT: 3) costruisci modello con gli HP campionati
    # EN: 3) build model with sampled hyperparameters
    model = QuantLSTM(
        n_features        = n_feat,
        lstm_hidden       = lstm_hidden,
        gru_hidden        = gru_hidden,
        mlp_hidden        = mcfg["mlp_hidden"],
        n_lstm_layers     = mcfg["lstm_layers"],
        dropout           = dropout,
        n_attention_heads = n_attention_heads,
        use_attention     = mcfg.get("use_attention", True),
        n_dynamic_features= n_dynamic,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(),
                             lr=learning_rate,
                             weight_decay=tcfg["weight_decay"])
    total_steps = MAX_EPOCHS * len(train_dl)
    sched       = CosineWarmup(opt, min(200, total_steps // 10), total_steps)

    use_amp   = tcfg["use_amp"] and device.type == "cuda"
    amp_sc    = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    # IT: 4) training con early stop locale + pruning Optuna (MedianPruner)
    # EN: 4) training with local early stop + Optuna pruning (MedianPruner)
    best_val   = float("inf")
    no_improve = 0

    # IT: checkpoint trial in tempfile — evita di sporcare models/ con centinaia di .pt
    # EN: per-trial checkpoint in tempfile — keeps models/ clean across many trials
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        ckpt_path = tf.name

    try:
        for epoch in range(MAX_EPOCHS):
            _run_epoch_train(model, train_dl, opt, amp_sc, sched, device, use_amp)
            val_loss = _run_epoch_val(model, val_dl, device)

            # IT: report a Optuna -> input per il pruner mediano
            # EN: report to Optuna -> feeds the median pruner
            trial.report(val_loss, epoch)

            # IT: il pruner taglia trial sotto la mediana storica
            # EN: pruner kills trials below the running median
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break
    finally:
        Path(ckpt_path).unlink(missing_ok=True)

    return best_val


# IT: main — carica dati, crea studio Optuna persistente, scrive best_params.json
# EN: main — load data, create persistent Optuna study, write best_params.json

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
    parser = argparse.ArgumentParser(
        description="Ricerca bayesiana iperparametri QuantLSTM con Optuna"
    )
    parser.add_argument("--n-trials",    type=int,   default=50,        help="Numero di trial (default: 50)")
    parser.add_argument("--study-name",  type=str,   default="quantsys", help="Nome dello studio Optuna")
    parser.add_argument("--timeout",     type=float, default=3600,      help="Timeout in secondi (default: 3600)")
    args = parser.parse_args()

    # IT: carica il dataset pre-processato (richiede 01_download_data.py)
    # EN: load the preprocessed dataset (requires 01_download_data.py)
    dataset_path = Path("data/lstm_dataset.npz")
    if not dataset_path.exists():
        print(
            f"ERRORE: {dataset_path} non trovato.\n"
            "Esegui prima: python scripts/01_download_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg    = load_config("config/default.yaml")
    device = setup_device(cfg)
    ensure_dirs("models")
    Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    data = np.load(str(dataset_path), allow_pickle=True)
    to_t = lambda k: torch.tensor(data[k], dtype=torch.float32)

    X_train, y_train = to_t("X_train"), to_t("y_train")
    X_val,   y_val   = to_t("X_val"),   to_t("y_val")
    n_feat   = X_train.shape[2]

    n_dynamic = (int(data["n_dynamic_features"][0])
                 if "n_dynamic_features" in data.files else None)

    log.info(
        f"Dataset caricato: train={tuple(X_train.shape)}  val={tuple(X_val.shape)}  "
        f"n_feat={n_feat}  n_dynamic={n_dynamic}"
    )

    # IT: studio persistente su SQLite — riprendibile in run successivi
    # EN: persistent SQLite study — resumable across runs
    import os as _os_opt
    _opt_arch = _os_opt.environ.get("QUANTSYS_ARCH", "lstm")
    _opt_dir  = Path("models") / _opt_arch
    _opt_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{_opt_dir}/optuna_{args.study_name}.db"
    study   = optuna.create_study(
        study_name    = args.study_name,
        storage       = storage,
        load_if_exists= True,
        direction     = "minimize",
        pruner        = optuna.pruners.MedianPruner(
            n_startup_trials = 5,
            n_warmup_steps   = 3,
        ),
    )

    completed_before = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
    log.info(
        f"Studio '{args.study_name}' caricato  "
        f"(trial completati precedentemente: {completed_before})"
    )

    # IT: chiusura che inietta dati/config nell'objective Optuna
    # EN: closure injecting data/config into the Optuna objective
    def _objective_wrapper(trial):
        return objective(
            trial, cfg, device,
            X_train, y_train, X_val, y_val,
            n_feat, n_dynamic,
        )

    study.optimize(
        _objective_wrapper,
        n_trials  = args.n_trials,
        timeout   = args.timeout,
        # IT: trial in errore -> FAIL, lo studio prosegue con i successivi
        # EN: failing trials -> FAIL, study continues with the next ones
        catch     = (Exception,),
    )

    # IT: estrae best trial e serializza i parametri per 02_train.py
    # EN: extract best trial and serialize params for 02_train.py
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("Nessun trial completato — impossibile salvare best_params.json.",
              file=sys.stderr)
        sys.exit(1)

    best = study.best_trial
    best_params = {
        "lstm_hidden":       best.params["lstm_hidden"],
        "gru_hidden":        best.params["gru_hidden"],
        "dropout":           round(best.params["dropout"], 6),
        "learning_rate":     round(best.params["learning_rate"], 8),
        "batch_size":        best.params["batch_size"],
        "forecast_horizon":  best.params["forecast_horizon"],
        "best_val_loss":     round(best.value, 6),
        "n_trials_completed": len(completed),
        "study_name":        args.study_name,
        "timestamp":         datetime.now().isoformat(),
    }

    cfg = load_config("config/default.yaml")
    out_path = Path(cfg["training"]["output_dir"]) / "best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(
        f"\nBest trial:  val_loss = {best.value:.4f}\n"
        f"  lstm_hidden      : {best.params['lstm_hidden']}\n"
        f"  gru_hidden       : {best.params['gru_hidden']}\n"
        f"  dropout          : {best.params['dropout']:.4f}\n"
        f"  learning_rate    : {best.params['learning_rate']:.6f}\n"
        f"  batch_size       : {best.params['batch_size']}\n"
        f"  forecast_horizon : {best.params['forecast_horizon']}\n"
        f"\nSalvato in {out_path}"
    )


if __name__ == "__main__":
    main()

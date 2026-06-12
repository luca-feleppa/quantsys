"""
Script 02 — Training con supporto Knowledge Distillation.

Modalità:
  - Standard:     python scripts/02_train.py
  - Distillation: python scripts/02_train.py --distill --teacher itransformer

La modalità distillation:
  1. Carica il teacher pre-addestrato
  2. Trasferisce i pesi delle output heads al modello student
  3. Usa loss mista: 0.7 × loss_reale + 0.3 × loss_distillazione
  4. Riduce le epoche al 60% (convergenza accelerata)
"""
import json, logging, math, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

# IT: alias per evitare shadowing da variabili locali con lo stesso nome
# EN: aliases to avoid shadowing by local variables with matching names
_json = json
_sh   = shutil

# IT: limiti CPU/BLAS letti dalla config (deve precedere import numpy/torch)
# EN: CPU/BLAS caps from config (must precede numpy/torch imports)
import yaml as _yaml
with open(Path(__file__).resolve().parent.parent / "config" / "default.yaml", encoding="utf-8") as _f:
    _cpu_frac = _yaml.safe_load(_f).get("hardware", {}).get("cpu_fraction", 0.5)
_cpu_limit = str(max(1, int(os.cpu_count() * _cpu_frac)))
os.environ.setdefault("OMP_NUM_THREADS",      _cpu_limit)
os.environ.setdefault("MKL_NUM_THREADS",      _cpu_limit)
os.environ.setdefault("OPENBLAS_NUM_THREADS",  _cpu_limit)
os.environ.setdefault("NUMEXPR_NUM_THREADS",   _cpu_limit)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(int(_cpu_limit))

from quantsys.utils import load_config, setup_logging, setup_device, ensure_dirs
from quantsys.model import QuantLSTM, student_t_nll, quantile_loss, EarlyStopping, set_clip_bounds

setup_logging()
log = logging.getLogger("quantsys.script.02")


# IT: accuratezza direzionale — frazione di segni predetti corretti
# EN: directional accuracy — fraction of correctly predicted signs
def directional_accuracy(y_true, y_pred):
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


# IT: set ridotto di metriche per logging frequente (DA + spearman globale)
# EN: lightweight metric set for frequent logging (DA + global spearman)
def prediction_metrics_fast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metriche leggere per validation intermedio: solo DA + spearman globale."""
    # IT: versione ridotta per logging frequente; full set in prediction_metrics
    # EN: lightweight set for frequent logging; see prediction_metrics for full
    da = directional_accuracy(y_true, y_pred)
    try:
        from scipy.stats import spearmanr
        with np.errstate(invalid='ignore'):
            corr, _ = spearmanr(y_true, y_pred)
        sp = float(corr) if not np.isnan(corr) else 0.0
    except Exception:
        sp = 0.0
    return {"directional_acc": da, "spearman": sp,
            "spearman_pvalue": 1.0, "weighted_hit_rate": 0.0,
            "ic_mean": 0.0, "icir": 0.0}


# IT: set completo di metriche predittive (DA, spearman, WHR, IC, ICIR)
# EN: full predictive metric set (DA, spearman, WHR, IC, ICIR)
def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Metriche di qualità predittiva complete (Fix 7).
    Gestisce correttamente i casi degeneri (array costanti, NaN).
    """
    from scipy.stats import spearmanr

    da = directional_accuracy(y_true, y_pred)

    try:
        with np.errstate(invalid='ignore'):
            corr, p_value = spearmanr(y_true, y_pred)
        spearman = float(corr)   if not np.isnan(corr)    else 0.0
        p_val    = float(p_value) if not np.isnan(p_value) else 1.0
    except Exception:
        spearman, p_val = 0.0, 1.0

    # IT: Weighted Hit Rate — pesa i segni corretti per la magnitudine di y
    # EN: Weighted Hit Rate — weighs correct signs by |y|, so big moves count more
    correct_mask = np.sign(y_true) == np.sign(y_pred)
    total_abs    = np.abs(y_true).sum()
    weighted_hr  = float(np.abs(y_true[correct_mask]).sum() / (total_abs + 1e-10))

    # IT: IC/ICIR su K sub-periodi non sovrapposti (~temporal slice).
    #     Fix 2026-06-02: precedente window=50 era inflato da autocorrelazione
    #     (target h=30 → sample consecutivi condividono 29 candele → spearman locale
    #     misurava persistenza del segnale, non skill). K=5 slice da ≥1000 sample
    #     ciascuna sono indipendenti e Spearman su slice ≈ skill genuino.
    # EN: IC/ICIR over K non-overlapping temporal slices.
    #     Fix 2026-06-02: previous window=50 was inflated by autocorrelation
    #     (target h=30 → consecutive samples share 29 candles → local spearman
    #     measured signal persistence, not skill). K=5 slices of ≥1000 samples
    #     each are independent → Spearman per slice ≈ genuine skill.
    n_periods       = 5
    min_per_period  = 1000
    if len(y_true) >= n_periods * min_per_period:
        slice_size = len(y_true) // n_periods
        ics = []
        for k in range(n_periods):
            s, e = k * slice_size, (k + 1) * slice_size
            yt = y_true[s:e]; yp = y_pred[s:e]
            if yt.std() < 1e-10 or yp.std() < 1e-10:
                continue
            with np.errstate(invalid='ignore'):
                ic, _ = spearmanr(yt, yp)
            if not np.isnan(ic):
                ics.append(ic)
        if len(ics) > 1:
            ics_arr = np.array(ics)
            ic_mean = float(ics_arr.mean())
            icir    = float(ic_mean / (ics_arr.std() + 1e-10))
        else:
            ic_mean, icir = spearman, 0.0
    else:
        # IT: test troppo piccolo per slice indipendenti → fallback su Spearman globale
        # EN: test set too small for independent slices → fall back to global Spearman
        ic_mean, icir = spearman, 0.0

    return {
        "directional_acc":    da,
        "spearman":           spearman,
        "spearman_pvalue":    p_val,
        "weighted_hit_rate":  weighted_hr,
        "ic_mean":            ic_mean,
        "icir":               icir,
    }


# IT: scheduler: warmup lineare seguito da decadimento cosine fino a min_frac
# EN: scheduler: linear warmup followed by cosine decay down to min_frac
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


# IT: una epoca di training (loss asimmetrica + CRPS + distill + mixup + grad accum)
# EN: one training epoch (asymmetric loss + CRPS + distill + mixup + grad accum)
def run_train(model, loader, opt, scaler, sched, device, use_amp, has_macro,
              asym_alpha: float = 2.0, asym_threshold: float = 0.002,
              crps_weight: float = 0.0, grad_accum_steps: int = 1,
              multitask_alpha: float = 0.7, multitask_threshold: float = 0.0001,
              use_distillation: bool = False, distill_alpha: float = 0.3,
              batch_size: int = 32, dv_lambda: float = 0.0,
              input_noise_std: float = 0.0,
              use_sample_weights: bool = False,
              mixup_alpha: float = 0.0,
              crps_distill_weight: float = 0.0):
    """Training con loss asimmetrica + CRPS ausiliario + distillation + direction-value loss.

    Se use_distillation=True, il dataloader contiene soft labels come tensori
    extra (mu, ls2, lnu) che vengono estratti dal batch — shuffle-safe.

    Se use_sample_weights=True, il dataloader contiene un tensore di pesi
    per-sample come ULTIMO elemento del batch. I pesi sono proporzionali a
    |target| / std(target), cosi' i grandi movimenti contribuiscono di piu'
    ai gradienti rispetto al rumore laterale.
    """
    from quantsys.model.distillation import distillation_loss_t_student
    from quantsys.model import direction_value_loss

    # IT: in regime distill il teacher già calibra sigma/nu nelle soft labels;
    #     CRPS sulla supervised loss creerebbe gradienti competitivi sulla varianza.
    # EN: under distill the teacher already calibrates sigma/nu via soft labels;
    #     keeping CRPS on the supervised loss would fight that signal.
    _crps_effective = crps_distill_weight if use_distillation else crps_weight

    model.train(); total = 0.0
    _grad_norms_acc = []  # IT: norme pre-clip per detect exploding/vanishing
                          # EN: pre-clip norms to detect exploding/vanishing grads
    opt.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        # IT: sample weights, se presenti, sono SEMPRE l'ultimo tensore del batch
        # EN: when present, sample weights are ALWAYS the last tensor in the batch
        if use_sample_weights:
            batch, sw_batch = list(batch[:-1]), batch[-1].to(device, non_blocking=True)
        else:
            batch, sw_batch = list(batch), None

        if use_distillation:
            if has_macro:
                Xb, Xm, yb, t_mu, t_ls2, t_lnu = [x.to(device, non_blocking=True) for x in batch]
            else:
                Xb, yb, t_mu, t_ls2, t_lnu = [x.to(device, non_blocking=True) for x in batch]; Xm = None
        elif has_macro:
            Xb, Xm, yb = [x.to(device, non_blocking=True) for x in batch]
        else:
            Xb, yb = [x.to(device, non_blocking=True) for x in batch]; Xm = None
        if input_noise_std > 0:
            Xb = Xb + torch.randn_like(Xb) * input_noise_std  # IT: data augmentation gaussiana | EN: gaussian input noise

        # IT: mixup temporale (B,T,F): lam unico sul batch per AMP stabile;
        #     mixa target/macro/soft labels per coerenza, weights/dir restano sull'originale.
        # EN: temporal mixup (B,T,F): single lam per batch for AMP stability;
        #     mixes target/macro/soft labels; weights/dir labels follow the original i.
        if mixup_alpha > 0:
            lam = float(torch.distributions.Beta(
                torch.tensor(mixup_alpha), torch.tensor(mixup_alpha)
            ).sample().item())
            lam = max(lam, 1.0 - lam)  # IT: bias verso sample originale | EN: bias toward the original sample
            perm = torch.randperm(Xb.size(0), device=Xb.device)
            Xb = lam * Xb + (1.0 - lam) * Xb[perm]
            yb = lam * yb + (1.0 - lam) * yb[perm]
            if Xm is not None:
                Xm = lam * Xm + (1.0 - lam) * Xm[perm]
            if use_distillation:
                t_mu  = lam * t_mu  + (1.0 - lam) * t_mu[perm]
                t_ls2 = lam * t_ls2 + (1.0 - lam) * t_ls2[perm]
                t_lnu = lam * t_lnu + (1.0 - lam) * t_lnu[perm]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(Xb, Xm) if has_macro else model(Xb)

            # IT: ramo loss in base alla testa di output del modello
            # EN: branch on model output head (quantile vs t-Student)
            if model.loss_type == "quantile":
                quantile_preds = out[0]
                main_loss = quantile_loss(yb, quantile_preds,
                                         sample_weights=sw_batch)
            else:
                mu, log_s2, log_nu = out[0], out[1], out[2]
                main_loss = student_t_nll(yb, mu, log_s2, log_nu,
                                         asymmetry_alpha=asym_alpha,
                                         large_move_threshold=asym_threshold,
                                         crps_weight=_crps_effective,
                                         sample_weights=sw_batch)

            # IT: multitask head — classifica SHORT/HOLD/LONG con soft label
            #     basata su |y| (più |y| alto -> più confidenza nella direzione).
            # EN: multitask head — SHORT/HOLD/LONG with soft labels driven by |y|
            #     (larger |y| -> stronger directional confidence).
            if model.use_multitask:
                dir_logits = out[-1]
                thr = multitask_threshold
                conf = torch.tanh(yb.abs() / max(thr, 1e-8))
                soft_labels = torch.zeros(yb.shape[0], 3, device=yb.device)
                soft_labels[:, 1] = 1.0 - conf  # IT/EN: HOLD
                soft_labels[:, 2] = torch.where(yb > 0, conf, torch.zeros_like(conf))  # IT/EN: LONG
                soft_labels[:, 0] = torch.where(yb < 0, conf, torch.zeros_like(conf))  # IT/EN: SHORT
                dir_loss = -(soft_labels * F.log_softmax(dir_logits, dim=-1)).sum(dim=-1).mean()
                loss_real = multitask_alpha * main_loss + (1 - multitask_alpha) * dir_loss
            else:
                loss_real = main_loss

            # IT: direction-value loss — penalizza segno sbagliato di mu
            # EN: direction-value loss — penalises wrong-sign mu predictions
            if dv_lambda > 0 and model.loss_type != "quantile":
                loss_real = loss_real + direction_value_loss(yb, mu, lambda_dv=dv_lambda)

            # IT: distillation loss — soft labels lette dal batch (shuffle-safe)
            # EN: distillation loss — soft labels pulled from batch (shuffle-safe)
            if use_distillation:
                # IT: per testa quantile mappiamo q50/IQR a mu/log-sigma-equiv per
                #     riusare la distillation_loss della t-Student.
                # EN: for quantile heads we map q50/IQR to mu/log-sigma equivalents
                #     so we can reuse the t-Student distillation loss.
                if model.loss_type == "quantile":
                    s_mu  = out[0][:, 2]
                    s_ls2 = (out[0][:, 4] - out[0][:, 0]).clamp(min=1e-6)
                    s_lnu = torch.full_like(s_mu, 5.0)
                else:
                    s_mu, s_ls2, s_lnu = out[0], out[1], out[2]

                d_loss = distillation_loss_t_student(
                    s_mu, s_ls2, s_lnu, t_mu, t_ls2, t_lnu
                )
                loss = (1.0 - distill_alpha) * loss_real + distill_alpha * d_loss
            else:
                loss = loss_real

            loss = loss / grad_accum_steps
        loss_val = loss.item() * grad_accum_steps
        # IT: NaN/Inf skip per non corrompere le statistiche di AMP scaler
        # EN: NaN/Inf skip to avoid poisoning AMP scaler statistics
        if math.isnan(loss_val) or math.isinf(loss_val):
            opt.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        total += loss_val

        # IT: step optimizer ogni grad_accum_steps (gradient accumulation)
        # EN: optimizer step every grad_accum_steps (gradient accumulation)
        if (step + 1) % grad_accum_steps == 0 or step + 1 == len(loader):
            scaler.unscale_(opt)
            # IT: salviamo la norma pre-clip per monitorare exploding/vanishing
            # EN: keep pre-clip norm to monitor exploding/vanishing gradients
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
            try:
                _grad_norms_acc.append(float(gnorm.item()))
            except Exception:
                pass
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()

    # IT: appende le grad-norm sull'oggetto opt — recuperate dal main loop
    # EN: stash grad-norm stats on opt so the main loop can fetch them
    if _grad_norms_acc:
        import numpy as _np_g
        _opt_state = getattr(opt, "_qs_gradnorm_stats", None)
        opt._qs_gradnorm_stats = {
            "mean": float(_np_g.mean(_grad_norms_acc)),
            "p95":  float(_np_g.percentile(_grad_norms_acc, 95)),
            "max":  float(max(_grad_norms_acc)),
        }
    return total / len(loader)


# IT: una epoca di eval — NLL simmetrica + raccolta mu/sigma/nu per le metriche
# EN: one eval epoch — symmetric NLL + collect mu/sigma/nu for the metrics
def run_eval(model, loader, device, has_macro, full_metrics: bool = False):
    """Validation con loss simmetrica standard per confrontabilità tra run.

    Returns (loss, mu_arr, y_arr, metrics, sigma_arr, nu_arr).
    sigma_arr e nu_arr sono sempre calcolati per evitare un secondo forward pass.
    """
    # IT: SWA AveragedModel non espone loss_type direttamente; va via .module
    # EN: SWA AveragedModel hides loss_type — resolve via .module once
    loss_type = getattr(model, "loss_type", None)
    if loss_type is None and hasattr(model, "module"):
        loss_type = getattr(model.module, "loss_type", "t_student")
    if loss_type is None:
        loss_type = "t_student"

    model.eval(); total, mus, ys, sigs, nus = 0.0, [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            if has_macro:
                Xb, Xm, yb = [x.to(device, non_blocking=True) for x in batch]
                out = model(Xb, Xm)
            else:
                Xb, yb = [x.to(device, non_blocking=True) for x in batch]
                out = model(Xb)

            if loss_type == "quantile":
                quantile_preds = out[0]
                total += quantile_loss(yb, quantile_preds).item()
                mu_batch = quantile_preds[:, 2]  # IT: q50 come punto | EN: q50 as point estimate
                iqr = (quantile_preds[:, 3] - quantile_preds[:, 1]).abs()  # IT: proxy di sigma | EN: proxy for sigma
                sigs.append(iqr.cpu().numpy())
                nus.append(np.full(len(iqr), 10.0))  # IT: nu fittizio | EN: placeholder nu
            else:
                mu_batch = out[0]
                total += student_t_nll(yb, out[0], out[1], out[2]).item()
                sigs.append((F.softplus(out[1])+1e-6).sqrt().cpu().numpy())
                nus.append((F.softplus(out[2])+2.0+1e-6).cpu().numpy())

            mus.append(mu_batch.cpu().numpy()); ys.append(yb.cpu().numpy())
    mu_arr = np.concatenate(mus); y_arr = np.concatenate(ys)
    sigma_arr = np.concatenate(sigs); nu_arr = np.concatenate(nus)
    _mfn = prediction_metrics if full_metrics else prediction_metrics_fast
    metrics = _mfn(y_arr, mu_arr)
    return total / len(loader), mu_arr, y_arr, metrics, sigma_arr, nu_arr


_GRAD_CLIP = 1.0  # IT: sovrascritto da main() da config | EN: overridden by main() from config


# IT: allinea i regime label Markov-Switching ai sample di validation (merge_asof)
# EN: align Markov-Switching regime labels to validation samples (merge_asof)
def _load_val_regimes(data) -> "np.ndarray | None":
    """Allinea i regime label (Markov-Switching) ai sample di validation.

    Legge `data/regime_probs.parquet` (prodotto da 01b_download_macro.py) che
    contiene `regime_dominant` per ogni timestamp giornaliero. Allinea con
    `t_val` dal dataset npz tramite merge_asof backward (ultimo regime noto).

    Returns:
        np.ndarray (N_val,) di int regime label, oppure None se mancano dati.
    """
    if "t_val" not in data.files:
        log.info("Stratified val: t_val non nel dataset → skip per-regime metrics")
        return None
    reg_path = Path("data") / "regime_probs.parquet"
    if not reg_path.exists():
        log.info(f"Stratified val: {reg_path} non trovato → skip per-regime metrics")
        return None
    try:
        import pandas as pd
        df_reg = pd.read_parquet(reg_path)
        if "regime_dominant" not in df_reg.columns:
            log.info("Stratified val: regime_dominant non presente nel parquet → skip")
            return None
        # IT: serve un indice timestamp confrontabile con t_val
        # EN: needs a timestamp index/column compatible with t_val
        if not isinstance(df_reg.index, pd.DatetimeIndex):
            tcol = next((c for c in ("open_time", "timestamp", "date") if c in df_reg.columns), None)
            if tcol is None:
                log.info("Stratified val: nessuna colonna timestamp nel parquet → skip")
                return None
            df_reg = df_reg.set_index(pd.to_datetime(df_reg[tcol])).sort_index()
        else:
            df_reg = df_reg.sort_index()
        # IT: merge_asof richiede stessa risoluzione (ns) e niente tz mismatch
        # EN: merge_asof needs identical dt resolution (ns) and tz-naive on both sides
        def _to_ns_naive(idx_or_series):
            s = pd.to_datetime(idx_or_series)
            if getattr(s, "tz", None) is not None:
                s = s.tz_convert("UTC").tz_localize(None)
            return s.astype("datetime64[ns]")
        df_reg.index = _to_ns_naive(df_reg.index)
        t_val_raw = data["t_val"]
        t_val = _to_ns_naive(pd.Index(pd.to_datetime(t_val_raw)))
        df_val = pd.DataFrame({"_t": t_val})
        merged = pd.merge_asof(
            df_val.sort_values("_t"),
            df_reg[["regime_dominant"]].reset_index().rename(columns={df_reg.index.name or "index": "_t"}),
            on="_t", direction="backward"
        )
        regimes = merged["regime_dominant"].to_numpy()
        # IT: merge_asof riordina, riportiamo i regime nell'ordine originale di t_val
        # EN: merge_asof sorts the result — restore original t_val ordering
        order = np.argsort(np.argsort(t_val.values))
        regimes = regimes[order]
        return regimes
    except Exception as e:
        log.warning(f"Stratified val: errore caricamento regime — {e}")
        return None


# IT: NLL t-Student calcolata separatamente per ogni regime di mercato
# EN: t-Student NLL computed separately for each market regime
def _per_regime_nll(mu_arr, sigma_arr, nu_arr, y_arr, regimes) -> dict:
    """Computa val_nll separato per ogni regime (NLL della t-Student).

    Returns: dict {regime_id: nll_value}. NaN regime skippati.
    """
    if regimes is None or len(regimes) != len(y_arr):
        return {}
    # IT: NLL t-Student con sigma/nu già nello spazio naturale (formula chiusa sotto)
    # EN: t-Student NLL with sigma/nu already in natural space (closed form below)
    # log p(y|mu,sigma,nu) = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*pi)
    #                        - log(sigma) - ((nu+1)/2)*log(1 + (y-mu)^2/(nu*sigma^2))
    from scipy.special import gammaln
    sigma2 = np.clip(sigma_arr.astype(np.float64), 1e-8, None) ** 2
    nu = np.clip(nu_arr.astype(np.float64), 2.01, None)
    z2 = (y_arr.astype(np.float64) - mu_arr.astype(np.float64)) ** 2 / (nu * sigma2)
    nll = -(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
            - 0.5 * np.log(sigma2) - ((nu + 1) / 2) * np.log1p(z2))
    valid = ~np.isnan(regimes.astype(float))
    out = {}
    for r in np.unique(regimes[valid]):
        mask = (regimes == r) & valid
        if mask.sum() < 10:
            continue
        out[int(r)] = float(nll[mask].mean())
    return out


# IT: parsing dei flag CLI di distillation (--distill, --teacher, alpha schedule)
# EN: parse distillation CLI flags (--distill, --teacher, alpha schedule)
def _parse_distill_args():
    """Parse --distill and --teacher CLI flags."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--distill", action="store_true",
                        help="Enable knowledge distillation from teacher")
    parser.add_argument("--teacher", type=str, default="itransformer",
                        help="Teacher architecture (default: itransformer)")
    parser.add_argument("--multi-teacher", action="store_true",
                        help="Use all 3 architectures as weighted teachers")
    parser.add_argument("--distill-alpha", type=float, default=0.6,
                        help="Initial weight for distillation loss (default: 0.6; scheduled to --distill-alpha-final)")
    parser.add_argument("--distill-alpha-final", type=float, default=0.1,
                        help="Final distillation alpha after decay (default: 0.1)")
    parser.add_argument("--distill-alpha-decay-epochs", type=int, default=20,
                        help="Epoche per decay lineare di distill_alpha (default: 20)")
    parser.add_argument("--student-epoch-fraction", type=float, default=0.6,
                        help="Fraction of epochs for student training (default: 0.6)")
    parser.add_argument("--n-ensemble", type=int, default=None,
                        help="Override n_ensemble from config (used by --distill pipeline)")
    parser.add_argument("--mc-samples", type=int, default=1,
                        help="MC Dropout samples per teacher in distillation (default: 1 = deterministico). "
                             "Valori tipici: 5-10 per soft labels stocastiche che catturano uncertainty teacher")
    args, _ = parser.parse_known_args()
    return args


# IT: entrypoint — setup, dati, loop ensemble, training, eval test, export artefatti
# EN: entrypoint — setup, data, ensemble loop, training, test eval, artifact export
def main():
    global _GRAD_CLIP
    # IT: Forza UTF-8 su stdout/stderr — evita UnicodeEncodeError cp1252 sui banner Unicode quando
    #     l'output è rediretto/in pipe su Windows (es. `02_train.py *> file.log`, o background).
    #     Bug osservato 2026-06-06 (distill in background → exit 1 sul print finale, modello già
    #     salvato). Stesso fix di 04_live_signals.py / run_all.py / 99_replay_live_vs_training.py.
    # EN: Force UTF-8 on stdout/stderr — avoids cp1252 UnicodeEncodeError on Unicode banners when
    #     output is redirected/piped on Windows. The model is saved before the crashing banner.
    import sys as _sys
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    cfg   = load_config("config/default.yaml")
    _GRAD_CLIP = cfg.get("training", {}).get("grad_clip_norm", 1.0)
    # 2026-05-15: opt-in SN solo su mu_head (anti-overfit). Default False = legacy
    # (compatibile checkpoint pre-2026-05-15).
    from quantsys.model import set_sn_on_mu_only
    set_sn_on_mu_only(bool(cfg.get("training", {}).get("sn_on_mu_only", False)))

    distill_args = _parse_distill_args()
    use_distillation = distill_args.distill
    use_multi_teacher = distill_args.multi_teacher
    teacher_arch     = distill_args.teacher
    distill_alpha    = distill_args.distill_alpha
    distill_alpha_final = distill_args.distill_alpha_final
    distill_alpha_decay_epochs = distill_args.distill_alpha_decay_epochs
    student_epoch_frac = distill_args.student_epoch_fraction
    mc_samples = max(1, int(getattr(distill_args, "mc_samples", 1)))

    # IT: crea dir esperimento con timestamp — storicizza ogni run (config + metriche)
    # EN: create timestamped experiment dir — archives each run (config + metrics)
    exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    _out_dir_early = Path(cfg["training"]["output_dir"])
    exp_dir  = _out_dir_early / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # IT: salva la config usata (riproducibilità)
    # EN: persist the config used (reproducibility)
    try:
        shutil.copy("config/default.yaml", exp_dir / "config.yaml")
    except Exception as _e:
        log.warning(f"Impossibile copiare config: {_e}")

    # IT: salva il git hash se disponibile (tracciabilità codice)
    # EN: record the git hash when available (code traceability)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=str(Path(__file__).parent.parent)
        ).decode().strip()
    except Exception:
        git_hash = "no-git"

    exp_meta = {
        "timestamp": exp_name,
        "git_hash":  git_hash,
        "python":    sys.version.split()[0],
        "config":    cfg,
    }
    with open(exp_dir / "meta.json", "w", encoding="utf-8") as _f:
        json.dump(exp_meta, _f, indent=2, default=str)

    log.info(f"Experiment tracking: {exp_dir}  (git={git_hash})")

    tcfg  = cfg["training"]; mcfg = cfg["model"]
    hwcfg = cfg["hardware"]; mccfg = cfg.get("macro", {})
    device = setup_device(cfg)
    ensure_dirs(tcfg["output_dir"])
    out_dir = Path(tcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data  = np.load("data/lstm_dataset.npz", allow_pickle=True)
    to_t  = lambda k: torch.from_numpy(data[k].astype(np.float32))

    X_tr, y_tr = to_t("X_train"), to_t("y_train")
    X_vl, y_vl = to_t("X_val"),   to_t("y_val")
    X_te, y_te = to_t("X_test"),  to_t("y_test")
    n_feat = X_tr.shape[2]

    # ── Stratified validation per regime (2026-05-15) ───────────────────────
    # IT: carica i regime label per ogni sample di val; se mancano → None (graceful)
    # EN: load regime labels per val sample; if absent → None (graceful degrade)
    val_regimes = _load_val_regimes(data)
    if val_regimes is not None:
        _uniq, _cnt = np.unique(val_regimes[~np.isnan(val_regimes.astype(float))], return_counts=True)
        log.info(f"Stratified val: {len(val_regimes)} sample, distribuzione regime: "
                 + ", ".join(f"r{int(r)}={int(c)} ({c/len(val_regimes):.0%})"
                             for r, c in zip(_uniq, _cnt)))

    # IT: clip bounds adattivi (p0.1/p99.9 per-feature da X_train); no leakage val/test
    # EN: adaptive clip bounds (per-feature p0.1/p99.9 from X_train); no val/test leakage
    log.info("Calcolo clip bounds adattivi da X_train (p0.1 / p99.9) ...")
    _X_flat = X_tr.reshape(-1, n_feat).numpy()
    _clip_lo = np.nanpercentile(_X_flat, 0.1, axis=0).astype(np.float32)
    _clip_hi = np.nanpercentile(_X_flat, 99.9, axis=0).astype(np.float32)
    clip_lo_t = torch.from_numpy(_clip_lo).to(device)   # (F,) — broadcastable su (B,T,F)
    clip_hi_t = torch.from_numpy(_clip_hi).to(device)
    del _X_flat

    # IT: pre-clip una volta sull'intero dataset (no clamp per-batch); clip interno
    #     resta per inference live su dati non pre-clippati.
    # EN: pre-clip once over the whole dataset (no per-batch clamp); in-model clip
    #     stays for live inference on non-pre-clipped data.
    _clip_lo_cpu = torch.from_numpy(_clip_lo)
    _clip_hi_cpu = torch.from_numpy(_clip_hi)
    X_tr = X_tr.clamp(_clip_lo_cpu, _clip_hi_cpu)
    X_vl = X_vl.clamp(_clip_lo_cpu, _clip_hi_cpu)
    X_te = X_te.clamp(_clip_lo_cpu, _clip_hi_cpu)
    del _clip_lo_cpu, _clip_hi_cpu
    if n_feat > 57:
        log.info(f"  ret_kurt_20: [{_clip_lo[57]:.2f}, {_clip_hi[57]:.2f}]  "
                 f"log_ret: [{_clip_lo[9]:.2f}, {_clip_hi[9]:.2f}]")
    else:
        log.info(f"  feature[0]: [{_clip_lo[0]:.2f}, {_clip_hi[0]:.2f}]  "
                 f"n_feat={n_feat}")

    # IT: dual-stream — legge il confine feature dinamiche/strutturali dal dataset
    # EN: dual-stream — read the dynamic/structural feature boundary from the dataset
    n_dynamic = int(data["n_dynamic_features"][0]) if "n_dynamic_features" in data.files else None
    if n_dynamic is not None:
        log.info(f"Dual-stream: {n_dynamic} feature dinamiche + {n_feat-n_dynamic} strutturali")
    else:
        log.info("Single-stream (n_dynamic_features non nel dataset)")

    has_macro = (mcfg.get("use_macro", True)
                 and "X_macro_train" in data.files
                 and "X_macro_val"   in data.files)
    n_macro = 0

    if has_macro:
        Xm_tr = to_t("X_macro_train"); Xm_vl = to_t("X_macro_val")
        Xm_te = to_t("X_macro_test") if "X_macro_test" in data.files \
                else Xm_vl[:1].expand(X_te.shape[0], -1)
        n_macro = Xm_tr.shape[1]
        log.info(f"MacroEncoder ATTIVO: {n_macro} features → {mccfg.get('embed_dim',16)} dim")
    else:
        log.info("MacroEncoder non presente — QuantLSTM standard.")

    # ── Sample weights proporzionali a |target| (large moves contribute more) ──
    # IT: pesi per-sample ∝ |y|/std(y) — i grandi movimenti pesano di più nel loss
    # EN: per-sample weights ∝ |y|/std(y) — big moves contribute more to the loss
    _sw_alpha = tcfg.get("sample_weight_alpha", 0.0)
    _use_sw   = _sw_alpha > 0.0
    if _use_sw:
        _y_std = y_tr.std().clamp(min=1e-8)
        sample_weights_tr = 1.0 + _sw_alpha * (y_tr.abs() / _y_std)
        log.info(f"Sample weighting ATTIVO: alpha={_sw_alpha:.1f}  "
                 f"y_std={_y_std:.6f}  "
                 f"w_range=[{sample_weights_tr.min():.2f}, {sample_weights_tr.max():.2f}]  "
                 f"w_mean={sample_weights_tr.mean():.2f}")
    else:
        sample_weights_tr = None
        log.info("Sample weighting disabilitato (sample_weight_alpha=0)")

    _nw = hwcfg["num_workers"]
    if sys.platform == "win32":
        _nw = 0
    kw = dict(pin_memory=hwcfg["pin_memory"], num_workers=_nw,
              persistent_workers=_nw > 0,
              prefetch_factor=4 if _nw > 0 else None)

    _bs_train = tcfg["batch_size"]
    _bs_eval  = _bs_train * 4
    # IT: sample_weights_tr è SEMPRE l'ultimo tensore del TensorDataset train;
    #     run_train() lo estrae prima dell'unpacking del resto.
    # EN: sample_weights_tr is ALWAYS the last tensor of the train TensorDataset;
    #     run_train() pops it before unpacking the rest.
    if has_macro:
        _tr_tensors = [X_tr, Xm_tr, y_tr] + ([sample_weights_tr] if _use_sw else [])
        train_dl = DataLoader(TensorDataset(*_tr_tensors), _bs_train, shuffle=True,  **kw)
        val_dl   = DataLoader(TensorDataset(X_vl, Xm_vl, y_vl), _bs_eval,  shuffle=False, **kw)
        test_dl  = DataLoader(TensorDataset(X_te, Xm_te, y_te), _bs_eval,  shuffle=False, **kw)
    else:
        _tr_tensors = [X_tr, y_tr] + ([sample_weights_tr] if _use_sw else [])
        train_dl = DataLoader(TensorDataset(*_tr_tensors), _bs_train, shuffle=True,  **kw)
        val_dl   = DataLoader(TensorDataset(X_vl, y_vl), _bs_eval,  shuffle=False, **kw)
        test_dl  = DataLoader(TensorDataset(X_te, y_te), _bs_eval,  shuffle=False, **kw)

    _arch = mcfg.get("architecture", "lstm")
    if _arch == "itransformer":
        model_type = "QuantiTransformer"
    elif _arch == "tft":
        model_type = "QuantTFT"
    elif _arch == "tcnmamba":
        model_type = "QuantTCNMamba"
    elif _arch == "nhits":
        model_type = "QuantNHiTS"
    elif has_macro:
        model_type = "QuantLSTMWithMacro"
    else:
        model_type = "QuantLSTM"
    use_amp    = tcfg["use_amp"] and device.type == "cuda"

    # ── Knowledge Distillation: carica teacher e genera soft labels ──────
    # IT: carica il/i teacher e precalcola le soft labels allineate al train set
    # EN: load teacher(s) and precompute soft labels aligned to the train set
    teacher_preds_train = None
    if use_distillation:
        from quantsys.model.distillation import (
            load_teacher, generate_teacher_predictions, transfer_output_heads,
            generate_multi_teacher_predictions, compute_teacher_weights,
        )
        current_arch = mcfg.get("architecture", "lstm")
        if current_arch == teacher_arch and not use_multi_teacher:
            log.warning(f"Distillation disattivata: student ({current_arch}) == teacher ({teacher_arch})")
            use_distillation = False
        else:
            log.info(f"Knowledge Distillation: teacher={teacher_arch} -> student={current_arch}")
            log.info(f"  distill_alpha={distill_alpha} (initial, schedulato a {distill_alpha_final} in {distill_alpha_decay_epochs} epoche)")
            log.info(f"  multi_teacher={use_multi_teacher}, student_epochs={student_epoch_frac:.0%}")

            # IT: dataloader non-shuffled — predizioni teacher allineate ai sample
            # EN: non-shuffled dataloader — teacher predictions aligned to samples
            if has_macro:
                _train_dl_ordered = DataLoader(
                    TensorDataset(X_tr, Xm_tr, y_tr), _bs_train, shuffle=False, **kw)
            else:
                _train_dl_ordered = DataLoader(
                    TensorDataset(X_tr, y_tr), _bs_train, shuffle=False, **kw)

            if use_multi_teacher:
                from quantsys.model.ensemble import get_distillation_archs
                all_archs = get_distillation_archs(cfg)
                log.info(f"  Multi-teacher: archs={all_archs} (da config/default.yaml)")
                log.info("  Multi-teacher: generazione soft labels pesate da tutti i modelli...")
                arch_weights = compute_teacher_weights(all_archs)
                teacher_preds_train = generate_multi_teacher_predictions(
                    all_archs, arch_weights, _train_dl_ordered, device, has_macro,
                    mc_samples=mc_samples)
                teacher_arch = "multi-teacher"
                teacher_model = load_teacher(
                    max(arch_weights, key=arch_weights.get), device)
            else:
                teacher_model = load_teacher(teacher_arch, device)
                log.info("  Generazione soft labels dal teacher...")
                teacher_preds_train = generate_teacher_predictions(
                    teacher_model, _train_dl_ordered, device, has_macro,
                    mc_samples=mc_samples)

            log.info(f"  Soft labels generate: {teacher_preds_train['mu'].shape[0]} campioni")
            del _train_dl_ordered

            # IT: ricrea il train dataloader con le soft labels integrate (shuffle-safe);
            #     sample_weights_tr (se attivo) resta SEMPRE l'ultimo tensore.
            # EN: rebuild the train dataloader with soft labels embedded (shuffle-safe);
            #     sample_weights_tr (if active) stays ALWAYS the last tensor.
            _t_mu  = teacher_preds_train["mu"]
            _t_ls2 = teacher_preds_train["ls2"]
            _t_lnu = teacher_preds_train["lnu"]
            if has_macro:
                _distill_tensors = [X_tr, Xm_tr, y_tr, _t_mu, _t_ls2, _t_lnu] + \
                                   ([sample_weights_tr] if _use_sw else [])
                train_dl = DataLoader(
                    TensorDataset(*_distill_tensors),
                    _bs_train, shuffle=True, **kw)
            else:
                _distill_tensors = [X_tr, y_tr, _t_mu, _t_ls2, _t_lnu] + \
                                   ([sample_weights_tr] if _use_sw else [])
                train_dl = DataLoader(
                    TensorDataset(*_distill_tensors),
                    _bs_train, shuffle=True, **kw)
            log.info("  Training dataloader ricreato con soft labels integrate (shuffle-safe)")

            # IT: riduci le epoche dello student (convergenza accelerata col teacher)
            # EN: cut student epochs (teacher accelerates convergence)
            original_epochs = tcfg["epochs"]
            tcfg["epochs"] = max(10, int(original_epochs * student_epoch_frac))
            log.info(f"  Epoche ridotte: {original_epochs} -> {tcfg['epochs']} "
                     f"({student_epoch_frac:.0%})")

            # IT: forza n_ensemble=1 — la distillation usa 1 modello per architettura
            # EN: force n_ensemble=1 — distillation uses one model per architecture
            if tcfg.get("n_ensemble", 1) > 1:
                log.info(f"  n_ensemble forzato a 1 (era {tcfg['n_ensemble']}): "
                         f"distillation usa 1 modello per architettura")
                tcfg["n_ensemble"] = 1

            # IT: tieni il teacher in memoria per il transfer heads (no ricaricamento)
            # EN: keep teacher in memory for output-head transfer (avoid reload)
            _teacher_for_transfer = teacher_model

    # IT: stima costo computazionale (mostrata una volta prima dei loop ensemble)
    # EN: compute-cost estimate (printed once before the ensemble loops)
    ram_gb    = X_tr.element_size() * X_tr.nelement() / 1e9
    n_batches = len(train_dl)
    eta_min   = n_batches * 0.05 / 60
    log.info(
        f"Dataset: X_train={tuple(X_tr.shape)} ({ram_gb:.1f} GB)  "
        f"{n_batches} batch/epoca  ETA ~{eta_min:.0f} min/epoca"
    )

    n_ensemble = tcfg.get("n_ensemble", 1)
    if distill_args.n_ensemble is not None:
        n_ensemble = distill_args.n_ensemble
        tcfg["n_ensemble"] = n_ensemble
        log.info(f"n_ensemble override da CLI: {n_ensemble}")
    models_dir = out_dir

    # IT: Pre-check anti-stale warning-only (bug 2026-06-10): un run con n_ensemble
    #     inferiore ai membri numerati già su disco aggiorna solo best_model.pt
    #     (o un sottoinsieme dei membri), lasciando checkpoint stale che
    #     EnsembleModel.load preferirà silenziosamente al nuovo best.
    # EN: Warning-only anti-stale pre-check (2026-06-10 bug): a run with n_ensemble
    #     lower than the numbered members already on disk only updates best_model.pt
    #     (or a subset of the members), leaving stale checkpoints that
    #     EnsembleModel.load will silently prefer over the new best.
    _existing_members = sorted(out_dir.glob("best_model_[0-9]*.pt"))
    if len(_existing_members) >= 2 and n_ensemble < len(_existing_members):
        _updated = ("solo best_model.pt" if n_ensemble == 1
                    else f"solo i primi {n_ensemble} membri numerati")
        log.warning(
            f"ANTI-STALE: in {out_dir} ci sono {len(_existing_members)} membri numerati "
            f"(best_model_*.pt) ma n_ensemble={n_ensemble}: questo run aggiornerà "
            f"{_updated}, lasciando membri stale che EnsembleModel.load preferirà "
            f"al posto del nuovo best. Rimedio: rimuovi/archivia i membri numerati "
            f"o ri-allena con n-ensemble pieno. | ANTI-STALE: {out_dir} holds "
            f"{len(_existing_members)} numbered members but n_ensemble={n_ensemble}: "
            f"this run will leave stale members that EnsembleModel.load will prefer "
            f"over the new best. Remedy: remove/archive the numbered members or "
            f"retrain with the full n-ensemble."
        )

    # IT: history/modello finali — riferiti al membro 0 (o all'unico) per l'export
    # EN: final history/model — refer to member 0 (or the single one) for export
    history    = None
    model      = None
    test_mu    = None
    test_y     = None
    test_metrics = None
    sig_a      = None
    nu_a       = None
    elapsed    = 0.0

    for ensemble_idx in range(n_ensemble):
        seed = 42 + ensemble_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        log.info(f"Ensemble {ensemble_idx+1}/{n_ensemble} (seed={seed})")

        # ── Crea modello fresco ──────────────────────────────────────────────
        # IT: istanzia un modello nuovo per il membro corrente in base all'arch
        # EN: instantiate a fresh model for the current member per the chosen arch
        architecture = mcfg.get("architecture", "lstm")

        if architecture == "itransformer":
            from quantsys.model import QuantiTransformer
            _n_dyn = n_dynamic if n_dynamic is not None else n_feat
            _T     = mcfg.get("window_size", 120)
            _model = QuantiTransformer(
                n_features      = n_feat,
                T               = _T,
                n_dynamic       = _n_dyn,
                n_macro         = n_macro if has_macro else 0,
                d_model         = mcfg.get("tft_d_model", 128),
                n_heads         = mcfg.get("tft_n_heads", 4),
                n_layers        = mcfg.get("tft_n_layers", 3),
                dropout         = mcfg.get("tft_dropout", 0.1),
                # Nuovi parametri
                patch_size      = mcfg.get("patch_size", 1),
                drop_path_rate  = mcfg.get("drop_path_rate", 0.0),
                use_multitask   = mcfg.get("use_multitask", False),
                loss_type       = mcfg.get("loss_type", "t_student"),
                n_output_experts= mcfg.get("n_output_experts", 1),
                use_revin        = mcfg.get("use_revin", False),
                revin_target_idx = mcfg.get("revin_target_idx", 0),
            ).to(device)
            log.info(f"Architettura: QuantiTransformer  d_model={mcfg.get('tft_d_model', 128)}  layers={mcfg.get('tft_n_layers', 3)}  T={_T}")
        elif architecture == "tft":
            from quantsys.model import QuantTFT
            # n_dynamic=None → single-stream: entrambi gli stream ricevono tutte le feature
            _n_dyn    = n_dynamic if n_dynamic is not None else n_feat
            _n_struct = n_feat - _n_dyn  # 0 se single-stream
            _model = QuantTFT(
                n_dynamic    = _n_dyn,
                n_structural = _n_struct,   # 0 = single-stream, handled inside QuantTFT
                n_macro      = n_macro if has_macro else 0,
                d_model      = mcfg.get("tft_d_model", 64),
                n_heads      = mcfg.get("tft_n_heads", 4),
                dropout      = mcfg.get("tft_dropout", 0.1),
            ).to(device)
            log.info(f"Architettura: QuantTFT  d_model={mcfg.get('tft_d_model', 64)}")
        elif architecture == "tcnmamba":
            from quantsys.model import QuantTCNMamba
            _n_dyn = n_dynamic if n_dynamic is not None else n_feat
            _model = QuantTCNMamba(
                n_features         = n_feat,
                d_model            = mcfg.get("d_model", 128),
                tcn_layers         = mcfg.get("tcn_layers", 4),
                tcn_kernel         = mcfg.get("tcn_kernel", 3),
                mamba_layers       = mcfg.get("mamba_layers", 3),
                mamba_d_state      = mcfg.get("mamba_d_state", 16),
                mamba_expand       = mcfg.get("mamba_expand", 2),
                dropout            = mcfg.get("dropout", 0.1),
                drop_path_rate     = mcfg.get("drop_path_rate", 0.0),
                n_dynamic_features = _n_dyn,
                use_multitask      = mcfg.get("use_multitask", False),
                loss_type          = mcfg.get("loss_type", "t_student"),
                n_output_experts   = mcfg.get("n_output_experts", 1),
                use_revin          = mcfg.get("use_revin", False),
                revin_target_idx   = mcfg.get("revin_target_idx", 0),
            ).to(device)
            log.info(
                f"Architettura: QuantTCNMamba  d_model={mcfg.get('d_model', 128)}"
                f"  tcn_layers={mcfg.get('tcn_layers', 4)}  mamba_layers={mcfg.get('mamba_layers', 3)}"
            )
        elif architecture == "nhits":
            from quantsys.model import QuantNHiTS
            _n_dyn = n_dynamic if n_dynamic is not None else n_feat
            _T     = mcfg.get("window_size", 120)
            _model = QuantNHiTS(
                n_features         = n_feat,
                T                  = _T,
                n_dynamic_features = _n_dyn,
                n_macro            = n_macro if has_macro else 0,
                d_model            = mcfg.get("d_model", 128),
                hidden             = mcfg.get("nhits_hidden", 256),
                n_stacks           = mcfg.get("nhits_stacks", 3),
                pool_kernels       = tuple(mcfg.get("nhits_pool_kernels", [8, 4, 1])),
                n_blocks_per_stack = mcfg.get("nhits_blocks_per_stack", 1),
                n_mlp_layers       = mcfg.get("nhits_mlp_layers", 2),
                dropout            = mcfg.get("dropout", 0.1),
                loss_type          = mcfg.get("loss_type", "t_student"),
                use_multitask      = mcfg.get("use_multitask", False),
                n_output_experts   = mcfg.get("n_output_experts", 1),
                use_revin          = mcfg.get("use_revin", False),
                revin_target_idx   = mcfg.get("revin_target_idx", 0),
            ).to(device)
            log.info(
                f"Architettura: QuantNHiTS  d_model={mcfg.get('d_model', 128)}"
                f"  stacks={mcfg.get('nhits_stacks', 3)}"
                f"  kernels={mcfg.get('nhits_pool_kernels', [8, 4, 1])}"
            )
        elif has_macro:
            from quantsys.macro.regime import QuantLSTMWithMacro
            _model = QuantLSTMWithMacro(
                n_price_features    = n_feat, n_macro_features=n_macro,
                lstm_hidden         = mcfg["lstm_hidden"], gru_hidden=mcfg["gru_hidden"],
                mlp_hidden          = mcfg["mlp_hidden"], macro_embed_dim=mccfg.get("embed_dim",16),
                n_lstm_layers       = mcfg["lstm_layers"], dropout=mcfg["dropout"],
                n_dynamic_features  = n_dynamic,
            ).to(device)
        else:
            _model = QuantLSTM(
                n_features         = n_feat,
                lstm_hidden        = mcfg["lstm_hidden"],
                gru_hidden         = mcfg["gru_hidden"],
                mlp_hidden         = mcfg["mlp_hidden"],
                n_lstm_layers      = mcfg["lstm_layers"],
                dropout            = mcfg["dropout"],
                n_dynamic_features = n_dynamic,
                # Nuovi parametri
                use_multitask      = mcfg.get("use_multitask", False),
                loss_type          = mcfg.get("loss_type", "t_student"),
                n_output_experts   = mcfg.get("n_output_experts", 1),
            ).to(device)

        # IT: imposta i clip bounds nel modello — salvati nel ckpt, riusati al load
        # EN: set clip bounds on the model — saved in the ckpt, reused at load time
        if hasattr(_model, "clip_lo"):
            set_clip_bounds(_model, _clip_lo, _clip_hi)

        # ── Knowledge Distillation: trasferisci pesi output heads ────────
        # IT: copia i pesi delle output heads dal teacher allo student (solo membro 0)
        # EN: copy output-head weights from teacher to student (member 0 only)
        _heads_transferred = False
        if use_distillation and ensemble_idx == 0:
            n_xfer = transfer_output_heads(_teacher_for_transfer, _model)
            log.info(f"  Transfer output heads: {n_xfer} parametri copiati dal teacher")
            _heads_transferred = (n_xfer > 0)
            del _teacher_for_transfer
            torch.cuda.empty_cache() if device.type == "cuda" else None

        # ── Optimizer ───────────────────────────────────────────────────────
        # IT: TFT usa macro_proj invece di macro_encoder — controlla entrambi
        # EN: TFT uses macro_proj instead of macro_encoder — check both
        _macro_mod = None
        if has_macro:
            if hasattr(_model, "macro_encoder"):
                _macro_mod = _model.macro_encoder
            elif hasattr(_model, "macro_proj"):
                _macro_mod = _model.macro_proj

        # ── Output heads (per LR discriminato post-transfer) ──────────────
        # IT: heads già calibrate dal teacher → lr ridotto in warmup mentre il body
        #     si adatta, evitando di "rovinarle". Solo se il transfer è avvenuto.
        # EN: teacher-calibrated heads → reduced lr during warmup while the body
        #     adapts, to avoid wrecking them. Only when transfer actually happened.
        _heads_warmup_epochs = int(tcfg.get("heads_warmup_epochs", 10))
        _heads_lr_factor     = float(tcfg.get("heads_lr_factor", 0.1))
        _heads_params = []
        if _heads_transferred:
            for hn in ("out_mu", "out_logsig2", "out_lognu",
                       "mu_head", "ls2_head", "lnu_head"):
                m = getattr(_model, hn, None)
                if m is not None and isinstance(m, torch.nn.Module):
                    _heads_params.extend(list(m.parameters()))
        _heads_ids = {id(p) for p in _heads_params}

        # IT: optimizer in 1-3 gruppi LR distinti: body / macro / heads
        # EN: optimizer with 1-3 distinct LR groups: body / macro / heads
        _macro_params = list(_macro_mod.parameters()) if _macro_mod is not None else []
        _macro_ids    = {id(p) for p in _macro_params}
        _body_params  = [p for p in _model.parameters()
                          if id(p) not in _macro_ids and id(p) not in _heads_ids]

        _opt_groups = [{"params": _body_params, "lr": tcfg["learning_rate"], "name": "body"}]
        if _macro_params:
            _opt_groups.append({"params": _macro_params,
                                 "lr": tcfg["learning_rate"] / 10, "name": "macro_proj"})
        if _heads_params:
            _opt_groups.append({"params": _heads_params,
                                 "lr": tcfg["learning_rate"] * _heads_lr_factor,
                                 "name": "heads"})
            log.info(f"  LR discriminato heads transferite: lr={tcfg['learning_rate']*_heads_lr_factor:.2e} "
                     f"per i primi {_heads_warmup_epochs} epoche, poi {tcfg['learning_rate']:.2e}")
        _opt = torch.optim.AdamW(_opt_groups, weight_decay=tcfg["weight_decay"])

        grad_accum_steps = tcfg.get("gradient_accumulation_steps", 1)
        _eff_steps = max(1, len(train_dl) // grad_accum_steps)
        _total_s   = tcfg["epochs"] * _eff_steps

        # IT: scheduler mutuamente esclusivi: CosineWarmup (per-batch) O per-epoch;
        #     usarli insieme fa conflitto sul LR.
        # EN: mutually exclusive schedulers: CosineWarmup (per-batch) OR per-epoch;
        #     combining them conflicts on the LR.
        if tcfg.get("lr_scheduler") == "plateau":
            _scheduler   = None
            _epoch_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                _opt, mode="min",
                patience=tcfg.get("lr_scheduler_patience", 8),
                factor=tcfg.get("lr_scheduler_factor", 0.5),
                min_lr=1e-6,
            )
        elif tcfg.get("lr_scheduler") == "cosine":
            _scheduler   = None
            _epoch_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                _opt,
                T_0=tcfg.get("lr_scheduler_T0", 10),
                T_mult=tcfg.get("lr_scheduler_T_mult", 2),
            )
        else:
            _scheduler   = CosineWarmup(_opt, min(500, _total_s//10), _total_s)
            _epoch_sched = None
        _amp_sc = torch.amp.GradScaler(device=device.type, enabled=use_amp)

        # IT: SWA — media i pesi per convergere su minimi piatti (più generalizzanti)
        # EN: SWA — averages weights to converge on flat minima (better generalization)
        _use_swa = tcfg.get("use_swa", False)
        _swa_start_frac = tcfg.get("swa_start_frac", 0.6)
        _swa_start_epoch = max(1, int(tcfg["epochs"] * _swa_start_frac))
        _swa_model = None
        if _use_swa:
            from torch.optim.swa_utils import AveragedModel, SWALR
            _swa_model = AveragedModel(_model)
            _swa_sched = SWALR(_opt, swa_lr=tcfg.get("swa_lr", 1e-5))
            log.info(f"SWA attivo: inizio a epoca {_swa_start_epoch}, swa_lr={tcfg.get('swa_lr', 1e-5)}")

        save_name = f"best_model_{ensemble_idx}.pt" if n_ensemble > 1 else "best_model.pt"
        _ckpt     = str(models_dir / save_name)
        _es       = EarlyStopping(patience=tcfg["patience"], path=_ckpt)
        _history  = {"train_nll": [], "val_nll": [], "val_dir_acc": [], "lr": [], "gap": []}

        # IT: detector overfit basato sul gap train-val NLL; soglia negativa (NLL),
        #     streak di N epoche, disabilitato nei primi warmup epoch (shift normale).
        # EN: overfit detector on the train-val NLL gap; negative threshold (NLL),
        #     N-epoch streak, disabled during warmup epochs (early shift is normal).
        _gap_threshold       = float(tcfg.get("gap_overfit_threshold", -0.5))
        _gap_overfit_streak  = int(tcfg.get("gap_overfit_streak",   3))
        _gap_overfit_warmup  = int(tcfg.get("gap_overfit_warmup_epochs", 10))
        _gap_consec          = 0

        log.info(f"Training  device={device}  AMP={use_amp}  macro={has_macro}  batch={tcfg['batch_size']}×{grad_accum_steps}={tcfg['batch_size']*grad_accum_steps}")
        log.info(f"Overfit guard: gap_threshold={_gap_threshold}, streak={_gap_overfit_streak} epoche, warmup={_gap_overfit_warmup} epoche")
        _t0 = time.time()
        vl_nll, da, sp = 0.0, 0.0, 0.0   # IT: init pre-loop per epoch senza val | EN: pre-loop init for epochs without val

        for epoch in range(1, tcfg["epochs"] + 1):
            # IT: ripristina lr pieno sulle heads transferite a fine warmup
            # EN: restore full lr on transferred heads once warmup ends
            if _heads_params and epoch == _heads_warmup_epochs + 1:
                for grp in _opt.param_groups:
                    if grp.get("name") == "heads":
                        grp["lr"] = tcfg["learning_rate"]
                log.info(f"  Heads warmup terminato a epoch {epoch}: lr restored a {tcfg['learning_rate']:.2e}")
            # IT: schedule lineare di distill_alpha (initial→final); ignorato se non-distill
            # EN: linear distill_alpha schedule (initial→final); ignored when non-distill
            if use_distillation and distill_alpha_decay_epochs > 0:
                _t = min(1.0, (epoch - 1) / float(distill_alpha_decay_epochs))
                _alpha_now = distill_alpha + (distill_alpha_final - distill_alpha) * _t
            else:
                _alpha_now = distill_alpha
            tr_nll = run_train(
                _model, train_dl, _opt, _amp_sc, _scheduler, device, use_amp, has_macro,
                asym_alpha          = tcfg.get("asymmetry_alpha",       2.0),
                asym_threshold      = tcfg.get("asymmetry_threshold",   0.002),
                crps_weight         = tcfg.get("crps_weight",           0.1),
                grad_accum_steps    = grad_accum_steps,
                multitask_alpha     = mcfg.get("multitask_alpha",       0.7),
                multitask_threshold = mcfg.get("multitask_threshold",   0.0001),
                use_distillation    = use_distillation,
                distill_alpha       = _alpha_now,
                batch_size          = _bs_train,
                dv_lambda           = tcfg.get("dv_lambda",             0.0),
                input_noise_std     = tcfg.get("input_noise_std",       0.01),
                use_sample_weights  = _use_sw,
                mixup_alpha         = tcfg.get("mixup_alpha",            0.0),
                crps_distill_weight = tcfg.get("crps_distill_weight",    0.0),
            )
            vl_nll, _all_mu, _all_y, _val_metrics, _val_sig, _val_nu = run_eval(_model, val_dl, device, has_macro)
            da = _val_metrics["directional_acc"]
            sp = _val_metrics["spearman"]
            do_val = True
            # IT: log NLL per-regime ogni 5 epoche (riduce lo spam di log)
            # EN: log per-regime NLL every 5 epochs (cuts log spam)
            if val_regimes is not None and (epoch == 1 or epoch % 5 == 0):
                _per_reg = _per_regime_nll(_all_mu, _val_sig, _val_nu, _all_y, val_regimes)
                if _per_reg:
                    _spread = max(_per_reg.values()) - min(_per_reg.values())
                    _history.setdefault("val_nll_per_regime", []).append(_per_reg)
                    _history.setdefault("val_nll_regime_spread", []).append(_spread)
                    log.info("  ↳ val_nll per regime: " +
                             ", ".join(f"r{r}={v:+.3f}" for r, v in sorted(_per_reg.items())) +
                             f"  spread={_spread:.3f}")
            lr_now = _scheduler.get_last_lr()[0] if _scheduler else _opt.param_groups[0]["lr"]
            if _epoch_sched is not None:
                if isinstance(_epoch_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    _epoch_sched.step(vl_nll)
                else:
                    _epoch_sched.step(epoch)
            current_lr = _opt.param_groups[0]["lr"]
            _gap = tr_nll - vl_nll   # IT: più negativo = train ≪ val = overfit | EN: more negative = train ≪ val = overfit
            _history["train_nll"].append(tr_nll); _history["val_nll"].append(vl_nll)
            _history["val_dir_acc"].append(da);   _history["lr"].append(lr_now)
            _history["gap"].append(_gap)
            _history.setdefault("val_spearman", []).append(sp)
            _alpha_log = f"  α={_alpha_now:.3f}" if use_distillation else ""
            _gn_stats = getattr(_opt, "_qs_gradnorm_stats", None)
            _gn_log = f"  ‖∇‖μ={_gn_stats['mean']:.2f}/p95={_gn_stats['p95']:.2f}" if _gn_stats else ""
            _history.setdefault("grad_norm_mean", []).append(_gn_stats["mean"] if _gn_stats else None)
            _history.setdefault("grad_norm_p95",  []).append(_gn_stats["p95"]  if _gn_stats else None)
            log.info(
                f"[ens {ensemble_idx+1}/{n_ensemble}] "
                f"Ep {epoch:3d}  train={tr_nll:+.5f}  val={vl_nll:+.5f}{'*' if do_val else ' '}  "
                f"gap={_gap:+.4f}  DA={da:.3f}  ρ={sp:+.4f}  lr={lr_now:.2e}  clr={current_lr:.2e}"
                f"{_gn_log}{_alpha_log}"
            )
            if _swa_model is not None and epoch >= _swa_start_epoch:
                _swa_model.update_parameters(_model)
                _swa_sched.step()

            # IT: early-stop su gap negativo per N epoche consecutive (train memorizza);
            #     skip nei primi warmup epoch per evitare falsi positivi.
            # EN: early-stop on negative gap for N consecutive epochs (train memorizing);
            #     skipped during warmup epochs to avoid false positives.
            if epoch > _gap_overfit_warmup:
                if _gap < _gap_threshold:
                    _gap_consec += 1
                    if _gap_consec >= _gap_overfit_streak:
                        log.warning(
                            f"Overfit detector triggered a epoch {epoch}: "
                            f"gap={_gap:+.3f} < {_gap_threshold} per {_gap_consec} epoche consecutive — stop"
                        )
                        break
                else:
                    _gap_consec = 0

            if do_val and _es(vl_nll, _model):
                log.info(f"Early stopping a epoch {epoch}"); break

        elapsed += time.time() - _t0
        _es.restore(_model)

        if _swa_model is not None:
            # IT: update_bn ricalibra le stat BatchNorm con un full pass; modelli
            #     solo-LayerNorm non hanno BN → skip (loop costoso e inutile).
            # EN: update_bn recalibrates BatchNorm stats via a full pass; LayerNorm-only
            #     models have no BN → skip (a costly, useless loop).
            _has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                          nn.SyncBatchNorm))
                          for m in _model.modules())
            if _has_bn:
                torch.optim.swa_utils.update_bn(train_dl, _swa_model, device=device)
            else:
                log.info("  SWA: nessun BatchNorm rilevato → skip update_bn (LayerNorm only)")
            swa_vl, _, _, _, _, _ = run_eval(_swa_model, val_dl, device, has_macro)
            if swa_vl < _es.best:
                log.info(f"SWA migliore: val={swa_vl:+.5f} vs best={_es.best:+.5f} — uso SWA")
                torch.save(_swa_model.module.state_dict(), _ckpt)
                _model.load_state_dict(_swa_model.module.state_dict())
            else:
                log.info(f"SWA peggiore: val={swa_vl:+.5f} vs best={_es.best:+.5f} — ignoro SWA")

        # IT: copia il ckpt come best_model.pt per backward-compat (solo membro 0)
        # EN: copy the ckpt as best_model.pt for backward-compat (member 0 only)
        if n_ensemble > 1 and ensemble_idx == 0:
            _sh.copy(_ckpt, str(models_dir / "best_model.pt"))

        # IT: valutazione sul test set per il report finale (membro 0 o singolo)
        # EN: test-set evaluation for the final report (member 0 or the single one)
        if ensemble_idx == 0:
            n_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
            log.info(f"Modello: {model_type}  |  Parametri: {n_params:,}")
            best_val_nll = _es.best
            _, test_mu, test_y, test_metrics, sig_a, nu_a = run_eval(_model, test_dl, device, has_macro, full_metrics=True)
            history = _history
            model   = _model

        log.info(f"Ensemble {ensemble_idx+1}/{n_ensemble} completato → {_ckpt}")

    from scipy.stats import t as t_dist
    # IT: z90 per-sample col proprio nu (non il medio): t(3)→2.35 vs t(10)→1.81
    # EN: per-sample z90 with its own nu (not the mean): t(3)→2.35 vs t(10)→1.81
    z90_per_sample = t_dist.ppf(0.95, df=np.clip(nu_a, 2.01, None))
    cov90 = np.mean(
        (test_y >= test_mu - z90_per_sample * sig_a) &
        (test_y <= test_mu + z90_per_sample * sig_a)
    )

    da_t  = test_metrics["directional_acc"]
    sp_t  = test_metrics["spearman"]
    whr_t = test_metrics["weighted_hit_rate"]
    icir_t= test_metrics["icir"]

    cfg_out = {
        **mcfg,
        "n_features":          int(n_feat),
        "n_dynamic_features":  n_dynamic,
        "n_macro":             int(n_macro),
        "has_macro":           has_macro,
        "macro_embed_dim":     mccfg.get("embed_dim", 16),
        "model_type":          model_type,
        "device":              str(device),
        "n_params":            n_params,
        "n_attention_heads":   mcfg.get("n_attention_heads", 4),
        "use_attention":       mcfg.get("use_attention", True),
        "tft_d_model":         mcfg.get("tft_d_model", 64),
        "tft_n_heads":         mcfg.get("tft_n_heads", 4),
        "tft_n_layers":        mcfg.get("tft_n_layers", 3),
        "tft_dropout":         mcfg.get("tft_dropout", 0.1),
        "window_size":         mcfg.get("window_size", 120),
        "loss_type":           mcfg.get("loss_type", "t_student"),
        "use_multitask":       mcfg.get("use_multitask", False),
        "n_output_experts":    mcfg.get("n_output_experts", 1),
        "patch_size":          mcfg.get("patch_size", 1),
        "drop_path_rate":      mcfg.get("drop_path_rate", 0.0),
        # TCNMamba parameters
        "d_model":             mcfg.get("d_model", 128),
        "tcn_layers":          mcfg.get("tcn_layers", 4),
        "tcn_kernel":          mcfg.get("tcn_kernel", 3),
        "mamba_layers":        mcfg.get("mamba_layers", 3),
        "mamba_d_state":       mcfg.get("mamba_d_state", 16),
        "mamba_expand":        mcfg.get("mamba_expand", 2),
        # Distillation info
        "distilled":           use_distillation,
        "teacher_arch":        teacher_arch if use_distillation else None,
        "distill_alpha":             distill_alpha if use_distillation else None,
        "distill_alpha_final":       distill_alpha_final if use_distillation else None,
        "distill_alpha_decay_epochs": distill_alpha_decay_epochs if use_distillation else None,
    }
    with open(out_dir/"config.json","w",  encoding="utf-8") as f: json.dump(cfg_out, f, indent=2)
    with open(out_dir/"history.json","w", encoding="utf-8") as f: json.dump(history, f, indent=2)
    np.savez_compressed(out_dir/"test_predictions.npz", mu=test_mu, sigma=sig_a, nu=nu_a, y_true=test_y)

    # ── Reliability diagram (calibration plot) ─────────────────────────────
    # IT: misura la calibrazione di sigma: per ogni livello CI conta gli y entro;
    #     diagonale=perfetto, sopra=under-confident, sotto=over-confident.
    # EN: measures sigma calibration: per CI level counts the y inside; diagonal=
    #     perfect, above=under-confident, below=over-confident.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import t as _t_dist
        _nu_clip = np.clip(nu_a, 2.01, None)
        _levels = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
        _empirical = []
        for lev in _levels:
            _z = _t_dist.ppf(0.5 + lev / 2, df=_nu_clip)
            _within = (test_y >= test_mu - _z * sig_a) & (test_y <= test_mu + _z * sig_a)
            _empirical.append(float(_within.mean()))
        _empirical = np.array(_empirical)
        _ece = float(np.mean(np.abs(_empirical - _levels)))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfetta calibrazione")
        ax.plot(_levels, _empirical, "o-", linewidth=2, markersize=8, label=f"{model_type}")
        ax.set_xlabel("Confidenza nominale (livello CI)")
        ax.set_ylabel("Copertura empirica (frazione y entro CI)")
        ax.set_title(f"Reliability diagram — {model_type}  (ECE={_ece:.3f})")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out_dir / "reliability_diagram.png", dpi=100)
        plt.close(fig)
        with open(out_dir / "calibration.json", "w", encoding="utf-8") as _cf:
            json.dump({
                "levels": _levels.tolist(),
                "empirical_coverage": _empirical.tolist(),
                "ece": _ece,
            }, _cf, indent=2)
        log.info(f"Reliability diagram salvato: ECE={_ece:.3f} → {out_dir/'reliability_diagram.png'}")
    except Exception as _e_cal:
        log.warning(f"Reliability diagram fallito: {_e_cal}")

    # IT: aggiorna PipelineState con la config del modello addestrato (Fix 6)
    # EN: update PipelineState with the trained model's config (Fix 6)
    _ps_path = out_dir / "pipeline_state.pkl"
    # IT: fallback — cerca pipeline_state.pkl in altre arch (scaler condivisi)
    # EN: fallback — look for pipeline_state.pkl in other archs (scalers are shared)
    if not _ps_path.exists():
        _legacy_ps = Path("models/pipeline_state.pkl")
        _root_ps = _legacy_ps if _legacy_ps.exists() else None
        if _root_ps is None:
            _cur_arch = os.environ.get("QUANTSYS_ARCH", "lstm")
            for _alt in ("lstm", "itransformer", "tcnmamba", "tft", "nhits"):
                if _alt == _cur_arch:
                    continue
                _alt_ps = Path("models") / _alt / "pipeline_state.pkl"
                if _alt_ps.exists():
                    _root_ps = _alt_ps
                    break
        if _root_ps:
            shutil.copy(_root_ps, _ps_path)
            log.info(f"PipelineState copiato da {_root_ps} → {_ps_path}")
        else:
            log.warning(
                "pipeline_state.pkl mancante in tutte le arch. "
                "Esegui 01_download_data.py per generarlo."
            )
    try:
        from quantsys.utils import PipelineState
        state = PipelineState.load(str(_ps_path))
    except Exception as e:
        log.warning(f"PipelineState load fallito (non critico): {e}")
        state = None
    if state is not None:
        # IT: guard anti-stale — se l'interval del pkl arch-locale non coincide con la
        #     config corrente, il pkl è di un dataset precedente (es. 1m sotto pivot 1h):
        #     prova la copia canonica di 01_download_data, altrimenti fail-fast. Salvarlo
        #     stale propagherebbe target_scale/scaler errati alla denormalizzazione.
        # EN: anti-stale guard — if the arch-local pkl interval mismatches the current
        #     config, the pkl belongs to a previous dataset (e.g. 1m under the 1h pivot):
        #     try 01_download_data's canonical copy, else fail fast. Saving it stale would
        #     propagate wrong target_scale/scalers into denormalization.
        from quantsys.utils import interval_minutes_from_cfg
        _cfg_im = interval_minutes_from_cfg(cfg)
        if state.interval_minutes != _cfg_im:
            _stale_im = state.interval_minutes
            _canon = Path("models/pipeline_state.pkl")
            _fixed = False
            if _canon.exists():
                _cs = PipelineState.load(str(_canon))
                if _cs.interval_minutes == _cfg_im:
                    state = _cs
                    _fixed = True
                    log.warning(
                        f"PipelineState arch-locale stale ({_stale_im}min): "
                        f"sostituito con la copia canonica {_canon} ({_cfg_im}min)."
                    )
            if not _fixed:
                raise RuntimeError(
                    f"PipelineState stale: {_ps_path} è a {_stale_im}min ma la config "
                    f"è a {_cfg_im}min, e nessuna copia canonica valida in models/pipeline_state.pkl. "
                    f"Rilancia scripts/01_download_data.py per rigenerarlo."
                )
        state.set_model_config(cfg_out)
        state.save(str(_ps_path))

    # IT: copia best_model + PipelineState nella exp dir (senza scaler il ckpt è inutile)
    # EN: copy best_model + PipelineState into the exp dir (no scalers = unusable ckpt)
    best_model_src = out_dir / "best_model.pt"
    if best_model_src.exists():
        try:
            _sh.copy(best_model_src, exp_dir / "best_model.pt")
        except Exception as _e:
            log.warning(f"Impossibile copiare best_model.pt: {_e}")
    pipeline_state_src = out_dir / "pipeline_state.pkl"
    if pipeline_state_src.exists():
        try:
            _sh.copy(pipeline_state_src, exp_dir / "pipeline_state.pkl")
        except Exception as _e:
            log.warning(f"Impossibile copiare pipeline_state.pkl: {_e}")

    # IT: salva le metriche finali nella experiment dir
    # EN: persist the final metrics into the experiment dir
    _train_losses = history.get("train_nll", [])
    _val_losses   = history.get("val_nll", [])
    _val_da       = history.get("val_dir_acc", [])
    exp_results = {
        "train_loss_final":          float(_train_losses[-1]) if _train_losses else None,
        "val_loss_final":            float(_val_losses[-1])   if _val_losses   else None,
        "best_val_loss":             float(min(_val_losses))  if _val_losses   else None,
        "n_epochs_trained":          len(_train_losses),
        "directional_accuracy_val":  float(_val_da[-1])       if _val_da       else None,
    }
    with open(exp_dir / "results.json", "w", encoding="utf-8") as _f:
        _json.dump(exp_results, _f, indent=2, default=str)

    log.info(f"Experiment salvato → {exp_dir}/")
    log.info("  Per confrontare esperimenti: ls models/experiments/")

    print(f"""
{'═'*58}
  02 · TRAINING · COMPLETATO  ({elapsed/60:.1f} min)
{'═'*58}
  Modello           : {model_type}
  Macro embedding   : {'✓ ATTIVO' if has_macro else '✗ assente'}
  Distillation      : {'✓ multi-teacher alpha=' + str(distill_alpha) if use_distillation and use_multi_teacher else '✓ teacher=' + teacher_arch + ' alpha=' + str(distill_alpha) if use_distillation else '✗ standard'}
  DV Joint Loss     : {'✓ lambda=' + str(tcfg.get('dv_lambda', 0.0)) if tcfg.get('dv_lambda', 0.0) > 0 else '✗ disabilitata'}

  ── Metriche predittive (test set) ──────────────────
  Directional Acc   : {da_t:.3f}  ({da_t*100:.1f}%)
  Spearman ρ        : {sp_t:+.4f}  (p={test_metrics['spearman_pvalue']:.3f})
  Weighted Hit Rate : {whr_t:.3f}  (pesato per |Δ prezzo|)
  IC medio          : {test_metrics['ic_mean']:+.4f}  (Spearman su 5 sub-periodi)
  ICIR              : {icir_t:+.4f}  (mean/std sub-periodi; >1.0 = stabile)

  ── Calibrazione distribuzione ──────────────────────
  90%% CI coverage   : {cov90:.3f}  (target 0.90)
  σ medio           : {sig_a.mean():.5f}
  ν medio           : {nu_a.mean():.2f}
  Best val NLL      : {best_val_nll:+.5f}
  Parametri         : {n_params:,}

  {'✓ Calibrazione OK'   if abs(cov90-0.90)<0.05 else '⚠ Controlla calibrazione'}
  {'✓ Segnale predittivo' if sp_t > 0.03         else '△ Spearman basso — valuta più dati'}
  {'✓ ICIR consistente'   if icir_t > 1.0        else '△ Segnale inconsistente tra sub-periodi'}

  Ensemble          : {n_ensemble} modelli
  Checkpoint → {str(models_dir / 'best_model.pt')}
  → Prossimo: python scripts/03_backtest.py
""")


if __name__ == "__main__":
    main()

"""
Probe temporanea (PERF AUDIT) — profilo di ~20 step di training iTransformer.
Temporary probe (PERF AUDIT) — profile of ~20 iTransformer training steps.

Replica il path production di scripts/02_train.py (loss_type=quantile +
multitask, AMP on, grad_accum=2, batch 64, DataLoader num_workers=0 su Windows)
riusando `run_train` importato dallo script reale — nessuna copia della loss.
Replicates the production path of scripts/02_train.py (quantile loss +
multitask, AMP on, grad_accum=2, batch 64, DataLoader num_workers=0 on Windows)
by importing the real `run_train` — no copy of the loss.

Output: tempo/step, breakdown CPU vs CUDA da torch.profiler, occupazione GPU.
NON scrive in models/ (usa QUANTSYS_MODELS_ROOT su scratch se serve).

Uso / Usage:
  python scripts/archive/perf_probe/bench_train_profile.py [--steps 20] [--compile]
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# IT: dir temporanea di sistema — la probe non scrive mai in models/ o data/.
# EN: system temp dir — this probe never writes into models/ or data/.
import tempfile  # noqa: E402
SCRATCH = Path(tempfile.gettempdir())


# IT: importa 02_train.py per nome-file (il modulo inizia con una cifra).
# EN: import 02_train.py by file path (module name starts with a digit).
def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "qs_train02", str(ROOT / "scripts" / "02_train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--compile", action="store_true",
                    help="wrap del modello in torch.compile (lever a)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    from quantsys.utils import load_config, setup_device
    from quantsys.model import QuantiTransformer, set_clip_bounds  # noqa: F401
    t02 = _load_train_module()

    cfg = load_config(str(ROOT / "config/default.yaml"))
    # IT: overlay arch itransformer (come fa run_all/02 con --arch).
    # EN: itransformer arch overlay (as run_all/02 do with --arch).
    import yaml
    with open(ROOT / "config/arch/itransformer.yaml", encoding="utf-8") as f:
        arch = yaml.safe_load(f)
    for sec, vals in arch.items():
        cfg.setdefault(sec, {}).update(vals)

    mcfg, tcfg = cfg["model"], cfg["training"]
    device = setup_device(cfg)
    bs = args.batch or tcfg["batch_size"]
    use_amp = (not args.no_amp) and tcfg.get("use_amp", True)

    # ── dati reali dal npz di produzione ────────────────────────────────────
    t0 = time.perf_counter()
    d = np.load(str(ROOT / "data/lstm_dataset.npz"), allow_pickle=True)
    n_need = (args.warmup + args.steps + 5) * bs * tcfg.get("gradient_accumulation_steps", 1)
    X = torch.from_numpy(d["X_train"][:n_need].astype(np.float32))
    y = torch.from_numpy(d["y_train"][:n_need].astype(np.float32))
    has_macro = "X_macro_train" in d.files and mcfg.get("use_macro", True)
    Xm = torch.from_numpy(d["X_macro_train"][:n_need].astype(np.float32)) if has_macro else None
    n_feat = X.shape[2]
    n_dyn = int(d["n_dynamic_features"][0]) if "n_dynamic_features" in d.files else None
    n_macro = Xm.shape[1] if has_macro else 0
    print(f"npz slice load: {time.perf_counter()-t0:.2f}s  X={tuple(X.shape)} "
          f"macro={n_macro} n_dyn={n_dyn}")

    tensors = [X, Xm, y] if has_macro else [X, y]
    dl = DataLoader(TensorDataset(*tensors), bs, shuffle=True,
                    pin_memory=cfg["hardware"]["pin_memory"], num_workers=0)

    model = QuantiTransformer(
        n_features=n_feat, T=mcfg.get("window_size", 120), n_dynamic=n_dyn,
        n_macro=n_macro, d_model=mcfg.get("tft_d_model", 128),
        n_heads=mcfg.get("tft_n_heads", 4), n_layers=mcfg.get("tft_n_layers", 3),
        dropout=mcfg.get("tft_dropout", 0.1), patch_size=mcfg.get("patch_size", 1),
        drop_path_rate=mcfg.get("drop_path_rate", 0.0),
        loss_type=mcfg.get("loss_type", "quantile"),
        use_multitask=mcfg.get("use_multitask", False),
        n_output_experts=mcfg.get("n_output_experts", 1),
        use_revin=mcfg.get("use_revin", False),
        revin_target_idx=mcfg.get("revin_target_idx", 0),
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: QuantiTransformer d_model={mcfg.get('tft_d_model')} "
          f"layers={mcfg.get('tft_n_layers')} patch={mcfg.get('patch_size')} "
          f"params={n_par:,}  AMP={use_amp}  bs={bs}")

    if args.compile:
        t0 = time.perf_counter()
        model = torch.compile(model)
        print(f"torch.compile() wrap: {time.perf_counter()-t0:.2f}s (lazy)")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                            weight_decay=tcfg["weight_decay"])
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    ga = tcfg.get("gradient_accumulation_steps", 1)

    kw = dict(multitask_alpha=mcfg.get("multitask_alpha", 0.7),
              multitask_threshold=mcfg.get("multitask_threshold", 1e-4),
              grad_accum_steps=ga, batch_size=bs,
              input_noise_std=tcfg.get("input_noise_std", 0.01),
              mixup_alpha=tcfg.get("mixup_alpha", 0.0))

    # ── warmup (compile/cudnn autotune/allocator) ───────────────────────────
    warm_ds = TensorDataset(*[t[:args.warmup * bs] for t in tensors])
    warm_dl = DataLoader(warm_ds, bs, shuffle=False, num_workers=0)
    t0 = time.perf_counter()
    t02.run_train(model, warm_dl, opt, scaler, None, device, use_amp, has_macro, **kw)
    torch.cuda.synchronize()
    t_warm = time.perf_counter() - t0
    print(f"warmup {args.warmup} step: {t_warm:.2f}s "
          f"({t_warm/args.warmup*1000:.1f} ms/step incl. compile)")

    # ── steady-state timing ─────────────────────────────────────────────────
    meas_ds = TensorDataset(*[t[args.warmup * bs:(args.warmup + args.steps) * bs]
                              for t in tensors])
    meas_dl = DataLoader(meas_ds, bs, shuffle=False, num_workers=0,
                         pin_memory=cfg["hardware"]["pin_memory"])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t02.run_train(model, meas_dl, opt, scaler, None, device, use_amp, has_macro, **kw)
    torch.cuda.synchronize()
    t_meas = time.perf_counter() - t0
    ms = t_meas / args.steps * 1000
    print(f"\nSTEADY: {args.steps} step in {t_meas:.3f}s → {ms:.2f} ms/step "
          f"({bs/ (t_meas/args.steps):,.0f} sample/s)")

    # IT: proiezione a epoca intera (n_train reale dal npz).
    # EN: full-epoch projection (real n_train from the npz).
    n_train = int(d["X_train"].shape[0])
    print(f"proiezione epoca: {n_train:,} sample / bs {bs} = {n_train//bs:,} step "
          f"→ {n_train//bs * ms/1000/60:.2f} min/epoca (solo train, no eval)")

    # ── torch.profiler ──────────────────────────────────────────────────────
    prof_dl = DataLoader(meas_ds, bs, shuffle=False, num_workers=0,
                         pin_memory=cfg["hardware"]["pin_memory"])
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False,
    ) as prof:
        t02.run_train(model, prof_dl, opt, scaler, None, device, use_amp, has_macro, **kw)
        torch.cuda.synchronize()

    ka = prof.key_averages()
    print("\n=== top 18 per CUDA time ===")
    print(ka.table(sort_by="self_cuda_time_total", row_limit=18))
    print("\n=== top 12 per CPU time ===")
    print(ka.table(sort_by="self_cpu_time_total", row_limit=12))

    tot_cuda = sum(e.self_device_time_total for e in ka)
    tot_cpu = sum(e.self_cpu_time_total for e in ka)
    n_kernels = sum(e.count for e in ka if e.self_device_time_total > 0)
    print(f"\nself CUDA totale : {tot_cuda/1e3:9.1f} ms su {args.steps} step "
          f"({tot_cuda/1e3/args.steps:.2f} ms/step)")
    print(f"self CPU totale  : {tot_cpu/1e3:9.1f} ms "
          f"({tot_cpu/1e3/args.steps:.2f} ms/step)")
    print(f"wall             : {ms:.2f} ms/step")
    print(f"→ occupazione GPU stimata = cuda/wall = "
          f"{(tot_cuda/1e3/args.steps)/ms*100:5.1f}%")
    print(f"→ lanci kernel CUDA (eventi con device time) = {n_kernels} "
          f"su {args.steps} step = {n_kernels/args.steps:.0f}/step")


if __name__ == "__main__":
    main()

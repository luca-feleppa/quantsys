"""
Temporary probe (LEVER B) — is torch.compile(cudagraphs) USABLE for training?

The speed-up (1.56x) is already measured. This answers the questions that
decide whether the lever can be adopted, which the audit marked "to be verified":

  ① REPRODUCIBILITY. Dropout 0.3 and drop_path 0.2 are active in training. Under
     CUDA Graph capture the RNG is handled with philox offsets: do two runs with the
     same seed give the same weights? If NOT, the lever costs training
     reproducibility — on this project, probably disqualifying.
     Required baseline: is eager reproducible with itself? (cudnn_benchmark
     is true in config, so it is not a given.)
  ② EQUIVALENCE. Do the weights after N compiled steps match the eager ones? Not
     expected (different kernels and reduction order); the MAGNITUDE is what matters.
  ③ VARIABLE SHAPE. The last batch of the epoch is partial (51882 % 64 = 42):
     does graph capture handle it or blow up/recompile?

Usage: python scripts/archive/perf_probe/bench_compile_reproducibility.py
"""
import sys
import time
from pathlib import Path

import numpy as np
# pandas before torch/sklearn — package DLL-init invariant.
import pandas as pd  # noqa: F401,E402

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

N_STEPS = 25
BS = 64


def _load(cfg):
    d = np.load(str(ROOT / "data/lstm_dataset.npz"), allow_pickle=True)
    n = (N_STEPS + 6) * BS
    X = torch.from_numpy(d["X_train"][:n].astype(np.float32))
    y = torch.from_numpy(d["y_train"][:n].astype(np.float32))
    Xm = torch.from_numpy(d["X_macro_train"][:n].astype(np.float32))
    meta = (X.shape[2], int(d["n_dynamic_features"][0]), Xm.shape[1])
    return X, Xm, y, meta


def _mk(cfg, meta, device, seed=0):
    from quantsys.model import QuantiTransformer
    m = cfg["model"]
    # seed BEFORE construction: deterministic weight init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return QuantiTransformer(
        n_features=meta[0], T=m.get("window_size", 120), n_dynamic=meta[1],
        n_macro=meta[2], d_model=m.get("tft_d_model", 128),
        n_heads=m.get("tft_n_heads", 4), n_layers=m.get("tft_n_layers", 3),
        dropout=m.get("tft_dropout", 0.1), patch_size=m.get("patch_size", 1),
        drop_path_rate=m.get("drop_path_rate", 0.0),
        loss_type=m.get("loss_type", "quantile"),
        use_multitask=m.get("use_multitask", False),
        n_output_experts=m.get("n_output_experts", 1),
    ).to(device)


def _train(model, X, Xm, y, cfg, device, seed, n_steps=N_STEPS, bs=BS):
    """N deterministic steps; returns the final weights on CPU."""
    from quantsys.model import quantile_loss
    mcfg, tcfg = cfg["model"], cfg["training"]
    use_amp = tcfg.get("use_amp", True)
    ma = mcfg.get("multitask_alpha", 0.7)
    thr = mcfg.get("multitask_threshold", 1e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                            weight_decay=tcfg["weight_decay"])
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    # seed BEFORE the loop: pins dropout/drop_path/input noise.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.train()
    noise = tcfg.get("input_noise_std", 0.01)
    for s in range(n_steps):
        a, b = s * bs, (s + 1) * bs
        Xb = X[a:b].to(device); Xmb = Xm[a:b].to(device); yb = y[a:b].to(device)
        if noise > 0:
            Xb = Xb + torch.randn_like(Xb) * noise
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(Xb, Xmb)
            main = quantile_loss(yb, out[0])
            dl = out[-1]
            conf = torch.tanh(yb.abs() / max(thr, 1e-8))
            sl = torch.zeros(yb.shape[0], 3, device=yb.device)
            sl[:, 1] = 1.0 - conf
            sl[:, 2] = torch.where(yb > 0, conf, torch.zeros_like(conf))
            sl[:, 0] = torch.where(yb < 0, conf, torch.zeros_like(conf))
            loss = ma * main + (1 - ma) * (
                -(sl * F.log_softmax(dl, -1)).sum(-1).mean())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       tcfg.get("grad_clip_norm", 0.5))
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    # `_orig_mod` = module under the torch.compile wrapper.
    base = getattr(model, "_orig_mod", model)
    return {k: v.detach().float().cpu().clone()
            for k, v in base.state_dict().items()}


def _cmp(a: dict, b: dict, label: str) -> bool:
    keys = sorted(set(a) | set(b))
    n_id, worst, worst_k = 0, 0.0, ""
    for k in keys:
        if k not in a or k not in b:
            print(f"  chiave asimmetrica: {k}")
            return False
        if torch.equal(a[k], b[k]):
            n_id += 1
            continue
        d = (a[k] - b[k]).abs().max().item()
        scale = max(a[k].abs().max().item(), 1e-12)
        if d / scale > worst:
            worst, worst_k = d / scale, k
    ident = n_id == len(keys)
    print(f"{label}: tensori bit-identici {n_id}/{len(keys)}"
          + ("  → IDENTICI" if ident
             else f"  worst rel {worst:.3e} su '{worst_k}'"))
    return ident


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    import yaml
    import torch._dynamo as dynamo
    from quantsys.utils import load_config, setup_device

    cfg = load_config(str(ROOT / "config/default.yaml"))
    with open(ROOT / "config/arch/itransformer.yaml", encoding="utf-8") as f:
        for sec, vals in yaml.safe_load(f).items():
            cfg.setdefault(sec, {}).update(vals)
    device = setup_device(cfg)
    X, Xm, y, meta = _load(cfg)
    print(f"dati {tuple(X.shape)}  dropout={cfg['model'].get('tft_dropout')} "
          f"drop_path={cfg['model'].get('drop_path_rate')} "
          f"cudnn_benchmark={cfg['hardware'].get('cudnn_benchmark')}\n")

    # ── ① EAGER reproducibility (baseline: is it a given?) ──────────────────
    print("=== ① RIPRODUCIBILITA' ===")
    w_e1 = _train(_mk(cfg, meta, device), X, Xm, y, cfg, device, seed=1234)
    w_e2 = _train(_mk(cfg, meta, device), X, Xm, y, cfg, device, seed=1234)
    eager_repro = _cmp(w_e1, w_e2, "eager vs eager (stesso seed)     ")

    # ── compiled, twice with the same seed ──────────────────────────────────
    dynamo.reset()
    t0 = time.perf_counter()
    mc1 = torch.compile(_mk(cfg, meta, device), backend="cudagraphs")
    w_c1 = _train(mc1, X, Xm, y, cfg, device, seed=1234)
    t_first = time.perf_counter() - t0
    dynamo.reset()
    mc2 = torch.compile(_mk(cfg, meta, device), backend="cudagraphs")
    w_c2 = _train(mc2, X, Xm, y, cfg, device, seed=1234)
    comp_repro = _cmp(w_c1, w_c2, "compiled vs compiled (stesso seed)")
    print(f"  (primo run compiled, incl. compilazione: {t_first:.1f}s)")

    # ── ② eager vs compiled equivalence ─────────────────────────────────────
    print("\n=== ② EQUIVALENZA eager vs compiled ===")
    _cmp(w_e1, w_c1, "eager vs compiled (stesso seed)   ")

    # ── ③ variable shape (last partial batch) ───────────────────────────────
    print("\n=== ③ SHAPE VARIABILE (ultimo batch parziale) ===")
    n_last = 51882 % BS
    print(f"  ultimo batch reale di un'epoca = {n_last} campioni")
    dynamo.reset()
    m3 = torch.compile(_mk(cfg, meta, device), backend="cudagraphs")
    try:
        _train(m3, X, Xm, y, cfg, device, seed=7, n_steps=3, bs=BS)
        t0 = time.perf_counter()
        _train(m3, X[:n_last * 3], Xm[:n_last * 3], y[:n_last * 3],
               cfg, device, seed=7, n_steps=2, bs=n_last)
        print(f"  batch parziale ({n_last}): OK "
              f"(ricompilazione/ricattura {time.perf_counter()-t0:.2f}s)")
    except Exception as e:
        print(f"  batch parziale ({n_last}): FALLITO — "
              f"{type(e).__name__}: {str(e)[:200]}")

    # ── verdict ─────────────────────────────────────────────────────────────
    print("\n=== VERDETTO ===")
    print(f"eager riproducibile    : {eager_repro}")
    print(f"compiled riproducibile : {comp_repro}")
    if comp_repro and eager_repro:
        print("→ la leva NON costa riproducibilita': adottabile con un gate")
    elif eager_repro and not comp_repro:
        print("→ la leva COSTA la riproducibilita' del training: squalificante")
    elif not eager_repro:
        print("→ eager NON e' gia' riproducibile: la domanda cambia — il "
              "training non e' deterministico nemmeno oggi")


if __name__ == "__main__":
    main()

"""
Probe temporanea (PERF AUDIT) — compute-bound o launch-bound?
Temporary probe (PERF AUDIT) — compute-bound or launch-bound?

Tre misure decisive, senza overhead di torch.profiler:
  A) sweep del batch size: se il wall/step e' quasi piatto al crescere di B,
     il tempo e' dominato dal LANCIO dei kernel, non dal loro calcolo.
  B) fwd+bwd puro su un batch GPU-resident (nessun DataLoader, nessun H2D):
     isola il costo del modello da quello del data path.
  C) costo del DataLoader da solo (iterazione senza modello).
Three decisive measurements without torch.profiler overhead:
  A) batch-size sweep: flat wall/step vs B ⇒ kernel LAUNCH bound.
  B) pure fwd+bwd on a GPU-resident batch (no DataLoader, no H2D).
  C) DataLoader-only iteration cost.

Uso / Usage: python scripts/archive/perf_probe/bench_train_scaling.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def build_model(cfg, n_feat, n_dyn, n_macro, device):
    from quantsys.model import QuantiTransformer
    m = cfg["model"]
    return QuantiTransformer(
        n_features=n_feat, T=m.get("window_size", 120), n_dynamic=n_dyn,
        n_macro=n_macro, d_model=m.get("tft_d_model", 128),
        n_heads=m.get("tft_n_heads", 4), n_layers=m.get("tft_n_layers", 3),
        dropout=m.get("tft_dropout", 0.1), patch_size=m.get("patch_size", 1),
        drop_path_rate=m.get("drop_path_rate", 0.0),
        loss_type=m.get("loss_type", "quantile"),
        use_multitask=m.get("use_multitask", False),
        n_output_experts=m.get("n_output_experts", 1),
    ).to(device)


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    import yaml
    from quantsys.utils import load_config, setup_device
    from quantsys.model import quantile_loss
    import torch.nn.functional as F

    cfg = load_config(str(ROOT / "config/default.yaml"))
    with open(ROOT / "config/arch/itransformer.yaml", encoding="utf-8") as f:
        for sec, vals in yaml.safe_load(f).items():
            cfg.setdefault(sec, {}).update(vals)
    mcfg, tcfg = cfg["model"], cfg["training"]
    device = setup_device(cfg)
    use_amp = tcfg.get("use_amp", True)

    d = np.load(str(ROOT / "data/lstm_dataset.npz"), allow_pickle=True)
    N = 40000
    X = torch.from_numpy(d["X_train"][:N].astype(np.float32))
    y = torch.from_numpy(d["y_train"][:N].astype(np.float32))
    Xm = torch.from_numpy(d["X_macro_train"][:N].astype(np.float32))
    n_feat, n_dyn, n_macro = X.shape[2], int(d["n_dynamic_features"][0]), Xm.shape[1]
    del d
    print(f"dati: X={tuple(X.shape)} macro={n_macro}\n")

    model = build_model(cfg, n_feat, n_dyn, n_macro, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    # IT: replica della loss production (quantile 0.7 + CE direzionale 0.3).
    # EN: production loss replica (0.7 quantile + 0.3 directional CE).
    ma, thr = mcfg.get("multitask_alpha", 0.7), mcfg.get("multitask_threshold", 1e-4)

    def loss_fn(out, yb):
        main = quantile_loss(yb, out[0])
        dl = out[-1]
        conf = torch.tanh(yb.abs() / max(thr, 1e-8))
        sl = torch.zeros(yb.shape[0], 3, device=yb.device)
        sl[:, 1] = 1.0 - conf
        sl[:, 2] = torch.where(yb > 0, conf, torch.zeros_like(conf))
        sl[:, 0] = torch.where(yb < 0, conf, torch.zeros_like(conf))
        dloss = -(sl * F.log_softmax(dl, dim=-1)).sum(dim=-1).mean()
        return ma * main + (1 - ma) * dloss

    def step(Xb, Xmb, yb, sync_item: bool):
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(Xb, Xmb)
            loss = loss_fn(out, yb)
        if sync_item:
            _ = loss.item()          # IT: sync GPU→CPU come in run_train | EN: GPU→CPU sync as in run_train
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)

    # ── B) fwd+bwd puro, batch GPU-resident ─────────────────────────────────
    print("=== B) fwd+bwd puro (batch GPU-resident, nessun DataLoader/H2D) ===")
    print(f"{'batch':>6} {'ms/step':>9} {'sample/s':>11} {'ms/step (no .item())':>21}")
    for bs in (32, 64, 128, 256, 512, 1024):
        Xb = X[:bs].to(device); Xmb = Xm[:bs].to(device); yb = y[:bs].to(device)
        for _ in range(8):
            step(Xb, Xmb, yb, True)
        torch.cuda.synchronize()
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            step(Xb, Xmb, yb, True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
        t0 = time.perf_counter()
        for _ in range(n):
            step(Xb, Xmb, yb, False)
        torch.cuda.synchronize()
        ms_ni = (time.perf_counter() - t0) / n * 1000
        print(f"{bs:>6} {ms:>9.2f} {bs/ms*1000:>11,.0f} {ms_ni:>21.2f}")
        del Xb, Xmb, yb
        torch.cuda.empty_cache()

    # ── C) DataLoader da solo ───────────────────────────────────────────────
    print("\n=== C) DataLoader da solo (num_workers=0, pin_memory=True) ===")
    for bs in (64, 256):
        dl = DataLoader(TensorDataset(X[:bs * 60], Xm[:bs * 60], y[:bs * 60]),
                        bs, shuffle=True, num_workers=0, pin_memory=True)
        t0 = time.perf_counter()
        nb = 0
        for b in dl:
            _ = [t.to(device, non_blocking=True) for t in b]
            nb += 1
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / nb * 1000
        print(f"batch={bs:>4}  {ms:6.2f} ms/batch  (collate+pin+H2D, senza modello)")

    # ── A) step completo con DataLoader ─────────────────────────────────────
    print("\n=== A) step completo (DataLoader + fwd/bwd), come in produzione ===")
    print(f"{'batch':>6} {'ms/step':>9} {'sample/s':>11} {'min/epoca(51882)':>18}")
    for bs in (64, 128, 256, 512):
        dl = DataLoader(TensorDataset(X[:bs * 45], Xm[:bs * 45], y[:bs * 45]),
                        bs, shuffle=True, num_workers=0, pin_memory=True)
        it = iter(dl)
        for _ in range(10):
            b = next(it)
            step(*[t.to(device, non_blocking=True) for t in
                   (b[0], b[1], b[2])], True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 0
        for b in dl:
            Xb, Xmb, yb = [t.to(device, non_blocking=True) for t in b]
            step(Xb, Xmb, yb, True)
            n += 1
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
        print(f"{bs:>6} {ms:>9.2f} {bs/ms*1000:>11,.0f} "
              f"{51882/bs*ms/1000/60:>18.2f}")


if __name__ == "__main__":
    main()

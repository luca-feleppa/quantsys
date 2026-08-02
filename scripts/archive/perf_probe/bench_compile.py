"""
Probe temporanea (PERF AUDIT) — lever (a): torch.compile su iTransformer.
Temporary probe (PERF AUDIT) — lever (a): torch.compile on the iTransformer.

Misura: presenza di spectral_norm sul path production, graph breaks,
tempo di compilazione, ms/step eager vs compiled, e scarto numerico
del forward (eager vs compiled) sullo stesso input in eval/fp32.
Measures: spectral_norm presence on the production path, graph breaks,
compile time, eager vs compiled ms/step, and forward numeric delta
(eager vs compiled) on identical input in eval/fp32.

Uso / Usage: python scripts/archive/perf_probe/bench_compile.py [--mode default]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import yaml
    import torch.nn.functional as F
    from quantsys.utils import load_config, setup_device
    from quantsys.model import QuantiTransformer, quantile_loss

    cfg = load_config(str(ROOT / "config/default.yaml"))
    with open(ROOT / "config/arch/itransformer.yaml", encoding="utf-8") as f:
        for sec, vals in yaml.safe_load(f).items():
            cfg.setdefault(sec, {}).update(vals)
    mcfg, tcfg = cfg["model"], cfg["training"]
    device = setup_device(cfg)
    use_amp = tcfg.get("use_amp", True)
    bs = args.batch

    d = np.load(str(ROOT / "data/lstm_dataset.npz"), allow_pickle=True)
    X = torch.from_numpy(d["X_train"][:bs * 4].astype(np.float32)).to(device)
    y = torch.from_numpy(d["y_train"][:bs * 4].astype(np.float32)).to(device)
    Xm = torch.from_numpy(d["X_macro_train"][:bs * 4].astype(np.float32)).to(device)
    n_feat, n_dyn, n_macro = X.shape[2], int(d["n_dynamic_features"][0]), Xm.shape[1]
    del d

    def mk():
        torch.manual_seed(0)
        return QuantiTransformer(
            n_features=n_feat, T=mcfg.get("window_size", 120), n_dynamic=n_dyn,
            n_macro=n_macro, d_model=mcfg.get("tft_d_model", 128),
            n_heads=mcfg.get("tft_n_heads", 4), n_layers=mcfg.get("tft_n_layers", 3),
            dropout=mcfg.get("tft_dropout", 0.1), patch_size=mcfg.get("patch_size", 1),
            drop_path_rate=mcfg.get("drop_path_rate", 0.0),
            loss_type=mcfg.get("loss_type", "quantile"),
            use_multitask=mcfg.get("use_multitask", False),
            n_output_experts=mcfg.get("n_output_experts", 1),
        ).to(device)

    m_eager = mk()

    # ── ① spectral_norm sul path production? ────────────────────────────────
    from torch.nn.utils.parametrize import is_parametrized
    parametrized = [n for n, mod in m_eager.named_modules() if is_parametrized(mod)]
    print(f"loss_type production = {mcfg.get('loss_type')!r}")
    print(f"moduli con parametrizations (spectral_norm): "
          f"{parametrized if parametrized else 'NESSUNO / NONE'}")
    print(f"params={sum(p.numel() for p in m_eager.parameters()):,}\n")

    ma, thr = mcfg.get("multitask_alpha", 0.7), mcfg.get("multitask_threshold", 1e-4)

    def loss_fn(out, yb):
        main = quantile_loss(yb, out[0])
        dl = out[-1]
        conf = torch.tanh(yb.abs() / max(thr, 1e-8))
        sl = torch.zeros(yb.shape[0], 3, device=yb.device)
        sl[:, 1] = 1.0 - conf
        sl[:, 2] = torch.where(yb > 0, conf, torch.zeros_like(conf))
        sl[:, 0] = torch.where(yb < 0, conf, torch.zeros_like(conf))
        return ma * main + (1 - ma) * (-(sl * F.log_softmax(dl, -1)).sum(-1).mean())

    def bench(model, tag, n=40, warm=12):
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
        scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

        def one():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(X[:bs], Xm[:bs]), y[:bs])
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        for _ in range(warm):
            one()
        torch.cuda.synchronize()
        t_warm = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            one()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
        print(f"{tag:<28} warmup({warm} step)={t_warm:7.2f}s   steady={ms:6.2f} ms/step")
        return ms

    ms_eager = bench(m_eager, "eager (produzione)")

    # ── ② compile: conta i graph break ──────────────────────────────────────
    import torch._dynamo as dynamo
    dynamo.reset()
    expl = None
    try:
        m_probe = mk()
        expl = dynamo.explain(lambda a, b: m_probe(a, b))(X[:bs], Xm[:bs])
        print(f"\ndynamo.explain: graph={expl.graph_count}  "
              f"graph_breaks={expl.graph_break_count}  ops={expl.op_count}")
        for i, r in enumerate(expl.break_reasons[:8]):
            print(f"  break {i+1}: {getattr(r, 'reason', r)}")
    except Exception as e:
        print(f"\ndynamo.explain fallito: {type(e).__name__}: {e}")

    # ── ③ compiled bench ────────────────────────────────────────────────────
    dynamo.reset()
    m_c = mk()
    m_c.load_state_dict(m_eager.state_dict())
    t0 = time.perf_counter()
    try:
        m_comp = torch.compile(m_c, mode=args.mode)
        ms_comp = bench(m_comp, f"compiled ({args.mode})", warm=12)
        print(f"\nspeedup steady = {ms_eager/ms_comp:.2f}×  "
              f"({ms_eager:.2f} → {ms_comp:.2f} ms/step)")
        print(f"tempo totale probe compiled (incl. compilazione) = "
              f"{time.perf_counter()-t0:.1f}s")
    except Exception as e:
        print(f"\ntorch.compile FALLITO: {type(e).__name__}: {e}")
        return

    # ── ④ scarto numerico forward eager vs compiled (eval, fp32) ────────────
    m_eager.eval(); m_comp.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        a = m_eager(X[:bs], Xm[:bs])
        torch.manual_seed(1)
        b = m_comp(X[:bs], Xm[:bs])
    for i, (ta, tb) in enumerate(zip(a, b)):
        if not torch.is_tensor(ta):
            continue
        dmax = (ta - tb).abs().max().item()
        rel = dmax / max(ta.abs().max().item(), 1e-12)
        print(f"  out[{i}] shape={tuple(ta.shape)}  |Δ|max={dmax:.3e}  "
              f"rel={rel:.3e}  bit-identico={'SI' if dmax == 0.0 else 'NO'}")


if __name__ == "__main__":
    main()

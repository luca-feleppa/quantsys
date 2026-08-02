"""
Probe temporanea (LEVA A) — quale stimatore dei clip bounds e' piu' CORRETTO?
Temporary probe (LEVER A) — which clip-bounds estimator is more CORRECT?

Non e' una domanda di velocita'. `02_train.py` calcola i clip bounds p0.1/p99.9
su `X_train.reshape(-1, F)`, cioe' sulla vista ESPANSA delle finestre: con
window=120 e stride=1 ogni barra compare ~120 volte, ma le barre di BORDO
compaiono meno. L'alternativa e' calcolarli sulle barre DISTINTE (200x piu'
veloce). Le due stime differiscono; questa probe stabilisce se la differenza
e' SEGNALE o ARTEFATTO, con tre prove:

  A) MECCANISMO — la vista espansa e' esattamente "barre distinte pesate per
     molteplicita'"? Se si', la differenza e' interamente il ri-peso dei bordi.
  B) RUMORE — la differenza fra i due stimatori e' grande o piccola rispetto
     alla variabilita' campionaria dello stimatore stesso (bootstrap)?
  C) IMPATTO A VALLE — quante celle vengono effettivamente clippate in modo
     diverso, e di quanto? E' la sola quantita' che il training vede davvero.

Uso / Usage: python scripts/archive/perf_probe/test_clip_bounds_correctness.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

W = 120          # IT/EN: window_size (model.window_size)
P_LO, P_HI = 0.1, 99.9


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    d = np.load(str(ROOT / "data/lstm_dataset.npz"), allow_pickle=True)
    X = d["X_train"]
    names = [str(x) for x in d["feature_names"]]
    n_tr, w, F = X.shape
    assert w == W, f"window inattesa: {w}"
    print(f"X_train = {X.shape}  ({X.nbytes/1e9:.2f} GB)  F={F}")

    # ── struttura a BLOCCHI: X_train NON e' contiguo ────────────────────────
    # IT: `create_windows` scarta le finestre contenenti NaN (`valid` mask), quindi
    #     X_train e' fatto di piu' blocchi contigui separati da salti. Ignorarlo
    #     (assumendo X[j,0,:] = barra j per ogni j) produce una ricostruzione
    #     SBAGLIATA in silenzio — verificato: la prova A fallisce. I punti di
    #     rottura si trovano confrontando X[j+1,:-1] con X[j,1:].
    # EN: `create_windows` drops NaN-containing windows (`valid` mask), so X_train
    #     is made of several contiguous blocks separated by jumps. Ignoring that
    #     (assuming X[j,0,:] = bar j for every j) yields a SILENTLY wrong
    #     reconstruction — verified: proof A fails. Break points are found by
    #     comparing X[j+1,:-1] against X[j,1:].
    breaks = []
    step = 2000
    for s in range(0, n_tr - 1, step):
        e = min(s + step, n_tr - 1)
        eq = (X[s:e, 1:, :] == X[s + 1:e + 1, :-1, :]).all(axis=(1, 2))
        breaks.extend((s + np.flatnonzero(~eq)).tolist())
    edges = [0] + [b + 1 for b in breaks] + [n_tr]
    print(f"discontinuita' trovate: {len(breaks)} -> {len(edges)-1} blocchi contigui")
    print(f"  blocchi: {[(edges[i], edges[i+1]) for i in range(len(edges)-1)]}")

    # IT: barre DISTINTE per blocco: first-bar di ogni finestra + coda dell'ultima.
    #     Molteplicita': dentro un blocco di nb finestre, la barra b compare in
    #     ogni (j,k) con j+k=b, 0<=j<nb, 0<=k<W.
    # EN: DISTINCT bars per block: each window's first bar + the last window's tail.
    #     Multiplicity: within a block of nb windows, bar b appears in every (j,k)
    #     with j+k=b, 0<=j<nb, 0<=k<W.
    parts, mults = [], []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        nb = e - s
        parts.append(X[s:e, 0, :])
        parts.append(X[e - 1, 1:, :])
        bb = np.arange(nb + W - 1)
        mults.append(np.minimum(W - 1, bb) - np.maximum(0, bb - nb + 1) + 1)
    distinct = np.concatenate(parts, axis=0)
    mult = np.concatenate(mults)
    n_bars = distinct.shape[0]
    assert mult.sum() == n_tr * W, "molteplicita' incoerenti"
    print(f"barre distinte = {n_bars:,}   (vista espansa = {n_tr*W:,} righe, "
          f"ridondanza {n_tr*W/n_bars:.1f}x)\n")
    b = np.arange(n_bars)
    n_edge = int((mult < W).sum())
    deficit = int((W - mult).sum())
    print("=== A) MECCANISMO ===")
    print(f"barre con molteplicita' piena ({W}): {int((mult==W).sum()):,}")
    print(f"barre di bordo sotto-pesate      : {n_edge:,} "
          f"({n_edge/n_bars*100:.2f}% delle barre)")
    print(f"deficit di peso totale           : {deficit:,} su {n_tr*W:,} "
          f"({deficit/(n_tr*W)*100:.3f}% del peso)")

    # IT: verifica che l'espansione sia SOLO ripetizione: ricostruisco la vista
    #     espansa su poche colonne via np.repeat e confronto i percentili.
    # EN: verify the expansion is ONLY repetition: rebuild the expanded view on a
    #     few columns via np.repeat and compare percentiles.
    probe_cols = [0, 9, 20, 57, F - 1]
    flat = X.reshape(-1, F)
    exp_p = np.percentile(flat[:, probe_cols], [P_LO, P_HI], axis=0)
    rep = np.repeat(distinct[:, probe_cols], mult, axis=0)
    rep_p = np.percentile(rep, [P_LO, P_HI], axis=0)
    ok = np.array_equal(exp_p, rep_p)
    print(f"ricostruzione 'distinte x molteplicita' == vista espansa : "
          f"{'IDENTICA' if ok else 'DIVERSA'}  (colonne {probe_cols})")
    if not ok:
        print(f"   max |diff| = {np.abs(exp_p-rep_p).max():.3e}")
    print("   -> la vista espansa NON contiene informazione in piu': e' la\n"
          "      stessa popolazione, con i bordi sotto-pesati.\n")
    del rep

    # ── i due stimatori ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    lo_exp, hi_exp = np.percentile(flat, [P_LO, P_HI], axis=0)
    t_exp = time.perf_counter() - t0
    t0 = time.perf_counter()
    lo_dis, hi_dis = np.percentile(distinct, [P_LO, P_HI], axis=0)
    t_dis = time.perf_counter() - t0
    print(f"tempo: espanso {t_exp:6.2f} s | distinto {t_dis:5.2f} s "
          f"-> {t_exp/t_dis:.0f}x\n")

    # ── B) la differenza e' dentro il rumore dello stimatore? ───────────────
    # IT: bootstrap sulle BARRE (l'unita' campionaria vera: le finestre non sono
    #     indipendenti, sono 120 copie sfalsate della stessa storia).
    # EN: bootstrap over BARS (the true sampling unit: windows are not
    #     independent, they are 120 shifted copies of the same history).
    print("=== B) RUMORE — bootstrap sulle barre (B=300) ===")
    rng = np.random.default_rng(42)
    B = 300
    boot_lo = np.empty((B, F), dtype=np.float64)
    boot_hi = np.empty((B, F), dtype=np.float64)
    for i in range(B):
        idx = rng.integers(0, n_bars, n_bars)
        s = distinct[idx]
        boot_lo[i], boot_hi[i] = np.percentile(s, [P_LO, P_HI], axis=0)
    sd_lo = boot_lo.std(axis=0, ddof=1)
    sd_hi = boot_hi.std(axis=0, ddof=1)

    diff_lo = np.abs(lo_exp - lo_dis)
    diff_hi = np.abs(hi_exp - hi_dis)
    # IT: differenza fra stimatori in unita' di deviazione standard campionaria.
    # EN: estimator difference in units of the sampling standard deviation.
    z_lo = diff_lo / np.maximum(sd_lo, 1e-12)
    z_hi = diff_hi / np.maximum(sd_hi, 1e-12)
    z_all = np.concatenate([z_lo, z_hi])
    print(f"|espanso - distinto| in unita' di SD bootstrap:")
    print(f"   mediana {np.median(z_all):.3f}   p90 {np.percentile(z_all,90):.3f}"
          f"   max {z_all.max():.3f}")
    print(f"   colonne con z > 1 (differenza sopra il rumore): "
          f"{int((z_all>1).sum())}/{2*F}")
    print(f"   colonne con z > 2                              : "
          f"{int((z_all>2).sum())}/{2*F}")
    worst = int(np.argmax(z_all))
    wn = names[worst % F] if worst < 2*F else "?"
    print(f"   peggiore: {wn} (z={z_all.max():.2f})")
    print("   -> z<1 significa che i due stimatori distano MENO di quanto lo\n"
          "      stimatore stesso oscilli ri-campionando i dati.\n")

    # ── C) impatto a valle: quante celle cambiano davvero ───────────────────
    # IT: e' la sola quantita' che il training vede. Applico i due clip alle
    #     barre distinte (il clip e' elemento-per-elemento: la ripetizione non
    #     cambia le frazioni).
    # EN: the only quantity training sees. Apply both clips to the distinct bars
    #     (clipping is elementwise: repetition does not change the fractions).
    print("=== C) IMPATTO A VALLE ===")
    c_exp = np.clip(distinct, lo_exp, hi_exp)
    c_dis = np.clip(distinct, lo_dis, hi_dis)
    n_cells = distinct.size
    frac_exp = float(((distinct < lo_exp) | (distinct > hi_exp)).sum()) / n_cells
    frac_dis = float(((distinct < lo_dis) | (distinct > hi_dis)).sum()) / n_cells
    delta = np.abs(c_exp - c_dis)
    n_diff = int((delta > 0).sum())
    print(f"celle clippate: espanso {frac_exp*100:.4f}%  |  "
          f"distinto {frac_dis*100:.4f}%   (atteso ~0.2%)")
    print(f"celle che DIFFERISCONO fra i due: {n_diff:,}/{n_cells:,} "
          f"({n_diff/n_cells*100:.4f}%)")
    if n_diff:
        nz = delta[delta > 0]
        # IT: scala di riferimento = IQR della feature (i dati sono z-score robusti).
        # EN: reference scale = feature IQR (data are robust z-scores).
        iqr = np.percentile(distinct, 75, axis=0) - np.percentile(distinct, 25, axis=0)
        med_iqr = float(np.median(iqr))
        print(f"   |Δ| su quelle celle: mediana {np.median(nz):.4f}  "
              f"p99 {np.percentile(nz,99):.4f}  max {nz.max():.4f}")
        print(f"   IQR mediana delle feature = {med_iqr:.3f} -> "
              f"|Δ| mediana = {np.median(nz)/med_iqr:.4f} IQR")
    print()

    # ── sintesi ─────────────────────────────────────────────────────────────
    print("=== SINTESI ===")
    print(f"differenza relativa max sui bound: "
          f"lo {np.max(diff_lo/np.maximum(np.abs(lo_dis),1e-9))*100:.2f}%  "
          f"hi {np.max(diff_hi/np.maximum(np.abs(hi_dis),1e-9))*100:.2f}%")
    print(f"frazione di celle del dataset che cambierebbero: "
          f"{n_diff/n_cells*100:.4f}%")
    print(f"z mediano (differenza / rumore dello stimatore): "
          f"{np.median(z_all):.3f}")


if __name__ == "__main__":
    main()

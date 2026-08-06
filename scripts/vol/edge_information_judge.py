# IT: GIUDICE E1 — L'EDGE NN-vs-IV HA CONTENUTO PREDITTIVO SULLA VARIANZA REALIZZATA?
#     Pre-registrazione vincolante: STATUS.md, sezione E1 del 2026-07-31.
#     La linea vol ha dimostrato che il NN batte una baseline ECONOMETRICA (HAR-C).
#     Questo giudice chiede una cosa piu' severa: batte la previsione del MERCATO?
#     Per ogni expiry giornaliera si confrontano due numeri gia' registrati con cio'
#     che poi e' successo:
#       x = edge = log(rv_pred / var_iv)   (chi vince secondo il NN)
#       y = log(RV_realizzata / var_iv)    (chi ha vinto davvero)
#     Denominatore comune per costruzione, quindi sign(x) = sign(y) e' esattamente
#     l'evento tradabile e non un proxy.
#     ⚠ COSTANTI HARDCODED PERCHE' PRE-REGISTRATE. Cambiarle a risultati visti e'
#     goalpost-moving: la pre-reg vieta ogni variante senza una NUOVA registrazione.
#     ⚠ DUE STADI. Stadio 1 = ESPLORATIVO sulla finestra gia' registrata: nessuna
#     soglia, NESSUN VERDETTO, hypothesis-generating soltanto (il campione porta
#     informazione parziale sull'esito, perche' la PnL di quelle expiry ha gia'
#     fallito il gate del 30/07). Stadio 2 = CONFERMATIVO su expiry liquidate DOPO
#     il commit della pre-reg, con guard fail-fast a n<40.
#     Read-only: non tocca 04b, ne' i modelli, ne' il path production.
# EN: E1 JUDGE — DOES THE NN-vs-IV EDGE CARRY PREDICTIVE CONTENT ABOUT REALIZED
#     VARIANCE? Binding pre-registration: STATUS.md, E1 section, 2026-07-31.
#     The vol line proved the NN beats an ECONOMETRIC baseline (HAR-C). This judge
#     asks the harder question: does it beat the MARKET's forecast? For each daily
#     expiry it compares two already-recorded numbers with what then happened:
#       x = edge = log(rv_pred / var_iv)   (who wins according to the NN)
#       y = log(realized RV / var_iv)      (who actually won)
#     Common denominator by construction, so sign(x) = sign(y) is exactly the
#     tradable event and not a proxy.
#     ⚠ CONSTANTS HARDCODED BECAUSE PRE-REGISTERED.
#     ⚠ TWO STAGES. Stage 1 = EXPLORATORY on the already-recorded window: no
#     thresholds, NO VERDICT. Stage 2 = CONFIRMATORY on expiries settling AFTER the
#     pre-registration commit, with a fail-fast guard at n<40.
#     Read-only: touches neither 04b, nor the models, nor the production path.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantsys.model.vol_metrics import diebold_mariano, qlike, qlike_series  # noqa: E402

FORECASTS = ROOT / "results" / "vol_paper" / "forecasts.parquet"
CANDLES_1H = ROOT / "data" / "raw_candles.parquet"
CANDLES_1M = ROOT / "data" / "raw_candles_1m_l2.parquet"
OUT_JSON = ROOT / "results" / "vol_paper" / "edge_information_stage{stage}.json"

# ─────────────────────── COSTANTI PRE-REGISTRATE / PRE-REGISTERED ───────────────────────
H_BARS = 30                 # IT/EN: orizzonte del target e tenor di var_iv / target horizon and var_iv tenor
DECISION_LO, DECISION_HI = 27.0, 33.0   # IT/EN: finestra del tick di decisione (ore prima dell'expiry)
EXPIRY_HOUR_UTC = 8         # IT/EN: dailies Deribit alle 08:00 UTC
HAC_LAG = 1                 # IT: sovrapposizione 6h su 30 fra expiry adiacenti, zero oltre
DM_H = 2                    # IT/EN: n_eff = n/2, conservativo / conservative
BLOCK = 2                   # IT/EN: moving-block bootstrap, assorbe il lag 1
N_BOOT = 10000
ALPHA = 0.05
N_MIN_STAGE2 = 40
# IT: stadio 2 = expiry liquidate DOPO il commit della pre-registrazione (de47191).
# EN: stage 2 = expiries settling AFTER the pre-registration commit (de47191).
STAGE2_CUTOFF = pd.Timestamp("2026-08-01 00:00", tz="UTC")


def hourly_close() -> pd.Series:
    # IT: serie oraria dei close. Le candele di produzione sono la sorgente primaria,
    #     MA la loro ultima riga e' una barra ancora in formazione al momento del
    #     download: verificato che 1166 ore su 1167 coincidono col 1m e che l'unica
    #     divergenza e' esattamente quell'ultima riga. Quindi: produzione fino alla
    #     penultima ora, 1m-derivate da li' in poi.
    # EN: hourly close series. Production candles are the primary source, BUT their
    #     last row is a bar still forming at download time: verified that 1166 of
    #     1167 hours match the 1m file and that the single divergence is exactly
    #     that last row. So: production up to the penultimate hour, 1m-derived after.
    def _load(p):
        d = pd.read_parquet(p)
        return d.set_index(pd.to_datetime(d["open_time"], utc=True)).sort_index()

    h = _load(CANDLES_1H)["close"].iloc[:-1]
    m = _load(CANDLES_1M)
    # IT: solo ore COMPLETE (60 barre): un'ora parziale darebbe un close anticipato.
    # EN: complete hours only (60 bars): a partial hour would give an early close.
    grp = m["close"].groupby(m.index.floor("h"))
    m_h = grp.last()[grp.size() == 60]
    s = pd.concat([h, m_h[m_h.index > h.index.max()]]).sort_index()
    return s[~s.index.duplicated(keep="first")]


def realized_rv(close: pd.Series, t0: pd.Timestamp, n_bars: int = H_BARS):
    # IT: RV = somma dei log-rendimenti ORARI al quadrato sulle n_bars ore successive
    #     a t0 — LA DEFINIZIONE DI TARGET DEL MODELLO. Serve percio' n_bars+1 close
    #     consecutivi; se ne manca uno la expiry e' NON OSSERVABILE (None), mai zero.
    #     ⚠ Disambiguazione della pre-reg, fissata PRIMA di qualunque risultato: il
    #     testo dice "sulle 30 ore da tick di decisione a expiry"; l'orizzonte
    #     operativo e' 30 BARRE DAL TICK, perche' lo stesso paragrafo impone che x e
    #     y siano commensurabili e sia rv_pred (30 barre) sia var_iv (tenor 30h) sono
    #     quantita' a 30 ore fisse, non "fino a scadenza".
    # EN: RV = sum of squared HOURLY log-returns over the n_bars hours after t0 — THE
    #     MODEL'S OWN TARGET DEFINITION. Needs n_bars+1 consecutive closes; if one is
    #     missing the expiry is UNOBSERVABLE (None), never zero.
    #     ⚠ Pre-reg disambiguation, fixed BEFORE any result: the operative horizon is
    #     30 BARS FROM THE TICK, because the same paragraph requires x and y to be
    #     commensurable and both rv_pred (30 bars) and var_iv (30h tenor) are fixed
    #     30-hour quantities, not "until expiry".
    want = pd.date_range(t0, periods=n_bars + 1, freq="h")
    px = close.reindex(want)
    if px.isna().any():
        return None
    r = np.diff(np.log(px.to_numpy()))
    return float(np.sum(r ** 2))


def build_panel(fc: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    # IT: un'osservazione per expiry giornaliera. Il tick di decisione e' l'ULTIMO
    #     con rv_pred/var_iv finiti nella finestra [E-33h, E-27h]: la finestra e'
    #     larga perche' la copertura tick e' 84.3% anche sul VPS (restart notturni),
    #     e cercare un'ora esatta scarterebbe expiry per un buco di logging.
    # EN: one observation per daily expiry. The decision tick is the LAST one with
    #     finite rv_pred/var_iv inside [E-33h, E-27h]: the window is wide because
    #     tick coverage is 84.3% even on the VPS (nightly restarts), and requiring an
    #     exact hour would discard expiries over a logging gap.
    days = pd.date_range(fc["candle_ts"].min().normalize(),
                         fc["candle_ts"].max().normalize() + pd.Timedelta("2D"), freq="D")
    rows, unobs = [], {"no_tick": 0, "no_rv": 0}
    for d in days:
        E = d + pd.Timedelta(hours=EXPIRY_HOUR_UTC)
        w = fc[(fc["candle_ts"] >= E - pd.Timedelta(hours=DECISION_HI))
               & (fc["candle_ts"] <= E - pd.Timedelta(hours=DECISION_LO))]
        w = w[np.isfinite(w["rv_pred"]) & np.isfinite(w["var_iv"]) & (w["var_iv"] > 0)]
        if len(w) == 0:
            unobs["no_tick"] += 1
            continue
        t = w.iloc[-1]
        rv = realized_rv(close, t["candle_ts"])
        if rv is None or rv <= 0:
            unobs["no_rv"] += 1
            continue
        # IT: naive = RV trailing sulle 30 ore PRECEDENTI il tick (zero parametri).
        # EN: naive = trailing RV over the 30 hours BEFORE the tick (zero parameters).
        rv_naive = realized_rv(close, t["candle_ts"] - pd.Timedelta(hours=H_BARS))
        rows.append({
            "expiry": E, "decision_ts": t["candle_ts"],
            "rv_pred": float(t["rv_pred"]), "var_iv": float(t["var_iv"]),
            "x_edge": float(np.log(t["rv_pred"] / t["var_iv"])),
            "rv_real": rv, "y_out": float(np.log(rv / t["var_iv"])),
            "rv_naive": rv_naive,
        })
    df = pd.DataFrame(rows)
    df.attrs["unobservable"] = unobs
    return df


def block_bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng) -> tuple:
    # IT: IC per Spearman con moving-block bootstrap (blocco 2): le expiry adiacenti
    #     condividono 6h di esposizione, un bootstrap iid sottostimerebbe la varianza.
    # EN: Spearman CI via moving-block bootstrap (block 2): adjacent expiries share
    #     6h of exposure, an iid bootstrap would understate the variance.
    from scipy.stats import spearmanr
    n = len(x)
    nb = int(np.ceil(n / BLOCK))
    starts_pool = np.arange(0, n - BLOCK + 1)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        st = rng.choice(starts_pool, size=nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in st])[:n]
        out[b] = spearmanr(x[idx], y[idx]).statistic
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def hac_mean_test(z: np.ndarray, null: float) -> dict:
    # IT: media di z contro `null` con varianza HAC (Bartlett, lag 1) — stessa
    #     macchina del DM, applicata a una serie di indicatori 0/1.
    # EN: mean of z against `null` with HAC variance (Bartlett, lag 1) — the same
    #     machinery as the DM, applied to a 0/1 indicator series.
    from scipy import stats
    n = len(z)
    dev = z - z.mean()
    var = float(np.mean(dev ** 2))
    for j in range(1, HAC_LAG + 1):
        var += 2.0 * (1.0 - j / (HAC_LAG + 1.0)) * float(np.mean(dev[j:] * dev[:-j]))
    se = np.sqrt(max(var, 1e-18) / n)
    t = (z.mean() - null) / se
    return {"mean": float(z.mean()), "se_hac": float(se), "t": float(t),
            "p_value": float(1.0 - stats.norm.cdf(t)), "n": n}


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Giudice E1 / E1 judge")
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True,
                    help="1 = esplorativo (nessun verdetto) / 2 = confermativo")
    # IT: --count-only = monitoraggio SICURO del campione, automatizzabile. Il guard
    #     n<40 -> NO_RUN protegge solo SOTTO soglia: a n>=40 uno `--stage 2` nudo
    #     calcola le tre condizioni, stampa il verdetto e scrive il report, cioe'
    #     eseguirebbe il run one-shot confermativo per automazione invece che per
    #     decisione. Con questo flag il giudice si ferma alla conta e non produce
    #     nessun numero decisionale a NESSUN n (stessa semantica del --count-only
    #     del comparatore MFIV).
    # EN: --count-only = SAFE sample monitoring, automatable. The n<40 -> NO_RUN guard
    #     only protects BELOW threshold: at n>=40 a bare `--stage 2` computes the three
    #     conditions, prints the verdict and writes the report — i.e. it would fire the
    #     confirmatory one-shot run by automation rather than by decision. With this
    #     flag the judge stops at the count and produces no decisional number at ANY n
    #     (same semantics as the MFIV comparator's --count-only).
    ap.add_argument("--count-only", action="store_true",
                    help="stampa solo la conta del campione, nessuna statistica e nessun report "
                         "/ prints the sample count only, no statistics and no report")
    args = ap.parse_args()

    fc = pd.read_parquet(FORECASTS)
    fc["candle_ts"] = pd.to_datetime(fc["candle_ts"], utc=True)
    close = hourly_close()
    panel = build_panel(fc, close)

    if args.stage == 2:
        panel = panel[panel["expiry"] >= STAGE2_CUTOFF].reset_index(drop=True)

    n = len(panel)
    print("=" * 78)
    print(f"E1 — CONTENUTO PREDITTIVO DELL'EDGE NN-vs-IV | STADIO {args.stage}")
    print("=" * 78)
    print(f"expiry osservabili / observable expiries : {n}")
    print(f"non osservabili / unobservable           : {panel.attrs.get('unobservable', {})}")
    if n:
        print(f"finestra / window                        : {panel['expiry'].min():%Y-%m-%d}"
              f" -> {panel['expiry'].max():%Y-%m-%d}")

    # IT: uscita anticipata del monitoraggio. Costruire il pannello E' la conta —
    #     un'expiry e' osservabile solo se la sua RV e' calcolabile — ma x, y e le
    #     statistiche restano in memoria e non vengono ne' stampate ne' scritte:
    #     l'operatore vede quante osservazioni ci sono, mai quanto valgono.
    # EN: monitoring early exit. Building the panel IS the count — an expiry is
    #     observable only if its RV is computable — but x, y and the statistics stay
    #     in memory and are neither printed nor written: the operator sees how many
    #     observations exist, never what they are worth.
    if args.count_only:
        print(f"\nCOUNT_ONLY — nessuna statistica calcolata, nessun report scritto "
              f"(soglia/threshold stadio 2 n>={N_MIN_STAGE2}). / no statistics, no report written.")
        return 0

    # IT: guard fail-fast anti-peeking: sotto n_min lo stadio 2 NON calcola nulla.
    # EN: fail-fast anti-peeking guard: below n_min stage 2 computes NOTHING.
    if args.stage == 2 and n < N_MIN_STAGE2:
        print(f"\nNO_RUN — n={n} < {N_MIN_STAGE2} pre-registrato. Nessun numero decisionale "
              f"calcolato, nessun report scritto. / no decisional number computed.")
        return 0
    if n < 10:
        print("\nNESSUNA CONCLUSIONE — campione troppo piccolo / sample too small.")
        return 0

    x = panel["x_edge"].to_numpy()
    y = panel["y_out"].to_numpy()

    # ① accordo di segno / sign agreement
    sa = hac_mean_test((np.sign(x) == np.sign(y)).astype(float), 0.5)
    # ② Spearman + IC bootstrap a blocchi
    from scipy.stats import spearmanr
    rho = float(spearmanr(x, y).statistic)
    lo, hi = block_bootstrap_spearman(x, y, np.random.default_rng(12345))

    # ④ controllo positivo: NN vs naive persistence in QLIKE
    ok = panel["rv_naive"].notna().to_numpy()
    ctrl = None
    if ok.sum() >= 10:
        rv_true = panel.loc[ok, "rv_real"].to_numpy()
        l_nn = qlike_series(rv_true, panel.loc[ok, "rv_pred"].to_numpy())
        l_nv = qlike_series(rv_true, panel.loc[ok, "rv_naive"].to_numpy())
        dm = diebold_mariano(l_nn, l_nv, h=DM_H, lag=HAC_LAG)
        ctrl = {"qlike_nn": qlike(rv_true, panel.loc[ok, "rv_pred"].to_numpy()),
                "qlike_naive": qlike(rv_true, panel.loc[ok, "rv_naive"].to_numpy()),
                "dm": dm, "n": int(ok.sum())}

    print(f"\n① ACCORDO DI SEGNO / sign agreement   : {sa['mean']:.4f}  "
          f"(t={sa['t']:+.3f}, p={sa['p_value']:.4f}, HAC lag {HAC_LAG})")
    print(f"② SPEARMAN rho(x,y)                   : {rho:+.4f}  "
          f"IC95 bootstrap a blocchi [{lo:+.4f}, {hi:+.4f}]")
    if ctrl:
        d = ctrl["dm"]
        print(f"④ CONTROLLO POSITIVO NN vs naive      : QLIKE {ctrl['qlike_nn']:.5f} vs "
              f"{ctrl['qlike_naive']:.5f}  (DM={d['dm_hln']:+.3f}, p={d['p_value']:.4f}, n={ctrl['n']})")
        print(f"   -> il NN batte la naive: {'SI' if d['p_value'] < ALPHA and ctrl['qlike_nn'] < ctrl['qlike_naive'] else 'NO'}")

    res = {"stage": args.stage, "n": n, "sign_agreement": sa,
           "spearman": {"rho": rho, "ci95": [lo, hi], "block": BLOCK, "n_boot": N_BOOT},
           "positive_control": ctrl, "unobservable": panel.attrs.get("unobservable", {}),
           "window": [str(panel["expiry"].min()), str(panel["expiry"].max())]}

    if args.stage == 1:
        print("\n" + "-" * 78)
        print("STADIO 1 — ESPLORATIVO. NESSUN VERDETTO, NESSUNA SOGLIA APPLICATA.")
        print("I numeri sopra sono hypothesis-generating e non possono essere citati")
        print("come evidenza confermativa ne' cambiare le soglie dello stadio 2.")
        print("-" * 78)
        res["verdict"] = "ESPLORATIVO - nessun verdetto / exploratory - no verdict"
    else:
        c1 = sa["p_value"] < ALPHA and sa["mean"] > 0.5
        c2 = lo > 0.0
        c4 = bool(ctrl and ctrl["dm"]["p_value"] < ALPHA and ctrl["qlike_nn"] < ctrl["qlike_naive"])
        verdict = ("NESSUNA CONCLUSIONE" if not c4 else ("PASS" if (c1 and c2) else "FAIL"))
        print(f"\n① {c1} · ② {c2} · ③ True (n={n}>={N_MIN_STAGE2}) · ④ {c4}")
        print(f"VERDETTO / VERDICT: {verdict}")
        res.update({"conditions": {"c1_sign": c1, "c2_spearman": c2, "c3_n": True, "c4_control": c4},
                    "verdict": verdict})

    out = Path(str(OUT_JSON).format(stage=args.stage))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nreport -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

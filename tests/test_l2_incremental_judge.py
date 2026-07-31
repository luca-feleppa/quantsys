# IT: Test del giudice B1 stadio 1 (pre-reg STATUS 2026-07-31). Il rischio dominante di
#     questo giudice NON e' un bug aritmetico ma un LEAKAGE silenzioso: il target somma
#     H ore successive, quindi le finestre si sovrappongono e addestrare su
#     un'osservazione il cui target non e' ancora accaduto produce un numero
#     perfettamente plausibile e completamente falso. I test (a) inchiodano l'embargo,
#     (b) verificano che il target non guardi il presente, (c) provano che la macchina
#     SA rilevare un effetto quando c'e' e NON lo inventa quando non c'e'.
# EN: Tests for the B1 stage-1 judge (STATUS 2026-07-31 pre-reg). This judge's dominant
#     risk is NOT an arithmetic bug but silent LEAKAGE: the target sums H following
#     hours, so windows overlap and training on an observation whose target has not
#     happened yet yields a perfectly plausible and completely false number. The tests
#     (a) pin the embargo down, (b) verify the target does not look at the present,
#     (c) prove the machinery CAN detect an effect when present and does NOT invent one
#     when absent.
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "l2_judge", ROOT / "scripts" / "vol" / "l2_incremental_judge.py")
J = importlib.util.module_from_spec(_spec)
sys.modules["l2_judge"] = J
_spec.loader.exec_module(J)


def _synth_price(path: Path, days: int = 40, seed: int = 0) -> Path:
    # IT: serie 1m sintetica con volatilita' variabile (serve una RV non costante).
    # EN: synthetic 1m series with time-varying volatility (a non-constant RV is needed).
    rng = np.random.default_rng(seed)
    n = days * 1440
    vol = 1e-4 * (1.0 + 0.5 * np.sin(np.arange(n) / 1440.0))
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol))
    pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
        "close": close,
    }).to_parquet(path, index=False)
    return path


# ── (a) embargo · embargo ────────────────────────────────────────────────────

def test_embargo_il_train_si_ferma_a_i_meno_h(monkeypatch):
    # IT: per prevedere i, il train deve arrivare a i-H incluso, cioe' avere i-H+1
    #     righe. Le ultime H-1 osservazioni sono in embargo perche' il loro target
    #     non e' ancora osservabile a fine ora t_i.
    # EN: to forecast i, training must stop at i-H inclusive, i.e. have i-H+1 rows.
    #     The last H-1 observations are embargoed because their target is not yet
    #     observable at the end of hour t_i.
    n, burn = 60, 20
    frame = pd.DataFrame({
        "y": np.arange(n, dtype=float),
        "xc_h": np.arange(n, dtype=float),
        "xc_w": np.zeros(n), "xc_m": np.zeros(n),
    })
    seen = []
    real = np.linalg.lstsq

    def spy(A, b, rcond=None):
        seen.append(len(A))
        return real(A, b, rcond=rcond)

    monkeypatch.setattr(np.linalg, "lstsq", spy)
    J.expanding_oos(frame, J.HAR_C_COLS, burn=burn, h=J.H)
    expected = [i - J.H + 1 for i in range(burn, n)]
    assert seen == expected, f"embargo violato / embargo violated: {seen[:5]} vs {expected[:5]}"


def test_nessuna_previsione_prima_del_burn_in():
    n, burn = 40, 15
    frame = pd.DataFrame({"y": np.random.default_rng(0).normal(size=n),
                          "xc_h": np.zeros(n), "xc_w": np.zeros(n), "xc_m": np.zeros(n)})
    p = J.expanding_oos(frame, J.HAR_C_COLS, burn=burn)
    assert np.isnan(p[:burn]).all() and np.isfinite(p[burn:]).all()


def test_i_due_modelli_sono_valutati_sugli_stessi_punti():
    # IT/EN: il confronto appaiato richiede maschere identiche / paired = identical masks
    rng = np.random.default_rng(1)
    n = 200
    frame = pd.DataFrame({c: rng.normal(size=n) for c in
                          ["y"] + J.HAR_C_COLS + J.L2_COLS})
    a = J.expanding_oos(frame, J.HAR_C_COLS)
    b = J.expanding_oos(frame, J.HAR_C_COLS + J.L2_COLS)
    assert np.array_equal(np.isnan(a), np.isnan(b))


def test_annidamento_baseline_dentro_candidato():
    assert set(J.HAR_C_COLS) < set(J.HAR_C_COLS + J.L2_COLS)
    assert len(J.L2_COLS) == 3


# ── (b) il target non guarda il presente · target does not look at the present ──

def test_il_target_esclude_l_ora_corrente(tmp_path):
    # IT: y[t] copre le ore t+1..t+H. Perturbare i rendimenti DENTRO l'ora t non deve
    #     cambiare y[t]; perturbare l'ora t+1 deve cambiarlo. E' il test che separa un
    #     target causale da uno che include il presente.
    # EN: y[t] covers hours t+1..t+H. Perturbing returns INSIDE hour t must not change
    #     y[t]; perturbing hour t+1 must. This separates a causal target from one that
    #     includes the present.
    p = _synth_price(tmp_path / "px.parquet")
    base = J.build_price_frame(str(p)).dropna()
    t = base.index[len(base) // 2]

    def perturb(hour_offset: int) -> pd.DataFrame:
        # IT: si perturbano SOLO i minuti INTERNI (1..58) dell'ora bersaglio. Scalare
        #     tutta l'ora sposterebbe anche il rendimento di CONFINE fra l'ultimo minuto
        #     di quest'ora e il primo della successiva — rendimento che appartiene
        #     all'ora successiva, quindi il test fallirebbe per un motivo giusto ma
        #     diverso da quello che vuole misurare.
        # EN: ONLY the INTERIOR minutes (1..58) of the target hour are perturbed. Scaling
        #     the whole hour would also move the BOUNDARY return between this hour's last
        #     minute and the next hour's first — a return that belongs to the next hour,
        #     so the test would fail for a correct but different reason.
        df = pd.read_parquet(p)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        lo = t + pd.Timedelta(hours=hour_offset)
        inner = ((df["open_time"] >= lo + pd.Timedelta("1min"))
                 & (df["open_time"] < lo + pd.Timedelta("59min")))
        df.loc[inner, "close"] = df.loc[inner, "close"] * 1.05
        q = tmp_path / f"px_{hour_offset}.parquet"
        df.to_parquet(q, index=False)
        return J.build_price_frame(str(q)).dropna()

    assert perturb(0).loc[t, "y"] == pytest.approx(base.loc[t, "y"]), \
        "y[t] cambia perturbando l'ora t: il target include il presente / target leaks"
    assert perturb(1).loc[t, "y"] != pytest.approx(base.loc[t, "y"]), \
        "y[t] non reagisce all'ora t+1: il target non copre l'orizzonte / horizon not covered"


def test_componente_continua_non_supera_la_rv(tmp_path):
    # IT/EN: C = min(RV, BV) ⇒ log C ≤ log RV per costruzione / by construction
    p = _synth_price(tmp_path / "px2.parquet", days=35, seed=3)
    f = J.build_price_frame(str(p)).dropna()
    assert (f["xc_h"] <= f["naive"] + 1e-9).all()


# ── (c) potere di rilevazione · detection power ──────────────────────────────

def test_rileva_un_effetto_quando_c_e():
    # IT: controllo positivo sintetico — una feature che E' il target deve far vincere
    #     il candidato in modo schiacciante. Se questo test fallisce, un FAIL sul dato
    #     reale non e' interpretabile perche' la macchina non saprebbe vedere nulla.
    # EN: synthetic positive control — a feature that IS the target must make the
    #     candidate win overwhelmingly. If this fails, a FAIL on real data is
    #     uninterpretable because the machinery could not see anything.
    rng = np.random.default_rng(7)
    n = 400
    y = rng.normal(-8, 1.0, n)
    frame = pd.DataFrame({
        "y": y, "xc_h": rng.normal(size=n), "xc_w": rng.normal(size=n),
        "xc_m": rng.normal(size=n),
        "ofi_abs": y + rng.normal(0, 0.01, n),          # IT/EN: quasi il target / almost the target
        "log_depth": rng.normal(size=n), "dimb25_abs": rng.normal(size=n),
    })
    pb = J.expanding_oos(frame, J.HAR_C_COLS)
    pc = J.expanding_oos(frame, J.HAR_C_COLS + J.L2_COLS)
    m = ~np.isnan(pb)
    err_b = np.mean((frame["y"].values[m] - pb[m]) ** 2)
    err_c = np.mean((frame["y"].values[m] - pc[m]) ** 2)
    assert err_c < 0.01 * err_b, f"effetto non rilevato / effect not detected: {err_c} vs {err_b}"


def test_non_inventa_un_effetto_quando_non_c_e():
    # IT: controllo negativo — feature L2 puro rumore: il candidato non deve migliorare
    #     in modo materiale (soglia ② della pre-reg). Con 3 parametri in piu' su dati
    #     casuali ci si aspetta un lieve PEGGIORAMENTO fuori campione.
    # EN: negative control — pure-noise L2 features: the candidate must not improve
    #     materially (pre-reg threshold ②). With 3 extra parameters on random data a
    #     slight out-of-sample WORSENING is expected.
    rng = np.random.default_rng(11)
    n = 400
    frame = pd.DataFrame({c: rng.normal(size=n) for c in ["y"] + J.HAR_C_COLS + J.L2_COLS})
    pb = J.expanding_oos(frame, J.HAR_C_COLS)
    pc = J.expanding_oos(frame, J.HAR_C_COLS + J.L2_COLS)
    m = ~np.isnan(pb)
    err_b = np.mean((frame["y"].values[m] - pb[m]) ** 2)
    err_c = np.mean((frame["y"].values[m] - pc[m]) ** 2)
    assert err_c > J.RATIO_MAX * err_b, "rumore puro supera la soglia di materialita' / noise passes"


# ── costanti pre-registrate · pre-registered constants ──────────────────────

def test_costanti_pre_registrate_invariate():
    # IT: sentinella anti-goalpost: cambiare una di queste = NUOVA pre-registrazione.
    # EN: anti-goalpost sentinel: changing any of these = NEW pre-registration.
    assert (J.H, J.BURN, J.ALPHA, J.RATIO_MAX, J.N_MIN) == (3, 120, 0.01, 0.97, 240)
    assert J.L2_COLS == ["ofi_abs", "log_depth", "dimb25_abs"]
    assert J.HAR_C_COLS == ["xc_h", "xc_w", "xc_m"]

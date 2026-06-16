# Schema del paper · Paper outline — "Are Price and Volume Enough?"

🇮🇹 Scheletro operativo del paper. Lingua del manoscritto: **inglese** (sotto, lo scheletro è in EN perché è il testo di lavoro). Genere: **negative result pre-registrato con positive control** — il valore sta nel contrasto pari/dispari sullo stesso identico setup, non nei singoli numeri. Venue candidate: arXiv `q-fin.ST` (preprint) → *Journal of Financial Data Science* o *Journal of Forecasting* (la JFDS accetta volentieri studi empirici ML-vs-econometria; in alternativa *Quantitative Finance* short communication). Lunghezza target: 8-12 pagine + appendice riproducibilità.

**EN** Working skeleton of the paper. Manuscript language: **English** (the skeleton below is in EN as the working text). Genre: **pre-registered negative result with a positive control** — the value lies in the even/odd contrast on the very same setup, not in any single number. Candidate venues: arXiv `q-fin.ST` (preprint) → *Journal of Financial Data Science* or *Journal of Forecasting* (JFDS welcomes ML-vs-econometrics empirical studies; alternatively a *Quantitative Finance* short communication). Target length: 8-12 pages + reproducibility appendix.

---

## Working title

**"Are Price and Volume Enough? Even Moments Are Predictable, Odd Moments Are Not: Pre-Registered Evidence from Bitcoin"**

(alt: *"What Candles Know: The Even-Moment Boundary of Price–Volume Information in Bitcoin"*)

## One-sentence thesis

> Using a single fixed pipeline (104 price/volume/funding features, identical splits, pre-registered gates), deep ensembles and classical econometrics agree: the conditional information in OHLCV candles is confined to **even moments** of the return distribution — realized variance is strongly predictable out-of-sample (−30% QLIKE vs HAR-RV at hourly resolution), while every **odd-moment object** (return sign/value, cross-sectional return ranks, signed semivariance asymmetry) is indistinguishable from noise, for neural networks and linear models alike.

---

## Structure

### 1. Introduction (~1.5 pp)
- Hook: ML4F literature reports both spectacular successes and failures on return prediction; rarely on the *same* pipeline with pre-registration. We ask a sharper question: *which functionals of the conditional distribution does price/volume information actually identify?*
- Contributions (bulleted):
  1. **Even/odd moment dichotomy** on one fixed pipeline: RV strongly predictable (positive control), direction + signed jump variation not — for NN *and* econometric baselines (→ it's the information set, not the model class).
  2. **Resolution specificity**: the NN edge over HAR-RV exists at hourly RV (h=30h) and vanishes at 30-minute RV — a boundary, not a blanket claim.
  3. **Methodological warning**: systematic val→test sign-instability of directional metrics (documented at 1m, 1h, and for linear baselines) — in-sample/validation skill on odd moments is anti-informative here.
  4. **Pre-registration discipline** as first-class methodology in a quant ML study (gates written before runs, kill-on-fail, test split touched once).
- What this paper is NOT: not a trading-strategy paper; the economic layer (costs, cross-sectional spreads) appears only as corroborating evidence of the magnitude wall.

### 2. Related work (~1 p)
- Vol predictability: Corsi (2009) HAR-RV; Patton (2011) QLIKE robustness; ML-beats-HAR strand (e.g., Bucci; Christensen et al.).
- Good/bad volatility & signed jumps: Barndorff-Nielsen–Kinnebrock–Shephard (2010); Patton–Sheppard (2015) — we test the *forward* version of their decomposition.
- Return (un)predictability & EMH in crypto; ML4F negative-results and replication-crisis literature (pre-registration in finance).
- iTransformer / modern TS architectures (brief — architecture is instrumental, not the point).

### 3. Data and protocol (~1.5 pp)
- BTC/USDT perpetual, Binance, 1h candles 2019→2026 (65k bars); 1m robustness window (381d). Funding rate series. 104 features (86 dynamic + 18 structural): returns, VWAP, technicals, CVD, volatility, volume profile, time-of-day, funding — *price/volume-measurable only* (this defines the information set under test).
- Targets, all at horizon h=30 bars: (i) `Σ log-ret` (odd, 1st moment); (ii) `log RV` (even, 2nd); (iii) `log(RS⁺/RS⁻)` forward signed-semivariance ratio (odd-like object inside the vol family).
- Chronological 80/10/10 split; global RobustScaler fit on train only; embargo/purged walk-forward where applicable; **pre-registration registry** (STATUS.md excerpts in appendix): every gate (metric+threshold+min-N) timestamped before the run; val-first, test once, zero post-hoc iterations.
- Anti-lookahead engineering (causal regime filter, no smoothed probabilities, bit-parity train↔inference) — 1 paragraph, details in appendix.

### 4. Models (~1 p)
- NN: iTransformer 5-seed ensemble (t-Student NLL + asymmetric-sign penalty + CRPS; total-variance law for σ_ens). Same hyperparameters across all three targets (tuned once on the directional task — conservative *against* the vol claim).
- Econometric baselines, all OLS/logit fit on train only:
  - direction: "HAR-mean" OLS on [r_h, r_7d, r_30d], sign logit, momentum persistence, train-mean;
  - variance: HAR-RV (Corsi), naive persistence;
  - asymmetry: HAR-RS (Patton–Sheppard regressors), naive lratio, train-mean.
- Judges: QLIKE on RV levels (even), MSE on log-ratio (asymmetry), Spearman/sign-DA (direction).

### 5. Results (~3 pp) — one table per claim, numbers from RESULTS_MAP.md
- **5.1 Direction (odd #1).** NN: val ρ +0.19 → test −0.04 (p=0.001); econometric baselines table (all |ρ|≤0.05, sign-flipping val→test); cross-sectional IC +0.014 (t=1.86, spread 1.5 bps vs 26 bps cost). Figure: val-vs-test scatter of directional metrics across all models/experiments → the anti-correlation cloud.
- **5.2 Realized variance (even). The positive control.** QLIKE table (NN 0.257 / HAR 0.368 / naive 0.807 on test; val→test *coherent*). Resolution boundary: 1m ratio 1.013 → FAIL. Figure: predicted vs realized log-RV on test; sub-period ICIR stability.
- **5.3 Signed semivariance (odd #2).** MSE table — headline: **HAR-RS underperforms the unconditional mean on test**; NN matches the constant. The asymmetry is unpredictable *for everyone*.
- **5.4 Synthesis figure.** One chart, the paper's centerpiece: each experiment on an axis "even ↔ odd object" × "OOS skill vs baseline" — even side clusters positive, odd side clusters at zero.
- (5.5 optional, if matured) Forward economic test: NN-RV vs Deribit implied variance at 30h tenor, pre-registered straddle rule on testnet — preliminary n, reported as outlook.

### 6. Discussion (~1 p)
- Information-theoretic reading: candle aggregation is (nearly) sign-symmetric — squared-return functionals survive aggregation, signed functionals don't. Odd-moment prediction plausibly requires *exogenous* state: order-flow imbalance, depth (L2), positioning — exactly what OHLCV destroys.
- Why val→test anti-correlation: weak overfit on a stationary-vol but non-stationary-direction generating process; implications for model selection on odd moments (validation is a trap).
- Economic corollary: even where direction had rank-information (Quiet regime, cross-section), magnitude ≪ costs — the even/odd boundary coincides with the tradability boundary for price/volume strategies.

### 7. Limitations & future work (~0.5 p)
- Single asset, single venue, one architecture family seriously tuned; 1m robustness on a shorter window; semivariance tested in ratio form only; L2/order-book information set as the natural falsification path of the "odd needs exogenous info" conjecture; RV-vs-IV forward test as the economic continuation.

### 8. Conclusion (~0.25 p)

### Appendices
- A. Pre-registration excerpts (verbatim, dated) + outcome ledger (all kills listed).
- B. Feature list (104) + anti-lookahead engineering details.
- C. Reproducibility: repo layout, judges (`scripts/vol/dev_vols_qlike.py`, `scripts/vol/dev_vols_rs_judge.py`, `scripts/research/paper_01_dir_baselines.py`), seeds, hardware, runtimes.

---

## Cose da fare per arrivare al draft · TODO to reach a draft

🇮🇹 1) **Fig. 5.4** (sintesi pari/dispari) e **Fig. 5.1** (cloud val-vs-test) — script di plotting dai JSON in `results/` (effort S). 2) Tabelle LaTeX dai JSON (generabili via script, no copia a mano). 3) Stesura §1-§4 (il materiale è tutto in `RESULTS_MAP.md` + STATUS). 4) Decidere se aspettare n≥30 del forward test per §5.5 o pubblicare senza. 5) Bibliografia (≈25 voci, metà già citate qui).

**EN** 1) **Fig. 5.4** (even/odd synthesis) and **Fig. 5.1** (val-vs-test cloud) — plotting script from the JSONs in `results/` (effort S). 2) LaTeX tables from the JSONs (script-generated, no hand-copying). 3) Write §1-§4 (material fully covered by `RESULTS_MAP.md` + STATUS). 4) Decide whether to wait for the forward test's n≥30 for §5.5 or publish without. 5) Bibliography (≈25 entries, half already cited here).

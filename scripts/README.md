# Mappa degli script · Script map

🇮🇹 Lo **spine numerato** (`00→99`) è piatto in `scripts/` e codifica la **fase** della pipeline (dati→train→backtest→live), **non** la linea di ricerca. La distinzione vol-volatilità vs direzionale-rendimenti **non** è una biforcazione di codice: è lo stesso motore (`quantsys/`) con un interruttore di config (`features.target_type`: `log_rv` per la vol, `ret` per i rendimenti). Gli script **non numerati** specifici di linea vivono in sottocartelle (`vol/`, `research/`, `archive/`) e usano `Path(__file__).resolve().parents[2]` → vanno lanciati dalla **root di progetto**.

**EN** The **numbered spine** (`00→99`) is flat in `scripts/` and encodes the pipeline **phase** (data→train→backtest→live), **not** the research line. The vol-volatility vs directional-returns split is **not** a code fork: it is the same engine (`quantsys/`) with a config switch (`features.target_type`: `log_rv` for vol, `ret` for returns). Line-specific **non-numbered** scripts live in subfolders (`vol/`, `research/`, `archive/`) and use `Path(__file__).resolve().parents[2]` → run them from the **project root**.

---

## Spine numerato (condiviso) · Numbered spine (shared)

| Script | Fase / Phase | Linea / Line |
|---|---|---|
| `00_check_setup.py` · `00_test_binance_testnet.py` | setup / smoke | shared |
| `01_download_data.py` · `01_update_data.py` · `01b_download_macro.py` | dati + macro/regime | shared |
| `01c_iv_poller.py` | poller IV Deribit (forward) → `data/iv/` | vol (gate NN-RV vs IV) |
| `01d_orderbook_recorder.py` | recorder order-book L2 Binance (forward) → `data/orderbook/` | B1 direzionale (raccolta) |
| `02_train.py` · `02b_walkforward_validate.py` · `02c_optuna_search.py` | training / validazione | shared (target da config) |
| `02d_cafn_joint_train.py` | training CONGIUNTO CAFN + 3 modelli (probe pre-registrato, inerte; output `models/cafn/`) | research (coordinatore) |
| `03_backtest.py` | backtest trading | **direzionale** (no senso su vol) |
| `04_live_signals.py` | live / paper trading WS | **direzionale** |
| `04b_vol_paper.py` · `04c_vol_paper_baselines.py` | forward test vol + gate baselines | **vol** |
| `05_analyze_signals.py` · `07_verify_teacher.py` | analisi segnali / confronto teacher | shared |
| `06_dashboard.py` | Deribit Options Risk Terminal (SPA HTTP+Plotly: vol surface 3D, smile/term, OI by strike, Greche, PCR, tab Trades + payoff) — dati Deribit pubblici, GPU-free, indipendente dalla pipeline ML | standalone (market data) |
| `99_replay_live_vs_training.py` | diagnostica parity (BLOCKER #1) | shared |

## Sottocartelle per linea · Per-line subfolders

| Cartella / Folder | Contenuto / Content | Linea / Line |
|---|---|---|
| `vol/` | **Forecasting/giudici:** `dev_vols_macro_append.py` (ri-appende X_macro), `dev_vols_qlike.py` (giudice QLIKE vs HAR-RV), `dev_vols_quantile_judge.py` (A2a: calibrazione quantili q10-q90, coverage+pinball vs HAR-quantile — FAIL 2026-07-08), `dev_vols_member_weights.py` (A5: pesi membro per-QLIKE, fit/eval su metà val — FAIL 2026-07-08), `dev_vols_rs_judge.py` (giudice MSE semivarianza), `step0_xarch_corr.py` (STEP 0 kill-check correlazione cross-arch), `wf_har_baseline.py` (baseline HAR per-fold del walk-forward). **Short-vol arm:** `short_vol_arm.py` (sim offline forward test), `short_vol_hist_backtest.py` (backtest storico strutturale FHS-GJR-GARCH 2019→2026), `short_vol_premium_validate.py` (robustness del premio VRP), `short_vol_regime_decomp.py` (decomposizione regime/anno + equity/drawdown). **IVS relative-value:** `ivs_scout.py` (scouting smile IV Deribit), `ivs_rv_backtest.py` (backtest net-of-cost reversione residui smile — KILL). **Delta-hedge (B2/A1):** `hedge_dry_run.py` (dry-run retrospettivo della leg perp su `exec_diag.jsonl` A6: PnL inverse esatto, δ_raw vs δ_adj, regressione hedge-ratio, varianza hedged vs unhedged — pre-studio del design v2, da rilanciare a gate chiuso). **Helper condiviso:** `_chain_io.py` (lettura option-chain `data/iv/chain/` con cache LRU). | **vol** |
| `research/` | `paper_01_dir_baselines.py` (baseline econometriche direzionali = negative-control del paper) | direzionale / paper |
| `vps/` | sync lato casa del VPS collector 24/7 (deploy in `deploy/vps/`): `pull_vps_data.ps1` (scp → `data/vps_staging/`), `merge_vps_data.py` (merge dedup → copia canonica + heartbeat staleness) | infra (collector 01c/01d) |
| `archive/` | probe chiusi: `xs_*` (cross-sectional KILL), `dev_step0_regime_sigma.py` | morti / dead |

🇮🇹 **Stato delle linee (agg. 2026-06-26):** la **vol** è l'unico segnale PASS OOS (NN batte HAR-RV del 30% QLIKE) → linea pubblicabile. Forecasting/giudici confermati robusti al purged k-fold (2026-06-22). Secondo braccio attivo: **short-vol systematic** (`short_vol_*`), edge VRP strutturale confermato sul backtest storico 2019→2026 (n=2538; audit 2026-06-26: edge **Trending-driven** non Stress, CI block-bootstrap>0, N_eff≈N) — gate vero = live n≥20. Probe **IVS relative-value** = KILL net-of-cost (vivrebbe solo da market-maker). Il **direzionale** non ha alpha OOS a nessun timeframe (corpus di KILL in `STATUS.md`) → resta come negative-control scientifico per il paper "Are price and volume enough?". Stesso repo, stesso motore: vedi `CLAUDE.md` § NOMENCLATURA.

**EN** **Line status (upd. 2026-06-26):** **vol** is the only OOS PASS signal (NN beats HAR-RV by 30% QLIKE) → publishable line. Forecasting/judges confirmed robust under purged k-fold (2026-06-22). Active second arm: **short-vol systematic** (`short_vol_*`), structural VRP edge confirmed on the 2019→2026 historical backtest (n=2538; 2026-06-26 audit: edge **Trending-driven** not Stress, block-bootstrap CIs>0, N_eff≈N) — the real gate is live n≥20. The **IVS relative-value** probe = KILL net-of-cost (would only live as a market-maker). **Directional** has no OOS alpha at any timeframe (KILL corpus in `STATUS.md`) → kept as the scientific negative-control for the "Are price and volume enough?" paper. Same repo, same engine: see `CLAUDE.md` § NOMENCLATURA.

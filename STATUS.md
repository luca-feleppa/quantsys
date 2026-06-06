# STATUS.md — Continuity Log

> Aggiornato alla fine di ogni ciclo di lavoro (direttiva permanente #3 di `CLAUDE.md`).
> Prima di iniziare qualsiasi task: **leggi questo file**. Memoria di lungo periodo dettagliata: `~/.claude/projects/E--quantsys-project/memory/`.

---

## 🕒 Ultimo aggiornamento: 2026-06-06 (distill misurato + roadmap A1.1 hardening live)

## ✅ Sotto-sessione 2026-06-06

- **Student distillato MISURATO (chiude la domanda OOS):** distillato N-HiTS multi-teacher (teacher=iTrans) e backtestato single-arch (test). Risultato **IDENTICO a 4 decimali** al N-HiTS standalone: return −3.57%, Sharpe −28.96, PF 0.21, WR 35%, 17 trade. I `best_model.pt` hanno **hash diversi** (modelli genuinamente diversi) ma stesso esito di trading → conferma empirica che **la distillation non cambia l'OOS** (corr 0.995 resa manifesta). Baseline N-HiTS ripristinato da backup. **La leva NON è la variante di modello.**
- **`run_all.py`:** `--arch` → `--n-ensemble 5` (default, override via flag); `--distill` resta a 1. + fix UTF-8 `--help` (cp1252). Committato e pushato su main (`92d7beb`).
- **Roadmap A1.1 — catch-up contiguo `candle_buffer`** (`scripts/04_live_signals.py`, `warmup()`): via `fetch_klines(start_time=last)` colma il gap tra il bootstrap parquet (può essere vecchio di giorni) e "ora"; mirror legacy reso dedup-safe (fallback). **Verificato con smoke test:** "+2211 candele REST → ultima 2026-06-06 00:12", 2 segnali emessi sul buffer contiguo, zero errori. Risolve il buco temporale che le feature a lookback lungo (ma200m/vp) attraversavano.
- **Fix cp1252 in `scripts/02_train.py`** (3ª occorrenza, aveva causato l'exit 1 "failed" del distill in background — il modello era comunque salvato): reconfigure UTF-8 stdout/stderr in `main()`.
- Recon roadmap A1 (2 subagent): catch-up candele + meccanismo funding. FundingRatePoller (Stage 4.4) resta come miglioria minore (funding cambia ogni 8h, ffill'd → workaround adeguato).
- **B2 esplorato e CHIUSO negativo (2 step de-risk):**
  - **Step 0** (`scripts/dev_step0_regime_sigma.py`, no-training): la mixture-of-universes (σ regime-condizionata) **accantonata** — aggiunge solo +0.0155 nats sopra una ricalibrazione σ globale; R1 Trend resta NLL 2.05 con σ-oracolo = μ-error irriducibile. MA ha scoperto che **σ è ~3× troppo grande** (std(z)=0.37/0.665/0.41, scale globale ottimo 0.33).
  - **Step 0.5** (flag `QUANTSYS_SIGMA_SCALE` in `03_backtest.py`, sweep val): ricalibrare σ verso il basso **peggiora monotonicamente** il backtest (return 4.03%→1.33%, PF 1.88→1.16). La σ larga disabilita di fatto gli stop → hold-to-horizon, migliore per edge debole. **NLL-calibrazione e PnL in conflitto; ottimo trading ≈1.0.** Flag inerte.
  - **Bilancio:** tutti i lever model/backtest-side sono esauriti (distill, ensemble, pesi, rank-harvest, mixture, σ-recal). Restano solo **A (paper-trading = verità forward, pronto)** e **B1 (order-book L2 = informazione nuova, progetto-dati a sé, accantonato)**.

## 🕒 2026-06-05 (BLOCKER #1 Stage 5 CHIUSO + Tier-1 rank esaurito)

## ✅✅ BLOCKER #1 RISOLTO — Stage 5 chiuso (parity live↔training)

- **Stato reale > doc-tracker:** Stage 4.6/4.7 (integrazione in `LiveEngine.__init__` + rimozione `_pad_or_truncate` → `assert window.shape[-1]==104`) erano **già nel codice**. Il LiveEngine carica `EnsembleModel.load_heterogeneous` (riga 934) che **non** espone `predict_with_uncertainty` → `_predict` cade sempre sul ramo **deterministico** (MC-dropout non scatta in produzione).
- **Refactor parity-safe** (`scripts/04_live_signals.py`): estratto `LiveEngine._deterministic_predict(model, window, xm, ps, device)` — nucleo deterministico (forward + denormalizzazione z→raw) **condiviso** da `_predict` (ramo ensemble) e dal parity test → il test esercita il path reale, zero drift.
- **Gate 1 — parity FEATURE:** `tests/test_live_training_parity.py` (5/5) + `99_replay_live_vs_training.py` → **max|Δ|=0.000e+00** su finestra (120,104).
- **Gate 2 — parity SEGNALE (Stage 5, nuovo):** stessi due percorsi feature attraverso `_deterministic_predict` + `SignalGenerator` → **Δμ=0, Δσ=0, side identico**. Codificato in `test_signal_parity_live_vs_offline` + sezione Gate 2 nello script di replay.
- **Test obsoleto aggiornato:** `TestBlocker1Documentation` (in `test_recent_fixes.py`) riscritto — il legacy `LiveFeatureBuffer` (39 feat) è deprecato (solo ATR/sanity), non più il path feature. **Suite: 21/21 verdi.**
- **Smoke test live (Stage 4.10) ESEGUITO 2026-06-05 — SUPERATO.** WS Binance reale connesso e stabile, warmup OK, pipeline 104-feature end-to-end su candele live, segnali emessi in `results/itransformer/live_signals.jsonl` (es. μ=8.3e-4, σ=0.0175 raw, prob_up=0.518, side=NONE — coerente con la scala del parity test). **Lo smoke test ha trovato e risolto 3 bug live reali** (`scripts/04_live_signals.py`): (1) crash `UnicodeEncodeError` cp1252 sul banner `run()` → reconfigure UTF-8 stdout/stderr in `main()`; (2) `AttributeError .dt` su `open_time` dtype misto (WS=epoch-ms int vs bootstrap=Timestamp) → helper `LiveCandleBuffer._norm_ts`; (3) `ValueError mix tz-aware/naive` (parquet tz-aware vs WS tz-naive) → uniformati a tz-naive UTC alla sorgente. Più fix test pre-esistente `test_circuit_breaker_blocks_new_positions` (stato DD irrealistico vs nuova logica recovery) + nuovo `test_circuit_breaker_auto_recovers...`. **Suite full: 125 passed, 1 skipped, 0 failed.**
- **Residuo:** solo avvio paper-trading 2-4 settimane (richiede dati `raw_candles.parquet` aggiornati: nello smoke il buffer bootstrap era fermo al 2026-06-04, gap colmato solo dal WS live).

## ✅ Appena completato (sessione Tier-1 harvest dell'edge ordinale)

- **Fix ① — Cadenza decisionale = orizzonte** (`scripts/03_backtest.py`, env `QUANTSYS_DECISION_CADENCE`, default `0`=off, `"h"`=`forecast_horizon`).
  Razionale: un segnale a orizzonte h tradato ogni candela genera h bet sovrapposti/autocorrelati (breadth effettiva ≪ nominale, IR≈IC·√breadth). Gate causale: nuova entry solo se `i − _last_entry_i ≥ cadence`. **Gli exit (SL/TP/trailing/circuit-breaker) restano ogni candela.** Inerte di default.
- **Fix ② — Esposizione continua rank-based, regime-gated** (env `QUANTSYS_RANK_EXPOSURE=1`/`QUANTSYS_RANK_REGIME`/`QUANTSYS_RANK_BAND`/`QUANTSYS_RANK_MIN_SIGMA`/`QUANTSYS_RANK_WIN`).
  `r`=percentile causale di μ nel buffer ∈[0,1]; `s`=2r−1; no-trade band `|s|<band` (deadzone=isteresi); `conviction=(|s|−band)/(1−band)` → scala il Kelly con continuità (`dist.conviction → RiskManager._size`). Attivo SOLO nel regime target (Quiet di default), NONE altrove. Sostituisce concettualmente il rank-entry **discreto** (che distrugge l'informazione ordinale). Inerte di default.
- **Verificato:** `py_compile` OK; sanity-check matematico del mapping rank→conviction e del gate cadenza (band=0.5 ⇒ trade solo top/bottom 25%; cadence=0 ⇒ baseline invariato).
- **Doc sincronizzate:** `CLAUDE.md` (nuovi flag), `AVVIO.md`+`AVVIO.en.md` (paragrafo harvest edge ordinale).

## 🧪 Esito validazione su VAL — Fix ①② FALLISCONO decisamente (NON promossi)

Confronto su `QUANTSYS_BACKTEST_SPLIT=val`, ensemble eterogeneo 3 archi, `cadence=h`, `rank_exposure=1`, `band=0.5`, `regime=Quiet`, `min_sigma=0`:

| Metrica         | Baseline (clean) | **Fix ①②**        |
|-----------------|------------------|--------------------|
| Total return    | **+4.03%**       | **−2.24%**         |
| Sharpe          | +12.89           | **−45.31**         |
| Profit Factor   | **1.88**         | **0.22**           |
| Win Rate        | 61.0%            | **25.7%**          |
| N° trade        | 41               | 35                 |
| avg_hold        | 188 candele      | 56 candele         |
| Close reasons   | MAX_HOLD 24 / SIGNAL 11 | **SIGNAL 27** / STOP_LOSS 6 |

- **Diagnosi meccanicistica:** il segnale rank come *entry direzionale* è **anti-predittivo su val** (WR 20-27% in entrambi i bucket di vol). Inoltre l'esposizione continua flippa di segno → 27/35 chiusure per **SIGNAL** (avg_hold crolla 188→56): la PnL realizzata è dominata dal path SL/TP/flip, **NON** dal rendimento a orizzonte-30 su cui è misurato lo Spearman +0.13÷0.19. L'edge di rango NON sopravvive alla macchina di realizzazione del trade.
- ⚠ **Nota sulla baseline val (+4%, PF 1.88):** è il lato favorevole — val→test anti-correlano (shift strutturale). Il baseline su **test** resta negativo (−1.77%). Non leggere il +4% come successo.
- **DECISIONE:** Fix ①② **NON promossi**; restano env-flag **inerti/reversibili**. Stato production invariato (i file di test non sono stati toccati; le run val usano suffisso `_val`). Artefatti diagnostici: `models/itransformer/metrics_val_{baseline,fix12}.json`.

**Test di isolamento orizzonte-locked (`QUANTSYS_HORIZON_EXIT=h`, chiusura TEMPORALE pura a 30 candele, bypassa SL/TP/SIGNAL):** `close_reasons={MAX_HOLD:41}`, avg_hold 30.0 esatto. Esito val: return **−0.97%**, PF **0.49**, WR **29.3%** (low_vol bucket WR 18%, 33 trade). Migliora vs Fix①② (PF 0.22→0.49: il path SL/TP/flip era drag aggiuntivo) **ma resta negativo**. ⇒ **anche realizzando la PnL esattamente al rendimento a orizzonte-30, l'edge di rango è anti-predittivo OOS (WR<50%).** Lo Spearman in-sample NON è un edge direzionale tradabile. **Filone rank/soglia ESAURITO** (3ª conferma). Non inseguire l'inversione di segno (overfit val, inversione regime già non-robusta). Artefatto: `metrics_val_rank_hx.json`.

## ✅ Sessione precedente (val-backtest split + esito edge Quiet)

- **Verifica CLAUDE.md vs codice** — allineato. Corretto unico ref sfasato: AMP-off in inference `ensemble.py:170` → **`:275`**.
- **Passo 1 — modalità val-backtest implementata** in `scripts/03_backtest.py`:
  `QUANTSYS_BACKTEST_SPLIT=val|test` (default `test`). Carica `X_{split}/y_{split}/t_{split}` e `X_macro_{split}`,
  allinea OHLCV (`raw_candles.parquet`) e regimi (`regime_probs.parquet`) su `t_{split}`.
  Output **suffissati** per `val` (`metrics_val.json`, `equity_curve_val.npz`, `trades_val.csv`,
  `dashboard_results_val.json`) → la run val **non clobbera** i file production-clean (risolve vecchio problema #4 per il caso val).
- **Passo 2 — backtest Quiet eseguito su val** (`q=0.10, σ≥0.004`, ensemble eterogeneo 3 archi).
- **Documentazione sincronizzata:** `CLAUDE.md` (nuova env var), `AVVIO.md`+`AVVIO.en.md` (esito validazione).

## 🧪 Esito chiave — l'edge Quiet rank-entry NON regge su val (Passo 3)

| Metrica            | Test (2026-06-04) | **Val (held-out, 2026-06-05)** |
|--------------------|-------------------|--------------------------------|
| Return             | −0.74%            | **−0.22%**                     |
| N° trade           | 34                | **13**                         |
| Win Rate           | 47%               | **38.5%**                      |
| Profit Factor      | —                 | **0.84** (<1 = perdita)        |
| Sharpe             | —                 | **−4.04**                      |

- Distribuzione regimi su val: **R0 Quiet 1428 (14%) / R1 Trending 1488 / R2 Stress 7187 (71%)** → periodo dominato da Stress, pochissimo Quiet ⇒ solo 13 trade, **sotto-campione non significativo**.
- Criterio Passo 3 (`return ≥ break-even, segno coerente`) **NON soddisfatto** → l'edge è **overfit del test**.
- **DECISIONE:** rank-entry Quiet **NON promosso** a regola nel `SignalGenerator`; resta env-flag **inerte/reversibile**. Stato production = **ensemble eterogeneo pulito** (≈ iTrans standalone).

## ⚠️ Problemi aperti

1. **L'edge tradabile a soglia/rank non esiste OOS** con questo dataset. L'unico segnale stabile (Spearman Quiet) è di rango ma troppo debole/raro per coprire le fee. Direzioni residue: (a) accumulare trade reali via paper-trading; (b) ripensare l'entry (non |μ|, non rank semplice).
2. **BLOCKER #1 (live) — RISOLTO 2026-06-05 (parity codice chiusa).** Resta solo il residuo OPERATIVO: smoke test live via WS Binance reale (Stage 4.10, non eseguito in questa sessione — richiede connessione) + avvio paper-trading. ⚠ I segnali ora riflettono il backtest, ma **il backtest è negativo OOS** (l'edge a soglia/rank è esaurito): il paper-trading serve ad accumulare trade reali, non c'è aspettativa di Sharpe>0 a priori.
3. **`dashboard_results.json`/`metrics.json` di `models/itransformer`** restano dell'ultima run di **test** (production); i nuovi file `*_val.*` sono separati. Se serve uno stato test production-clean fresco, rilanciare il backtest senza env sperimentali (comando sotto).

## ▶️ Azione esatta da cui ripartire

**Fix ①② validati su val e FALLITI (vedi sopra) → archiviati come flag inerti.** L'unica iterazione *principled* rimasta sul meccanismo, se la si vuole esaurire prima di mollare:
- **Test di isolamento orizzonte-locked:** chiusura puramente TEMPORALE a esattamente `h=30` candele (disabilitare SL/TP/trailing/flip-SIGNAL per i trade rank), così la PnL realizzata coincide col rendimento a orizzonte su cui vive lo Spearman. È l'unico modo di testare *davvero* l'edge di rango isolato dalla macchina di realizzazione. **Prior basso** (WR val 26% suggerisce che la traduzione direzionale del rango è debole/negativa OOS) — non inseguire l'inversione di segno (= overfit del val; l'inversione regime non è robusta, vedi storia).

**BLOCKER #1 parity CHIUSO (vedi sopra).** Prossimi passi possibili, in ordine:
1. **Smoke test live reale (Stage 4.10):** `$env:QUANTSYS_ARCH="itransformer"; python scripts/04_live_signals.py` — verificare connessione WS Binance, warmup buffer, primo segnale entro ~2 min, nessun warning "feature mismatch". (Richiede rete; non eseguito qui.)
2. **Avvio paper-trading** per accumulare trade reali OOS (unico canale di verità non contaminato). Solo dopo, decidere se valgono i 2 fix proposti (AFD / DILATE) — vedi analisi in chat: prima eventualmente riattivare FFD fisso `d=0.4` (parity-safe ora che il buffer è 50k), non l'AFD adattivo.
3. Filone rank/soglia: **chiuso** (3ª conferma OOS). Flag `QUANTSYS_RANK_EXPOSURE`/`QUANTSYS_DECISION_CADENCE`/`QUANTSYS_HORIZON_EXIT` restano inerti/reversibili.

Comando per stato **test production-clean** (da rilanciare prima di chiudere se servono i file di test aggiornati):
`Remove-Item Env:\QUANTSYS_QUIET_RANK_Q,Env:\QUANTSYS_QUIET_MIN_SIGMA,Env:\QUANTSYS_BACKTEST_SPLIT,Env:\QUANTSYS_BACKTEST_SINGLE_ARCH -EA SilentlyContinue; $env:QUANTSYS_ARCH="itransformer"; python scripts/03_backtest.py`

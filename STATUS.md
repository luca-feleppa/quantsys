# STATUS.md — Continuity Log

> Aggiornato alla fine di ogni ciclo di lavoro (direttiva permanente #3 di `CLAUDE.md`).
> Prima di iniziare qualsiasi task: **leggi questo file**. Memoria di lungo periodo dettagliata: `~/.claude/projects/E--quantsys-project/memory/`.

---

## 🕒 Ultimo aggiornamento: 2026-06-06 (probe cross-sectional eseguito → KILL)

## 🔴 PROBE CROSS-SECTIONAL IC — ESEGUITO → VERDETTO **KILL**

- **Implementato e girato** (fan-out 3 subagent + orchestrazione): `quantsys/data/universe.py` (PerpUniverse top-N), `scripts/xs_01_download.py` (16 perp USDT scaricati: ADA AVAX BNB BTC DOGE ENA ETH FIL LINK LTC NEAR SOL SUI WLD XRP ZEC, raw+funding, stesso span), `scripts/xs_02_panel_signals.py` (applica l'ensemble esistente per-asset, denormalize z→raw, grid 30 candele → `data/xs/mu_panel.parquet`, 261.900 righe/16 simboli), `scripts/xs_03_ic_report.py` (Spearman cross-sezionale, sub-periodi non sovrapposti, verdetto pre-registrato). Report: `results/xs/ic_report.json`.
- **Risultato OOS:** mean cross-sectional IC **+0.0138**, t-stat **+1.86** (<2), ICIR 0.035, sub-periodi [+0.043, −0.016, +0.0002, +0.036, +0.005] = **4/5 positivi**. Tradability: spread lordo top-bottom **+1.5 bps/step**, **netto −0.00245/step** (~−43 ann, 13bps/leg).
- **VERDETTO KILL** (gate: |t|≥2 FALSE, ≥4/5 TRUE, net>0 FALSE → serve tutti e 3). **Non costruire il portfolio layer.**
- **Lettura:** μ ha skill di rango cross-sezionale **debolmente positiva** (direzione giusta, NON anti-predittiva come nel single-asset WR 29%), ma l'effetto (~1.5 bps) è **~17× sotto il costo** (~26 bps round-trip long-short). **Il muro è la MAGNITUDINE, non il segno.** Stesso pattern strutturale: rank reale ma troppo debole per le fee. Il de-risk economico ha risparmiato il build M.
- **Nota infra:** GPU 8GB satura con A+xs_02 in parallelo → contesa CUDA. Paper-trading A **FERMATO** per liberare la GPU (xs_02 da 28min-stallo → 28s/simbolo). A NON riavviato (vedi sotto: poco valore ora).

## ▶️ Sotto-sessione 2026-06-06 (paper-trading live + pivot)

- **STRADA A — paper-trading AVVIATO e sano (processo persistente in background).** `$env:QUANTSYS_ARCH="itransformer"; python scripts/04_live_signals.py` (log dedicato `logs/paper_trading_live.log`, log pulito `logs/quantsys_20260606_134026.log`). Warmup OK: ensemble eterogeneo 3 archi, 104 feature, funding 5941 obs, bootstrap 50k + **catch-up A1.1 +2899 candele REST** → ultima 2026-06-06 11:40, **WebSocket connesso** `btcusdt@kline_1m`, equity paper $10.000, feature in calcolo su candele live. Segnali in `results/itransformer/live_signals.jsonl`. ⚠ Se la macchina riavvia, **va rilanciato** (non è un servizio). Aspettativa onesta: piatto/negativo — è validazione forward + readiness, non scommessa Sharpe.
- **PIVOT — fan-out di 4 subagent (studi di fattibilità read-only).** Sintesi ranked (dettaglio completo in memoria `pivot_fanout_2026_06_06`):
  1. **Cross-sectional multi-asset** — **prior tradabile più alto + test più economico.** Long-short rank market-neutral top-N perp: usa solo l'ordinamento (= il nostro Spearman) e cancella il beta comune. **De-risk quasi-gratis: misurare la Spearman cross-sezionale di μ PRIMA di costruire il portfolio layer** (se ≈0 → stop). Backfillable ora. Effort M. Success: net Sharpe ≥1.0 test, +in ≥4/5 sub-periodi, fees<40% gross.
  2. **Timeframe → 1h** — cost/σ 1m=1.9-3.3× → 1h~0.25-0.42×. Serve re-download multi-anno + re-tune lookback hardcoded a 1m. Effort M. Rischio: cost-fragile; anti-correlazione val→test è del *metodo*.
  3. **Target → volatilità** — **PREDICIBILE MA NON TRADABILE** sul perimetro attuale (no strumento; Deribit = progetto > B1). Valore: esperimento S che chiude B2 (vol vs HAR-RV, QLIKE) + jump/no-trade gate difensivo per A.
  4. **Asset class → ES 1m** — inefficienza session-mechanical (documentata, causale). Dati ~free (Databento). Rewrite data-layer + ~2/3 riuso. Effort M. Rischio: già HFT-arbitraggiato; leakage roll.
- **B1 (order-book L2)** resta l'asse "informazione nuova" non spremuto, ortogonale (memoria `future_orderbook_l2`).

## 🕒 2026-06-06 (merge doc bilingue single-file)

## ✅ Sotto-sessione 2026-06-06 (doc bilingue fuse)

- **Fusi i 4 file `.md` gemelli IT/EN in un unico file bilingue paragrafo-per-paragrafo.** Coppie fuse → file base, gemelli eliminati: `AVVIO.md`(+`AVVIO.en.md`), `README.md`(+`README.it.md`), `TEORIA.md`(+`TEORIA.en.md`), `docs/MODEL_IMPROVEMENTS.md`(+`.en.md`). Formato: heading bilingue `IT · EN`; corpo con paragrafo IT (prefisso `🇮🇹`) seguito da EN (prefisso `**EN**`); blocchi di codice emessi una volta, tabelle duplicate IT/EN; puntatori "versione X in Y" rimossi; cross-reference `.en.md`/`.it.md` reindirizzati ai file base (le menzioni storiche dentro MODEL_IMPROVEMENTS restano come record).
- **Metodo:** script one-shot `scripts/_merge_bilingual.py` (poi rimosso) con allineamento **sezione-per-sezione via difflib** su chiave heading language-neutral (numeri/date/emoji/`Step X` + ancora primaria `Stage|Phase|Fase|Step|Fix #N`) e **resync corpo** via ancore inline-code/numeri. Validato con check di adiacenza marker: AVVIO 38/38 e README 26/26 perfetti; TEORIA ha 1 blocco IT-only legittimo (`Perché T=120`, assente nell'EN); MODEL ha la sola sezione IT-only `RESUME 2026-06-04` + Stage 4 con heading `✅ COMPLETATO · 🚧 IN PROGRESS` (l'EN era stale, drift pre-esistente reso esplicito).
- **Doc-convention aggiornata in `CLAUDE.md`** (direttiva #2 + nomenclatura): single-file bilingue, NON ricreare i gemelli.
- ⚠ **Drift residuo da sanare (non bloccante):** l'EN di alcune sezioni era più vecchio dell'IT (evidente in `MODEL_IMPROVEMENTS` Stage 4). Ora visibile nello stesso file → riallineare le due lingue alla prossima modifica di quelle sezioni.

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

**STATO al 2026-06-06: TUTTI i lever model/backtest-side esauriti e dimostrati negativi OOS** (distill≡baseline, ensemble corr 0.995, pesi dinamici anti-correlano, rank-harvest fallito, mixture-of-universes non vale, ricalibrazione σ peggiora). L'infrastruttura è SOLIDA: BLOCKER #1 chiuso (parity bit-perfect), live engine robusto (catch-up A1.1, smoke test superato, 3 cp1252 fixati), `run_all.py` aggiornato (`--arch`→5 / `--distill`→1). **Non restano tweak di modello con prior non-nullo.**

**▶️ STATO 2026-06-06 (aggiornato):** Strada A avviata e poi **fermata** (poco valore validare forward un modello in pivot + serviva la GPU). Pivot **studiato** (fan-out 4 assi). **Probe cross-sectional IC ESEGUITO → KILL** (vedi sezione in cima). **SESSIONE IN PAUSA dopo commit.**

**▶️ AZIONE ESATTA ALLA RIPRESA — scegliere fra le 3 strade residue (cross-sectional già escluso):**
La diagnosi del KILL è precisa: il rank di μ è cross-sezionalmente **debolmente positivo (segno giusto)** ma **~17× sotto i costi** → **il muro è la MAGNITUDINE, non il segno.** Questo ordina le opzioni:
1. **★ TIMEFRAME → 1h (RACCOMANDATO — attacca direttamente la diagnosi):** a 1h il rapporto cost/σ scende da ~1.9-3.3× (1m) a ~0.25-0.42× perché il movimento di barra cresce ~√Δt e il costo è fisso. Passi: `config/default.yaml` `interval:1h` + `start_time` multi-anno (~2019, il resample del file 381g è troppo sottile: 1h~9k barre); **re-tune dei lookback hardcoded a 1m** in `quantsys/features/__init__.py` (`_structural_features` ~l.536/547, `price_vs_ma200m` ~l.559 `rolling(200)`=minuti, `_funding_features` `rolling(1440)`→`rolling(24)`, `features.windows`) e il regime-clock in `quantsys/macro/regime.py` (`_build_btc_hourly_df` resample 1m→1h diventa no-op); rebuild dataset; train iTrans single-arch; backtest cost-aware OOS col filtro magnitudo **già nel repo** (`min_expected_ret`/`min_snr`) a 13 E 23 bps. Success: Sharpe≥1.0, PF≥1.3, ≥80 trade, net>0 a entrambi i costi. Effort M ~2-3g. Rischio: cost-fragile + l'anti-corr val→test è del *metodo*.
2. **VOL-S (ortogonale, costo S):** re-target a RV (one-line `target=Σlog_ret²` in `FeatureBuilder._returns` ~l.137), baseline HAR-RV+GJR-GARCH, giudice **QLIKE** OOS; success = batte HAR-RV ≥5% su test. Chiude scientificamente B2 + estrae il **jump/no-trade gate difensivo**. NON è alpha (vol predicibile ma non tradabile sul perimetro spot/perp).
3. **ES 1m (esci dalla crypto, effort M ~1-2 sett.):** porta l'architettura su index futures (inefficienza session-mechanical, documentata/causale). Dati ~free per le barre (Databento PAYG); rewrite data-layer + ~5-8 feature (drop funding, VWAP reset per sessione, add time-of-session/overnight-gap). Rischio: già HFT-arbitraggiato; leakage del roll continuous-contract.

**B1 order-book L2** resta l'asse informazione-NUOVA non spremuto, ortogonale a tutti e 3 (memoria `future_orderbook_l2`).
**Riavvio paper-trading A** (`$env:QUANTSYS_ARCH="itransformer"; python scripts/04_live_signals.py`): solo SE si decide di validare forward un modello (NON ora; la readiness è già verificata). ⚠ GPU 8GB NON regge A + training/inferenza in parallelo.

Memorie di riferimento per riprendere: `pivot_fanout_2026_06_06` (4 assi + esito KILL), `roadmap_post_blocker1_2026_06`, `future_orderbook_l2`, `mixture_of_universes_design`.

Comando stato **test production-clean** (azzera TUTTI gli env sperimentali prima di un backtest "production"):
`Remove-Item Env:\QUANTSYS_RANK_EXPOSURE,Env:\QUANTSYS_DECISION_CADENCE,Env:\QUANTSYS_HORIZON_EXIT,Env:\QUANTSYS_SIGMA_SCALE,Env:\QUANTSYS_QUIET_RANK_Q,Env:\QUANTSYS_QUIET_MIN_SIGMA,Env:\QUANTSYS_BACKTEST_SPLIT,Env:\QUANTSYS_BACKTEST_SINGLE_ARCH -EA SilentlyContinue; $env:QUANTSYS_ARCH="itransformer"; python scripts/03_backtest.py`

# ROADMAP — Vol Book v2 · IT · EN

> 🇮🇹 Backlog architetturale e strategico emerso dall'audit anti-overfit + analisi book futures/opzioni (sessione 2026-07-07). Ogni item segue il protocollo sperimentale di `CLAUDE.md` (gate pre-registrato in `STATUS.md`, val-first, env-flag inerte, esito scritto anche se negativo).
> **EN** Architectural and strategic backlog from the anti-overfit audit + futures/options book analysis (2026-07-07 session). Every item follows the experimental protocol in `CLAUDE.md` (pre-registered gate in `STATUS.md`, val-first, inert env-flag, outcome written even if negative).

---

## 0 · Verdetti dell'audit anti-overfit (chiusi, non ri-testare) · Anti-overfit audit verdicts (closed, do not re-test)

🇮🇹 Audit del 2026-07-07 sulle strategie di generalizzazione (purged CV, FrAug, RevIN, iTransformer/TCN-Mamba/N-HiTS internals). Conclusioni **negative o già-presenti** — registrate qui come vaccino anti re-test:
**EN** 2026-07-07 audit of generalization strategies (purged CV, FrAug, RevIN, iTransformer/TCN-Mamba/N-HiTS internals). **Negative or already-present** conclusions — recorded here as re-test vaccine:

| Strategia · Strategy | Verdetto · Verdict |
|---|---|
| Purged k-fold + embargo | ✅ già presente e corretto (`walk_forward_folds`, embargo 168 ≥ T+h=150) · already present and correct |
| FrAug / freq-domain augmentation | ❌ NO in forma canonica: le 104 feature hanno dipendenze deterministiche incrociate → il masking spettrale per-canale produce vettori cross-feature incoerenti, corrompe l'attention inter-feature di iTransformer · NO in canonical form: per-channel spectral masking breaks cross-feature consistency |
| RevIN sulla linea vol | ❌ lasciare OFF: il livello locale di vol È il segnale (persistence HAR); denorm inconsistente col target `log_rv` (scala colonna return ≠ scala log-RV fwd) · keep OFF: local vol level IS the signal; denorm inconsistent with `log_rv` target |
| ProtoNorm / LCD | ❌ skip — raffinamenti multi-entità, ROI trascurabile su serie singola + RobustScaler + regime detector · skip, negligible ROI on a single series |
| MC-Dropout nel path live | ❌ non riattivare: la legge varianza totale dell'ensemble è uno stimatore epistemico superiore E preserva la parity bit-perfect · do not re-enable: ensemble total-variance law is superior and preserves parity |
| Interpolazione gerarchica N-HiTS | ❌ non applicabile: target scalare aggregato (Σ log-ret su h), nessuna traiettoria multi-step da interpolare · not applicable: scalar aggregated target, no multi-step trajectory |
| RSI come feature | ⚠ NON esiste nel feature set (rimosso: ridondante con vol_std+lag_ret, `features/__init__.py:431`) · does NOT exist in the feature set (pruned) |

---

## A · Miglioramenti modello/infra (priorità = prior × effort⁻¹) · Model/infra improvements

### A1 🔴 Delta-hedge del book opzioni col perpetual Deribit · Options-book delta-hedge via Deribit perpetual
🇮🇹 **Il singolo upgrade più giustificato del progetto** — vedi sezione B. Effort M. **Post-gate n≥20** (non toccare `04b` a campione aperto).
**EN** The single most justified upgrade — see section B. Effort M. **Post-gate n≥20** (do not touch `04b` mid-sample).

### A2 🔴 Output distribuzionale su log-RV (quantile/CRPS, coda destra) · Distributional log-RV output (quantile/CRPS, right tail)
> ⚫ **ESITO 2026-07-10 — A2-CONFORME FAIL → FILONE A2 CHIUSO DEL TUTTO.** Ricalibrazione split-conformal (gate pre-registrato: shift additivo per livello su prefisso val, giudizio su suffisso, NN e HAR trattati identicamente; `dev_vols_quantile_judge.py --conformal`): FAIL su coverage (q50 0.676 — il bias di locazione del NN deriva nel tempo, lo shift costante non basta; HAR-conf 0.5305 quasi perfetta) E su pinball q90 (NN-conf 0.1387 > HAR-conf 0.1334). **HAR-q90 = stimatore di coda definitivo per il sizing v2.** Diagnostica: il NN domina centro/coda sinistra (q10/q25/q50) — coerente col PASS QLIKE su μ; la coda destra (l'errore costoso short-vol) resta di HAR. Dettaglio: `STATUS.md` 2026-07-10. · **OUTCOME 2026-07-10 — A2-CONFORMAL FAIL → A2 LINE FULLY CLOSED.** Split-conformal recalibration (pre-registered; NN and HAR treated identically): FAIL on coverage (NN location bias drifts over time; constant shift insufficient) AND on q90 pinball (0.1387 vs 0.1334). **HAR-q90 = definitive tail estimator for v2 sizing.** NN dominates center/left tail — consistent with the QLIKE PASS on μ; the costly right tail belongs to HAR.
> ⚫ **ESITO 2026-07-08 — A2a FAIL, A2b MORTO.** La testa quantile era GIÀ nei checkpoint PASS (`loss_type: quantile`, q10-q90): estratta e giudicata (`scripts/vol/dev_vols_quantile_judge.py`, gate pre-registrato) → coverage sopra target a TUTTI i livelli (q50→0.73, q90→0.97: distribuzione shiftata in alto) e pinball q90 NN **perde** da HAR+quantili-residui (0.160 vs 0.144) → il retrain q95 non è giustificato. Per il sizing v2: HAR-q90 (vince oggi) o ricalibrazione conforme da pre-registrare. Dettaglio in `STATUS.md` 2026-07-08. · **OUTCOME 2026-07-08 — A2a FAIL, A2b DEAD.** Quantile head was already in the PASS checkpoints; extracted and judged → coverage above target at every level (upward-shifted distribution) and NN q90 pinball **loses** to HAR+residual-quantiles → q95 retrain unjustified. For v2 sizing: HAR-q90 (currently wins) or a pre-registered conformal recalibration.
🇮🇹 Il rischio short-vol vive nel quantile alto della RV forward (kurtosis residui ≈19.7). La testa quantile esiste già in tutte le arch (`loss_type=quantile`): abilitarla sul target `log_rv` dà q90/q95 predetti → sizing e kill-switch guidati dalla coda, non da σ t-Student. Giudice da estendere: pinball + coverage empirica. Effort S, val-first (`QUANTSYS_VOLS_SPLIT=val`).
**EN** Short-vol risk lives in the upper quantile of forward RV (residual kurtosis ≈19.7). The quantile head already exists in every arch: enabling it on `log_rv` yields predicted q90/q95 → tail-driven sizing and kill-switch instead of t-Student σ. Judge extension: pinball + empirical coverage. Effort S, val-first.

### A3 🟠 Mixture-of-universes (design B2 in memoria) — prior RIALZATO · prior RAISED
🇮🇹 Backbone condiviso + 3 teste-regime, soft-gate causale su `regime_prob_0/1/2` (già persistite in `data/regime_probs.parquet`). Era accantonato ("migliora σ, non la direzione"); l'audit 2026-06-26 ha stabilito che l'edge short-vol è **Trending-driven** → la calibrazione σ regime-condizionata è ora direttamente monetizzabile. Forward contract invariato. Effort M. Design: memoria `mixture_of_universes_design`.
**EN** Shared backbone + 3 regime heads, causal soft-gate on `regime_prob_0/1/2`. Was shelved ("improves σ, not direction"); the 2026-06-26 audit showed the short-vol edge is **Trending-driven** → regime-conditional σ calibration is now directly monetizable. Forward contract unchanged. Effort M.

### A4 🟠 Feature jump/continuous HAR-CJ come INPUT · HAR-CJ jump/continuous features as INPUT
🇮🇹 Il probe semivarianza è fallito come **target** (momenti dispari); la decomposizione bipower/jump variation come **input** è ortogonale al kill: componenti C e J hanno persistence diverse (Andersen–Bollerslev–Diebold) e separarle migliora il forecast del momento pari. Causale, calcolabile dal path 1h. Effort S ma richiede rigen dataset + retrain → **accodare a un retrain già pianificato**.
**EN** The semivariance probe failed as a **target** (odd moments); bipower/jump-variation decomposition as **input** is orthogonal to that kill: C and J components have different persistence and separating them improves even-moment forecasts. Causal, computable from the 1h path. Effort S but needs dataset regen + retrain → **queue behind an already-planned retrain**.

### A5 🟡 Pesi ensemble per-QLIKE (linea vol) · QLIKE-based ensemble weights (vol line)
> ⚫ **ESITO 2026-07-08 — FAIL (variante 5-seed).** Pesi 1/QLIKE fittati su 1ª metà val, valutati su 2ª (`scripts/vol/dev_vols_member_weights.py`, gate pre-registrato ≤0.97): ratio 0.9925 (+0.75%, sotto soglia), pesi quasi-uniformi [0.168-0.220], best-single peggio dell'ensemble → uniformi già ottimali (diversità intra-arch = rumore di seed). Variante cross-arch subordinata al distill 5-seed (prior basso). · **OUTCOME 2026-07-08 — FAIL (5-seed variant).** 1/QLIKE weights fit on 1st val half, judged on 2nd: ratio 0.9925 (below the ≤0.97 gate), near-uniform weights, best-single worse than the ensemble → uniform already optimal (intra-arch diversity = seed noise). Cross-arch variant subordinate to the 5-seed distill (low prior).
🇮🇹 I pesi dinamici usano inverse-NLL; per la vol il giudice canonico è QLIKE (robusto alla proxy RV, penalizza asimmetricamente l'under-prediction = l'errore costoso per lo short-vol). Sostituire la metrica nel blend per la linea vol. Effort S.
**EN** Dynamic weights use inverse-NLL; for vol the canonical judge is QLIKE (proxy-robust, asymmetric penalty on under-prediction = the costly error for short-vol). Swap the blend metric for the vol line. Effort S.

### A6 🟡 Realismo esecuzione in `04b_vol_paper.py` — **PRE-gate, solo colonne diagnostiche** · Execution realism — PRE-gate, diagnostic columns only
> ✅ **IMPLEMENTATO 2026-07-08** — `log_exec_diag()` in `04b`, output `results/vol_paper/exec_diag.jsonl` (1 riga/tick: bid/ask/mark/IV/greeks per leg + delta netto + half-spread; posizione aperta → leg in essere, flat → straddle ATM ipotetico). Fail-soft, regola pre-registrata intatta, processo live riavviato col nuovo codice. · **IMPLEMENTED 2026-07-08** — fail-soft, pre-registered rule untouched, live process restarted.
🇮🇹 Fill al mark = zero-spread, e l'IVS è morto sullo spread. **Senza toccare le costanti pre-registrate:** loggare bid/ask reali dal book + delta teorico della posizione a ogni tick orario → a gate chiuso il PnL si rilegge net-of-half-spread e si stima offline il valore dell'hedge (alimenta A1). Effort S. ✅ Unico item eseguibile SUBITO.
**EN** Mark-price fills = zero spread, and IVS died on the spread. **Without touching pre-registered constants:** log real bid/ask + theoretical position delta at every hourly tick → post-gate, re-read PnL net-of-half-spread and estimate hedge value offline (feeds A1). Effort S. ✅ The only item executable NOW.

### A7 🟡 Risk layer greeks-aware · Greeks-aware risk layer
🇮🇹 Il `RiskManager` è delta-one (Kelly, SL ATR, CB su DD nozionale). Il book opzioni richiede: cap di **vega netta** per posizione, circuit breaker su vega-loss mark-to-market, margin simulation Deribit (inverse/portfolio margin). Necessario solo quando il sizing passa da 1 contratto fisso a Kelly-su-edge. Effort M, post-gate.
**EN** The `RiskManager` is delta-one. An options book needs: net-**vega** cap per position, mark-to-market vega-loss circuit breaker, Deribit margin simulation (inverse/portfolio margin). Needed only when sizing moves from fixed 1 contract to Kelly-on-edge. Effort M, post-gate.

### A8 🟢 Quick-win training già cablati (solo config) · Already-wired training quick-wins (config-only)
🇮🇹 Entrambi misurabili sul giudice QLIKE dove val→test è coerente; gate pre-registrato prima di girare:
**EN** Both measurable on the QLIKE judge where val→test is coherent; pre-register the gate before running:
- `mixup_alpha: 0.2` — implementato in `02_train.py` (mixup temporale, lam biasato verso l'originale), oggi 0.0. Unica augmentation che preserva la coerenza cross-feature per costruzione; su target log-RV il mix di y = media geometrica delle RV (coerente). · Implemented, currently 0.0; the only augmentation that preserves cross-feature consistency by construction.
- `drop_path_rate: 0.05–0.1` — DropPath con scheduling lineare già cablato in `QuantiTransformer`/TCN-Mamba, oggi 0.0. · Already wired, currently 0.0.

### A9 🟢 Blocco MaxPool parallelo in N-HiTS (componente jump) · Parallel MaxPool block in N-HiTS (jump component)
🇮🇹 Il codice usa AvgPool (passa-basso, deviazione dal paper che usa MaxPool). Per il target di vol gli spike sono informativi (jump della RV): un blocco MaxPool in parallelo cattura la componente jump. Effort S, prior moderato, val-first sul QLIKE.
**EN** Code uses AvgPool (low-pass; paper uses MaxPool). For the vol target spikes are informative (RV jumps): a parallel MaxPool block captures the jump component. Effort S, moderate prior, val-first on QLIKE.

### A10 🟡 Sparsity dell'attention (solo se A8-DropPath non basta) · Attention sparsity (only if A8-DropPath is insufficient)
🇮🇹 Penalità entropica sulle mappe di attention di `iTransformerLayer` (F=104 token, correlazioni spurie). ⚠ Costo: materializzare la matrice rinuncia a Flash-Attention su quel path (stabilità fp16). Priorità: DropPath ≫ sparsity. Effort M.
**EN** Entropy penalty on `iTransformerLayer` attention maps. ⚠ Cost: materializing the matrix forfeits Flash-Attention on that path (fp16 stability). Priority: DropPath ≫ sparsity. Effort M.

### A11 🟢 Attribution PnL gamma/theta/vega per trade (offline, PRE-gate OK) · Per-trade gamma/theta/vega PnL attribution (offline, PRE-gate OK)
> ✅ **IMPLEMENTATO 2026-07-14** — `scripts/vol/pnl_attribution.py`: decomposizione ex-post del PnL di ogni trade chiuso in delta/gamma/theta/vega/residuo dalla serie A6 (`exec_diag.jsonl`), con coverage dichiarata dei tick mancanti. Solo lettura, zero impatto sul forward test. · **IMPLEMENTED 2026-07-14** — read-only, zero impact on the forward test.
🇮🇹 Il PnL dello straddle delta-hedgiato è ∫½ΓS²(σ²_real−σ²_impl)dt + repricing vega + fee. L'attribution per-intervallo (Δ·ΔS + ½Γ·ΔS² + ν·Δiv + Θ·Δt vs ΔV effettivo) verifica che i trade vincenti vincano **per il motivo giusto** (RV vs IV, non direzione/vega). Rafforza l'interpretazione del gate n≥20 senza toccarlo. Effort S.
**EN** Delta-hedged straddle PnL is ∫½ΓS²(σ²_real−σ²_impl)dt + vega repricing + fees. Per-interval attribution (Δ·ΔS + ½Γ·ΔS² + ν·Δiv + Θ·Δt vs realized ΔV) verifies winning trades win **for the right reason** (RV vs IV, not direction/vega). Sharpens the n≥20 gate's interpretation without touching it. Effort S.

### A12 🟠 Banda di hedge gamma-scalata (Whalley–Wilmott) · Gamma-scaled hedge band (Whalley–Wilmott)
> ✅ **CODICE 2026-07-14, INERTE** — `--hedge-band-mode ww` + `--hedge-ww-lambda` in `04b` (default `fixed` = design attuale bit-identico). Attivazione: DENTRO la pre-registrazione hedged-vs-unhedged, dopo confronto offline fixed-vs-ww sul dry-run A6. · **CODE 2026-07-14, INERT** — activate INSIDE the hedged-vs-unhedged pre-registration, after the offline fixed-vs-ww comparison on the A6 dry-run.
🇮🇹 La no-trade band ottimale sotto costi proporzionali scala con Γ^(2/3) (half-width asintotica W–W 1997: `(3·k·S·Γ²/2λ)^(1/3)`). Sulle dailies il Γ ATM cresce di ordini di grandezza verso scadenza → banda fissa = churn a inizio vita E delta nudo a fine vita. Fail-soft: greeks assenti → fallback alla banda fissa; clip a [band/4, 4·band]. Effort S (fatto), λ da congelare alla pre-registrazione.
**EN** The optimal no-trade band under proportional costs scales with Γ^(2/3) (W–W 1997 asymptotic half-width `(3·k·S·Γ²/2λ)^(1/3)`). On dailies ATM Γ grows by orders of magnitude toward expiry → a fixed band churns early AND leaves naked delta late. Fail-soft: missing greeks → fixed-band fallback; clipped to [band/4, 4·band]. λ frozen at pre-registration.

### A13 🟠 Pin risk: early-close T−x + gamma cap di libro · Pin risk: T−x early-close + book gamma cap
> ✅ **CODICE 2026-07-14, INERTE** — (a) `--pin-close-hours`/`--pin-close-band` in `04b` (default OFF = hold-to-expiry pre-registrato intatto); (b) `max_net_gamma` in `GreeksLimits` (default None = nessun cap). Attivazione: pre-registrazione sizing v2, post-gate n≥20. · **CODE 2026-07-14, INERT** — activate at the v2 sizing pre-registration, post-gate n≥20.
🇮🇹 Nelle ultime ore il Γ ATM esplode e il PnL marginale è pin-risk (posizione di S vs K), non più la bet RV-vs-IV. (a) early-close quando restano ≤x ore E |S−K|/S è nella pin region: incassa, rinuncia alle ore a Sharpe peggiore; (b) cap sul Γ netto di libro (pattern `_cap_scale`) contro convessità corta concentrata a scadenza — rilevante per il braccio short-vol. Entrambi cambiano regole pre-registrate → SOLO v2.
**EN** In the final hours ATM Γ explodes and marginal PnL is pin risk (S vs K), no longer the RV-vs-IV bet. (a) early-close when ≤x hours remain AND |S−K|/S is inside the pin region; (b) net book-Γ cap (`_cap_scale` pattern) against expiry-concentrated short convexity — relevant for the short-vol arm. Both alter pre-registered rules → v2 ONLY.

### A14 🟡 Sizing vega-normalizzato · Vega-normalized sizing
> ✅ **CODICE 2026-07-14, INERTE** — `--size-mode vega` + `--size-vega-target` in `04b` (default `contracts` = 1 contratto fisso pre-registrato). Attivazione: pre-registrazione sizing v2 (con A7 nel critical path), post-gate n≥20. · **CODE 2026-07-14, INERT** — activate at the v2 sizing pre-registration (with A7 on the critical path), post-gate n≥20.
🇮🇹 A size fissa l'esposizione di vol per trade varia col tenor all'entry (~22-30h) → bet non uniformi in spazio-vol, statistica del gate sporca. Normalizzare per la vega di struttura all'entry (`amount = target_vega/Σν`, step 0.1, cap fail-safe) rende ogni trade la stessa bet. Col ladder multi-expiry ([[idea in memoria]]) diventa quasi obbligatorio: senza, i rung corti dominano il Γ di libro. Fallback fail-soft a size fissa se greeks assenti.
**EN** At fixed size, per-trade vol exposure varies with entry tenor (~22-30h) → non-uniform bets in vol space, noisier gate statistics. Normalizing by entry structure vega (`amount = target_vega/Σν`, 0.1 step, fail-safe cap) makes every trade the same bet. With the multi-expiry ladder it becomes near-mandatory: without it, short rungs dominate book Γ. Fail-soft fallback to fixed size when greeks are missing.

---

## B · Verdetto strategico: book a due strumenti (futures + opzioni) · Strategic verdict: two-instrument book

### B1 ❌ Futures direzionali con leva (opzioni a copertura) — NO, dimostrato · Leveraged directional futures (options as hedge) — NO, proven
🇮🇹 L'edge direzionale è stato falsificato con rigore: 1m (muro costi), 1h (4/4 gate falliti), soglia/rango/cadenza/esposizione-continua/semivarianza — tutti FAIL OOS. Sintesi: **momenti dispari non generalizzano, pari sì**. La leva moltiplica un edge ≈0 e costi certi: E[PnL] = leva·(0 − costi) < 0, monotono nella leva. La copertura con opzioni **paga il VRP** che l'altro braccio raccoglie → book internamente contraddittorio. Filone chiuso.
**EN** Directional edge falsified rigorously across every probe — odd moments don't generalize OOS. Leverage multiplies a ≈0 edge and certain costs: E[PnL] = leverage·(0 − costs) < 0, monotone in leverage. Option protection **pays the VRP** the other arm harvests → internally contradictory book. Line closed.

### B2 ✅ Futures come delta-hedge del book opzioni — SÌ · Futures as options-book delta-hedge — YES
🇮🇹 Il perp Deribit non è una bet: isola il PnL di volatilità. Risultato di robustezza (El Karoui / Carr–Madan), posizione delta-hedgiata:

```
PnL = ∫ ½·Γ_t·S_t²·(σ²_impl − σ²_real) dt
```

= puro harvest di (IV²−RV²), esattamente la quantità che il NN predice (+30% QLIKE vs HAR). Oggi hold-to-expiry senza hedge: PnL = premio − |S_T − K|, contaminato dal termine direzionale impredicibile. Benefici: (1) varianza per trade ↓↓ → il gate n≥20 acquista potere statistico; (2) il sospetto "edge Trending-driven" diventa testabile pulito (il termine di trend sparisce per costruzione); (3) stessa venue → margine incrociato, spread perp ~1bp. La "leva" = solo efficienza di margine, non driver di rendimento.
**EN** The Deribit perp is not a bet: it isolates vol PnL (Carr–Madan robustness result above) = pure (IV²−RV²) harvest, exactly what the NN predicts. Current unhedged hold-to-expiry PnL is contaminated by the unpredictable directional term. Benefits: per-trade variance ↓↓ (gate n≥20 gains power); the "Trending-driven edge" hypothesis becomes cleanly testable; same venue → cross-margin. "Leverage" = margin efficiency only.

🇮🇹 **Costi/caveat da mettere nel design:** hedging discreto orario → errore O(Γ·(ΔS)²) su ~30 ribilanciamenti/tenor, da misurare; funding del perp nel carry (dati funding già raccolti — riuso); opzioni Deribit inverse/coin-settled → delta in termini BTC, attenzione alla convenzione; fee perp per ribilanciamento (trascurabili a 1 contratto, da contare net al sizing vero).
**EN** **Design caveats:** hourly discrete hedging → O(Γ·(ΔS)²) error over ~30 rebalances/tenor, to be measured; perp funding in carry (funding data already collected — reuse); Deribit inverse options → BTC-terms delta convention; perp fees per rebalance.

### B3 Sequencing (vincolante) · Sequencing (binding)
🇮🇹
1. **ORA:** A6 — logging bid/ask + delta teorico in `04b` (solo colonne diagnostiche, costanti pre-registrate INTATTE, il gate n≥20 chiude sul design congelato).
2. **POST-GATE (~metà luglio):** `04b` v2 con leg delta-hedge sul perp testnet; **pre-registrare** il confronto hedged vs unhedged (Sharpe per trade, PnL net-of-funding) — includendo la scelta banda fixed-vs-ww (A12, confronto offline sul dry-run A6). In parallelo: A2 (quantili RV) + A5 (pesi QLIKE) → alimentano il sizing v2; poi A7 (risk greeks-aware) + A13 (pin-close/gamma cap) + A14 (sizing vega-normalizzato) quando il sizing diventa Kelly-su-edge. A11 (attribution) è offline: usabile in QUALSIASI momento.
3. **Opportunistici (a retrain pianificato):** A4 (HAR-CJ input); quick-win A8 quando si rigira un training comunque; A3 (mixture-of-universes) come esperimento modello a sé con gate QLIKE pre-registrato.

**EN**
1. **NOW:** A6 — bid/ask + theoretical-delta logging in `04b` (diagnostic columns only, pre-registered constants UNTOUCHED, the n≥20 gate closes on the frozen design).
2. **POST-GATE (~mid July):** `04b` v2 with a perp delta-hedge leg on testnet; **pre-register** hedged vs unhedged comparison (per-trade Sharpe, PnL net-of-funding) — including the fixed-vs-ww band choice (A12, offline comparison on the A6 dry-run). In parallel: A2 (RV quantiles) + A5 (QLIKE weights) → feed v2 sizing; then A7 + A13 (pin-close/gamma cap) + A14 (vega-normalized sizing) when sizing becomes Kelly-on-edge. A11 (attribution) is offline: usable at ANY time.
3. **Opportunistic (at a planned retrain):** A4 (HAR-CJ inputs); A8 quick-wins whenever a training run happens anyway; A3 (mixture-of-universes) as a standalone model experiment with a pre-registered QLIKE gate.

🇮🇹 **Sintesi:** la parte finale del progetto diventa un book a due strumenti con i ruoli **opzioni = veicolo dell'edge (vol), futures = copertura che lo purifica** — mai il contrario.
**EN** **Summary:** the project's final stage becomes a two-instrument book where **options = edge vehicle (vol), futures = the hedge that purifies it** — never the reverse.

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
🇮🇹 Il rischio short-vol vive nel quantile alto della RV forward (kurtosis residui ≈19.7). La testa quantile esiste già in tutte le arch (`loss_type=quantile`): abilitarla sul target `log_rv` dà q90/q95 predetti → sizing e kill-switch guidati dalla coda, non da σ t-Student. Giudice da estendere: pinball + coverage empirica. Effort S, val-first (`QUANTSYS_VOLS_SPLIT=val`).
**EN** Short-vol risk lives in the upper quantile of forward RV (residual kurtosis ≈19.7). The quantile head already exists in every arch: enabling it on `log_rv` yields predicted q90/q95 → tail-driven sizing and kill-switch instead of t-Student σ. Judge extension: pinball + empirical coverage. Effort S, val-first.

### A3 🟠 Mixture-of-universes (design B2 in memoria) — prior RIALZATO · prior RAISED
🇮🇹 Backbone condiviso + 3 teste-regime, soft-gate causale su `regime_prob_0/1/2` (già persistite in `data/regime_probs.parquet`). Era accantonato ("migliora σ, non la direzione"); l'audit 2026-06-26 ha stabilito che l'edge short-vol è **Trending-driven** → la calibrazione σ regime-condizionata è ora direttamente monetizzabile. Forward contract invariato. Effort M. Design: memoria `mixture_of_universes_design`.
**EN** Shared backbone + 3 regime heads, causal soft-gate on `regime_prob_0/1/2`. Was shelved ("improves σ, not direction"); the 2026-06-26 audit showed the short-vol edge is **Trending-driven** → regime-conditional σ calibration is now directly monetizable. Forward contract unchanged. Effort M.

### A4 🟠 Feature jump/continuous HAR-CJ come INPUT · HAR-CJ jump/continuous features as INPUT
🇮🇹 Il probe semivarianza è fallito come **target** (momenti dispari); la decomposizione bipower/jump variation come **input** è ortogonale al kill: componenti C e J hanno persistence diverse (Andersen–Bollerslev–Diebold) e separarle migliora il forecast del momento pari. Causale, calcolabile dal path 1h. Effort S ma richiede rigen dataset + retrain → **accodare a un retrain già pianificato**.
**EN** The semivariance probe failed as a **target** (odd moments); bipower/jump-variation decomposition as **input** is orthogonal to that kill: C and J components have different persistence and separating them improves even-moment forecasts. Causal, computable from the 1h path. Effort S but needs dataset regen + retrain → **queue behind an already-planned retrain**.

### A5 🟡 Pesi ensemble per-QLIKE (linea vol) · QLIKE-based ensemble weights (vol line)
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
2. **POST-GATE (~metà luglio):** `04b` v2 con leg delta-hedge sul perp testnet; **pre-registrare** il confronto hedged vs unhedged (Sharpe per trade, PnL net-of-funding). In parallelo: A2 (quantili RV) + A5 (pesi QLIKE) → alimentano il sizing v2; poi A7 (risk greeks-aware) quando il sizing diventa Kelly-su-edge.
3. **Opportunistici (a retrain pianificato):** A4 (HAR-CJ input); quick-win A8 quando si rigira un training comunque; A3 (mixture-of-universes) come esperimento modello a sé con gate QLIKE pre-registrato.

**EN**
1. **NOW:** A6 — bid/ask + theoretical-delta logging in `04b` (diagnostic columns only, pre-registered constants UNTOUCHED, the n≥20 gate closes on the frozen design).
2. **POST-GATE (~mid July):** `04b` v2 with a perp delta-hedge leg on testnet; **pre-register** hedged vs unhedged comparison (per-trade Sharpe, PnL net-of-funding). In parallel: A2 (RV quantiles) + A5 (QLIKE weights) → feed v2 sizing; then A7 when sizing becomes Kelly-on-edge.
3. **Opportunistic (at a planned retrain):** A4 (HAR-CJ inputs); A8 quick-wins whenever a training run happens anyway; A3 (mixture-of-universes) as a standalone model experiment with a pre-registered QLIKE gate.

🇮🇹 **Sintesi:** la parte finale del progetto diventa un book a due strumenti con i ruoli **opzioni = veicolo dell'edge (vol), futures = copertura che lo purifica** — mai il contrario.
**EN** **Summary:** the project's final stage becomes a two-instrument book where **options = edge vehicle (vol), futures = the hedge that purifies it** — never the reverse.

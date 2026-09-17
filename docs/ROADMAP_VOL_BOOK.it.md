🇮🇹 Italiano · [🇬🇧 English](ROADMAP_VOL_BOOK.md)

# ROADMAP — Vol Book: stato e residui

**Riconciliata il 2026-09-10.** Coda corrente e preregistrazioni vincolanti: [STATUS.md](../STATUS.md), sezioni in testa. Questo file conserva gli ID e i residui utili; non autorizza training, modifiche live o nuove letture dei gate. Gli esiti completi restano in [THEORY.it.md](../THEORY.it.md) §12 e nello storico di stato.

## Priorità correnti

E1 stadio 2 è **CHIUSO, NESSUNA CONCLUSIONE** (10/09): niente seconde letture né inversione del segnale senza nuova preregistrazione. FT1 è implementato-inerte e **non avviato**: audit di potenza fatto il 15/09, con emendamento 1 adottato (① declassata a controllo pre-calcolato, claim di ②a col suo limite, conteggio degli esiti dell'esecutore, momento di lettura, ②b/③b misurabili) — il giudice FT1 è in codice con test sentinella dal 17/09 (`scripts/vol/ft1_execution_judge.py`); vintage macro deciso il 17/09: nessuna promozione (FT1 non legge i forecast); go-live solo su istruzione esplicita e con ledger flat dopo settlement. Il contatore E1 della routine è stato ritirato il 17/09.

## Residui conservati

| ID | Stato e vincolo | Fonte |
|---|---|---|
| A3 / A3-bis | Regime-MoE **eseguito il 19/07**, nessuna conclusione: r1=657<800. Parcheggiato; A8-BIS è fallito, quindi quel ramo di riapertura è decaduto. Nuovo campione e nuova preregistrazione prima di rivalutare. | [Implementazione](MODEL_IMPROVEMENTS.it.md), STATUS 19–20/07 |
| A7 | Skeleton greeks-risk non cablato. Il FAIL hedged non autorizza l'ingresso nel critical path né il sizing HAR-q90; serve una decisione di disegno separata. | MODEL_IMPROVEMENTS, THEORY §12.2 |
| A9 | MaxPool parallelo N-HiTS implementato-inerte; nessun PASS documentato. Non è una riapertura automatica della classe training chiusa: nuova ipotesi e preregistrazione necessarie. | `config/arch/nhits.yaml`, `tests/test_nhits_maxpool.py`, STATUS |
| A13 / A13a | Pin-close e gamma cap inerti. Pin-close parcheggiato: unità utile = trigger, n_trig≥20 nel disegno storico; scegliere offline appaiato o forward eseguito. E1 non conclusivo non promuove né declassa automaticamente la leva. | STATUS, `04b --pin-close-hours/--pin-close-band` |
| A14 | Sizing vega inerte (`--size-mode vega`, `--size-vega-target`). Non attivare sulla base della vecchia checklist post-v1: nuova decisione e preregistrazione sul design corrente. | STATUS, `04b` |
| MacroNormalizer | Pin implementato-inerte (`--macro-norm` su 04b e replay), riferimento dichiarato 20260730. Pin del normalizzatore e promozione del parquet sono decisioni distinte; nessuna attivazione presunta. | [START](../START.it.md), STATUS 31/07 e stato corrente |
| B1 / L2 | Stadio 1 non conclusivo per controllo positivo fallito. Non riaprire automaticamente a h=3. A h=30 attendere n_eff=216; date subordinate alla continuità. Nessun produttore permanente delle kline 1m. | STATUS 10/08 e stato corrente |
| CAFN | Parcheggiato, prior basso; riapertura solo con nuova definizione del perimetro e gate. | [START](../START.it.md), STATUS 20/07 |
| Replay C1 | La vecchia checklist lasciava pendente il confronto completo su tick live/replay futuri sovrapposti dopo il funding-refresh. Non confonderlo con la parità del solo refactor C2; conservarlo come verifica da riconciliare prima di dichiarare copertura completa. | STATUS 18/07 |
| Dashboard | D1–D5 conservati (D6 fatto 2026-09-17): proposte ancora utili, non preregistrazioni. Le viste di risultati non possono anticipare gate o descrittivi vincolati. | [DASHBOARD_IMPROVEMENTS](DASHBOARD_IMPROVEMENTS.it.md) |

## Voci esaurite rimosse dalla coda

| ID | Esito |
|---|---|
| v1 n=20 / n=30 | FAIL 0/3 il 18/07 e 30/07; i campioni e i caveat sono nello storico. |
| A1 / B2, hedge v2 | FAIL 2/3 l'11/08: varianza −55,3%, drag −0,647 SE oltre budget −0,25; 76% fee. Wind-down completato il 13/08. |
| A12, banda WW | Codice inerte; confronto pre-v2 senza dominanza, scelta fixed 0,30. Nessuna nuova attivazione autorizzata. |
| A8-BIS / B4-bis / A10 | Mixup FAIL 20/07; DVOL-feature FAIL 23/07; sparsity FAIL 30/07 con manipulation check superato. Classe training chiusa. |
| A4 / C1–C3 | HAR-CJ come input non è una via per riaprire la classe training. Smearing non adottato; HAR-C baseline adottata; ri-specificazione HAR-CJ esaurita. |
| MFIV | Derivazione completata; comparatore v2 FAIL 18/08, n=41. ATM resta comparatore, MFIV diagnostica; nessuna v3. |
| C2 / C4 infrastruttura | Refactor 2ter e greeks+sync completati il 18/07, documentati in README/START/THEORY. |
| R1 | Non è più un'attività da eseguire il 04/08: usare esito e vincoli di provenienza in STATUS/THEORY, senza promozione implicita del modello. |

**Ritiro checklist:** `POST_GATE_V1.md` e `RIPRESA.md` eliminati il 10/09 dopo riconciliazione. I residui sono nella tabella sopra; gli esiti e le preregistrazioni restano nello storico di STATUS e in Git. Le menzioni nelle voci storiche indicano i file presenti a quella data, non istruzioni correnti. Promemoria operativo conservato: rivalutare rinnovo/disdetta VPS verso dicembre 2026.

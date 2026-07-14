# Deploy collector 24/7 su VPS · 24/7 collector VPS deploy

🇮🇹 Kit per spostare i due collector leggeri (`01c_iv_poller`, `01d_orderbook_recorder`) su un VPS Linux always-on (Ubuntu 24.04 o Debian 12+; l'istanza reale monta Debian) (decisione 2026-06-24; acquisto netcup VPS Lite 1 G12s 2026-07-14). Obiettivo: eliminare i buchi PC-off nella serie IV (unico dato non rigenerabile), sbloccare B1 (book L2 continuo) e rendere replayabile offline il forward test `04b`. Nessun secret sul VPS: entrambi i collector usano solo endpoint pubblici non autenticati. Training/GPU restano a casa.

**EN** Kit to move the two lightweight collectors (`01c_iv_poller`, `01d_orderbook_recorder`) to an always-on Linux VPS (Ubuntu 24.04 or Debian 12+; the actual instance runs Debian) (2026-06-24 decision; netcup VPS Lite 1 G12s purchased 2026-07-14). Goal: remove PC-off gaps in the IV series (the only non-regenerable dataset), unblock B1 (continuous L2 book) and make the `04b` forward test replayable offline. No secrets on the VPS: both collectors only hit public unauthenticated endpoints. Training/GPU stay home.

## Sequenza di deploy · Deploy sequence

🇮🇹 **0. Geo-test — PRIMA di installare qualsiasi cosa.** Se Binance risponde 451 l'IP è geo-bloccato: rendi il VPS nella finestra di recesso, non c'è workaround.

**EN** **0. Geo-test — BEFORE installing anything.** If Binance returns 451 the IP is geo-blocked: return the VPS within the withdrawal window, there is no workaround.

```bash
# sul VPS appena provisionato / on the freshly provisioned VPS
curl -sO https://raw.githubusercontent.com/luca-feleppa/quantsys/main/deploy/vps/geo_test.sh \
  || scp deploy/vps/geo_test.sh root@<ip>:   # se il repo è privato / if the repo is private
bash geo_test.sh          # atteso/expected: VERDETTO PASS
```

🇮🇹 **1. Deploy key (repo privato).** Genera sul VPS una chiave dedicata e aggiungila su GitHub → repo → Settings → Deploy keys (read-only):

**EN** **1. Deploy key (private repo).** Generate a dedicated key on the VPS and add it on GitHub → repo → Settings → Deploy keys (read-only):

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "quantsys-vps"
cat ~/.ssh/id_ed25519.pub   # → incolla su GitHub / paste into GitHub
```

🇮🇹 **2. Setup one-shot** (da root; fa pacchetti, utente `quantsys`, ufw, clone, venv con torch-CPU, smoke `--once`, unit systemd attive):

**EN** **2. One-shot setup** (as root; does packages, `quantsys` user, ufw, clone, venv with CPU torch, `--once` smoke, active systemd units):

```bash
git clone git@github.com:luca-feleppa/quantsys.git /opt/quantsys   # solo la prima volta / first time only
bash /opt/quantsys/deploy/vps/setup_vps.sh
```

🇮🇹 **3. Verifica.** Log live e presenza dei parquet:

**EN** **3. Verify.** Live logs and parquet presence:

```bash
journalctl -u quantsys-iv -u quantsys-ob -f
find /opt/quantsys/data -name '*.parquet' -newermt '-1 hour'
```

🇮🇹 **4. Sync verso casa** (Windows, dalla root di progetto; scarica in `data/vps_staging/` e fa merge+heartbeat nella copia canonica):

**EN** **4. Sync back home** (Windows, from the project root; downloads into `data/vps_staging/` and merges+heartbeats into the canonical copy):

```powershell
.\scripts\vps\pull_vps_data.ps1   # host letto da config/secrets.yaml → vps.host (privato, gitignored)
```

## Semantica dei dati · Data semantics

🇮🇹 Il doppio poller (casa accesa + VPS) produce tick duplicati **by design**: il merge deduplica (`atm_30h`/`dvol` su `timestamp`; `chain/*` su `snapshot_ts+instrument_name`; `orderbook/*` su `timestamp+symbol`) e ordina, con scritture atomiche. La copia canonica resta quella di casa (`data/iv/`, `data/orderbook/`); il VPS è la sorgente di continuità e la seconda copia di ridondanza dell'asset IV. `04b` a casa continua a leggere il file locale (staleness ≤30 min) alimentato dal poller locale quando il PC è acceso. ⚠ I trade eventualmente replayati offline sulle ore PC-off NON entrano retroattivamente nel gate v1 (campione pre-registrato): vanno in file separati.

**EN** Dual polling (home on + VPS) duplicates ticks **by design**: the merge deduplicates (`atm_30h`/`dvol` on `timestamp`; `chain/*` on `snapshot_ts+instrument_name`; `orderbook/*` on `timestamp+symbol`) and sorts, with atomic writes. The canonical copy stays home (`data/iv/`, `data/orderbook/`); the VPS is the continuity source and the redundancy copy of the IV asset. `04b` at home keeps reading the local file (≤30 min staleness) fed by the local poller while the PC is on. ⚠ Any trades replayed offline over PC-off hours do NOT retroactively enter the v1 gate (pre-registered sample): they go to separate files.

## File del kit · Kit files

| File | 🇮🇹 | **EN** |
|---|---|---|
| `geo_test.sh` | Check 451 Binance + Deribit prod/testnet, pre-install | Binance 451 + Deribit prod/testnet check, pre-install |
| `setup_vps.sh` | Provisioning one-shot idempotente (root) | Idempotent one-shot provisioning (root) |
| `requirements-vps.txt` | Dipendenze minime collector (+ torch CPU a parte) | Minimal collector deps (+ CPU torch separately) |
| `quantsys-iv.service` | Unit systemd 01c (tick 10 min, `Restart=always`) | 01c systemd unit (10-min tick, `Restart=always`) |
| `quantsys-ob.service` | Unit systemd 01d (polling 5 s, `Restart=always`) | 01d systemd unit (5 s polling, `Restart=always`) |
| `../../scripts/vps/pull_vps_data.ps1` | Pull scp lato casa → staging | Home-side scp pull → staging |
| `../../scripts/vps/merge_vps_data.py` | Merge dedup → canonico + heartbeat staleness | Dedup merge → canonical + staleness heartbeat |

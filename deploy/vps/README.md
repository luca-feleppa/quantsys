🇬🇧 English · [🇮🇹 Italiano](README.it.md)

# 24/7 collector VPS deploy

Kit for the three lightweight collectors (`01c_iv_poller`, `01d_orderbook_recorder`, `01e_trades_recorder`) on an always-on Linux VPS (Ubuntu 24.04 or Debian 12+) (2026-06-24 decision; entry-level EU VPS purchased 2026-07-14; 01e added 2026-07-16). Goal: remove PC-off gaps in the IV series (non-regenerable data), unblock B1 (continuous L2 book), make the `04b` forward test replayable offline, and accumulate option trades for realized spreads (API retention ~24h: also not reconstructible ex-post). No secrets on the VPS: all collectors only hit public unauthenticated endpoints. Training/GPU stay home.

## Deploy sequence

**0. Geo-test — BEFORE installing anything.** If Binance returns 451 the IP is geo-blocked: return the VPS within the withdrawal window, there is no workaround.

```bash
# sul VPS appena provisionato / on the freshly provisioned VPS
curl -sO https://raw.githubusercontent.com/luca-feleppa/quantsys/main/deploy/vps/geo_test.sh \
  || scp deploy/vps/geo_test.sh root@<ip>:   # se il repo è privato / if the repo is private
bash geo_test.sh          # atteso/expected: VERDETTO PASS
```

**1. Deploy key (private repo).** Generate a dedicated key on the VPS and add it on GitHub → repo → Settings → Deploy keys (read-only):

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "quantsys-vps"
cat ~/.ssh/id_ed25519.pub   # → incolla su GitHub / paste into GitHub
```

⚠ The key is **passphrase-less out of necessity**: `git pull` runs unattended under systemd and a passphrase would block it. Mitigations: **read-only** deploy key **scoped to this repo only** (not a user key), `600` permissions, SSH-only `ufw`. The key used **from the workstation to the VPS** is a different thing: that one is interactive and **must** carry a passphrase.

**2. One-shot setup** (as root; does packages, `quantsys` user, ufw, clone, venv with CPU torch, `--once` smoke, active systemd units):

```bash
git clone git@github.com:luca-feleppa/quantsys.git /opt/quantsys   # solo la prima volta / first time only
bash /opt/quantsys/deploy/vps/setup_vps.sh
```

**3. Verify.** Live logs and parquet presence:

```bash
journalctl -u quantsys-iv -u quantsys-ob -u quantsys-trades -f
find /opt/quantsys/data -name '*.parquet' -newermt '-1 hour'
```

**4. Sync back home** (Windows, from the project root; downloads into `data/vps_staging/` and merges+heartbeats into the canonical copy):

```powershell
.\scripts\vps\pull_vps_data.ps1   # host letto da config/secrets.yaml → vps.host (privato, gitignored)
```

## Data semantics

Dual polling (home on + VPS) duplicates ticks **by design**: the merge deduplicates (`atm_30h`/`dvol` on `timestamp`; `chain/*` on `snapshot_ts+instrument_name`; `orderbook/*` on `timestamp+symbol`; `deribit_trades/*` on `trade_id`) and sorts, with atomic writes. The canonical copy stays home (`data/iv/`, `data/orderbook/`, `data/deribit_trades/`); the VPS is the continuity source and the redundancy copy of the IV asset. `01d` and `01e` live ONLY on the VPS (no home instance). `04b` at home keeps reading the local file (≤30 min staleness) fed by the local poller while the PC is on. ⚠ Any trades replayed offline over PC-off hours do NOT retroactively enter the v1 gate (pre-registered sample): they go to separate files.

## Kit files

| File | Description |
|---|---|
| `geo_test.sh` | Binance 451 + Deribit prod/testnet check, pre-install |
| `setup_vps.sh` | Idempotent one-shot provisioning (root) |
| `requirements-vps.txt` | Minimal collector deps (+ CPU torch separately) |
| `quantsys-iv.service` | 01c systemd unit (10-min tick, `Restart=always`) |
| `quantsys-ob.service` | 01d systemd unit (5 s polling, `Restart=always`) |
| `quantsys-trades.service` | 01e systemd unit (10-min tick, `Restart=always`) |
| `../../scripts/vps/pull_vps_data.ps1` | Home-side scp pull → staging |
| `../../scripts/vps/merge_vps_data.py` | Dedup merge → canonical + staleness heartbeat |

#!/usr/bin/env bash
# IT: SETUP ONE-SHOT del VPS collector QUANTSYS (Ubuntu 24.04 o Debian 12+,
#     eseguire da root su macchina APPENA provisionata, DOPO che geo_test.sh
#     è PASS). Unico requisito Python: >=3.11 (Debian 12 = 3.11, 13 = 3.13).
#     Fa tutto: pacchetti, utente dedicato, firewall, clone repo, venv con
#     torch-CPU, unit systemd, smoke test --once. Idempotente al meglio
#     (ri-eseguibile senza danni). NON tocca mai secret: i collector usano
#     solo endpoint pubblici non autenticati.
# EN: ONE-SHOT setup of the QUANTSYS collector VPS (Ubuntu 24.04 or Debian 12+,
#     run as root on a FRESHLY provisioned box, AFTER geo_test.sh PASSES).
#     Only Python requirement: >=3.11 (Debian 12 = 3.11, 13 = 3.13).
#     Does everything: packages, dedicated user, firewall, repo clone, venv
#     with CPU torch, systemd units, --once smoke test. Best-effort idempotent
#     (safe to re-run). Never touches secrets: the collectors only use public
#     unauthenticated endpoints.
set -euo pipefail

# ── Parametri / Parameters ────────────────────────────────────────────────────
# IT: repo privato → serve una Deploy Key read-only su GitHub (vedi README.md);
#     in alternativa esporta REPO_URL con un PAT https prima di lanciare.
# EN: private repo → needs a read-only GitHub Deploy Key (see README.md);
#     alternatively export REPO_URL with an https PAT before running.
REPO_URL="${REPO_URL:-git@github.com:luca-feleppa/quantsys.git}"
BRANCH="${BRANCH:-main}"
INSTALL_DIR="/opt/quantsys"
RUN_USER="quantsys"

echo "=== [1/7] Pacchetti base / base packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3-venv python3-pip ufw unattended-upgrades curl
timedatectl set-timezone UTC

echo "=== [2/7] Utente dedicato / dedicated user ==="
# IT: utente di servizio senza password (login solo via su/systemd, non SSH).
# EN: passwordless service user (login only via su/systemd, not SSH).
id "$RUN_USER" &>/dev/null || useradd --system --create-home --shell /bin/bash "$RUN_USER"

echo "=== [3/7] Firewall (solo SSH) / firewall (SSH only) ==="
ufw allow OpenSSH
ufw --force enable

echo "=== [4/7] Clone repo ==="
if [ ! -d "$INSTALL_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
else
    git -C "$INSTALL_DIR" pull --ff-only
fi

echo "=== [5/7] Venv + dipendenze (torch CPU) / venv + deps (CPU torch) ==="
cd "$INSTALL_DIR"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
# IT: ordine obbligato — torch CPU prima (indice dedicato), poi il resto, poi
#     il package quantsys senza deps (pyproject non dichiara dependencies).
# EN: mandatory order — CPU torch first (dedicated index), then the rest, then
#     the quantsys package without deps (pyproject declares no dependencies).
.venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --quiet -r deploy/vps/requirements-vps.txt
.venv/bin/pip install --quiet -e . --no-deps
# IT: directory di output dei collector (path CWD-relativi, vedi 01c/01d).
# EN: collector output directories (CWD-relative paths, see 01c/01d).
mkdir -p data/iv/chain data/orderbook data/deribit_trades logs
chown -R "$RUN_USER":"$RUN_USER" "$INSTALL_DIR"

echo "=== [6/7] Smoke test --once (tutti i collector / all collectors) ==="
sudo -u "$RUN_USER" .venv/bin/python scripts/01c_iv_poller.py --once
sudo -u "$RUN_USER" .venv/bin/python scripts/01d_orderbook_recorder.py --once
sudo -u "$RUN_USER" .venv/bin/python scripts/01e_trades_recorder.py --once
echo "--- parquet scritti / parquet written:"
find data/iv data/orderbook data/deribit_trades -name '*.parquet' -newermt '-10 minutes' | sed 's/^/    /'

echo "=== [7/7] Unit systemd (enable --now) ==="
cp deploy/vps/quantsys-iv.service deploy/vps/quantsys-ob.service \
   deploy/vps/quantsys-trades.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now quantsys-iv.service quantsys-ob.service quantsys-trades.service
sleep 3
systemctl --no-pager --lines=5 status quantsys-iv.service quantsys-ob.service quantsys-trades.service || true

# IT: 04b vol-paper (migrato dal 2026-07-18) — abilitato SOLO se i prerequisiti
#     NON-git sono già stati seedati (secrets testnet + modelli + stato forward
#     test, vedi header di quantsys-volpaper.service): su un VPS collector-only
#     questo blocco è un no-op esplicito.
# EN: 04b vol-paper (migrated 2026-07-18) — enabled ONLY when the non-git
#     prerequisites are already seeded (testnet secrets + models + forward-test
#     state, see the quantsys-volpaper.service header): on a collector-only VPS
#     this block is an explicit no-op.
if [ -f "$INSTALL_DIR/config/secrets.yaml" ] && [ -d "$INSTALL_DIR/models/itransformer" ]; then
    cp deploy/vps/quantsys-volpaper.service deploy/vps/quantsys-volpaper-restart.service \
       deploy/vps/quantsys-volpaper-restart.timer /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable --now quantsys-volpaper.service quantsys-volpaper-restart.timer
    systemctl --no-pager --lines=5 status quantsys-volpaper.service || true
else
    echo "SKIP quantsys-volpaper: prerequisiti non seedati (secrets/modelli) / prerequisites not seeded (secrets/models)"
fi

echo
echo "FATTO / DONE. Log live: journalctl -u quantsys-iv -u quantsys-ob -u quantsys-trades -f"
echo "Da casa / from home: scripts/vps/pull_vps_data.ps1 -VpsHost <user@ip>"

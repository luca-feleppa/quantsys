🇬🇧 English · [🇮🇹 Italiano](DASHBOARD_IMPROVEMENTS.it.md)

# Dashboard Improvements — proposal backlog

> **2026-09-10 review:** backlog retained: the terminal and local parser do not exhaust D1–D6. The July timing notes below are historical, not authorizations: hedge v2 is closed FAIL and its pre-hedge phase is no longer a priority. D4 remains an operational improvement; D1–D3 must respect forward-sample and descriptive-analysis restrictions. D5 is optional, D6 remains opportunistic. Current queue and gates: [STATUS.md](../STATUS.md).

> Created 2026-07-16 (pre-v1-gate-close session). Status: **PROPOSALS, none implemented**. Scope: `scripts/06_dashboard.py` (Deribit risk terminal, 4 tabs: Surface / Chain / Risk & Greeks / Trades; single-file, GPU-free, by design DECOUPLED from the ML models). Common thread: the real gains are not cosmetic — they are on-disk data no tab currently shows. No item touches pre-registered paths: it is all read-only layer.

## Items (value-ordered)

| # | Item | Effort | When |
|---|---|---|---|
| D1 | **VRP monitor tab**: `iv_30h` (H24 since 07-14) vs subsequent 30h realized vol from candles = realized VRP over time, overlaying the forward-test trades (entry, edge, outcome). Also makes the gate's pre/post-VPS hourly caveat visible | M | post-gate v1 |
| D2 | **Trades tab: from list to analysis**: cumulative BTC equity curve; entry-edge → realized-PnL scatter ("does edge predict PnL?" in one chart); A11 attribution panel from `results/vol_paper/attribution.parquet` (per-trade Δ/Γ/ν/Θ/residual, declared coverage — nobody reads that parquet today) | M | post-gate v1 (first) |
| D3 | **Realized-spreads panel** from the 01e data (`data/deribit_trades/`): \|price−mark\| distribution by moneyness/tenor/hour-of-day — natural consumer of the 2026-07-16 collector, direct input to post-gate executable costs | M | post-gate v1 |
| D4 | **Infra/position panel**: canonical-parquet freshness (visual IV/L2/trades heartbeat without `check_vps`), last-pull age; for the open position: expiry countdown + pin distance \|S−K\|/S (the A13 metric). All CLI+logs today | S | post-gate v1 (first) |
| D5 | **Venue δ vs local BS δ comparison** on the ATM straddle (`atm_greeks.parquet` series vs `bs_greeks`): visual validation of the raw/premium-adjusted convention, supports the pre-v2-hedge decision | S | after C4 (`--greeks` active) |
| D6 | **Technical hygiene**: consolidate the 4th Deribit parser copy (2026-07-16 audit: import `parse_instrument` from `quantsys/data/deribit.py`, keeping its own session+retry); optional SVI/SABR smile fit for display (max justified modelling upgrade) | S | opportunistic |

## Non-goals (rejected with rationale)

- **Wiring the ML models into the dashboard** — the decoupling is a won design (market terminal, no GPU contention, no risk of contaminating the production path).
- **Dupire / Variance-Gamma / mixed models for local greeks** — rejected 2026-07-16: sparse data (4 short expiries, illiquid wings) → unstable second derivatives; the delta would diverge from the venue's, the only decisional one (07-08 verdict). `bs_greeks` stays display-only.
- **WebSocket instead of REST polling** — complexity without benefit at the current cadence.

## Suggested practical order

D2 and D4 right after the v1 gate closes (they serve the evaluation and v2 operations) → D1 and D3 in the same window (shared data plumbing) → D5 once C4 is active → D6 opportunistically.

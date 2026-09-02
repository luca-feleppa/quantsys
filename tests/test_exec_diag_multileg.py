# IT: exec_diag a N gambe — prova di INERZIA sul record a 2 gambe e contratto del corpo.
#     Famiglia 3 delle regole di test: una generalizzazione deve lasciare il path
#     production bit-identico, verificato sui dati VERI (replay di exec_diag.jsonl) e non
#     solo su sintetici. Tre contratti:
#       (1) replay: exec_diag_aggregate riproduce ESATTAMENTE i 4 aggregati storici di ogni
#           riga di results/vol_paper/exec_diag.jsonl e non aggiunge chiavi su 2 gambe;
#       (2) corpo: con 4 gambe gli aggregati storici sono identici a quelli delle sole
#           prime due (le ali non li toccano) e i campi `_all` compaiono solo allora;
#       (3) dato mancante: un delta/quote None su una gamba qualsiasi → None, mai zero
#           (contratto anti-sotto-conteggio, famiglia 2).
# EN: N-leg exec_diag — INERTIA proof on the 2-leg record and body contract. Test-rule
#     family 3: a generalisation must leave the production path bit-identical, verified on
#     REAL data (replay of exec_diag.jsonl), not only on synthetics. Three contracts:
#     (1) replay reproduces every historical aggregate exactly and adds no key on 2 legs;
#     (2) with 4 legs the historical aggregates equal the first-two-legs ones and `_all`
#     fields appear only then; (3) a missing datum on any leg → None, never zero.
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "results" / "vol_paper" / "exec_diag.jsonl"
HIST_KEYS = ("straddle_delta", "net_delta", "half_spread_btc", "half_spread_frac")
ALL_KEYS = ("n_legs", "body_idx", "structure_delta_all",
            "half_spread_btc_all", "half_spread_frac_all")


@pytest.fixture(scope="module")
def vp():
    # IT/EN: importa 04b come modulo (nome che inizia per cifra → importlib da path).
    spec = importlib.util.spec_from_file_location(
        "volpaper_04b_diag", ROOT / "scripts" / "04b_vol_paper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["volpaper_04b_diag"] = mod
    spec.loader.exec_module(mod)
    return mod


def _leg(inst, bid, ask, mark, delta):
    return {"instrument": inst, "bid": bid, "ask": ask, "bid_size": 1.0, "ask_size": 1.0,
            "mark": mark, "mark_iv": 40.0, "bid_iv": 39.0, "ask_iv": 41.0,
            "underlying": 80000.0, "delta": delta, "gamma": 0.0001, "vega": 10.0,
            "theta": -50.0}


BODY = [_leg("BTC-10SEP26-80000-C", 0.010, 0.012, 0.011, 0.51),
        _leg("BTC-10SEP26-80000-P", 0.009, 0.011, 0.010, -0.49)]
WINGS = [_leg("BTC-10SEP26-88000-C", 0.0010, 0.0016, 0.0013, 0.08),
         _leg("BTC-10SEP26-72000-P", 0.0011, 0.0017, 0.0014, -0.07)]


@pytest.mark.skipif(not DIAG.exists(), reason="exec_diag.jsonl assente / absent")
def test_replay_two_leg_records_bit_identical(vp):
    # IT: ③a — ogni riga storica, ricalcolata dalle sue gambe, riproduce i 4 aggregati
    #     con uguaglianza ESATTA (stessa aritmetica, stesso ordine di somma) e senza
    #     chiavi in più. Il confronto è `==` sui float, non una tolleranza.
    # EN: ③a — every historical row, recomputed from its legs, reproduces the 4
    #     aggregates with EXACT equality and no extra keys. Float `==`, no tolerance.
    n = 0
    for line in DIAG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        legs = r.get("legs") or []
        if len(legs) != 2:
            continue
        agg = vp.exec_diag_aggregate(legs, int(r["side"]))
        assert set(agg) == set(HIST_KEYS), agg.keys()
        for k in HIST_KEYS:
            assert agg[k] == r.get(k), (r["ts"], k, agg[k], r.get(k))
        n += 1
    assert n > 100, n


def test_body_contract_with_four_legs(vp):
    # IT: le ali non toccano gli aggregati storici; i campi `_all` esistono solo a >2 gambe.
    # EN: wings never touch the historical aggregates; `_all` fields exist only at >2 legs.
    two = vp.exec_diag_aggregate(BODY, -1)
    four = vp.exec_diag_aggregate(BODY + WINGS, -1)
    assert set(two) == set(HIST_KEYS)
    assert set(four) == set(HIST_KEYS) | set(ALL_KEYS)
    for k in HIST_KEYS:
        assert four[k] == two[k]
    assert four["n_legs"] == 4 and four["body_idx"] == [0, 1]
    assert four["structure_delta_all"] == pytest.approx(0.51 - 0.49 + 0.08 - 0.07)
    assert four["half_spread_btc_all"] == pytest.approx(0.001 + 0.001 + 0.0003 + 0.0003)
    assert four["half_spread_frac_all"] == pytest.approx(
        four["half_spread_btc_all"] / (0.011 + 0.010 + 0.0013 + 0.0014))
    # IT/EN: il corpo pesa lo spread sul suo mark, non su quello della struttura
    assert two["half_spread_frac"] == pytest.approx(0.002 / 0.021)


def test_missing_datum_is_none_never_zero(vp):
    # IT: un delta None su un'ala azzera SOLO `structure_delta_all`; un bid None sul corpo
    #     azzera gli aggregati del corpo E quelli `_all` (che lo contengono). Mai 0.0.
    # EN: a None delta on a wing nulls ONLY `structure_delta_all`; a None bid on the body
    #     nulls body aggregates AND the `_all` ones (which include it). Never 0.0.
    w = [dict(WINGS[0], delta=None), WINGS[1]]
    a = vp.exec_diag_aggregate(BODY + w, 1)
    assert a["straddle_delta"] is not None and a["structure_delta_all"] is None
    b = [dict(BODY[0], bid=None), BODY[1]]
    a2 = vp.exec_diag_aggregate(b + WINGS, 1)
    assert a2["half_spread_btc"] is None and a2["half_spread_frac"] is None
    assert a2["half_spread_btc_all"] is None
    assert a2["straddle_delta"] is not None

"""
Regression tests: credentials must never reach models/**/pipeline_state.pkl.

`load_config` merges config/secrets.yaml over default.yaml, so the cfg handed to
`PipelineState.set_training_config` carries live credentials. Before the fix the
method did `dict(cfg)` (shallow) and every persisted state embedded them.

All credential values used here are FAKE and invented for the test.
"""
from __future__ import annotations

import pickle

from quantsys.utils import PipelineState, interval_minutes_from_cfg, redact_config


# Fake values: distinctive enough to be searched for inside the pickle bytes.
FAKE_FRED          = "FAKE_FRED_KEY_123"
FAKE_ALPACA_ID     = "FAKE_ALPACA_ID_456"
FAKE_ALPACA_SECRET = "FAKE_ALPACA_SECRET_789"
FAKE_DERIBIT       = "FAKE_DERIBIT_CLIENT_SECRET_321"
FAKE_VPS_HOST      = "fake-host.invalid"
FAKE_NESTED_TOKEN  = "FAKE_NESTED_TOKEN_654"

FAKE_VALUES = [FAKE_FRED, FAKE_ALPACA_ID, FAKE_ALPACA_SECRET,
               FAKE_DERIBIT, FAKE_VPS_HOST, FAKE_NESTED_TOKEN]


def _make_cfg() -> dict:
    """Mimics load_config output: default.yaml sections + secrets.yaml merge."""
    return {
        "data": {"interval": "1h", "symbol": "BTCUSDT"},
        "features": {"forecast_horizon": 30, "target_type": "log_rv"},
        "model": {"use_revin": True, "loss_type": "quantile"},
        "dashboard": {"host": "127.0.0.1", "port": 8050},
        # Credential key inside a legitimate, always-read section.
        "macro": {"fred_api_key": FAKE_FRED, "lookback_days": 365,
                  "nested": {"provider": {"access_token": FAKE_NESTED_TOKEN}}},
        # Sections that exist only because of the secrets.yaml merge.
        "vps": {"host": FAKE_VPS_HOST, "user": "deploy"},
        "alpaca": {"api_key_id": FAKE_ALPACA_ID, "api_secret_key": FAKE_ALPACA_SECRET},
        "deribit_testnet": {"client_id": "fake-client", "client_secret": FAKE_DERIBIT},
    }


class TestTrainingConfigRedaction:
    """(a) No credential value survives into the serialized PipelineState."""

    def test_secret_sections_are_dropped(self):
        state = PipelineState().set_training_config(_make_cfg())
        for section in ("vps", "alpaca", "deribit_testnet"):
            assert section not in state.training_config

    def test_secret_keys_are_replaced(self):
        state = PipelineState().set_training_config(_make_cfg())
        assert state.training_config["macro"]["fred_api_key"] == "<redacted>"
        # Redaction is depth-independent, not just top-of-section.
        assert (state.training_config["macro"]["nested"]["provider"]["access_token"]
                == "<redacted>")

    def test_no_fake_value_in_pickle_bytes(self):
        state = PipelineState().set_training_config(_make_cfg())
        blob = pickle.dumps(state)
        for value in FAKE_VALUES:
            assert value.encode() not in blob

    def test_non_secret_keys_survive(self):
        # dashboard.host is 127.0.0.1 in default.yaml: a bind address, not a secret.
        state = PipelineState().set_training_config(_make_cfg())
        assert state.training_config["dashboard"]["host"] == "127.0.0.1"
        assert state.training_config["macro"]["lookback_days"] == 365
        assert state.training_config["data"]["symbol"] == "BTCUSDT"


class TestCallerConfigUntouched:
    """(b) The aliasing regression: the caller keeps a working cfg."""

    def test_caller_cfg_keeps_its_credentials(self):
        cfg = _make_cfg()
        macro_section = cfg["macro"]
        PipelineState().set_training_config(cfg)
        # 01_download_data.py keeps using cfg after the call: it must be intact.
        assert cfg["macro"]["fred_api_key"] == FAKE_FRED
        assert cfg["macro"]["nested"]["provider"]["access_token"] == FAKE_NESTED_TOKEN
        assert cfg["alpaca"]["api_secret_key"] == FAKE_ALPACA_SECRET
        assert cfg["vps"]["host"] == FAKE_VPS_HOST
        # Same object, not a replaced one: nothing was rebound in place either.
        assert cfg["macro"] is macro_section

    def test_redacted_copy_shares_no_subdict(self):
        cfg = _make_cfg()
        clean = redact_config(cfg)
        assert clean["macro"] is not cfg["macro"]
        assert clean["macro"]["nested"] is not cfg["macro"]["nested"]
        # Mutating the copy must not reach the live config.
        clean["data"]["interval"] = "1m"
        assert cfg["data"]["interval"] == "1h"

    def test_lists_are_rebuilt_not_shared(self):
        cfg = {"features": {"windows": [5, 15, 60]}}
        clean = redact_config(cfg)
        assert clean["features"]["windows"] == [5, 15, 60]
        assert clean["features"]["windows"] is not cfg["features"]["windows"]


class TestReadersSurviveRedaction:
    """(c) The three fields actually read back from training_config."""

    def test_forecast_horizon(self):
        state = PipelineState().set_training_config(_make_cfg())
        assert state.forecast_horizon == 30

    def test_interval_is_derivable(self):
        state = PipelineState().set_training_config(_make_cfg())
        assert state.training_config["data"]["interval"] == "1h"
        assert interval_minutes_from_cfg(state.training_config) == 60

    def test_use_revin(self):
        state = PipelineState().set_training_config(_make_cfg())
        assert state.training_config["model"]["use_revin"] is True

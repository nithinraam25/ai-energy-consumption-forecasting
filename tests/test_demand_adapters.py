"""Unit tests for operational demand telemetry adapters (PJM Data Miner 2 & EIA API v2)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.data.demand_adapters import (
    PJMDataMinerAdapter,
    EIADemandAdapter,
    DemandTelemetryAuthError,
    DemandTelemetryUnavailableError,
    get_operational_demand_adapter,
)
from src.data.demand_contract import DemandContractViolationError


class TestDemandTelemetrySecurityAndDefaults:
    """Validate security invariants, environment variable extraction, and credential masking."""

    def test_unauthenticated_without_env_vars(self, monkeypatch):
        monkeypatch.delenv("PJM_API_KEY", raising=False)
        monkeypatch.delenv("EIA_API_KEY", raising=False)

        pjm = PJMDataMinerAdapter()
        assert not pjm.is_authenticated

        eia = EIADemandAdapter()
        assert not eia.is_authenticated

        with pytest.raises(DemandTelemetryAuthError, match="PJM Data Miner 2 authentication key"):
            pjm.fetch_recent_demand("2026-10-15T00:00:00Z", "2026-10-15T12:00:00Z")

        with pytest.raises(DemandTelemetryAuthError, match="EIA API v2 authentication key"):
            eia.fetch_recent_demand("2026-10-15T00:00:00Z", "2026-10-15T12:00:00Z")

    def test_api_keys_are_never_logged_or_exposed_in_str(self):
        secret_key = "super_secret_pjm_api_token_12345"
        adapter = PJMDataMinerAdapter(api_key=secret_key)
        rep = repr(adapter)
        s = str(adapter)
        assert secret_key not in rep
        assert secret_key not in s
        assert adapter.is_authenticated

    def test_factory_returns_unconfigured_when_no_keys(self, monkeypatch):
        monkeypatch.delenv("PJM_API_KEY", raising=False)
        monkeypatch.delenv("EIA_API_KEY", raising=False)
        adapter = get_operational_demand_adapter()
        assert not adapter.is_authenticated


class TestPJMDataMinerAdapter:
    """Validate PJM Data Miner 2 parsing, 5-min aggregation, and contract validation."""

    def test_successful_fetch_and_resampling(self):
        adapter = PJMDataMinerAdapter(api_key="valid_pjm_subscription_key")

        # Mock 5-minute instantaneous load records for 1 hour
        mock_items = []
        for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
            mock_items.append({
                "datetime_beginning_utc": f"2026-10-15 01:{minute:02d}",
                "instantaneous_load": 32000.0 + minute * 10,
            })

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": mock_items}

        with patch("requests.get", return_value=mock_resp) as mock_get:
            df = adapter.fetch_recent_demand("2026-10-15T01:00:00Z", "2026-10-15T02:00:00Z")
            assert mock_get.called
            # Verify headers sent masked key without leakage to logs
            headers = mock_get.call_args[1]["headers"]
            assert headers["Ocp-Apim-Subscription-Key"] == "valid_pjm_subscription_key"

        assert len(df) == 1
        assert df["consumption"].iloc[0] > 32000.0
        assert df["timestamp"].iloc[0] == pd.Timestamp("2026-10-15 01:00:00", tz="UTC")

    def test_out_of_bounds_demand_triggers_contract_violation(self):
        adapter = PJMDataMinerAdapter(api_key="valid_pjm_subscription_key")
        mock_items = [{
            "datetime_beginning_utc": "2026-10-15 01:00",
            "instantaneous_load": 999999.0,  # Physically impossible for PJM East
        }]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": mock_items}

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(DemandContractViolationError, match="violated demand contract"):
                adapter.fetch_recent_demand("2026-10-15T01:00:00Z", "2026-10-15T02:00:00Z")


class TestEIADemandAdapter:
    """Validate EIA API v2 client parsing and error handling."""

    def test_successful_fetch_eia(self):
        adapter = EIADemandAdapter(api_key="valid_eia_api_key_1234")

        mock_records = [
            {"period": "2026-10-15T01", "value": 31500.0},
            {"period": "2026-10-15T02", "value": 31200.0},
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": {"data": mock_records}}

        with patch("requests.get", return_value=mock_resp):
            df = adapter.fetch_recent_demand("2026-10-15T01:00:00Z", "2026-10-15T02:00:00Z")

        assert len(df) == 2
        assert list(df["consumption"]) == [31500.0, 31200.0]
        assert df["timestamp"].iloc[0] == pd.Timestamp("2026-10-15 01:00:00", tz="UTC")

    def test_eia_auth_rejection(self):
        adapter = EIADemandAdapter(api_key="invalid_eia_key")
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.raise_for_status.side_effect = Exception("403 Forbidden")

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(DemandTelemetryAuthError, match="rejected API credentials"):
                adapter.fetch_recent_demand("2026-10-15T01:00:00Z", "2026-10-15T02:00:00Z")

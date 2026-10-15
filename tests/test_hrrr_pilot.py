"""Deterministic unit tests for historical NOAA HRRR archive pilot fetcher."""

from pathlib import Path
import pytest

from src.weather.hrrr_pilot import (
    GenuineHRRRPilotFetcher,
    GenuineHRRRForecastRecord,
    HISTORICAL_HRRR_CACHE_DIR,
)


class TestGenuineHRRRPilotFetcher:
    """Validate NOAA HRRR S3 URL construction, byte parsing, and record schemas."""

    def test_url_construction(self):
        fetcher = GenuineHRRRPilotFetcher()
        grib_url, idx_url = fetcher.build_urls("20180712", 0, 1)

        expected_grib = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20180712/conus/hrrr.t00z.wrfsfcf01.grib2"
        expected_idx = f"{expected_grib}.idx"

        assert grib_url == expected_grib
        assert idx_url == expected_idx

    def test_url_construction_multi_digit_cycles_and_leads(self):
        fetcher = GenuineHRRRPilotFetcher()
        grib_url, idx_url = fetcher.build_urls("20180105", 18, 24)

        assert "hrrr.t18z.wrfsfcf24.grib2" in grib_url
        assert "hrrr.20180105" in grib_url
        assert idx_url == f"{grib_url}.idx"

    def test_cached_record_extraction(self):
        """If a cached GRIB2 message exists from pilot download, verify it parses deterministically."""
        cache_file = HISTORICAL_HRRR_CACHE_DIR / "hrrr_20180712_t00z_f01_tmp2m.grib2"
        if not cache_file.is_file():
            pytest.skip("Historical HRRR 2018-07-12 cache file not present in local test environment.")

        fetcher = GenuineHRRRPilotFetcher(cache_dir=HISTORICAL_HRRR_CACHE_DIR)
        rec = fetcher.fetch_and_decode_cycle("20180712", 0, 1, use_cache=True)

        assert isinstance(rec, GenuineHRRRForecastRecord)
        assert rec.forecast_lead_hours == 1
        assert rec.cycle_init_utc == "2018-07-12T00:00:00+00:00"
        assert rec.valid_timestamp_utc == "2018-07-12T01:00:00+00:00"
        assert rec.availability_cutoff_utc == "2018-07-12T00:50:00+00:00"
        assert rec.provenance == "GENUINE_NOAA_HRRR_AWS_BDP"

        # Check plausible meteorological values for July 12, 2018 around Philadelphia
        assert 65.0 <= rec.kphl_temp_f <= 95.0
        assert 60.0 <= rec.kewr_temp_f <= 90.0
        assert 65.0 <= rec.kilg_temp_f <= 95.0
        assert rec.grib_message_bytes > 1_000_000

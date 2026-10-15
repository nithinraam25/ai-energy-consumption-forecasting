"""Targeted NOAA HRRR GRIB2 metadata parsing and feasibility loader for Phase 7C.

Provides lightweight, bandwidth-efficient inspection of archived HRRR forecasts
via HTTP range requests, WMO GRIB2 section parsing, Lambert Conformal coordinate
projection, bilinear spatial weight calculation for KPHL, and operational latency evaluation.
"""

from typing import Any
import struct
import math
import requests
import numpy as np
import pandas as pd


# NOAA HRRR CONUS 3km Lambert Conformal Grid Constants (from Section 3 metadata)
HRRR_LATIN1_DEG = 38.5
HRRR_LATIN2_DEG = 38.5
HRRR_LOV_DEG = -97.5  # 262.5° E
HRRR_DX_M = 3000.0
HRRR_DY_M = 3000.0
HRRR_NX = 1799
HRRR_NY = 1059
HRRR_FIRST_LAT = 21.138123
HRRR_FIRST_LON = -122.719528
HRRR_EARTH_RADIUS_M = 6371229.0  # WMO standard sphere radius

# KPHL Station Coordinates (Philadelphia International Airport)
KPHL_LAT = 39.8719
KPHL_LON = -75.2411

DEFAULT_HRRR_S3_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


def parse_hrrr_idx(
    idx_content: str,
    target_var: str = "TMP",
    target_level: str = "2 m above ground",
) -> dict[str, Any] | None:
    """Parse a NOAA HRRR .idx index file to locate field byte offsets.

    The NOAA .idx format consists of colon-separated fields:
    `record_num:byte_offset:date_str:var:level:fcst_str:...`

    Args:
        idx_content: Plain text content of the .idx file.
        target_var: Target variable identifier (e.g. 'TMP'). Defaults to 'TMP'.
        target_level: Target vertical level (e.g. '2 m above ground').

    Returns:
        dict[str, Any] | None: Metadata dictionary with 'record_num', 'start_byte',
            'end_byte', 'byte_length', 'date_str', 'fcst_lead', or None if not found.
    """
    lines = [line.strip() for line in idx_content.strip().split("\n") if line.strip()]
    if not lines:
        return None

    parsed_records = []
    for line in lines:
        parts = line.split(":")
        if len(parts) >= 6:
            parsed_records.append({
                "record_num": int(parts[0]),
                "start_byte": int(parts[1]),
                "date_str": parts[2],
                "var": parts[3],
                "level": parts[4],
                "fcst": parts[5],
            })

    for i, rec in enumerate(parsed_records):
        if rec["var"] == target_var and rec["level"] == target_level:
            # End byte is the start of the next record minus 1 (or end of file if last)
            if i + 1 < len(parsed_records):
                end_byte = parsed_records[i + 1]["start_byte"] - 1
            else:
                end_byte = None

            byte_length = (end_byte - rec["start_byte"] + 1) if end_byte is not None else None
            return {
                "record_num": rec["record_num"],
                "start_byte": rec["start_byte"],
                "end_byte": end_byte,
                "byte_length": byte_length,
                "date_str": rec["date_str"],
                "var": rec["var"],
                "level": rec["level"],
                "fcst": rec["fcst"],
            }

    return None


def parse_grib2_metadata(grib_bytes: bytes) -> dict[str, Any]:
    """Parse WMO GRIB2 message binary sections to extract verifiable metadata.

    Decodes:
    - Section 0 (Indicator): Discipline and edition number.
    - Section 1 (Identification): Originating center and reference/init timestamp.
    - Section 3 (Grid Definition): Projection template, dimensions (Nx, Ny), resolution.
    - Section 4 (Product Definition): Parameter category, parameter number, lead time, level.
    - Section 5 (Data Representation): Packing template, reference value in Kelvin.

    Args:
        grib_bytes: Complete raw bytes of an isolated GRIB2 message.

    Returns:
        dict[str, Any]: Verified metadata extracted directly from binary headers.

    Raises:
        ValueError: If bytes do not start with b'GRIB' or terminate with b'7777'.
    """
    if len(grib_bytes) < 16 or not grib_bytes.startswith(b"GRIB"):
        raise ValueError("Invalid GRIB2 message: Header does not start with b'GRIB'.")
    if not grib_bytes.endswith(b"7777"):
        raise ValueError("Invalid GRIB2 message: Terminator does not end with b'7777'.")

    disc, edition, total_len = struct.unpack(">B B Q", grib_bytes[6:16])
    if edition != 2:
        raise ValueError(f"Unsupported GRIB edition: {edition} (expected 2).")

    meta: dict[str, Any] = {
        "discipline": disc,
        "edition": edition,
        "total_message_bytes": total_len,
    }

    offset = 16
    while offset < len(grib_bytes) - 4:
        sec_len, sec_num = struct.unpack(">I B", grib_bytes[offset:offset + 5])

        if sec_num == 1:
            center, subcenter, master_ver, local_ver, sig_time = struct.unpack(
                ">H H B B B", grib_bytes[offset + 5:offset + 12]
            )
            year, month, day, hour, minute, second = struct.unpack(
                ">H B B B B B", grib_bytes[offset + 12:offset + 19]
            )
            meta["originating_center"] = center
            meta["subcenter"] = subcenter
            meta["init_timestamp"] = pd.Timestamp(
                year=year, month=month, day=day, hour=hour, minute=minute, second=second, tz="UTC"
            )

        elif sec_num == 3:
            num_pts, = struct.unpack(">I", grib_bytes[offset + 6:offset + 10])
            gdt, = struct.unpack(">H", grib_bytes[offset + 12:offset + 14])
            meta["grid_template"] = gdt
            meta["num_grid_points"] = num_pts

            if gdt == 30:  # Lambert Conformal Conic
                nx, ny, la1, lo1 = struct.unpack(">I I i i", grib_bytes[offset + 30:offset + 46])
                lov, dx, dy = struct.unpack(">i I I", grib_bytes[offset + 51:offset + 63])
                latin1, latin2 = struct.unpack(">i i", grib_bytes[offset + 65:offset + 73])
                meta["nx"] = nx
                meta["ny"] = ny
                meta["first_lat_deg"] = la1 / 1e6
                meta["first_lon_deg"] = lo1 / 1e6
                meta["lov_deg"] = lov / 1e6
                meta["dx_m"] = dx / 1e3
                meta["dy_m"] = dy / 1e3
                meta["latin1_deg"] = latin1 / 1e6
                meta["latin2_deg"] = latin2 / 1e6

        elif sec_num == 4:
            pdt, = struct.unpack(">H", grib_bytes[offset + 7:offset + 9])
            cat, param, gen_proc = struct.unpack(">B B B", grib_bytes[offset + 9:offset + 12])
            time_unit, fcst_time = struct.unpack(">B I", grib_bytes[offset + 17:offset + 22])
            surf_type, surf_scale, surf_val = struct.unpack(">B B I", grib_bytes[offset + 22:offset + 28])
            meta["product_template"] = pdt
            meta["param_category"] = cat
            meta["param_number"] = param
            meta["time_unit_code"] = time_unit
            meta["forecast_lead_hours"] = fcst_time
            meta["surface_type"] = surf_type
            meta["surface_level_m"] = surf_val * (10 ** (-surf_scale))

        elif sec_num == 5:
            drt, = struct.unpack(">H", grib_bytes[offset + 9:offset + 11])
            ref_val, bin_scale, dec_scale, nbits = struct.unpack(
                ">f h h B", grib_bytes[offset + 11:offset + 20]
            )
            meta["data_representation_template"] = drt
            meta["reference_value_k"] = ref_val
            meta["binary_scale_factor"] = bin_scale
            meta["decimal_scale_factor"] = dec_scale
            meta["bits_per_value"] = nbits

        offset += sec_len

    if "init_timestamp" in meta and "forecast_lead_hours" in meta:
        meta["valid_timestamp"] = meta["init_timestamp"] + pd.Timedelta(hours=meta["forecast_lead_hours"])

    return meta


def project_latlon_to_hrrr_grid(
    lat_deg: float,
    lon_deg: float,
    earth_radius: float = HRRR_EARTH_RADIUS_M,
) -> tuple[float, float]:
    """Compute exact fractional (i, j) coordinates on the HRRR Lambert Conformal grid.

    Mathematical formulation for secant/tangent Lambert Conformal Conic:
        lat1 = lat2 = 38.5 deg
        n = sin(lat1)
        F = (cos(lat1) * tan(pi/4 + lat1/2)^n) / n
        rho = R * F / (tan(pi/4 + lat/2)^n)
        dlon = lon - LoV
        x = rho * sin(n * dlon)
        y = -rho * cos(n * dlon)
        i = (x - x_0) / dx
        j = (y - y_0) / dy

    Args:
        lat_deg: Geodetic latitude in degrees North.
        lon_deg: Geodetic longitude in degrees (-180 to 180 or 0 to 360).
        earth_radius: Sphere radius in meters. Defaults to 6,371,229.0 m.

    Returns:
        tuple[float, float]: (i_fractional, j_fractional) zero-indexed grid coordinates.
    """
    lat1 = math.radians(HRRR_LATIN1_DEG)
    lat2 = math.radians(HRRR_LATIN2_DEG)
    lov = math.radians(HRRR_LOV_DEG)

    if abs(lat1 - lat2) < 1e-7:
        n = math.sin(lat1)
    else:
        n = math.log(math.cos(lat1) / math.cos(lat2)) / math.log(
            math.tan(math.pi / 4 + lat2 / 2) / math.tan(math.pi / 4 + lat1 / 2)
        )

    f = (math.cos(lat1) * (math.tan(math.pi / 4 + lat1 / 2) ** n)) / n

    def _to_xy(phi_deg: float, lambda_deg: float) -> tuple[float, float]:
        phi = math.radians(phi_deg)
        lam = math.radians(lambda_deg)
        dlon = lam - lov
        dlon = (dlon + math.pi) % (2 * math.pi) - math.pi
        rho = earth_radius * f / (math.tan(math.pi / 4 + phi / 2) ** n)
        x = rho * math.sin(n * dlon)
        y = -rho * math.cos(n * dlon)
        return x, y

    x0, y0 = _to_xy(HRRR_FIRST_LAT, HRRR_FIRST_LON)
    x_pt, y_pt = _to_xy(lat_deg, lon_deg)

    i_frac = (x_pt - x0) / HRRR_DX_M
    j_frac = (y_pt - y0) / HRRR_DY_M
    return i_frac, j_frac


def compute_kphl_spatial_weights(
    kphl_lat: float = KPHL_LAT,
    kphl_lon: float = KPHL_LON,
) -> dict[str, Any]:
    """Calculate the 4 surrounding HRRR grid nodes and normalized bilinear weights for KPHL.

    Bilinear interpolation weighting formula:
        Let delta_i = i_frac - floor(i_frac), delta_j = j_frac - floor(j_frac)
        w_00 = (1 - delta_i) * (1 - delta_j)   at (i_0, j_0)
        w_10 = delta_i * (1 - delta_j)         at (i_0 + 1, j_0)
        w_01 = (1 - delta_i) * delta_j         at (i_0, j_0 + 1)
        w_11 = delta_i * delta_j               at (i_0 + 1, j_0 + 1)
        Sum(w_ab) = 1.000000

    Args:
        kphl_lat: Latitude of KPHL station. Defaults to 39.8719° N.
        kphl_lon: Longitude of KPHL station. Defaults to -75.2411° W.

    Returns:
        dict[str, Any]: Node indices, fractional offsets, and normalized weights.
    """
    i_frac, j_frac = project_latlon_to_hrrr_grid(kphl_lat, kphl_lon)
    i0 = math.floor(i_frac)
    j0 = math.floor(j_frac)
    delta_i = i_frac - i0
    delta_j = j_frac - j0

    w00 = (1.0 - delta_i) * (1.0 - delta_j)
    w10 = delta_i * (1.0 - delta_j)
    w01 = (1.0 - delta_i) * delta_j
    w11 = delta_i * delta_j

    total_weight = w00 + w10 + w01 + w11

    # Nearest integer node
    i_nearest = round(i_frac)
    j_nearest = round(j_frac)

    return {
        "kphl_coordinates": (kphl_lat, kphl_lon),
        "fractional_indices": (i_frac, j_frac),
        "nearest_node": (i_nearest, j_nearest),
        "nodes": [
            {"node": (i0, j0), "weight": w00},
            {"node": (i0 + 1, j0), "weight": w10},
            {"node": (i0, j0 + 1), "weight": w01},
            {"node": (i0 + 1, j0 + 1), "weight": w11},
        ],
        "weight_sum": total_weight,
    }


def kelvin_to_fahrenheit(temp_k: float) -> float:
    """Convert absolute temperature from Kelvin to degrees Fahrenheit.

    Mathematical formula:
        T_F = (T_K - 273.15) * 9/5 + 32.0

    Args:
        temp_k: Temperature in Kelvin.

    Returns:
        float: Temperature in degrees Fahrenheit.
    """
    return (float(temp_k) - 273.15) * 9.0 / 5.0 + 32.0


def evaluate_operational_eligibility(
    target_hour_utc: pd.Timestamp,
    cycle_init_utc: pd.Timestamp,
    forecast_lead_hours: int,
    forecast_origin_lead_minutes: int = 60,
    model_latency_minutes: int = 50,
) -> dict[str, Any]:
    """Evaluate whether an NWP forecast is operationally admissible for a target demand hour.

    Admissibility conditions:
        1. Target Matching: valid_time == target_hour_utc
        2. Zero Lookahead: availability_time <= forecast_origin_time

    Timeline arithmetic:
        forecast_origin_time = target_hour_utc - forecast_origin_lead_minutes
        availability_time = cycle_init_utc + model_latency_minutes
        buffer_minutes = forecast_origin_time - availability_time

    Args:
        target_hour_utc: Target energy demand timestamp (UTC).
        cycle_init_utc: NWP model cycle initialization timestamp (UTC).
        forecast_lead_hours: Forecast projection lead time in hours.
        forecast_origin_lead_minutes: Lead time of energy forecast cutoff (e.g. 60 min for 1h-ahead).
        model_latency_minutes: Assumed elapsed time for NWP run and dissemination. Defaults to 50 min.

    Returns:
        dict[str, Any]: Detailed evaluation containing admissibility boolean,
            buffer minutes, and latency sensitivity assessment.
    """
    valid_time = cycle_init_utc + pd.Timedelta(hours=forecast_lead_hours)
    forecast_origin = target_hour_utc - pd.Timedelta(minutes=forecast_origin_lead_minutes)
    simulated_avail = cycle_init_utc + pd.Timedelta(minutes=model_latency_minutes)

    valid_matches_target = (valid_time == target_hour_utc)
    available_before_origin = (simulated_avail <= forecast_origin)
    buffer_min = (forecast_origin - simulated_avail).total_seconds() / 60.0

    is_admissible = bool(valid_matches_target and available_before_origin)

    # Sensitivity analysis: what is the maximum latency before this run becomes late?
    max_tolerable_latency_min = (forecast_origin - cycle_init_utc).total_seconds() / 60.0

    return {
        "target_hour_utc": target_hour_utc,
        "cycle_init_utc": cycle_init_utc,
        "forecast_lead_hours": forecast_lead_hours,
        "valid_time_utc": valid_time,
        "forecast_origin_utc": forecast_origin,
        "simulated_avail_utc": simulated_avail,
        "valid_matches_target": valid_matches_target,
        "available_before_origin": available_before_origin,
        "buffer_minutes": buffer_min,
        "max_tolerable_latency_minutes": max_tolerable_latency_min,
        "is_admissible": is_admissible,
    }


def fetch_hrrr_sample_field(
    date_str: str,
    cycle_hour: int,
    lead_hour: int = 2,
    base_url: str = DEFAULT_HRRR_S3_BASE,
    timeout_sec: int = 15,
) -> tuple[dict[str, Any], bytes]:
    """Retrieve sample 2m temperature metadata and raw message bytes via targeted HTTP Range.

    Args:
        date_str: Date string in YYYYMMDD format (e.g. '20141001').
        cycle_hour: NWP initialization cycle hour (0 to 23).
        lead_hour: Forecast lead time in hours. Defaults to 2 (f02).
        base_url: S3 HTTP base URL.
        timeout_sec: Network timeout in seconds.

    Returns:
        tuple[dict[str, Any], bytes]: (Parsed metadata dict, raw GRIB2 message bytes).

    Raises:
        FileNotFoundError: If index file or variable is not found.
        RuntimeError: If HTTP request fails.
    """
    prefix = f"hrrr.{date_str}/conus/hrrr.t{cycle_hour:02d}z.wrfprsf{lead_hour:02d}.grib2"
    idx_url = f"{base_url}/{prefix}.idx"
    grib_url = f"{base_url}/{prefix}"

    # 1. Fetch index file (~35 KB)
    idx_resp = requests.get(idx_url, timeout=timeout_sec)
    if idx_resp.status_code != 200:
        raise FileNotFoundError(f"HRRR index not found at {idx_url} (HTTP {idx_resp.status_code})")

    idx_info = parse_hrrr_idx(idx_resp.text, target_var="TMP", target_level="2 m above ground")
    if idx_info is None or idx_info.get("end_byte") is None:
        raise FileNotFoundError(f"'TMP:2 m above ground' not located in index {idx_url}")

    start_byte = idx_info["start_byte"]
    end_byte = idx_info["end_byte"]

    # 2. Targeted HTTP Range request (~1.2 MB)
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    grib_resp = requests.get(grib_url, headers=headers, timeout=timeout_sec)
    if grib_resp.status_code not in (200, 206):
        raise RuntimeError(f"Range request failed for {grib_url} (HTTP {grib_resp.status_code})")

    meta = parse_grib2_metadata(grib_resp.content)
    meta["source_url"] = grib_url
    meta["byte_range"] = (start_byte, end_byte)
    meta["http_status"] = grib_resp.status_code

    return meta, grib_resp.content

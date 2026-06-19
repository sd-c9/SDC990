"""
simulator.py — Synthetic rod-pump telemetry generator for SB-Guard.

Real PDO field data is confidential, so this module fabricates physically
plausible multivariate time-series for a fleet of beam-lift wells. The
generator models four operating regimes that progress toward a Stuffing Box
(SB) leak caused by polished-rod misalignment:

    NORMAL  -> STAGE1 (72h) -> STAGE2 (48h) -> STAGE3 (24h) -> FAILURE

Each stage perturbs the underlying signals in a way that mirrors the physics
of a degrading stuffing box: lateral vibration rises first (misalignment),
then the surface dynacard area shifts, then friction climbs, and finally a
sudden load anomaly marks the failure event.

The raw output deliberately includes noise, slow sensor drift, sparse missing
samples and occasional outliers so downstream feature engineering and models
are exercised on realistic, messy data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


# --------------------------------------------------------------------------- #
# Per-well baseline "personality"
# --------------------------------------------------------------------------- #
def _well_baseline(rng: np.random.Generator, well_id: str) -> dict:
    """Draw a stable baseline operating point unique to each well."""
    return {
        "well_id": well_id,
        "spm": rng.uniform(5.0, 9.0),                 # strokes/min
        "pprl": rng.uniform(18000, 26000),            # lb
        "mprl": rng.uniform(6000, 11000),             # lb
        "fillage": rng.uniform(82, 97),               # %
        "gearbox_temp": rng.uniform(55, 70),          # degC
        "rod_stretch": rng.uniform(0.9, 1.1),         # normalized factor
        "flp": rng.uniform(9000, 15000),              # lb fluid load
        "lvi": rng.uniform(0.05, 0.18),               # baseline vibration
        "friction": rng.uniform(0.08, 0.16),          # SB friction coeff
        "tds": rng.uniform(0.02, 0.08),               # torque deviation
        "drift_dir": rng.choice([-1, 1], size=6),     # sensor drift signs
    }


# --------------------------------------------------------------------------- #
# Degradation profile
# --------------------------------------------------------------------------- #
def _stage_severity(hours_to_failure: np.ndarray) -> np.ndarray:
    """
    Map "hours remaining until failure" to a 0..1 degradation severity.

    Returns 0 far from failure and ramps smoothly toward 1 at the failure
    instant. Wells that never fail receive +inf hours_to_failure -> 0.
    """
    sev = np.zeros_like(hours_to_failure, dtype=float)
    h = hours_to_failure
    # Stage 1 window (72h -> 48h): gentle onset
    s1 = (h <= 72) & (h > 48)
    sev[s1] = 0.15 * (72 - h[s1]) / 24.0
    # Stage 2 window (48h -> 24h)
    s2 = (h <= 48) & (h > 24)
    sev[s2] = 0.15 + 0.35 * (48 - h[s2]) / 24.0
    # Stage 3 window (24h -> 0h): steep climb
    s3 = (h <= 24) & (h >= 0)
    sev[s3] = 0.50 + 0.50 * (24 - h[s3]) / 24.0
    return np.clip(sev, 0.0, 1.0)


def simulate_well(well_id: str,
                  fails: bool,
                  seed: int | None = None) -> pd.DataFrame:
    """
    Generate the full 6-month, 10-minute-interval record for one well.

    Parameters
    ----------
    well_id : str
        Identifier, e.g. "PDO-NIM-014".
    fails : bool
        Whether this well experiences an SB failure event in the window.
    seed : int, optional
        Per-well RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    base = _well_baseline(rng, well_id)

    n = cfg.SAMPLES_PER_DAY * cfg.DAYS_OF_DATA
    t_index = pd.date_range("2025-12-01", periods=n, freq=f"{cfg.SAMPLE_INTERVAL_MIN}min")
    minutes = np.arange(n) * cfg.SAMPLE_INTERVAL_MIN
    hours = minutes / 60.0
    day_phase = 2 * np.pi * (minutes % 1440) / 1440.0   # daily duty cycle

    # ---- failure timing ---------------------------------------------------- #
    if fails:
        # Place the failure event somewhere in the back half of the window.
        fail_idx = rng.integers(int(n * 0.55), int(n * 0.95))
        hours_to_failure = (fail_idx - np.arange(n)) * cfg.SAMPLE_INTERVAL_MIN / 60.0
        hours_to_failure[hours_to_failure < 0] = np.inf  # post-failure: well down
    else:
        fail_idx = -1
        hours_to_failure = np.full(n, np.inf)

    sev = _stage_severity(hours_to_failure)

    # ---- base diurnal + slow drift ---------------------------------------- #
    diurnal = 1.0 + 0.03 * np.sin(day_phase)
    drift = 1.0 + base["drift_dir"][0] * cfg.SENSOR_DRIFT_MAX * (np.arange(n) / n)

    # ---- core signals ------------------------------------------------------ #
    spm = base["spm"] * diurnal + rng.normal(0, 0.05, n)

    # Lateral vibration: primary misalignment indicator — rises sharply.
    lvi = (base["lvi"] * diurnal
           + sev * (0.9 + 0.4 * rng.random(n))            # strong stage growth
           + rng.normal(0, 0.015, n))

    # SB friction: derived coefficient climbs mostly in stage 2/3.
    friction = (base["friction"]
                + (sev ** 1.6) * 0.55
                + rng.normal(0, 0.01, n))

    # Torque deviation grows with misalignment.
    tds = base["tds"] + sev * 0.6 + rng.normal(0, 0.01, n)

    # Peak / min polished rod loads — card distortion in later stages.
    pprl = (base["pprl"] * drift * diurnal
            + sev * 2600 * np.sin(day_phase + sev)       # asymmetric distortion
            + rng.normal(0, 180, n))
    mprl = (base["mprl"] * diurnal
            - sev * 900
            + rng.normal(0, 120, n))

    # Pump fillage degrades modestly.
    fillage = np.clip(base["fillage"] - sev * 14 + rng.normal(0, 1.2, n), 25, 100)

    # Gearbox temp rises with friction load.
    gearbox_temp = (base["gearbox_temp"] * diurnal
                    + sev * 12 + friction * 18
                    + rng.normal(0, 0.8, n))

    rod_stretch = base["rod_stretch"] + sev * 0.25 + rng.normal(0, 0.01, n)
    flp = base["flp"] * diurnal + sev * 1500 + rng.normal(0, 90, n)

    # Surface card area (energy/stroke): proportional to load range * stroke.
    prlr = pprl - mprl
    sca = prlr * (1.0 + 0.4 * sev) * 0.001 * (1 + 0.05 * np.sin(day_phase))
    sca += rng.normal(0, sca.std() * 0.02 + 1e-6, n)

    # ---- failure event spike ---------------------------------------------- #
    label_24 = np.zeros(n, dtype=int)
    label_48 = np.zeros(n, dtype=int)
    label_72 = np.zeros(n, dtype=int)
    failure_flag = np.zeros(n, dtype=int)
    stage = np.zeros(n, dtype=int)

    if fails:
        # Multi-horizon labels: 1 if a failure occurs within the horizon.
        h2f = (fail_idx - np.arange(n)) * cfg.SAMPLE_INTERVAL_MIN / 60.0
        label_24[(h2f > 0) & (h2f <= 24)] = 1
        label_48[(h2f > 0) & (h2f <= 48)] = 1
        label_72[(h2f > 0) & (h2f <= 72)] = 1
        stage[(h2f > 48) & (h2f <= 72)] = 1
        stage[(h2f > 24) & (h2f <= 48)] = 2
        stage[(h2f > 0) & (h2f <= 24)] = 3

        # Sudden load anomaly at the failure instant + brief well-down window.
        end = min(fail_idx + cfg.SAMPLES_PER_HOUR * 3, n)
        pprl[fail_idx:end] += rng.uniform(4000, 7000)
        lvi[fail_idx:end] += rng.uniform(1.5, 2.5)
        friction[fail_idx:end] += rng.uniform(0.4, 0.7)
        failure_flag[fail_idx] = 1
        stage[fail_idx:end] = 4
        # After remediation the well returns to a clean baseline-ish state.
        if end < n:
            lvi[end:] = base["lvi"] + rng.normal(0, 0.02, n - end)
            friction[end:] = base["friction"] + rng.normal(0, 0.01, n - end)

    df = pd.DataFrame({
        "timestamp": t_index,
        "well_id": well_id,
        "SPM": spm,
        "PPRL": pprl,
        "MPRL": mprl,
        "PRLR": prlr,
        "SCA": sca,
        "pump_fillage": fillage,
        "LVI": lvi,
        "sb_friction": friction,
        "TDS": tds,
        "gearbox_temp": gearbox_temp,
        "rod_stretch": rod_stretch,
        "FLP": flp,
        "stage": stage,
        "failure_event": failure_flag,
        "label_24h": label_24,
        "label_48h": label_48,
        "label_72h": label_72,
        "well_fails": int(fails),
    })

    df = _inject_realism(df, rng)
    return df


def _inject_realism(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add outliers and missing data to the continuous sensor channels."""
    sensor_cols = ["SPM", "PPRL", "MPRL", "PRLR", "SCA", "pump_fillage",
                   "LVI", "sb_friction", "TDS", "gearbox_temp",
                   "rod_stretch", "FLP"]
    n = len(df)

    # Outliers: multiplicative spikes on random cells.
    n_out = int(n * cfg.OUTLIER_RATE)
    if n_out:
        rows = rng.integers(0, n, n_out)
        cols = rng.choice(sensor_cols, n_out)
        for r, c in zip(rows, cols):
            df.at[r, c] *= rng.uniform(1.5, 3.0)

    # Missing data: blank out random cells (NaN) — preprocessor will impute.
    n_miss = int(n * cfg.MISSING_DATA_RATE)
    if n_miss:
        rows = rng.integers(0, n, n_miss)
        cols = rng.choice(sensor_cols, n_miss)
        for r, c in zip(rows, cols):
            df.at[r, c] = np.nan

    return df


def fleet_failure_assignment(n_wells: int, seed: int) -> list[bool]:
    """Decide which wells in the fleet will experience a failure."""
    rng = np.random.default_rng(seed)
    n_fail = int(round(n_wells * cfg.FAILURE_WELL_FRACTION))
    flags = [True] * n_fail + [False] * (n_wells - n_fail)
    rng.shuffle(flags)
    return flags


# Geographic-flavoured well names for the southern Oman fleet.
_FIELDS = ["NIM", "MRM", "AML", "QRN", "SAY", "RML", "HBR", "YBL"]


def well_name(idx: int, rng: np.random.Generator) -> str:
    field = _FIELDS[idx % len(_FIELDS)]
    return f"PDO-{field}-{idx + 1:03d}"

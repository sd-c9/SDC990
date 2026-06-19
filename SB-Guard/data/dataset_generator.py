"""
dataset_generator.py — Build and persist the full SB-Guard training dataset.

Generates the simulated fleet, runs feature engineering, and writes:

    data/generated/fleet_raw.parquet        full raw telemetry (all wells)
    data/generated/fleet_features.parquet   engineered features + labels
    data/generated/fleet_features_sample.csv human-readable sample
    data/generated/well_index.csv            per-well metadata (fails y/n)

Run directly:  python -m data.dataset_generator
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config as cfg
from data import simulator, preprocessor


def generate(verbose: bool = True) -> dict:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    fail_flags = simulator.fleet_failure_assignment(cfg.N_WELLS, cfg.RANDOM_SEED)

    frames = []
    index_rows = []
    for i in range(cfg.N_WELLS):
        wid = simulator.well_name(i, rng)
        fails = fail_flags[i]
        df = simulator.simulate_well(wid, fails, seed=cfg.RANDOM_SEED + i)
        frames.append(df)
        index_rows.append({
            "well_id": wid,
            "fails": int(fails),
            "n_samples": len(df),
            "failure_events": int(df["failure_event"].sum()),
        })
        if verbose:
            print(f"  [{i+1:>2}/{cfg.N_WELLS}] {wid}  fails={fails}  rows={len(df)}")

    raw = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"Raw fleet telemetry: {raw.shape[0]:,} rows x {raw.shape[1]} cols")
        print("Engineering features ...")

    feats = preprocessor.engineer_features(raw)

    os.makedirs(cfg.DATASET_DIR, exist_ok=True)
    raw_path = os.path.join(cfg.DATASET_DIR, "fleet_raw.parquet")
    feat_path = os.path.join(cfg.DATASET_DIR, "fleet_features.parquet")
    sample_path = os.path.join(cfg.DATASET_DIR, "fleet_features_sample.csv")
    index_path = os.path.join(cfg.DATASET_DIR, "well_index.csv")

    raw.to_parquet(raw_path, index=False)
    feats.to_parquet(feat_path, index=False)
    feats.head(5000).to_csv(sample_path, index=False)
    pd.DataFrame(index_rows).to_csv(index_path, index=False)

    if verbose:
        print(f"Saved raw       -> {raw_path}")
        print(f"Saved features  -> {feat_path}")
        print(f"Saved sample    -> {sample_path}")
        print(f"Saved index     -> {index_path}")

    return {
        "raw_path": raw_path,
        "feat_path": feat_path,
        "index_path": index_path,
        "n_rows": len(feats),
        "n_wells": cfg.N_WELLS,
        "n_failing_wells": int(sum(fail_flags)),
    }


def load_features() -> pd.DataFrame:
    path = os.path.join(cfg.DATASET_DIR, "fleet_features.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Feature dataset not found. Run `python -m data.dataset_generator` first."
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    print("=" * 60)
    print("SB-Guard :: Synthetic dataset generation")
    print("=" * 60)
    summary = generate()
    print("-" * 60)
    print(f"Done. {summary['n_rows']:,} engineered rows across "
          f"{summary['n_wells']} wells "
          f"({summary['n_failing_wells']} with SB failures).")

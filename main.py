"""Sanitized reference pipeline for smartphone log-based TTE modeling.

This research-code snapshot demonstrates the core data flow used by the study:

1. align public MDF logs to a minute grid;
2. identify discharge-only segments;
3. construct instantaneous features and recursive multi-exponential tail states;
4. create participant-group train/validation/test splits;
5. build TTE origins without using future usage/context records as predictors;
6. compare transparent rate-based baselines and tail-state ablations.

The program contains no dataset rows, participant identifiers, machine-specific
paths, credentials, or hard-coded manuscript results. MDF schema variants may
require local adaptation. See README.md before using the code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls


DEFAULT_DATA_ROOT = Path("data/MDF")
DEFAULT_WORK_DIR = Path("artifacts")
DEFAULT_INITIAL_LEVELS = (0.20, 0.40, 0.60, 0.80, 1.00)
DEFAULT_THRESHOLDS = (0.05, 0.20)

EVENT_FILES: dict[str, tuple[str, ...]] = {
    "wifi_evt_cnt": ("wifi_scans.csv",),
    "cell_evt_cnt": ("cells.csv",),
    "call_evt_cnt": ("calls.csv",),
    "media_evt_cnt": ("multimedia.csv",),
    "p2p_evt_cnt": ("wifi_p2p_scans.csv",),
    "bt_scan_evt_cnt": ("bt_scan.csv",),
    "bt_conn_evt_cnt": ("bt_conn.csv",),
}

TAIL_TIME_CONSTANTS_SECONDS: dict[str, tuple[int, ...]] = {
    "wifi_evt_cnt": (30, 120, 600),
    "cell_evt_cnt": (30, 120, 900),
    "call_evt_cnt": (60, 300, 1200),
    "media_evt_cnt": (30, 120, 600),
    "p2p_evt_cnt": (30, 120, 600),
    "bt_scan_evt_cnt": (30, 120, 600),
    "bt_conn_evt_cnt": (30, 120, 600),
}

INSTANT_FEATURES = (
    "screen_on",
    "running_app_cnt",
    "running_category_cnt",
    "wifi_evt_cnt",
    "cell_evt_cnt",
    "call_evt_cnt",
    "media_evt_cnt",
    "p2p_evt_cnt",
    "bt_scan_evt_cnt",
    "bt_conn_evt_cnt",
)


def eprint(*values: object) -> None:
    print(*values, file=sys.stderr)


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse a comma-separated list into a sorted unique tuple."""
    values = tuple(sorted({float(item.strip()) for item in text.split(",") if item.strip()}))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return values


def json_ready(value: object) -> object:
    """Convert common NumPy/Path values for stable JSON output."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def read_csv_optional(path: Path, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    """Read an optional CSV without exposing its path in generated output."""
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        eprint(f"warning: skipped an unreadable optional CSV named {path.name}: {error}")
        return pd.DataFrame()


def choose_column(frame: pd.DataFrame, candidates: Iterable[str], *, required: bool = False) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ValueError(f"none of the required columns are present: {tuple(candidates)}")
    return None


def parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse MDF numeric timestamps or ISO-like strings as UTC."""
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if not finite.empty:
        median = float(np.median(np.abs(finite.to_numpy(float))))
        if median >= 1e14:
            unit = "us"
        elif median >= 1e11:
            unit = "ms"
        elif median >= 1e8:
            unit = "s"
        else:
            unit = None
        if unit is not None:
            return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def to_minute(series: pd.Series) -> pd.Series:
    return parse_timestamp(series).dt.floor("min")


def numeric_boolean(series: pd.Series) -> pd.Series:
    """Map common numeric/string states to 0/1 while preserving missing values."""
    numeric = pd.to_numeric(series, errors="coerce")
    output = pd.Series(np.nan, index=series.index, dtype=float)
    output.loc[numeric.notna()] = (numeric.loc[numeric.notna()] != 0).astype(float)
    text = series.astype(str).str.strip().str.lower()
    output.loc[text.isin({"true", "yes", "on", "charging"})] = 1.0
    output.loc[text.isin({"false", "no", "off", "discharging"})] = 0.0
    return output


def participant_directories(data_root: Path) -> list[Path]:
    """Find directories that contain the required battery log."""
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"dataset directory not found: {data_root.as_posix()}\n"
            "Download the public MDF dataset and see README.md for the expected layout."
        )
    directories = sorted(
        (path for path in data_root.iterdir() if path.is_dir() and (path / "battery.csv").is_file()),
        key=lambda path: path.name.casefold(),
    )
    if not directories:
        raise FileNotFoundError("no participant directories containing battery.csv were found")
    return directories


def battery_minute_table(participant_dir: Path, max_interpolation_gap: int) -> pd.DataFrame:
    raw = read_csv_optional(participant_dir / "battery.csv")
    if raw.empty:
        return pd.DataFrame()
    time_column = choose_column(raw, ("time", "timestamp", "datetime"), required=True)
    level_column = choose_column(raw, ("level", "soc", "battery_level", "percentage"), required=True)
    charging_column = choose_column(raw, ("charging", "is_charging", "charge"), required=False)

    work = pd.DataFrame(
        {
            "time_min": to_minute(raw[time_column]),
            "SOC_raw": pd.to_numeric(raw[level_column], errors="coerce"),
        }
    )
    if charging_column:
        work["charging_raw"] = numeric_boolean(raw[charging_column])
    else:
        work["charging_raw"] = 0.0
    work = work.dropna(subset=["time_min", "SOC_raw"]).sort_values("time_min")
    if work.empty:
        return pd.DataFrame()

    if float(work["SOC_raw"].quantile(0.95)) > 1.5:
        work["SOC_raw"] = work["SOC_raw"] / 100.0
    work["SOC_raw"] = work["SOC_raw"].clip(0.0, 1.0)

    minute = work.groupby("time_min", sort=True).agg(
        SOC_observed=("SOC_raw", "last"),
        charging=("charging_raw", "max"),
    )
    full_index = pd.date_range(minute.index.min(), minute.index.max(), freq="min", tz="UTC")
    minute = minute.reindex(full_index)
    minute.index.name = "time_min"
    minute["battery_observed"] = minute["SOC_observed"].notna().astype(np.int8)
    minute["SOC"] = minute["SOC_observed"].interpolate(
        method="linear", limit=max_interpolation_gap, limit_area="inside"
    )
    minute["charging"] = minute["charging"].ffill(limit=max_interpolation_gap).fillna(0.0)
    return minute


def screen_feature(participant_dir: Path, minute_index: pd.DatetimeIndex) -> pd.Series:
    raw = read_csv_optional(participant_dir / "display.csv")
    output = pd.Series(0.0, index=minute_index, name="screen_on")
    if raw.empty:
        return output
    time_column = choose_column(raw, ("time", "timestamp", "datetime"), required=False)
    state_column = choose_column(raw, ("state", "screen_state", "display_state"), required=False)
    if not time_column or not state_column:
        return output
    time_min = to_minute(raw[time_column])
    state = pd.to_numeric(raw[state_column], errors="coerce")
    finite = state[np.isfinite(state)]
    if finite.empty:
        return output
    unique = np.sort(finite.unique())
    # MDF mirrors differ in display-state encoding. For two or more ordered
    # numeric states, the largest state is treated as active; for binary states,
    # any nonzero value is active. Confirm this rule for a local dataset copy.
    if len(unique) >= 2 and unique[-1] > 1:
        active = (state == unique[-1]).astype(float)
    else:
        active = (state != 0).astype(float)
    grouped = pd.DataFrame({"time_min": time_min, "value": active}).dropna(subset=["time_min"])
    if grouped.empty:
        return output
    per_minute = grouped.groupby("time_min")["value"].last()
    output.loc[output.index.intersection(per_minute.index)] = per_minute.reindex(output.index).dropna()
    return output


def running_app_features(participant_dir: Path, minute_index: pd.DatetimeIndex) -> pd.DataFrame:
    raw = read_csv_optional(participant_dir / "running_apps.csv")
    output = pd.DataFrame(
        {"running_app_cnt": 0.0, "running_category_cnt": 0.0}, index=minute_index
    )
    if raw.empty:
        return output
    time_column = choose_column(raw, ("time", "timestamp", "datetime"), required=False)
    app_column = choose_column(raw, ("app", "package", "package_name"), required=False)
    category_column = choose_column(raw, ("category", "app_category"), required=False)
    if not time_column:
        return output
    work = pd.DataFrame({"time_min": to_minute(raw[time_column])})
    work["app"] = raw[app_column].astype(str) if app_column else "event"
    work["category"] = raw[category_column].astype(str) if category_column else "unknown"
    work = work.dropna(subset=["time_min"])
    if work.empty:
        return output
    grouped = work.groupby("time_min").agg(
        running_app_cnt=("app", "nunique"),
        running_category_cnt=("category", "nunique"),
    )
    common = output.index.intersection(grouped.index)
    output.loc[common, ["running_app_cnt", "running_category_cnt"]] = grouped.loc[
        common, ["running_app_cnt", "running_category_cnt"]
    ].to_numpy(float)
    return output


def event_count(participant_dir: Path, filenames: Sequence[str], minute_index: pd.DatetimeIndex) -> pd.Series:
    output = pd.Series(0.0, index=minute_index)
    for filename in filenames:
        raw = read_csv_optional(participant_dir / filename)
        if raw.empty:
            continue
        time_column = choose_column(raw, ("time", "timestamp", "datetime"), required=False)
        if not time_column:
            continue
        time_min = to_minute(raw[time_column]).dropna()
        if time_min.empty:
            continue
        counts = time_min.value_counts(sort=False).astype(float)
        common = output.index.intersection(counts.index)
        output.loc[common] = output.loc[common].to_numpy(float) + counts.loc[common].to_numpy(float)
    return output


def recursive_tail_state(events: np.ndarray, tau_seconds: float, dt_seconds: float = 60.0) -> np.ndarray:
    """Compute z_t = exp(-dt/tau) z_(t-1) + event_t."""
    if tau_seconds <= 0:
        raise ValueError("tail time constants must be positive")
    decay = math.exp(-dt_seconds / tau_seconds)
    state = np.zeros(len(events), dtype=float)
    previous = 0.0
    for index, event in enumerate(np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)):
        previous = decay * previous + max(float(event), 0.0)
        state[index] = previous
    return state


def add_tail_states(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for event_name, constants in TAIL_TIME_CONSTANTS_SECONDS.items():
        events = pd.to_numeric(output[event_name], errors="coerce").fillna(0.0).to_numpy(float)
        prefix = event_name.removesuffix("_evt_cnt")
        for tau in constants:
            output[f"{prefix}_z_tau{tau}"] = recursive_tail_state(events, float(tau))
    return output


def discharge_segment_ids(
    frame: pd.DataFrame,
    *,
    upward_jump_tolerance: float,
    maximum_observation_gap: int,
    minimum_segment_minutes: int,
) -> np.ndarray:
    """Assign local segment ids; invalid/non-discharge minutes receive -1."""
    soc = pd.to_numeric(frame["SOC"], errors="coerce").to_numpy(float)
    observed = frame["battery_observed"].to_numpy(bool)
    charging = pd.to_numeric(frame["charging"], errors="coerce").fillna(0.0).to_numpy(float)

    segment = np.full(len(frame), -1, dtype=np.int64)
    current = -1
    last_valid_index: int | None = None
    last_observed_index: int | None = None
    last_soc = float("nan")
    for index in range(len(frame)):
        if observed[index]:
            last_observed_index = index
        observation_gap = (
            index - last_observed_index if last_observed_index is not None else maximum_observation_gap + 1
        )
        valid = np.isfinite(soc[index]) and charging[index] < 0.5 and observation_gap <= maximum_observation_gap
        if not valid:
            last_valid_index = None
            last_soc = float("nan")
            continue
        starts_new = last_valid_index is None or index != last_valid_index + 1
        if np.isfinite(last_soc) and soc[index] > last_soc + upward_jump_tolerance:
            starts_new = True
        if starts_new:
            current += 1
        segment[index] = current
        last_valid_index = index
        last_soc = soc[index]

    counts = pd.Series(segment[segment >= 0]).value_counts()
    short_ids = set(counts[counts < minimum_segment_minutes].index.astype(int))
    if short_ids:
        segment[np.isin(segment, list(short_ids))] = -1
    return segment


@dataclass(frozen=True)
class PreparationConfig:
    data_root: Path
    output_dir: Path
    max_participants: int = 0
    max_interpolation_gap: int = 2
    maximum_observation_gap: int = 5
    minimum_segment_minutes: int = 15
    upward_jump_tolerance: float = 0.005


def prepare_dataset(config: PreparationConfig) -> Path:
    directories = participant_directories(config.data_root)
    if config.max_participants > 0:
        directories = directories[: config.max_participants]
    config.output_dir.mkdir(parents=True, exist_ok=True)

    participant_tables: list[pd.DataFrame] = []
    next_global_segment = 0
    skipped = 0
    for ordinal, directory in enumerate(directories, start=1):
        pseudonym = f"p{ordinal:04d}"
        try:
            minute = battery_minute_table(directory, config.max_interpolation_gap)
        except (ValueError, OSError) as error:
            eprint(f"warning: skipped participant {ordinal}: {error}")
            skipped += 1
            continue
        if minute.empty:
            skipped += 1
            continue

        minute["screen_on"] = screen_feature(directory, minute.index)
        minute = minute.join(running_app_features(directory, minute.index), how="left")
        for feature_name, filenames in EVENT_FILES.items():
            minute[feature_name] = event_count(directory, filenames, minute.index)
        minute = add_tail_states(minute)
        local_segment = discharge_segment_ids(
            minute,
            upward_jump_tolerance=config.upward_jump_tolerance,
            maximum_observation_gap=config.maximum_observation_gap,
            minimum_segment_minutes=config.minimum_segment_minutes,
        )
        global_segment = np.full(len(local_segment), -1, dtype=np.int64)
        local_ids = np.unique(local_segment[local_segment >= 0])
        for local_id in local_ids:
            global_segment[local_segment == local_id] = next_global_segment
            next_global_segment += 1

        minute["segment_id"] = global_segment
        minute["participant_id"] = pseudonym
        participant_tables.append(minute.reset_index())
        print(
            f"prepared participant {ordinal}/{len(directories)}: "
            f"{len(minute):,} minute rows, {len(local_ids)} retained segments"
        )

    if not participant_tables:
        raise RuntimeError("no usable participant tables were produced")
    combined = pd.concat(participant_tables, ignore_index=True)
    combined = combined.sort_values(["participant_id", "time_min"]).reset_index(drop=True)
    output_path = config.output_dir / "minute_data.parquet"
    combined.to_parquet(output_path, index=False)

    manifest = {
        "status": "research code snapshot; local MDF schema adaptation may be required",
        "sampling_interval_seconds": 60,
        "participant_count": int(combined["participant_id"].nunique()),
        "skipped_participant_directories": int(skipped),
        "minute_rows": int(len(combined)),
        "retained_discharge_segments": int(combined.loc[combined["segment_id"] >= 0, "segment_id"].nunique()),
        "columns": list(map(str, combined.columns)),
        "tail_time_constants_seconds": TAIL_TIME_CONSTANTS_SECONDS,
        "privacy": "source folder names replaced by run-local pseudonyms; no mapping saved",
    }
    write_json(config.output_dir / "preparation_manifest.json", manifest)
    return output_path


def participant_split(
    participant_ids: Sequence[str], seed: int, train_fraction: float, validation_fraction: float
) -> pd.DataFrame:
    ids = np.array(sorted(set(map(str, participant_ids))), dtype=object)
    if len(ids) < 3:
        raise ValueError("at least three prepared participants are required for train/validation/test evaluation")
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be smaller than 1")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = max(1, int(math.floor(train_fraction * len(ids))))
    n_validation = max(1, int(math.floor(validation_fraction * len(ids))))
    if n_train + n_validation >= len(ids):
        n_validation = 1
        n_train = len(ids) - 2
    split = np.full(len(ids), "test", dtype=object)
    split[:n_train] = "train"
    split[n_train : n_train + n_validation] = "validation"
    return pd.DataFrame({"participant_id": ids, "split": split})


def historical_rate(group: pd.DataFrame, origin: int, window_minutes: int) -> float:
    start = max(0, origin - window_minutes)
    time = pd.to_datetime(group["time_min"], utc=True)
    elapsed_seconds = float((time.iloc[origin] - time.iloc[start]).total_seconds())
    if elapsed_seconds <= 0:
        return float("nan")
    soc_start = float(group.iloc[start]["SOC"])
    soc_origin = float(group.iloc[origin]["SOC"])
    return max(soc_start - soc_origin, 0.0) / elapsed_seconds


def most_recent_profile(row: pd.Series, feature_names: Sequence[str]) -> dict[str, float]:
    """Return the singular most-recent sampled profile available at the origin."""
    profile: dict[str, float] = {}
    for feature in feature_names:
        value = pd.to_numeric(pd.Series([row.get(feature, np.nan)]), errors="coerce").iloc[0]
        profile[feature] = float(value) if np.isfinite(value) else float("nan")
    return profile


def construct_tte_origins(
    data: pd.DataFrame,
    split_table: pd.DataFrame,
    initial_levels: Sequence[float],
    thresholds: Sequence[float],
    history_window_minutes: int,
) -> pd.DataFrame:
    split_map = dict(zip(split_table["participant_id"], split_table["split"]))
    tail_features = sorted(column for column in data.columns if "_z_tau" in str(column))
    feature_names = [feature for feature in INSTANT_FEATURES if feature in data.columns] + tail_features
    rows: list[dict[str, float | int | str]] = []

    valid = data[data["segment_id"] >= 0].copy()
    for segment_id, group in valid.groupby("segment_id", sort=True):
        group = group.sort_values("time_min").reset_index(drop=True)
        soc = pd.to_numeric(group["SOC"], errors="coerce").to_numpy(float)
        time = pd.to_datetime(group["time_min"], utc=True)
        if len(group) < 3 or not np.isfinite(soc).all():
            continue
        participant_id = str(group.iloc[0]["participant_id"])
        for threshold in thresholds:
            for initial in initial_levels:
                if initial <= threshold:
                    continue
                # A segment must begin at or above the requested origin level.
                # This prevents treating a partial late fragment as a full origin.
                if soc[0] < initial - 0.02:
                    continue
                start_candidates = np.flatnonzero(soc <= initial + 1e-12)
                if not len(start_candidates):
                    continue
                origin = int(start_candidates[0])
                end_candidates = np.flatnonzero(soc[origin + 1 :] <= threshold + 1e-12)
                if not len(end_candidates):
                    # The segment ended before reaching the threshold. Because
                    # charging breaks segments, this excludes charge-first cases.
                    continue
                end = origin + 1 + int(end_candidates[0])
                observed_tte = float((time.iloc[end] - time.iloc[origin]).total_seconds() / 60.0)
                if not np.isfinite(observed_tte) or observed_tte <= 0:
                    continue
                observed_initial = float(soc[origin])
                distance = max(observed_initial - float(threshold), 0.0)
                if distance <= 0:
                    continue
                row: dict[str, float | int | str] = {
                    "segment_id": int(segment_id),
                    "participant_id": participant_id,
                    "split": str(split_map.get(participant_id, "ignore")),
                    "initial_level": float(initial),
                    "observed_initial": observed_initial,
                    "threshold": float(threshold),
                    "observed_tte_min": observed_tte,
                    "target_rate_per_second": distance / (observed_tte * 60.0),
                    "history_rate_per_second": historical_rate(group, origin, history_window_minutes),
                    "origin_time": time.iloc[origin].isoformat(),
                }
                row.update(most_recent_profile(group.iloc[origin], feature_names))
                rows.append(row)
    return pd.DataFrame(rows)


class NonnegativeRateModel:
    """Scaled NNLS model for transparent non-negative drain attribution."""

    def __init__(self, feature_names: Sequence[str]):
        self.feature_names = list(feature_names)
        self.fill_values: np.ndarray | None = None
        self.scale_values: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None

    def _matrix(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        values = frame.reindex(columns=self.feature_names).apply(pd.to_numeric, errors="coerce")
        matrix = values.to_numpy(dtype=float, copy=True)
        matrix[~np.isfinite(matrix)] = np.nan
        if fit:
            self.fill_values = np.nanmedian(matrix, axis=0)
            self.fill_values[~np.isfinite(self.fill_values)] = 0.0
            filled = np.where(np.isfinite(matrix), matrix, self.fill_values)
            self.scale_values = np.nanpercentile(filled, 95, axis=0)
            invalid_scale = ~np.isfinite(self.scale_values) | (self.scale_values <= 1e-12)
            self.scale_values[invalid_scale] = 1.0
        if self.fill_values is None or self.scale_values is None:
            raise RuntimeError("model preprocessing parameters are not fitted")
        filled = np.where(np.isfinite(matrix), matrix, self.fill_values)
        scaled = np.maximum(filled, 0.0) / self.scale_values
        return np.column_stack([np.ones(len(scaled)), scaled])

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> None:
        matrix = self._matrix(frame, fit=True)
        target = np.asarray(target, dtype=float)
        valid = np.isfinite(target) & (target > 0)
        if not valid.any():
            raise ValueError("the training target contains no positive finite rates")
        self.coefficients = nnls(matrix[valid], target[valid])[0]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("model has not been fitted")
        return self._matrix(frame, fit=False) @ self.coefficients

def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted) & (observed > 0)
    observed = observed[valid]
    predicted = predicted[valid]
    if not len(observed):
        return {key: float("nan") for key in ("MAE_min", "RMSE_min", "MAPE_percent", "bias_min", "R2")}
    residual = predicted - observed
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    return {
        "MAE_min": float(np.mean(np.abs(residual))),
        "RMSE_min": float(np.sqrt(np.mean(residual**2))),
        "MAPE_percent": float(np.mean(np.abs(residual) / observed) * 100.0),
        "bias_min": float(np.mean(residual)),
        "R2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan"),
    }


def representative_single_tail(all_tail_features: Sequence[str]) -> list[str]:
    """Choose one medium-scale state per event channel for the single-tail ablation."""
    selected: list[str] = []
    prefixes = sorted({feature.rsplit("_z_tau", 1)[0] for feature in all_tail_features})
    for prefix in prefixes:
        candidates = [feature for feature in all_tail_features if feature.startswith(prefix + "_z_tau")]
        candidates.sort(key=lambda name: abs(float(name.rsplit("tau", 1)[1]) - 120.0))
        if candidates:
            selected.append(candidates[0])
    return selected


@dataclass(frozen=True)
class EvaluationConfig:
    prepared_file: Path
    output_dir: Path
    seed: int = 7
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    history_window_minutes: int = 30
    initial_levels: tuple[float, ...] = DEFAULT_INITIAL_LEVELS
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS


def evaluate_tte(config: EvaluationConfig) -> None:
    if not config.prepared_file.is_file():
        raise FileNotFoundError(f"prepared table not found: {config.prepared_file.as_posix()}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(config.prepared_file)
    required = {"time_min", "SOC", "segment_id", "participant_id"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"prepared table is missing required columns: {sorted(missing)}")
    data["time_min"] = pd.to_datetime(data["time_min"], utc=True)

    split_table = participant_split(
        data["participant_id"].astype(str).unique(),
        config.seed,
        config.train_fraction,
        config.validation_fraction,
    )
    split_table.to_csv(config.output_dir / "split_assignments.csv", index=False)
    origins = construct_tte_origins(
        data,
        split_table,
        config.initial_levels,
        config.thresholds,
        config.history_window_minutes,
    )
    if origins.empty:
        raise RuntimeError("no complete TTE origins were found after filtering")
    origins.to_csv(config.output_dir / "tte_origins.csv", index=False)

    train = origins[origins["split"] == "train"].copy()
    validation = origins[origins["split"] == "validation"].copy()
    test = origins[origins["split"] == "test"].copy()
    if train.empty or validation.empty or test.empty:
        counts = origins["split"].value_counts().to_dict()
        raise RuntimeError(f"one or more origin splits are empty: {counts}")

    target_train = pd.to_numeric(train["target_rate_per_second"], errors="coerce").to_numpy(float)
    positive_train = target_train[np.isfinite(target_train) & (target_train > 0)]
    if not len(positive_train):
        raise RuntimeError("no positive finite training rates were constructed")
    rate_floor, rate_ceiling = np.percentile(positive_train, [1, 99])
    if rate_floor <= 0 or not np.isfinite(rate_ceiling) or rate_floor >= rate_ceiling:
        rate_floor = max(float(np.min(positive_train)), 1e-12)
        rate_ceiling = max(float(np.max(positive_train)), rate_floor * 1.01)

    available_instant = [feature for feature in INSTANT_FEATURES if feature in origins.columns]
    available_tail = sorted(column for column in origins.columns if "_z_tau" in str(column))
    available_single_tail = representative_single_tail(available_tail)
    model_features = {
        "instantaneous-only": ["history_rate_per_second"] + available_instant,
        "single-exponential-tail": ["history_rate_per_second"] + available_instant + available_single_tail,
        "multi-exponential-tail": ["history_rate_per_second"] + available_instant + available_tail,
    }

    rate_predictions: dict[str, np.ndarray] = {
        "constant-rate-baseline": np.full(len(test), float(np.median(positive_train))),
    }
    recent = pd.to_numeric(test["history_rate_per_second"], errors="coerce").to_numpy(float)
    recent[~np.isfinite(recent) | (recent <= 0)] = float(np.median(positive_train))
    rate_predictions["recent-history-baseline"] = recent

    for model_name, feature_names in model_features.items():
        model = NonnegativeRateModel(feature_names)
        model.fit(train, target_train)
        rate_predictions[model_name] = model.predict(test)

    observed = pd.to_numeric(test["observed_tte_min"], errors="coerce").to_numpy(float)
    distance = np.maximum(
        pd.to_numeric(test["observed_initial"], errors="coerce").to_numpy(float)
        - pd.to_numeric(test["threshold"], errors="coerce").to_numpy(float),
        0.0,
    )
    prediction_frame = test[
        [
            "segment_id",
            "participant_id",
            "origin_time",
            "initial_level",
            "threshold",
            "observed_initial",
            "observed_tte_min",
        ]
    ].copy()
    model_summary: list[dict[str, float | int | str]] = []
    tte_predictions: dict[str, np.ndarray] = {}
    for model_name, predicted_rate in rate_predictions.items():
        clipped_rate = np.clip(predicted_rate, rate_floor, rate_ceiling)
        predicted_tte = distance / (clipped_rate * 60.0)
        tte_predictions[model_name] = predicted_tte
        prediction_frame[model_name] = predicted_tte
        row: dict[str, float | int | str] = {"model": model_name, "n": int(len(test))}
        row.update(regression_metrics(observed, predicted_tte))
        model_summary.append(row)

    full_model_name = "multi-exponential-tail"
    group_summary: list[dict[str, float | int]] = []
    for (initial_level, threshold), index in test.groupby(
        ["initial_level", "threshold"], sort=True
    ).groups.items():
        positions = test.index.get_indexer(index)
        row: dict[str, float | int] = {
            "initial_level": float(initial_level),
            "threshold": float(threshold),
            "n": int(len(positions)),
        }
        row.update(regression_metrics(observed[positions], tte_predictions[full_model_name][positions]))
        group_summary.append(row)

    prediction_frame.to_csv(config.output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(model_summary).to_csv(config.output_dir / "model_comparison.csv", index=False)
    pd.DataFrame(group_summary).to_csv(config.output_dir / "horizon_metrics.csv", index=False)

    manifest = {
        "status": "fresh local evaluation from the research-code snapshot; not a bundled manuscript result",
        "seed": config.seed,
        "split_unit": "participant",
        "split_fractions": {
            "train": config.train_fraction,
            "validation": config.validation_fraction,
            "test": 1.0 - config.train_fraction - config.validation_fraction,
        },
        "participant_counts": split_table["split"].value_counts().sort_index().to_dict(),
        "origin_counts": origins["split"].value_counts().sort_index().to_dict(),
        "initial_levels": config.initial_levels,
        "thresholds": config.thresholds,
        "history_window_minutes": config.history_window_minutes,
        "information_boundary": (
            "predictor features come from the most recent sampling point and history at or before t0; "
            "the later SOC trajectory is used only to label threshold-crossing time"
        ),
        "charging_rule": "segments that end before threshold are excluded; they are not completed TTE labels",
        "models": list(rate_predictions),
    }
    write_json(config.output_dir / "evaluation_manifest.json", manifest)
    print(pd.DataFrame(model_summary).to_string(index=False))
    print(pd.DataFrame(group_summary).to_string(index=False))


def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--max-participants", type=int, default=0)
    parser.add_argument("--max-interpolation-gap", type=int, default=2)
    parser.add_argument("--maximum-observation-gap", type=int, default=5)
    parser.add_argument("--minimum-segment-minutes", type=int, default=15)
    parser.add_argument("--upward-jump-tolerance", type=float, default=0.005)


def add_evaluation_arguments(parser: argparse.ArgumentParser, include_prepared_file: bool) -> None:
    if include_prepared_file:
        parser.add_argument(
            "--prepared-file", type=Path, default=DEFAULT_WORK_DIR / "prepared" / "minute_data.parquet"
        )
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_DIR / "evaluation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--history-window-minutes", type=int, default=30)
    parser.add_argument(
        "--initial-levels", type=parse_float_list, default=DEFAULT_INITIAL_LEVELS,
        help="comma-separated normalized SOC-proxy levels",
    )
    parser.add_argument(
        "--thresholds", type=parse_float_list, default=DEFAULT_THRESHOLDS,
        help="comma-separated normalized SOC-proxy thresholds",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sanitized MDF preprocessing and smartphone TTE reference pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="align MDF logs and construct discharge features")
    add_prepare_arguments(prepare)

    evaluate = subparsers.add_parser("evaluate", help="evaluate held-out TTE origins")
    add_evaluation_arguments(evaluate, include_prepared_file=True)

    all_command = subparsers.add_parser("all", help="run preparation followed by evaluation")
    add_prepare_arguments(all_command)
    add_evaluation_arguments(all_command, include_prepared_file=False)
    return parser


def preparation_config_from_args(args: argparse.Namespace) -> PreparationConfig:
    return PreparationConfig(
        data_root=args.data_root,
        output_dir=args.work_dir / "prepared",
        max_participants=args.max_participants,
        max_interpolation_gap=args.max_interpolation_gap,
        maximum_observation_gap=args.maximum_observation_gap,
        minimum_segment_minutes=args.minimum_segment_minutes,
        upward_jump_tolerance=args.upward_jump_tolerance,
    )


def evaluation_config_from_args(
    args: argparse.Namespace, prepared_file: Path | None = None, output_dir: Path | None = None
) -> EvaluationConfig:
    return EvaluationConfig(
        prepared_file=prepared_file if prepared_file is not None else args.prepared_file,
        output_dir=output_dir if output_dir is not None else args.output_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        history_window_minutes=args.history_window_minutes,
        initial_levels=tuple(args.initial_levels),
        thresholds=tuple(args.thresholds),
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        output = prepare_dataset(preparation_config_from_args(args))
        print(f"prepared table: {output.as_posix()}")
        return
    if args.command == "evaluate":
        evaluate_tte(evaluation_config_from_args(args))
        return
    if args.command == "all":
        prepared = prepare_dataset(preparation_config_from_args(args))
        evaluate_tte(
            evaluation_config_from_args(
                args,
                prepared_file=prepared,
                output_dir=args.work_dir / "evaluation",
            )
        )
        return
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()

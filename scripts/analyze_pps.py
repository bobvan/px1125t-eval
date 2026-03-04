#!/usr/bin/env python3
"""
analyze_pps.py — PPS timing analysis for px1125t_eval.

chA = NEO-F10T PPS,  chB = SkyTraq PX1125T PPS
qErr sources: --timtp (F10T UBX-TIM-TP) and/or --psti (PX1125T $PSTI,00).
Both are optional; omit either to skip qErr correction for that channel.

Usage:
    python scripts/analyze_pps.py \\
        --ticc  data/foo_ticc.csv  \\
        --psti  data/foo_psti.csv  \\
        --timtp data/foo_timtp.csv \\
        --out   data/foo

    # TICC only (raw, no qErr correction):
    python scripts/analyze_pps.py \\
        --ticc data/foo_ticc.csv --out data/foo

Outputs:
    _pps_report.txt  — statistics and ADEV/TDEV summary
    _pps_diff.png    — raw vs qErr-corrected A−B time series
    _pps_adev.png    — ADEV(τ): raw and corrected
    _pps_tdev.png    — TDEV(τ): raw and corrected

Alignment notes:
  - TIM-TP / $PSTI,00 message N predicts qErr for PPS edge N+1.
    So TICC pair i is corrected by qErr[i-1] (shift by 1).
    The first TICC pair has no prior qErr and is dropped.
  - chA (F10T) maps to 'TOP'; chB (PX1125T) maps to 'BOT' internally.
  - Alignment by UTC wall-clock second (preferred): TICC host_timestamp
    floored to integer second S joins TIM-TP row at utc_s = S-1.
    This is unambiguous and requires no GPS offset search.
  - Fallback (old TICC CSVs without host_timestamp): GPS offset arithmetic
    with ±1 search.  UTC wall-clock from F10T TIM-TP or PSTI timestamps.
"""

import argparse
import sys
from pathlib import Path

import allantools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Wavelength table reused for optional CMC correlation
_C = 299_792_458.0
_WAVELENGTH: dict[str, float] = {
    "GPS-L1CA":   _C / 1_575_420_000.0,
    "GAL-E1C":    _C / 1_575_420_000.0,
    "GAL-E1B":    _C / 1_575_420_000.0,
    "GPS-L5I":    _C / 1_176_450_000.0,
    "GPS-L5Q":    _C / 1_176_450_000.0,
    "GAL-E5aI":   _C / 1_176_450_000.0,
    "GAL-E5aQ":   _C / 1_176_450_000.0,
    "BDS-B2aI":   _C / 1_176_450_000.0,
    "BDS-B1I":    _C / 1_561_098_000.0,
    "BDS-B1C":    _C / 1_575_420_000.0,
    "GPS-L2CL":   _C / 1_227_600_000.0,
    "GPS-L2CM":   _C / 1_227_600_000.0,
}


# ── loading ──────────────────────────────────────────────────────────── #

def load_ticc(path: Path) -> pd.DataFrame:
    """
    Load TICC CSV and pair chA/chB edges by integer second.

    Handles three CSV format generations:
      Gen 1: timestamp_s, channel          (legacy float)
      Gen 2: host_timestamp, timestamp_s, channel  (transitional float)
      Gen 3: host_timestamp, ref_sec, ref_ps, channel  (current int64)

    Also handles TICC power-cycle resets: the TICC timestamps restart from ~0
    after a reset.  If a backward jump > _RESET_THRESHOLD_S is detected in the
    file-order sequence, the data is split into sessions and the longest kept.

    All generations produce chA_ref_sec / chA_ref_ps / chB_ref_sec / chB_ref_ps
    as int64 columns, plus raw_diff_ps (int64) and raw_diff_s (float64).

    Asserts that no edge is within 100 ns of a second boundary.

    Returns DataFrame with columns:
        integer_sec, chA_ref_sec, chA_ref_ps, chB_ref_sec, chB_ref_ps,
        raw_diff_ps, raw_diff_s[, host_sec]
    Sorted by integer_sec; only seconds where both channels arrived.
    """
    _BOUNDARY_GUARD_S  = 100e-9   # 100 ns minimum distance from integer boundary
    _RESET_THRESHOLD_S = 60.0     # backward jump this large means a TICC reset

    df = pd.read_csv(path)
    cols = set(df.columns)

    # Detect generation and normalise to ref_sec / ref_ps int64 columns.
    if "ref_sec" in cols:                      # Gen 3
        df["ref_sec"] = df["ref_sec"].astype("int64")
        df["ref_ps"]  = df["ref_ps"].astype("int64")
        df["integer_sec"] = df["ref_sec"]
        seq_col = "ref_sec"   # column to use for reset detection
    else:                                       # Gen 1 or 2 (float timestamp_s)
        df["integer_sec"] = df["timestamp_s"].astype("int64")
        df["ref_sec"] = df["integer_sec"]
        df["ref_ps"]  = ((df["timestamp_s"] - df["integer_sec"]) * 1e12
                         ).round().astype("int64")
        seq_col = "timestamp_s"

    # Detect TICC resets: backward jumps in file-order timestamp sequence.
    jumps = df[seq_col].diff()
    reset_rows = jumps[jumps < -_RESET_THRESHOLD_S].index.tolist()
    if reset_rows:
        boundaries = [0] + reset_rows + [len(df)]
        sessions   = [df.iloc[boundaries[i]:boundaries[i+1]].copy()
                      for i in range(len(boundaries) - 1)]
        longest    = max(sessions, key=len)
        n_dropped  = len(df) - len(longest)
        print(f"  TICC: detected {len(reset_rows)} reset(s); "
              f"dropped {n_dropped} pre-reset row(s), using {len(longest)} rows.")
        df = longest.reset_index(drop=True)

    # host_timestamp → integer UTC second for UTC-join with TIM-TP.
    if "host_timestamp" in cols:
        host_ts = pd.to_datetime(df["host_timestamp"], utc=True)
        df["host_sec"] = (host_ts.astype("int64") // 1_000_000_000).astype(int)

    frac_s = df["ref_ps"] / 1e12
    bad = (frac_s < _BOUNDARY_GUARD_S) | (frac_s > 1.0 - _BOUNDARY_GUARD_S)
    if bad.any():
        raise ValueError(
            f"TICC: {bad.sum()} edge(s) within {_BOUNDARY_GUARD_S*1e9:.0f} ns "
            f"of a second boundary — possible straddling artefact"
        )

    piv_sec = (df.pivot_table(index="integer_sec", columns="channel",
                               values="ref_sec", aggfunc="first")
                 .rename(columns={"chA": "chA_ref_sec", "chB": "chB_ref_sec"}))
    piv_ps  = (df.pivot_table(index="integer_sec", columns="channel",
                               values="ref_ps",  aggfunc="first")
                 .rename(columns={"chA": "chA_ref_ps",  "chB": "chB_ref_ps"}))
    piv = (pd.concat([piv_sec, piv_ps], axis=1)
             .dropna()
             .reset_index()
             .sort_values("integer_sec")
             .reset_index(drop=True))
    for col in ("chA_ref_sec", "chB_ref_sec", "chA_ref_ps", "chB_ref_ps"):
        piv[col] = piv[col].astype("int64")

    piv["raw_diff_ps"] = ((piv["chA_ref_sec"] - piv["chB_ref_sec"])
                          * 1_000_000_000_000
                          + piv["chA_ref_ps"] - piv["chB_ref_ps"])
    piv["raw_diff_s"]  = piv["raw_diff_ps"].astype(float) * 1e-12

    if "host_sec" in df.columns:
        hs_map = df.groupby("integer_sec")["host_sec"].first()
        piv["host_sec"] = piv["integer_sec"].map(hs_map)
    return piv


def load_timtp(path: Path) -> dict[str, pd.DataFrame]:
    """
    Load TIM-TP CSV.  Returns dict keyed by receiver ('TOP', 'BOT'),
    each a DataFrame sorted by timestamp with columns:
        timestamp, qerr_ps, tow_ms, tow_s, utc_s, week
    tow_s is the integer GPS second (tow_ms // 1000).
    utc_s is the integer UTC second when the message was logged (for UTC join).
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["tow_s"] = (df["tow_ms"] // 1000).astype(int)
    # utc_s: integer UTC second when message was logged (for UTC-second join).
    df["utc_s"] = (df["timestamp"].astype("int64") // 1_000_000_000).astype(int)
    return {
        rx: grp.sort_values("timestamp").reset_index(drop=True)
        for rx, grp in df.groupby("receiver")
    }


# ── GPS-second join ──────────────────────────────────────────────────── #

def _gps_join(ticc: pd.DataFrame,
              top_df: pd.DataFrame,
              bot_df: pd.DataFrame,
              gps_offset: int) -> pd.DataFrame:
    """
    Join TICC pairs with TIM-TP qErr by GPS second.

    TICC integer_sec + gps_offset = GPS second S.
    TIM-TP at tow_s = S−1 predicted the PPS edge at GPS second S,
    so it supplies the qErr that corrects the TICC pair at GPS second S.

    Returns a copy of ticc with added columns:
        gps_sec, qerr_top_ps, qerr_bot_ps, utc_time
    Rows where no matching TIM-TP entry exists are dropped.
    """
    df = ticc.copy()
    df["gps_sec"] = df["integer_sec"] + gps_offset

    top_q  = top_df.set_index("tow_s")["qerr_ps"]   if not top_df.empty else pd.Series(dtype=float)
    bot_q  = bot_df.set_index("tow_s")["qerr_ps"]   if not bot_df.empty else pd.Series(dtype=float)
    top_ts = top_df.set_index("tow_s")["timestamp"]  if not top_df.empty else pd.Series(dtype=object)
    bot_ts = bot_df.set_index("tow_s")["timestamp"]  if not bot_df.empty else pd.Series(dtype=object)

    # tow_s = GPS_second − 1 is the TIM-TP that predicted this PPS edge
    corr_tow = df["gps_sec"] - 1

    # When a source is absent, use 0 (no correction) rather than NaN.
    # Only drop rows where an *available* source has no matching entry.
    df["qerr_top_ps"] = corr_tow.map(top_q) if not top_q.empty else 0
    df["qerr_bot_ps"] = corr_tow.map(bot_q) if not bot_q.empty else 0

    # UTC wall-clock: prefer TOP (F10T), fall back to BOT (PX1125T)
    if not top_ts.empty:
        df["utc_time"] = corr_tow.map(top_ts)
    elif not bot_ts.empty:
        df["utc_time"] = corr_tow.map(bot_ts)
    else:
        df["utc_time"] = pd.NaT

    drop_cols = ([("qerr_top_ps",)] if not top_q.empty else []) + \
                ([("qerr_bot_ps",)] if not bot_q.empty else [])
    drop_cols = [c for tup in drop_cols for c in tup]
    if drop_cols:
        df = df.dropna(subset=drop_cols)
    return df.reset_index(drop=True)


def _utc_join(ticc: pd.DataFrame,
              top_df: pd.DataFrame,
              bot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join TICC pairs with TIM-TP qErr by UTC wall-clock second.

    TICC host_sec = S  (integer UTC second when edge arrived at host).
    TIM-TP utc_s  = S−1 (when TIM-TP message was logged; it predicted PPS at S).

    No GPS offset arithmetic; no ±1 search uncertainty.
    Requires host_timestamp column in TICC CSV (logged since 2026-03-04).
    """
    df = ticc.copy()
    top_q  = top_df.set_index("utc_s")["qerr_ps"]  if not top_df.empty else pd.Series(dtype=float)
    bot_q  = bot_df.set_index("utc_s")["qerr_ps"]  if not bot_df.empty else pd.Series(dtype=float)
    top_ts = top_df.set_index("utc_s")["timestamp"] if not top_df.empty else pd.Series(dtype=object)
    bot_ts = bot_df.set_index("utc_s")["timestamp"] if not bot_df.empty else pd.Series(dtype=object)

    corr_utc = df["host_sec"] - 1   # TIM-TP at S-1 predicts PPS edge at S
    df["qerr_top_ps"] = corr_utc.map(top_q) if not top_q.empty else 0
    df["qerr_bot_ps"] = corr_utc.map(bot_q) if not bot_q.empty else 0

    if not top_ts.empty:
        df["utc_time"] = corr_utc.map(top_ts)
    elif not bot_ts.empty:
        df["utc_time"] = corr_utc.map(bot_ts)
    else:
        df["utc_time"] = pd.NaT

    drop_cols = []
    if not top_q.empty: drop_cols.append("qerr_top_ps")
    if not bot_q.empty: drop_cols.append("qerr_bot_ps")
    if drop_cols:
        df = df.dropna(subset=drop_cols)
    return df.reset_index(drop=True)


# ── qErr alignment validation ────────────────────────────────────────── #

def validate_alignment(ticc: pd.DataFrame,
                       timtp: dict[str, pd.DataFrame]) -> tuple:
    """
    Confirm qErr sign and GPS-second alignment.

    When TICC CSV contains host_timestamp (logged since 2026-03-04):
      Uses direct UTC-second join.  Only sign (+1/-1) is searched.
      Returns join_method="utc".

    Fallback (old data without host_timestamp):
      GPS offset arithmetic with delta search −1…+2.
      Expected: delta=0, sign=+1.
      Returns join_method="gps".

    Returns (raw_std_ns, results, naive_offset, join_method)
      results: list of (delta, sign, std_ns, n_pairs)
    """
    top = timtp.get("TOP", pd.DataFrame(columns=["qerr_ps", "tow_s"]))
    bot = timtp.get("BOT", pd.DataFrame(columns=["qerr_ps", "tow_s"]))
    raw_std_ns = float(ticc["raw_diff_s"].dropna().std() * 1e9)

    if top.empty and bot.empty:
        return raw_std_ns, [], 0, "gps"

    # UTC join: unambiguous when host_timestamp column is present.
    ref = top if not top.empty else bot
    if "host_sec" in ticc.columns and "utc_s" in ref.columns:
        joined = _utc_join(ticc, top, bot)
        n = len(joined)
        results = []
        for sign in (+1, -1):
            corr_ps = (joined["raw_diff_ps"]
                       + sign * (joined["qerr_top_ps"] - joined["qerr_bot_ps"]))
            std_ns = float(corr_ps.std() * 1e-3) if n > 1 else np.inf
            results.append((0, sign, std_ns, n))
        return raw_std_ns, results, 0, "utc"

    # GPS offset fallback: prefer TOP (F10T) as reference.
    if not top.empty:
        naive = int(top["tow_s"].iloc[0]) - int(ticc["integer_sec"].iloc[0])
    else:
        naive = int(bot["tow_s"].iloc[0]) - int(ticc["integer_sec"].iloc[0])

    results = []
    for delta in (-1, 0, +1, +2):
        joined = _gps_join(ticc, top, bot, naive + delta)
        n = len(joined)
        for sign in (+1, -1):
            corr_ps = (joined["raw_diff_ps"]
                       + sign * (joined["qerr_top_ps"] - joined["qerr_bot_ps"]))
            std_ns = float(corr_ps.std() * 1e-3) if n > 1 else np.inf
            results.append((delta, sign, std_ns, n))
    return raw_std_ns, results, naive, "gps"


# ── qErr correction ──────────────────────────────────────────────────── #

def apply_qerr(ticc: pd.DataFrame,
               timtp: dict[str, pd.DataFrame],
               gps_offset: int,
               sign: int = +1,
               psti_utc: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Correct TICC timestamps with qErr from TIM-TP using GPS-second join.

    TICC integer_sec + gps_offset = GPS second S.
    TIM-TP at tow_s = S−1 predicted the quantisation error of PPS edge S.

    Sign convention: corrected = measured + sign * qerr_ps * 1e-12.
    sign=+1 is the expected convention: positive qErr means the pulse fired
    that many ps early; adding qerr moves the timestamp to the true boundary.

    psti_utc: optional DataFrame with columns [tow_s, timestamp] used to
    establish real UTC wall-clock time even when qErr correction is skipped.

    Adds columns:
        gps_sec                              — GPS second for each pair
        qerr_top_ps, qerr_bot_ps            — corrections applied (ps)
        corr_diff_s                          — qErr-corrected A−B (s)
        utc_time                             — wall-clock UTC (GPS or synthetic)
    """
    top = timtp.get("TOP", pd.DataFrame(columns=["qerr_ps", "tow_s", "timestamp"]))
    bot = timtp.get("BOT", pd.DataFrame(columns=["qerr_ps", "tow_s"]))

    # No qErr available: return raw pairs with synthetic relative time axis.
    if top.empty and bot.empty:
        df = ticc.copy()
        df["qerr_top_ps"] = np.int64(0)
        df["qerr_bot_ps"] = np.int64(0)
        df["chA_corr_ps"] = df["chA_ref_ps"]
        df["chB_corr_ps"] = df["chB_ref_ps"]
        df["corr_diff_ps"] = df["raw_diff_ps"]
        df["corr_diff_s"]  = df["corr_diff_ps"].astype(float) * 1e-12
        df["_sign"] = sign
        # UTC time axis: use PSTI GPS timestamps if available, else synthetic.
        if psti_utc is not None and not psti_utc.empty:
            # Naive GPS offset from PSTI
            naive = int(psti_utc["tow_s"].iloc[0]) - int(df["integer_sec"].iloc[0])
            df["gps_sec"] = df["integer_sec"] + naive
            ts_map = psti_utc.set_index("tow_s")["timestamp"]
            df["utc_time"] = df["gps_sec"].map(ts_map)
            # Forward-fill any gaps (should be rare)
            df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True)
            df["utc_time"] = df["utc_time"].ffill().bfill()
        else:
            epoch = pd.Timestamp("1970-01-01", tz="UTC")
            df["utc_time"] = epoch + pd.to_timedelta(df["integer_sec"], unit="s")
        return df

    # qErr available: use UTC join when host_timestamp present, else GPS join.
    use_utc = ("host_sec" in ticc.columns and not top.empty
               and "utc_s" in top.columns)
    if use_utc:
        df = _utc_join(ticc, top, bot)
    else:
        df = _gps_join(ticc, top, bot, gps_offset)
    df["chA_corr_ps"] = df["chA_ref_ps"] + sign * df["qerr_top_ps"]
    df["chB_corr_ps"] = df["chB_ref_ps"] + sign * df["qerr_bot_ps"]
    df["corr_diff_ps"] = (df["raw_diff_ps"]
                          + sign * (df["qerr_top_ps"] - df["qerr_bot_ps"]))
    df["corr_diff_s"]  = df["corr_diff_ps"].astype(float) * 1e-12
    df["_sign"] = sign   # carry sign through for reporting
    return df


# ── ADEV / TDEV ──────────────────────────────────────────────────────── #

def compute_stability(phase_s: np.ndarray) -> dict:
    """
    Compute ADEV and TDEV from a 1-Hz phase time series (seconds).
    Returns dict with keys: taus_adev, adev, taus_tdev, tdev.
    Returns empty dict if the series is too short.
    """
    phase_s = phase_s[~np.isnan(phase_s)]
    if len(phase_s) < 8:
        return {}
    taus_a, adev, _, _ = allantools.adev(
        phase_s, rate=1.0, data_type="phase", taus="all")
    taus_t, tdev, _, _ = allantools.tdev(
        phase_s, rate=1.0, data_type="phase", taus="all")
    return {"taus_adev": taus_a, "adev": adev,
            "taus_tdev": taus_t, "tdev": tdev}


def individual_stability(df: pd.DataFrame) -> dict[str, dict]:
    """
    Compute ADEV/TDEV for each PPS channel individually and for the difference.

    Individual phase series: x[i] = (ref_ps[i] - ref_ps[0]) * 1e-12 seconds.
    The integer-second parts cancel exactly (ref_sec[i] - ref_sec[i-1] ≈ 1,
    gaps handled because the per-epoch phase is relative to the first epoch).
    The TICC clock's linear drift is common-mode in the A-B difference.

    Returns dict with keys 'chA_raw', 'chA_corr', 'chB_raw', 'chB_corr',
    'diff_raw', 'diff_corr', each a stability dict from compute_stability()
    (may be empty if data is too short or qErr unavailable).
    """
    def _phase_ps(ps_arr: np.ndarray) -> np.ndarray:
        """Phase residual from int64 ps array, returned as float seconds."""
        return (ps_arr - ps_arr[0]).astype(float) * 1e-12

    return {
        "chA_raw":  compute_stability(_phase_ps(df["chA_ref_ps"].values)),
        "chA_corr": compute_stability(_phase_ps(df["chA_corr_ps"].values)),
        "chB_raw":  compute_stability(_phase_ps(df["chB_ref_ps"].values)),
        "chB_corr": compute_stability(_phase_ps(df["chB_corr_ps"].values)),
        "diff_raw":  compute_stability(df["raw_diff_s"].values),
        "diff_corr": compute_stability(df["corr_diff_s"].values),
    }


# ── report ───────────────────────────────────────────────────────────── #

def _stab_rows(stab: dict, key_taus: list[int]) -> list[tuple]:
    """Return (tau, adev, tdev) rows for the given key taus, or empty list."""
    if not stab:
        return []
    rows = []
    for tau in key_taus:
        ia = np.searchsorted(stab["taus_adev"], tau)
        it = np.searchsorted(stab["taus_tdev"], tau)
        adev = stab["adev"][ia] if ia < len(stab["adev"]) else None
        tdev = stab["tdev"][it] * 1e9 if it < len(stab["tdev"]) else None
        rows.append((tau, adev, tdev))
    return rows


def _stab_section(a, header: str, raw: dict, corr: dict, key_taus: list[int],
                  has_corr: bool) -> None:
    """Append a combined ADEV/TDEV section (raw + optionally corrected)."""
    a(header)
    # Header row
    if has_corr:
        a(f"  {'τ (s)':>6s}  {'ADEV raw':>11s}  {'ADEV corr':>11s}"
          f"  {'TDEV raw (ns)':>13s}  {'TDEV corr (ns)':>14s}")
    else:
        a(f"  {'τ (s)':>6s}  {'ADEV':>11s}  {'TDEV (ns)':>12s}")
    raw_rows  = _stab_rows(raw,  key_taus)
    corr_rows = _stab_rows(corr, key_taus)
    for i, (tau, adev_r, tdev_r) in enumerate(raw_rows):
        if has_corr and i < len(corr_rows):
            _, adev_c, tdev_c = corr_rows[i]
            adev_r_s = f"{adev_r:.3e}" if adev_r is not None else "  —"
            adev_c_s = f"{adev_c:.3e}" if adev_c is not None else "  —"
            tdev_r_s = f"{tdev_r:.3f}"  if tdev_r is not None else "  —"
            tdev_c_s = f"{tdev_c:.3f}"  if tdev_c is not None else "  —"
            a(f"  {tau:>6d}  {adev_r_s:>11s}  {adev_c_s:>11s}"
              f"  {tdev_r_s:>13s}  {tdev_c_s:>14s}")
        else:
            adev_s = f"{adev_r:.3e}" if adev_r is not None else "  —"
            tdev_s = f"{tdev_r:.3f}"  if tdev_r is not None else "  —"
            a(f"  {tau:>6d}  {adev_s:>11s}  {tdev_s:>12s}")
    a("")


def write_report(df: pd.DataFrame,
                 indiv: dict[str, dict],
                 alignment: tuple, out_stem: Path) -> None:
    """
    Write the PPS timing report.

    indiv keys: 'chA_raw', 'chA_corr', 'chB_raw', 'chB_corr',
                'diff_raw', 'diff_corr'

    ADEV is reported as dimensionless σ_y (e.g. 9.5e-9).
    TDEV is reported in nanoseconds (σ_x × 1e9).
    """
    lines = []
    a = lines.append

    raw_ns  = df["raw_diff_s"]  * 1e9
    corr_ns = df["corr_diff_s"] * 1e9
    dur_h   = (df["utc_time"].max() - df["utc_time"].min()).total_seconds() / 3600
    has_corr = not (df["qerr_top_ps"] == 0).all() or not (df["qerr_bot_ps"] == 0).all()

    a("=" * 62)
    a("  px1125t_eval PPS / TICC report  (chA=F10T, chB=PX1125T)")
    a("=" * 62)
    a(f"  Start    : {df['utc_time'].min()}")
    a(f"  End      : {df['utc_time'].max()}")
    a(f"  Duration : {dur_h:.2f} h")
    a(f"  Pairs    : {len(df)}  (chA=F10T, chB=PX1125T)")
    a(f"  qErr     : {'applied' if has_corr else 'not available (raw only)'}")
    a("")

    raw_std_ns, align_results, naive_offset, join_method = alignment
    best = min(align_results, key=lambda x: x[2]) if align_results else None
    if join_method == "utc":
        a("── qErr alignment (UTC host_timestamp join) ─────────────────────")
        a(f"  Raw std  : {raw_std_ns:.3f} ns")
        a(f"  {'Sign':>5s}  {'N pairs':>7s}  {'std (ns)':>10s}")
        for delta, sign, std_ns, n_pairs in align_results:
            tag = "  ← best" if (best and delta == best[0] and sign == best[1]) else ""
            a(f"  {sign:>+5d}  {n_pairs:>7d}  {std_ns:>10.3f}{tag}")
    else:
        a("── qErr alignment (GPS offset search; no host_timestamp) ─────────")
        a(f"  Raw std  : {raw_std_ns:.3f} ns")
        a(f"  {'GPS Δ':>6s}  {'Sign':>5s}  {'N pairs':>7s}  {'std (ns)':>10s}")
        for delta, sign, std_ns, n_pairs in align_results:
            tags = []
            if delta == 0 and sign == +1:
                tags.append("← naive")
            if best and delta == best[0] and sign == best[1]:
                tags.append("← best")
            tag = "  " + " ".join(tags) if tags else ""
            a(f"  {delta:>+6d}  {sign:>+5d}  {n_pairs:>7d}  {std_ns:>10.3f}{tag}")
        if best and (best[0] != 0 or best[1] != +1):
            a(f"  *** GPS offset delta={best[0]:+d} sign={best[1]:+d} "
              f"(GPS offset={naive_offset + best[0]}) ***")
    a("")

    a("── A−B difference (chA − chB) ───────────────────────────")
    a(f"  Raw  mean : {raw_ns.mean():+.3f} ns     std : {raw_ns.std():.3f} ns")
    if has_corr:
        a(f"  Corr mean : {corr_ns.mean():+.3f} ns     std : {corr_ns.std():.3f} ns")
    a("")

    key_taus = [1, 10, 100, 1000]
    note = ("  ADEV: dimensionless Allan deviation σ_y(τ)  [standard form, no units]\n"
            "  TDEV: time deviation σ_x(τ), directly computed from phase series, in nanoseconds\n"
            "        (measures RMS timing uncertainty at averaging time τ)")
    a(note)
    a("")

    _stab_section(a, "── chA (F10T) individual PPS stability ──────────────────",
                  indiv["chA_raw"], indiv["chA_corr"], key_taus, has_corr)
    _stab_section(a, "── chB (PX1125T) individual PPS stability ───────────────",
                  indiv["chB_raw"], indiv["chB_corr"], key_taus, has_corr)
    _stab_section(a, "── A−B differential stability (agreement) ───────────────",
                  indiv["diff_raw"], indiv["diff_corr"], key_taus, has_corr)

    a("=" * 62)

    path = out_stem.parent / (out_stem.name + "_pps_report.txt")
    path.write_text("\n".join(lines) + "\n")
    print(f"Report  → {path}")


# ── plots ────────────────────────────────────────────────────────────── #

def plot_diff(df: pd.DataFrame, out_stem: Path) -> None:
    """Raw vs corrected A−B time series."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    t = df["utc_time"]
    raw_ns  = df["raw_diff_s"]  * 1e9
    corr_ns = df["corr_diff_s"] * 1e9

    axes[0].plot(t, raw_ns, color="steelblue", linewidth=0.7, label="raw")
    axes[0].axhline(raw_ns.mean(), color="navy", linewidth=0.8,
                    linestyle="--", label=f"mean {raw_ns.mean():+.1f} ns")
    axes[0].set_ylabel("A−B (ns)")
    axes[0].set_title(f"Raw  (std = {raw_ns.std():.2f} ns)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, corr_ns, color="tomato", linewidth=0.7, label="qErr-corrected")
    axes[1].axhline(corr_ns.mean(), color="darkred", linewidth=0.8,
                    linestyle="--", label=f"mean {corr_ns.mean():+.1f} ns")
    axes[1].set_ylabel("A−B (ns)")
    axes[1].set_title(f"qErr-corrected  (std = {corr_ns.std():.2f} ns)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.suptitle("PPS A−B time difference  (chA=F10T, chB=PX1125T)\n"
                 "mean offset = cable delay + receiver bias",
                 fontsize=11)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_pps_diff.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def _stability_plot(curves: list[tuple[str, str, str, np.ndarray, np.ndarray]],
                    ylabel: str, title: str, out_path: Path) -> None:
    """
    curves: list of (label, color, linestyle, taus, values)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, ls, taus, vals in curves:
        if taus is not None and len(taus) > 0:
            ax.loglog(taus, vals, color=color, linestyle=ls,
                      linewidth=1.2, label=label)
    ax.set_xlabel("τ (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Plot    → {out_path}")


def plot_adev(indiv: dict[str, dict], out_stem: Path) -> None:
    """
    ADEV plot: all three series (chA, chB, diff) raw and corrected.
    Y axis: dimensionless σ_y (no unit scaling).
    """
    def _curve(key, label, color, ls):
        stab = indiv.get(key, {})
        if stab:
            return (label, color, ls, stab["taus_adev"], stab["adev"])
        return None

    curves = [c for c in [
        _curve("chA_raw",  "chA (F10T) raw",       "steelblue", "-"),
        _curve("chA_corr", "chA (F10T) corrected",  "steelblue", "--"),
        _curve("chB_raw",  "chB (PX1125T) raw",     "tomato",    "-"),
        _curve("chB_corr", "chB (PX1125T) corrected","tomato",   "--"),
        _curve("diff_raw", "A−B raw",               "seagreen",  "-"),
        _curve("diff_corr","A−B corrected",          "seagreen",  "--"),
    ] if c is not None]

    _stability_plot(
        curves,
        ylabel="ADEV  σ_y(τ)  [dimensionless]",
        title="Allan deviation — individual PPS stability and differential agreement\n"
              "(chA/chB: absolute stability + TICC noise; A−B: differential, TICC cancels)",
        out_path=out_stem.parent / (out_stem.name + "_pps_adev.png"),
    )


def plot_tdev(indiv: dict[str, dict], out_stem: Path) -> None:
    """
    TDEV plot: all three series raw and corrected.
    Y axis: σ_x(τ) in nanoseconds.
    """
    def _curve(key, label, color, ls):
        stab = indiv.get(key, {})
        if stab:
            return (label, color, ls, stab["taus_tdev"], stab["tdev"] * 1e9)
        return None

    curves = [c for c in [
        _curve("chA_raw",  "chA (F10T) raw",        "steelblue", "-"),
        _curve("chA_corr", "chA (F10T) corrected",   "steelblue", "--"),
        _curve("chB_raw",  "chB (PX1125T) raw",      "tomato",    "-"),
        _curve("chB_corr", "chB (PX1125T) corrected","tomato",    "--"),
        _curve("diff_raw", "A−B raw",                "seagreen",  "-"),
        _curve("diff_corr","A−B corrected",           "seagreen",  "--"),
    ] if c is not None]

    _stability_plot(
        curves,
        ylabel="TDEV  σ_x(τ)  (ns)",
        title="Time deviation — individual PPS stability and differential agreement",
        out_path=out_stem.parent / (out_stem.name + "_pps_tdev.png"),
    )


# ── optional CMC correlation ─────────────────────────────────────────── #

def plot_cmc_correlation(df: pd.DataFrame, rawx_path: Path,
                         out_stem: Path, window_s: int = 60) -> None:
    """
    Compare rolling CMC noise (per receiver) with rolling PPS jitter.

    Hypothesis: epochs with higher multipath (higher CMC std dev) should
    show more PPS timing variability.  The Kalman filter in the receiver
    low-passes the multipath-to-PPS transfer, so correlation is expected
    at timescales of ~100 s or longer.
    """
    rawx = pd.read_csv(rawx_path, parse_dates=["timestamp"])
    rawx["timestamp"] = pd.to_datetime(rawx["timestamp"], utc=True)
    if "locktime_ms" in rawx.columns:
        rawx = rawx.rename(columns={"locktime_ms": "lock_duration_ms"})

    # CMC (code-minus-carrier)
    wl = rawx["signal_id"].map(_WAVELENGTH)
    cp_ok = (rawx["cp_valid"] == 1) & (rawx["half_cyc"] == 1)
    rawx["cmc_m"] = np.where(
        cp_ok & wl.notna(),
        rawx["pseudorange_m"] - wl * rawx["carrier_phase_cy"],
        np.nan,
    )
    per_sv_mean = (rawx.groupby(["receiver", "signal_id", "sv_id"])["cmc_m"]
                       .transform("mean"))
    rawx["cmc_det"] = rawx["cmc_m"] - per_sv_mean

    # Floor to integer second, then compute per-epoch std across all SVs
    rawx["ts_s"] = rawx["timestamp"].dt.floor("s")
    epoch_cmc = (rawx.dropna(subset=["cmc_det"])
                     .groupby(["ts_s", "receiver"])["cmc_det"]
                     .std()
                     .reset_index()
                     .rename(columns={"cmc_det": "cmc_std_m"}))

    top_cmc = (epoch_cmc[epoch_cmc["receiver"] == "TOP"]
               .set_index("ts_s")["cmc_std_m"]
               .sort_index())
    bot_cmc = (epoch_cmc[epoch_cmc["receiver"] == "BOT"]
               .set_index("ts_s")["cmc_std_m"]
               .sort_index())

    # Rolling std of corrected PPS diff, indexed by utc_time
    pps = df.set_index("utc_time")["corr_diff_s"].sort_index() * 1e9  # ns
    pps_roll = pps.rolling(window=window_s, center=True, min_periods=window_s // 2).std()

    # Rolling mean CMC (smoothed) per receiver
    top_roll = top_cmc.rolling(window=window_s, center=True, min_periods=window_s // 2).mean()
    bot_roll = bot_cmc.rolling(window=window_s, center=True, min_periods=window_s // 2).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Top panel: CMC std per receiver
    ax = axes[0]
    ax.plot(top_roll.index, top_roll.values, color="steelblue",
            linewidth=0.8, label="TOP CMC noise")
    ax.plot(bot_roll.index, bot_roll.values, color="tomato",
            linewidth=0.8, label="BOT CMC noise")
    ax.set_ylabel("CMC std (m)")
    ax.set_title(f"Rolling {window_s}s CMC noise (detrended, all signals combined)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom panel: PPS jitter
    ax = axes[1]
    ax.plot(pps_roll.index, pps_roll.values, color="seagreen",
            linewidth=0.8, label="PPS jitter (A−B std)")
    ax.set_ylabel("PPS std (ns)")
    ax.set_title(f"Rolling {window_s}s PPS jitter (qErr-corrected A−B)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.suptitle("CMC multipath noise vs PPS timing jitter\n"
                 "(correlation would indicate multipath influence on PPS)",
                 fontsize=11)
    fig.tight_layout()
    path = out_stem.parent / (out_stem.name + "_pps_cmc_corr.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")

    # Quantify: Pearson r on the overlapping time window
    common = pps_roll.index.intersection(top_roll.index)
    if len(common) > 10:
        r_top = np.corrcoef(pps_roll.reindex(common).dropna(),
                            top_roll.reindex(common).dropna())[0, 1]
        r_bot = np.corrcoef(pps_roll.reindex(common).dropna(),
                            bot_roll.reindex(common).dropna())[0, 1]
        print(f"  Pearson r(PPS jitter, TOP CMC noise) = {r_top:+.3f}")
        print(f"  Pearson r(PPS jitter, BOT CMC noise) = {r_bot:+.3f}")


# ── main ─────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(
        description="PPS timing analysis: TICC + PX1125T $PSTI + NEO-F10T TIM-TP"
    )
    ap.add_argument("--ticc",  required=True, help="_ticc.csv from log_timing.py")
    ap.add_argument("--psti",  default=None,
                    help="_psti.csv from log_timing.py (PX1125T qErr → chB correction)")
    ap.add_argument("--timtp", default=None,
                    help="_timtp.csv from log_timing.py (F10T qErr → chA correction)")
    ap.add_argument("--out",   required=True, help="Output filename stem")
    args = ap.parse_args()

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading TICC  : {args.ticc}")
    ticc = load_ticc(Path(args.ticc))
    print(f"  {len(ticc)} complete pairs (chA=F10T, chB=PX1125T)")

    # Build unified qErr dict keyed by "TOP" (chA=F10T) and "BOT" (chB=PX1125T).
    # Both sources use the same CSV schema (timtp_logger / psti_logger are compatible).
    timtp: dict = {}

    if args.timtp:
        print(f"Loading TIM-TP: {args.timtp}  (F10T → chA/TOP)")
        loaded = load_timtp(Path(args.timtp))
        # remap whatever receiver label was written to "TOP"
        for grp in loaded.values():
            timtp["TOP"] = grp
            print(f"  F10T: {len(grp)} rows  "
                  f"qerr [{grp['qerr_ps'].min():+d}, {grp['qerr_ps'].max():+d}] ps")
            break

    if args.psti:
        print(f"Loading $PSTI : {args.psti}  (PX1125T, logged for reference)")
        loaded = load_timtp(Path(args.psti))   # same schema
        for grp in loaded.values():
            print(f"  PX1125T: {len(grp)} rows  "
                  f"qerr [{grp['qerr_ps'].min():+d}, {grp['qerr_ps'].max():+d}] ps")
            print("  NOTE: $PSTI,00 qErr is uncorrelated with PPS timing (r<0.025).")
            print("        Logged for reference; NOT applied as a correction.")
            # Intentionally not added to timtp dict — see bead pe-i03.
            break

    if not timtp:
        print("No qErr correction applied — raw TICC analysis only.")

    print("Validating qErr alignment …")
    alignment = validate_alignment(ticc, timtp)
    raw_std_ns, align_results, naive_offset, join_method = alignment
    best = min(align_results, key=lambda x: x[2]) if align_results else (0, 1, raw_std_ns, 0)
    best_delta, best_sign, best_std, best_n = best
    best_gps_offset = naive_offset + best_delta
    print(f"  Method   : {join_method.upper()} join")
    print(f"  Raw std  : {raw_std_ns:.3f} ns")
    if join_method == "utc":
        print(f"  {'Sign':>5s}  {'N pairs':>7s}  {'std (ns)':>10s}")
        for delta, sign, std_ns, n_pairs in align_results:
            tag = "  ← best" if (delta == best_delta and sign == best_sign) else ""
            print(f"  {sign:>+5d}  {n_pairs:>7d}  {std_ns:>10.3f}{tag}")
    else:
        print(f"  {'GPS Δ':>6s}  {'Sign':>5s}  {'N pairs':>7s}  {'std (ns)':>10s}")
        for delta, sign, std_ns, n_pairs in align_results:
            tags = []
            if delta == 0 and sign == +1:
                tags.append("← naive")
            if delta == best_delta and sign == best_sign:
                tags.append("← best")
            tag = "  " + " ".join(tags) if tags else ""
            print(f"  {delta:>+6d}  {sign:>+5d}  {n_pairs:>7d}  {std_ns:>10.3f}{tag}")
        if best_delta != 0 or best_sign != +1:
            print(f"  *** ALIGNMENT: GPS offset delta={best_delta:+d}, sign={best_sign:+d} "
                  f"(using GPS offset={best_gps_offset}) ***", file=sys.stderr)

    # Build psti_utc: tow_s → timestamp mapping for UTC time axis even in raw-only mode.
    psti_utc = None
    if args.psti:
        _psti_all = load_timtp(Path(args.psti))
        for _grp in _psti_all.values():
            psti_utc = _grp[["tow_s", "timestamp"]].copy()
            break

    print("Applying qErr correction …")
    df = apply_qerr(ticc, timtp, gps_offset=best_gps_offset, sign=best_sign,
                    psti_utc=psti_utc)
    print(f"  {len(df)} pairs after join (GPS offset={best_gps_offset}, sign={best_sign:+d})")
    raw_ns  = df["raw_diff_s"]  * 1e9
    corr_ns = df["corr_diff_s"] * 1e9
    print(f"  Raw  : mean={raw_ns.mean():+.2f} ns  std={raw_ns.std():.2f} ns")
    if "corr_diff_s" in df.columns and not df["corr_diff_s"].isna().all():
        print(f"  Corr : mean={corr_ns.mean():+.2f} ns  std={corr_ns.std():.2f} ns")

    print("Computing ADEV/TDEV …")
    indiv = individual_stability(df)

    write_report(df, indiv, alignment, out_stem)
    plot_diff(df, out_stem)
    plot_adev(indiv, out_stem)
    plot_tdev(indiv, out_stem)

    print("Done.")


if __name__ == "__main__":
    main()

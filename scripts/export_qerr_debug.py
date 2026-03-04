#!/usr/bin/env python3
"""
export_qerr_debug.py — Export qErr debug CSV for gnuplot inspection.

pe-rig version: chA = NEO-F10T (--timtp), chB = PX1125T (--psti).
Both sources are optional; omit either to leave that channel uncorrected.

Outputs one row per aligned epoch with:
  epoch                    sequential integer
  utc_time                 wall-clock UTC (from F10T TIM-TP, or PSTI if absent)
  qerr_top_ps              raw F10T TIM-TP qErr (ps)       — chA
  qerr_bot_ps              raw PX1125T $PSTI,00 qErr (ps)  — chB (clamped at ±4200 ps)
  qerr_top_smooth_ps       rolling-median of qerr_top (--smooth epochs)
  qerr_bot_smooth_ps       rolling-median of qerr_bot
  qerr_diff_ps             qerr_top − qerr_bot
  ticc_interval_a_ps       chA PPS interval deviation from 1 s (ps)
  ticc_interval_b_ps       chB PPS interval deviation from 1 s (ps)
                           ← retrospective "true qErr" for PX1125T;
                             compare against qerr_bot_ps to see clamping
  ticc_cumphase_a_ps       cumulative phase walk from chA intervals (ps)
  ticc_cumphase_b_ps       cumulative phase walk from chB intervals (ps)
  ticc_cumphase_smooth_a_ps  long-window smooth of cumphase_a (--long-smooth epochs)
  ticc_cumphase_smooth_b_ps  long-window smooth of cumphase_b
  raw_diff_ns              raw chA − chB TICC difference (ns)

ticc_interval_b_ps is the per-epoch PPS interval deviation of chB (PX1125T).
It represents what $PSTI,00 qErr should report if unclamped — plotting both
reveals the ±4200 ps saturation limit of the SkyTraq firmware.

Usage:
    python scripts/export_qerr_debug.py \\
        --ticc  data/foo_ticc.csv  \\
        --timtp data/foo_timtp.csv \\   # F10T TIM-TP  (chA)
        --psti  data/foo_psti.csv  \\   # PX1125T PSTI (chB)
        --out   data/foo_qerr_debug.csv \\
        [--smooth 300] [--long-smooth 1200]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── loading (mirrors analyze_pps.py) ────────────────────────────────────── #

def load_ticc(path: Path) -> pd.DataFrame:
    """
    Load TICC CSV, pair chA/chB by integer second.

    Handles three CSV format generations:
      Gen 1: timestamp_s, channel
      Gen 2: host_timestamp, timestamp_s, channel
      Gen 3: host_timestamp, ref_sec, ref_ps, channel

    Returns DataFrame with int64 columns chA_ref_sec, chA_ref_ps,
    chB_ref_sec, chB_ref_ps, raw_diff_ps, raw_diff_ns, and
    optionally host_sec.
    """
    _BOUNDARY_GUARD_S = 100e-9

    df = pd.read_csv(path)
    cols = set(df.columns)

    if "ref_sec" in cols:                      # Gen 3
        df["ref_sec"] = df["ref_sec"].astype("int64")
        df["ref_ps"]  = df["ref_ps"].astype("int64")
        df["integer_sec"] = df["ref_sec"]
    else:                                       # Gen 1 or 2
        df["integer_sec"] = df["timestamp_s"].astype("int64")
        df["ref_sec"] = df["integer_sec"]
        df["ref_ps"]  = ((df["timestamp_s"] - df["integer_sec"]) * 1e12
                         ).round().astype("int64")

    if "host_timestamp" in cols:
        host_ts = pd.to_datetime(df["host_timestamp"], utc=True)
        _epoch = pd.Timestamp("1970-01-01", tz="UTC")
        df["host_sec"] = (host_ts - _epoch).dt.total_seconds().astype("int64")

    frac_s = df["ref_ps"] / 1e12
    bad = (frac_s < _BOUNDARY_GUARD_S) | (frac_s > 1.0 - _BOUNDARY_GUARD_S)
    if bad.any():
        raise ValueError(f"TICC: {bad.sum()} edge(s) within 100 ns of second boundary")

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
    piv["raw_diff_ns"] = piv["raw_diff_ps"].astype(float) * 1e-3

    if "host_sec" in df.columns:
        hs_map = df.groupby("integer_sec")["host_sec"].first()
        piv["host_sec"] = piv["integer_sec"].map(hs_map)
    return piv


def load_timtp(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["tow_s"] = (df["tow_ms"] // 1000).astype(int)
    _epoch = pd.Timestamp("1970-01-01", tz="UTC")
    df["utc_s"] = (df["timestamp"] - _epoch).dt.total_seconds().astype("int64")
    return {
        rx: grp.sort_values("timestamp").reset_index(drop=True)
        for rx, grp in df.groupby("receiver")
    }


# ── join helpers (mirrors analyze_pps.py) ────────────────────────────────── #

def gps_join(ticc: pd.DataFrame,
             top_df: pd.DataFrame,
             bot_df: pd.DataFrame,
             gps_offset: int) -> pd.DataFrame:
    """GPS-second join fallback (for TICC CSV without host_timestamp).
    bot_df may be empty; rows are only dropped for sources that are present."""
    df = ticc.copy()
    df["gps_sec"] = df["integer_sec"] + gps_offset
    corr_tow = df["gps_sec"] - 1   # TIM-TP at S-1 predicts PPS edge at S

    top_q  = top_df.set_index("tow_s")["qerr_ps"] if not top_df.empty else pd.Series(dtype=float)
    bot_q  = bot_df.set_index("tow_s")["qerr_ps"] if not bot_df.empty else pd.Series(dtype=float)
    top_ts = top_df.set_index("tow_s")["timestamp"] if not top_df.empty else pd.Series(dtype=object)

    df["qerr_top_ps"] = corr_tow.map(top_q) if not top_q.empty else np.int64(0)
    df["qerr_bot_ps"] = corr_tow.map(bot_q) if not bot_q.empty else np.int64(0)
    df["utc_time"]    = corr_tow.map(top_ts) if not top_ts.empty else pd.NaT

    drop_cols = []
    if not top_q.empty: drop_cols.append("qerr_top_ps")
    if not bot_q.empty: drop_cols.append("qerr_bot_ps")
    if drop_cols:
        df = df.dropna(subset=drop_cols)
    return df.reset_index(drop=True)


def utc_join(ticc: pd.DataFrame,
             top_df: pd.DataFrame,
             bot_df: pd.DataFrame) -> pd.DataFrame:
    """UTC-second join when host_timestamp column is present (preferred).
    bot_df may be empty; rows are only dropped for sources that are present."""
    df = ticc.copy()
    corr_utc = df["host_sec"] - 1

    top_q  = top_df.set_index("utc_s")["qerr_ps"]  if not top_df.empty else pd.Series(dtype=float)
    bot_q  = bot_df.set_index("utc_s")["qerr_ps"]  if not bot_df.empty else pd.Series(dtype=float)
    top_ts = top_df.set_index("utc_s")["timestamp"] if not top_df.empty else pd.Series(dtype=object)
    bot_ts = bot_df.set_index("utc_s")["timestamp"] if not bot_df.empty else pd.Series(dtype=object)

    df["qerr_top_ps"] = corr_utc.map(top_q) if not top_q.empty else np.int64(0)
    df["qerr_bot_ps"] = corr_utc.map(bot_q) if not bot_q.empty else np.int64(0)
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


def best_gps_offset(ticc: pd.DataFrame,
                    top: pd.DataFrame,
                    bot: pd.DataFrame) -> tuple[int, int]:
    """
    Return (gps_offset, sign) that minimises corrected-diff std.
    Uses UTC join when host_timestamp present; GPS offset search otherwise.
    """
    ref = top if not top.empty else bot
    if ref.empty:
        return 0, +1

    if "host_sec" in ticc.columns and "utc_s" in ref.columns:
        joined = utc_join(ticc, top, bot)
        best_std = np.inf
        best_sign = +1
        for sign in (+1, -1):
            corr_ps = (joined["raw_diff_ps"]
                       + sign * (joined["qerr_top_ps"] - joined["qerr_bot_ps"]))
            std = float(corr_ps.std() * 1e-3)   # ps → ns
            if std < best_std:
                best_std = std
                best_sign = sign
        return 0, best_sign   # offset unused by utc_join, 0 is a sentinel

    naive = int(ref["tow_s"].iloc[0]) - int(ticc["integer_sec"].iloc[0])
    best_std = np.inf
    best_offset, best_sign = naive, +1

    for delta in (-1, 0, +1, +2):
        joined = gps_join(ticc, top, bot, naive + delta)
        if joined.empty:
            continue
        for sign in (+1, -1):
            corr_ps = (joined["raw_diff_ps"]
                       + sign * (joined["qerr_top_ps"] - joined["qerr_bot_ps"]))
            std = float(corr_ps.std() * 1e-3)   # ps → ns
            if std < best_std:
                best_std = std
                best_offset, best_sign = naive + delta, sign

    return best_offset, best_sign


# ── TICC cumulative phase walk ───────────────────────────────────────────── #

def cumulative_phase(ref_sec: np.ndarray, ref_ps: np.ndarray,
                     long_window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From TICC int64 ref_sec/ref_ps arrays (already sorted by epoch), compute:
      interval_ps   — per-epoch PPS interval deviation from 1 s (ps)
                      (NaN for the first epoch — no prior interval)
      cumphase_ps   — cumulative sum of interval_ps (phase walk, ps)
      smooth_ps     — rolling-median of cumphase_ps (anticipated sawtooth trend)

    All arrays have the same length as ref_sec/ref_ps.
    Using int64 arithmetic avoids float64 precision loss at long TICC uptimes.
    """
    # interval deviation from 1 s, in ps:
    # (ref_sec[n] - ref_sec[n-1] - 1) * 1e12 + (ref_ps[n] - ref_ps[n-1])
    sec_diff = np.diff(ref_sec.astype("int64"))    # expected: 1 for no gap
    ps_diff  = np.diff(ref_ps.astype("int64"))
    deviations_ps = (sec_diff - 1) * 1_000_000_000_000 + ps_diff  # int64

    # Prepend NaN so length matches input arrays
    dev_f = deviations_ps.astype(float)
    interval_ps = np.concatenate([[np.nan], dev_f])
    cumphase_ps = np.concatenate([[np.nan], np.cumsum(dev_f)])

    smooth_ps = (pd.Series(cumphase_ps)
                   .rolling(long_window, center=True, min_periods=long_window // 4)
                   .median()
                   .values)

    return interval_ps, cumphase_ps, smooth_ps


# ── main ─────────────────────────────────────────────────────────────────── #

_EMPTY = pd.DataFrame(columns=["qerr_ps", "tow_s", "utc_s", "timestamp"])


def main():
    ap = argparse.ArgumentParser(
        description="Export qErr debug CSV for gnuplot inspection (pe rig)"
    )
    ap.add_argument("--ticc",        required=True,
                    help="_ticc.csv input")
    ap.add_argument("--timtp",       default=None,
                    help="_timtp.csv from F10T (chA/TOP); optional")
    ap.add_argument("--psti",        default=None,
                    help="_psti.csv from PX1125T (chB/BOT); optional")
    ap.add_argument("--out",         required=True, help="Output .csv path")
    ap.add_argument("--smooth",      type=int, default=300,
                    help="Rolling-median window for qErr smoothing (epochs, default 300)")
    ap.add_argument("--long-smooth", type=int, default=1200,
                    help="Rolling-median window for TICC phase smoothing (epochs, default 1200)")
    args = ap.parse_args()

    print(f"Loading TICC  : {args.ticc}")
    ticc = load_ticc(Path(args.ticc))
    print(f"  {len(ticc)} paired epochs")

    # Load and remap to TOP (chA=F10T) / BOT (chB=PX1125T)
    top = _EMPTY
    if args.timtp:
        print(f"Loading TIM-TP: {args.timtp}  (F10T → chA/TOP)")
        loaded = load_timtp(Path(args.timtp))
        for grp in loaded.values():
            top = grp
            print(f"  F10T: {len(grp)} rows  "
                  f"qerr [{grp['qerr_ps'].min():+d}, {grp['qerr_ps'].max():+d}] ps")
            break

    bot = _EMPTY
    if args.psti:
        print(f"Loading $PSTI : {args.psti}  (PX1125T → chB/BOT)")
        loaded = load_timtp(Path(args.psti))
        for grp in loaded.values():
            bot = grp
            print(f"  PX1125T: {len(grp)} rows  "
                  f"qerr [{grp['qerr_ps'].min():+d}, {grp['qerr_ps'].max():+d}] ps")
            break

    if top.empty and bot.empty:
        print("WARNING: no qErr source provided — outputting TICC-only columns.")

    gps_off, sign = best_gps_offset(ticc, top, bot)
    use_utc = "host_sec" in ticc.columns and (
        ("utc_s" in top.columns and not top.empty) or
        ("utc_s" in bot.columns and not bot.empty)
    )
    if use_utc:
        print(f"  Best alignment: UTC join, sign={sign:+d}")
        df = utc_join(ticc, top, bot)
    else:
        print(f"  Best alignment: gps_offset={gps_off}, sign={sign:+d}")
        df = gps_join(ticc, top, bot, gps_off)
    print(f"  {len(df)} epochs after join")

    # qErr smoothing (short window — shows receiver sawtooth)
    df["qerr_top_smooth_ps"] = (pd.Series(df["qerr_top_ps"].values)
                                  .rolling(args.smooth, center=True,
                                           min_periods=args.smooth // 4)
                                  .median()
                                  .values)
    df["qerr_bot_smooth_ps"] = (pd.Series(df["qerr_bot_ps"].values)
                                  .rolling(args.smooth, center=True,
                                           min_periods=args.smooth // 4)
                                  .median()
                                  .values)
    df["qerr_diff_ps"] = df["qerr_top_ps"] - df["qerr_bot_ps"]

    # TICC cumulative phase walk (independent oscillator estimate)
    # ticc is already sorted by integer_sec from load_ticc(), so no re-sort needed.
    n = len(df)
    intv_a, cum_a, sm_a = cumulative_phase(
        ticc["chA_ref_sec"].values, ticc["chA_ref_ps"].values, args.long_smooth)
    intv_b, cum_b, sm_b = cumulative_phase(
        ticc["chB_ref_sec"].values, ticc["chB_ref_ps"].values, args.long_smooth)

    # Align TICC phase arrays to the joined df length
    df["ticc_interval_a_ps"]       = intv_a[:n]
    df["ticc_interval_b_ps"]       = intv_b[:n]
    df["ticc_cumphase_a_ps"]       = cum_a[:n]
    df["ticc_cumphase_b_ps"]       = cum_b[:n]
    df["ticc_cumphase_smooth_a_ps"] = sm_a[:n]
    df["ticc_cumphase_smooth_b_ps"] = sm_b[:n]

    # Select and order output columns
    out = df[[
        "utc_time",
        "qerr_top_ps",
        "qerr_bot_ps",
        "qerr_top_smooth_ps",
        "qerr_bot_smooth_ps",
        "qerr_diff_ps",
        "ticc_interval_a_ps",
        "ticc_interval_b_ps",
        "ticc_cumphase_a_ps",
        "ticc_cumphase_b_ps",
        "ticc_cumphase_smooth_a_ps",
        "ticc_cumphase_smooth_b_ps",
        "raw_diff_ns",
    ]].copy()
    out.insert(0, "epoch", range(len(out)))

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows → {args.out}")
    print(f"  Columns: {', '.join(out.columns)}")
    print("Done.")


if __name__ == "__main__":
    main()

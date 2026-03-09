#!/usr/bin/env python3
"""
derive_true_qerr.py — Derive "true" PPS qErr from TICC timestamps via
least-squares drift removal, then compare against logged SkyTraq $PSTI,00
values to check for clipping at ±4200 ps.

Approach (precision-safe for float64):
  1. Load TICC chB timestamps (int64 ref_sec / ref_ps).
  2. UTC-align with SkyTraq $PSTI,00 qErr using host_timestamp join.
  3. In sliding windows (default 900 s, max 1000 s for float64 safety):
     - Zero-base TICC timestamps: t = (ref_sec - ref_sec[0]) + ref_ps/1e12.
       With ≤3 digits left + 11 digits right of decimal, float64 is exact
       to the TICC's 10 ps resolution (60 ps accuracy).
     - Linear least-squares fit captures oscillator drift.
     - Residuals = true PPS phase perturbation = "true qErr".
  4. Compare true qErr against logged SkyTraq qErr.

Outputs:
  _true_qerr.csv       — per-epoch: true vs logged qErr
  _true_qerr_scatter.png — scatter plot (clipping shows as vertical saturation)
  _true_qerr_series.png  — time series overlay
  _true_qerr_report.txt  — summary statistics

Usage:
    python scripts/derive_true_qerr.py \\
        --ticc  data/foo_ticc.csv \\
        --psti  data/foo_psti.csv \\
        --out   data/foo \\
        [--window 900] [--overlap 100]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── loading (shared with analyze_pps.py / export_qerr_debug.py) ────────── #

def load_ticc(path: Path) -> pd.DataFrame:
    """Load TICC CSV, pair chA/chB by integer second (Gen 3 only)."""
    _BOUNDARY_GUARD_S = 100e-9
    _RESET_THRESHOLD_S = 60.0

    df = pd.read_csv(path)
    cols = set(df.columns)

    if "ref_sec" in cols:
        df["ref_sec"] = df["ref_sec"].astype("int64")
        df["ref_ps"]  = df["ref_ps"].astype("int64")
        df["integer_sec"] = df["ref_sec"]
        seq_col = "ref_sec"
    else:
        df["integer_sec"] = df["timestamp_s"].astype("int64")
        df["ref_sec"] = df["integer_sec"]
        df["ref_ps"]  = ((df["timestamp_s"] - df["integer_sec"]) * 1e12
                         ).round().astype("int64")
        seq_col = "timestamp_s"

    # Detect TICC resets
    jumps = df[seq_col].diff()
    reset_rows = jumps[jumps < -_RESET_THRESHOLD_S].index.tolist()
    if reset_rows:
        boundaries = [0] + reset_rows + [len(df)]
        sessions = [df.iloc[boundaries[i]:boundaries[i+1]].copy()
                    for i in range(len(boundaries) - 1)]
        longest = max(sessions, key=len)
        n_dropped = len(df) - len(longest)
        print(f"  TICC: detected {len(reset_rows)} reset(s); "
              f"dropped {n_dropped} pre-reset row(s), using {len(longest)} rows.")
        df = longest.reset_index(drop=True)

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

    if "host_sec" in df.columns:
        hs_map = df.groupby("integer_sec")["host_sec"].first()
        piv["host_sec"] = piv["integer_sec"].map(hs_map)
    return piv


def load_psti(path: Path) -> pd.DataFrame:
    """Load $PSTI,00 CSV. Returns DataFrame with utc_s, qerr_ps, timestamp."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    _epoch = pd.Timestamp("1970-01-01", tz="UTC")
    df["utc_s"] = (df["timestamp"] - _epoch).dt.total_seconds().astype("int64")
    df["tow_s"] = (df["tow_ms"] // 1000).astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


# ── UTC join ──────────────────────────────────────────────────────────── #

def utc_join_channel(ticc: pd.DataFrame, qerr_df: pd.DataFrame,
                     channel: str = "chB") -> pd.DataFrame:
    """
    Join TICC channel timestamps with qErr source by UTC second.

    TICC host_sec S → qErr at utc_s = S-1 predicts PPS edge at S.
    channel: "chA" or "chB".
    qerr_df: DataFrame with utc_s, qerr_ps, timestamp columns.

    Returns DataFrame with {ch}_ref_sec, {ch}_ref_ps, logged_qerr_ps, utc_time.
    """
    if "host_sec" not in ticc.columns:
        raise ValueError("TICC CSV must have host_timestamp column for UTC join")

    sec_col = f"{channel}_ref_sec"
    ps_col  = f"{channel}_ref_ps"
    df = ticc[["integer_sec", sec_col, ps_col, "host_sec"]].copy()
    corr_utc = df["host_sec"] - 1  # qErr at S-1 predicts PPS at S

    qerr_map = qerr_df.set_index("utc_s")["qerr_ps"]
    ts_map   = qerr_df.set_index("utc_s")["timestamp"]

    df["logged_qerr_ps"] = corr_utc.map(qerr_map)
    df["utc_time"]       = corr_utc.map(ts_map)
    df = df.dropna(subset=["logged_qerr_ps"]).reset_index(drop=True)
    df["logged_qerr_ps"] = df["logged_qerr_ps"].astype("int64")
    # Normalise column names so downstream code doesn't need to know the channel
    df.rename(columns={sec_col: "ch_ref_sec", ps_col: "ch_ref_ps"}, inplace=True)
    return df


# ── windowed least-squares drift removal ──────────────────────────────── #

def derive_true_qerr(df: pd.DataFrame,
                     window: int = 900,
                     overlap: int = 100) -> pd.DataFrame:
    """
    Derive true qErr from TICC chB timestamps via windowed linear fit.

    For each window of `window` epochs:
      1. Zero-base: t_i = (ref_sec[i] - ref_sec[0]) + ref_ps[i] / 1e12
         Safe in float64: ≤3 digits left + 11 right of decimal.
      2. Reject outlier epochs (ref_ps > 1 us from window median).
      3. Fit line: t_i = a * i + b  (i = epoch index within window)
      4. Residual = t_i - fitted = true qErr (seconds)

    Windows advance by (window - overlap) epochs. Epochs covered by multiple
    windows use the window whose center is closest.

    Returns df with added columns:
      true_qerr_ps  — derived qErr in picoseconds (NaN for outlier epochs)
      outlier       — boolean, True for rejected epochs
    """
    _OUTLIER_THRESH_PS = 1_000_000  # 1 us — any ref_ps this far from median is bogus

    n = len(df)
    ref_sec = df["ch_ref_sec"].values   # int64
    ref_ps  = df["ch_ref_ps"].values    # int64

    # Global outlier detection: flag epochs where ref_ps is far from global median.
    # These are TICC mispairings (wrong PPS edge matched to wrong second).
    median_ps = np.median(ref_ps.astype(float))
    outlier_mask = np.abs(ref_ps.astype(float) - median_ps) > _OUTLIER_THRESH_PS
    n_outliers = int(outlier_mask.sum())
    if n_outliers > 0:
        print(f"  Flagged {n_outliers} outlier epoch(s) "
              f"(ref_ps > {_OUTLIER_THRESH_PS/1e6:.0f} us from median)")

    # Pre-allocate: each epoch gets the residual from its best (closest-center) window
    true_qerr_ps = np.full(n, np.nan, dtype=float)
    best_dist    = np.full(n, np.inf, dtype=float)  # distance to window center

    step = max(1, window - overlap)
    n_windows = 0

    for start in range(0, n, step):
        end = min(start + window, n)
        if end - start < 10:  # too few points for a meaningful fit
            break

        sl = slice(start, end)
        win_len = end - start
        center = start + win_len / 2.0

        # Mask: only fit non-outlier epochs
        win_good = ~outlier_mask[sl]
        n_good = int(win_good.sum())
        if n_good < 10:
            continue

        # Zero-base timestamps: subtract first ref_sec in this window
        base_sec = ref_sec[start]
        # t_i = (ref_sec[i] - base_sec) + ref_ps[i] / 1e12
        # ref_sec[i] - base_sec is a small integer (0..window), safe in float64
        t_zeroed = (ref_sec[sl] - base_sec).astype(float) + ref_ps[sl].astype(float) * 1e-12

        # Epoch index within window (0, 1, 2, ...)
        idx = np.arange(win_len, dtype=float)

        # Linear least-squares on good epochs only
        coeffs = np.polyfit(idx[win_good], t_zeroed[win_good], deg=1)
        fitted = np.polyval(coeffs, idx)
        residuals_s = t_zeroed - fitted  # seconds

        # Convert to picoseconds
        residuals_ps = residuals_s * 1e12

        # Assign to non-outlier epochs closer to this window's center
        for j in range(start, end):
            if outlier_mask[j]:
                continue
            dist = abs(j - center)
            if dist < best_dist[j]:
                best_dist[j] = dist
                true_qerr_ps[j] = residuals_ps[j - start]

        n_windows += 1

    print(f"  Fitted {n_windows} windows (size={window}, overlap={overlap})")

    df = df.copy()
    df["outlier"] = outlier_mask
    # Leave NaN for outlier epochs rather than forcing to int64
    df["true_qerr_ps"] = np.round(true_qerr_ps)
    return df


# ── plotting ──────────────────────────────────────────────────────────── #

def plot_scatter(df: pd.DataFrame, out_stem: Path) -> None:
    """Scatter: true qErr vs logged SkyTraq qErr. Clipping appears as
    vertical saturation at ±4200 ps on the x-axis."""
    fig, ax = plt.subplots(figsize=(8, 8))

    good = ~df["outlier"].values & ~np.isnan(df["true_qerr_ps"].values)
    logged = df.loc[good, "logged_qerr_ps"].values.astype(float)
    true   = df.loc[good, "true_qerr_ps"].values.astype(float)

    ax.scatter(logged, true, s=1, alpha=0.3, color="steelblue")

    # Clamp lines
    for v in (-4200, 4200):
        ax.axvline(v, color="red", linewidth=1, linestyle="--", alpha=0.7,
                   label=f"SkyTraq clamp {v:+d} ps" if v == 4200 else None)
        ax.axvline(v, color="red", linewidth=1, linestyle="--", alpha=0.7)

    # 1:1 reference line
    lim = max(abs(logged.min()), abs(logged.max()),
              abs(true.min()), abs(true.max())) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=0.8,
            linestyle=":", label="1:1 line")

    ax.set_xlabel("Logged SkyTraq qErr (ps)")
    ax.set_ylabel("True qErr from TICC least-squares (ps)")
    ax.set_title("SkyTraq qErr clipping check\n"
                 "(vertical spread at ±4200 = clipping)")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_stem.parent / (out_stem.name + "_true_qerr_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def plot_series(df: pd.DataFrame, out_stem: Path) -> None:
    """Time series: true qErr and logged qErr overlaid."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    good = ~df["outlier"].values & ~np.isnan(df["true_qerr_ps"].values)
    gdf = df[good].reset_index(drop=True)
    epochs = gdf.index.values
    logged = gdf["logged_qerr_ps"].values.astype(float)
    true   = gdf["true_qerr_ps"].values.astype(float)
    diff   = true - logged

    # Panel 1: both series
    axes[0].plot(epochs, true, linewidth=0.6, color="steelblue",
                 label="True qErr (TICC least-squares)", alpha=0.8)
    axes[0].plot(epochs, logged, linewidth=0.6, color="tomato",
                 label="Logged SkyTraq qErr", alpha=0.8)
    for v in (-4200, 4200):
        axes[0].axhline(v, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("qErr (ps)")
    axes[0].set_title("True vs logged qErr")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: difference
    axes[1].plot(epochs, diff, linewidth=0.5, color="seagreen", alpha=0.8)
    axes[1].set_ylabel("True − Logged (ps)")
    axes[1].set_title("Residual: true − logged qErr")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: histogram of difference
    axes[2].hist(diff[~np.isnan(diff)], bins=100, color="seagreen",
                 alpha=0.7, edgecolor="none")
    axes[2].set_xlabel("True − Logged (ps)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Distribution of true − logged qErr difference")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("PX1125T SkyTraq qErr validation\n"
                 "(if logged qErr is correct, difference should be noise-like)",
                 fontsize=11)
    fig.tight_layout()

    path = out_stem.parent / (out_stem.name + "_true_qerr_series.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot    → {path}")


def write_report(df: pd.DataFrame, out_stem: Path) -> None:
    """Summary statistics."""
    lines = []
    a = lines.append

    # Exclude outlier epochs from statistics
    good = ~df["outlier"].values if "outlier" in df.columns else np.ones(len(df), dtype=bool)
    good &= ~np.isnan(df["true_qerr_ps"].values)
    logged = df.loc[good, "logged_qerr_ps"].values.astype(float)
    true   = df.loc[good, "true_qerr_ps"].values.astype(float)
    diff   = true - logged

    n = int(good.sum())
    n_total = len(df)
    n_outliers = n_total - n
    n_at_pos_clamp = int(np.sum(logged >= 4200))
    n_at_neg_clamp = int(np.sum(logged <= -4200))
    n_clamped = n_at_pos_clamp + n_at_neg_clamp
    pct_clamped = 100.0 * n_clamped / n if n > 0 else 0

    # Correlation
    mask = ~(np.isnan(true) | np.isnan(logged))
    r = np.corrcoef(true[mask], logged[mask])[0, 1] if mask.sum() > 10 else float("nan")

    # When clamped vs unclamped: compare true qErr ranges
    clamped_mask = (logged >= 4200) | (logged <= -4200)
    unclamped_mask = ~clamped_mask

    a("=" * 62)
    a("  SkyTraq qErr validation — true vs logged")
    a("=" * 62)
    a(f"  Epochs     : {n}  ({n_outliers} outlier(s) excluded from {n_total} total)")
    a(f"  Pearson r  : {r:+.4f}")
    a("")
    a("── Logged SkyTraq qErr ──────────────────────────────────")
    a(f"  Range      : [{logged.min():+.0f}, {logged.max():+.0f}] ps")
    a(f"  Mean       : {np.nanmean(logged):+.1f} ps")
    a(f"  Std        : {np.nanstd(logged):.1f} ps")
    a(f"  At +4200   : {n_at_pos_clamp} ({100*n_at_pos_clamp/n:.1f}%)")
    a(f"  At -4200   : {n_at_neg_clamp} ({100*n_at_neg_clamp/n:.1f}%)")
    a(f"  Clamped    : {n_clamped} ({pct_clamped:.1f}%)")
    a("")
    a("── True qErr (TICC least-squares) ───────────────────────")
    a(f"  Range      : [{np.nanmin(true):+.0f}, {np.nanmax(true):+.0f}] ps")
    a(f"  Mean       : {np.nanmean(true):+.1f} ps")
    a(f"  Std        : {np.nanstd(true):.1f} ps")
    a("")
    a("── Difference (true − logged) ──────────────────────────")
    a(f"  Mean       : {np.nanmean(diff):+.1f} ps")
    a(f"  Std        : {np.nanstd(diff):.1f} ps")
    a(f"  RMS        : {np.sqrt(np.nanmean(diff**2)):.1f} ps")
    a("")

    if clamped_mask.any() and unclamped_mask.any():
        a("── Clamped vs unclamped epochs ─────────────────────────")
        a(f"  Unclamped: diff std = {np.nanstd(diff[unclamped_mask]):.1f} ps "
          f"(n={unclamped_mask.sum()})")
        a(f"  Clamped  : diff std = {np.nanstd(diff[clamped_mask]):.1f} ps "
          f"(n={clamped_mask.sum()})")
        a(f"  True qErr at clamped: range "
          f"[{np.nanmin(true[clamped_mask]):+.0f}, "
          f"{np.nanmax(true[clamped_mask]):+.0f}] ps")
        if pct_clamped > 5:
            a("  *** Significant clipping detected — true qErr exceeds ±4200 ps ***")
        a("")

    # Correction effectiveness test: does applying true qErr reduce chB variance?
    a("── Clipping verdict ────────────────────────────────────")
    if pct_clamped < 1:
        a("  No significant clipping: logged qErr rarely reaches ±4200 ps")
    elif pct_clamped < 10:
        a("  Mild clipping: logged qErr occasionally hits ±4200 ps")
    else:
        a("  Heavy clipping: logged qErr frequently saturates at ±4200 ps")
    if abs(r) > 0.7:
        verdict = "STRONG — logged qErr tracks true qErr well"
    elif abs(r) > 0.3:
        verdict = "MODERATE — partial agreement"
    else:
        verdict = "WEAK — logged qErr does not track true qErr"
    a(f"  Correlation (r={r:+.4f}): {verdict}")
    a("")
    a("=" * 62)

    path = out_stem.parent / (out_stem.name + "_true_qerr_report.txt")
    path.write_text("\n".join(lines) + "\n")
    print(f"Report  → {path}")


# ── main ──────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(
        description="Derive true PPS qErr from TICC timestamps and compare "
                    "against logged SkyTraq $PSTI,00 values"
    )
    ap.add_argument("--ticc",    required=True, help="_ticc.csv from log_timing.py")
    ap.add_argument("--psti",    default=None,
                    help="_psti.csv (PX1125T qErr → chB)")
    ap.add_argument("--timtp",   default=None,
                    help="_timtp.csv (F10T qErr → chA)")
    ap.add_argument("--channel", default=None, choices=["chA", "chB"],
                    help="TICC channel to analyze (default: auto from qErr source)")
    ap.add_argument("--out",     required=True, help="Output filename stem")
    ap.add_argument("--window",  type=int, default=900,
                    help="Window size in epochs/seconds (max 1000 for float64 safety, default 900)")
    ap.add_argument("--overlap", type=int, default=100,
                    help="Overlap between windows (default 100)")
    args = ap.parse_args()

    if not args.psti and not args.timtp:
        ap.error("Provide at least one qErr source: --psti or --timtp")

    if args.window > 1000:
        print(f"WARNING: window={args.window} exceeds 1000 s float64 safety limit; "
              f"clamping to 1000")
        args.window = 1000

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading TICC  : {args.ticc}")
    ticc = load_ticc(Path(args.ticc))
    print(f"  {len(ticc)} paired epochs")

    # Determine channel and qErr source
    if args.timtp and not args.psti:
        channel = args.channel or "chA"
        qerr_label = "F10T TIM-TP"
        print(f"Loading TIM-TP: {args.timtp}  (F10T → {channel})")
        qerr_all = load_psti(Path(args.timtp))  # same CSV schema
    elif args.psti and not args.timtp:
        channel = args.channel or "chB"
        qerr_label = "SkyTraq $PSTI"
        print(f"Loading $PSTI : {args.psti}  (PX1125T → {channel})")
        qerr_all = load_psti(Path(args.psti))
    else:
        ap.error("Provide exactly one qErr source: --psti or --timtp (not both)")

    print(f"  {len(qerr_all)} rows  "
          f"qerr [{qerr_all['qerr_ps'].min():+d}, {qerr_all['qerr_ps'].max():+d}] ps")

    print(f"UTC-joining TICC {channel} with {qerr_label} qErr …")
    df = utc_join_channel(ticc, qerr_all, channel=channel)
    print(f"  {len(df)} epochs after join")

    print(f"Deriving true qErr (window={args.window}, overlap={args.overlap}) …")
    df = derive_true_qerr(df, window=args.window, overlap=args.overlap)

    # Output CSV
    out_csv = df[["utc_time", "ch_ref_sec", "ch_ref_ps",
                  "logged_qerr_ps", "true_qerr_ps"]].copy()
    out_csv.insert(0, "epoch", range(len(out_csv)))
    csv_path = out_stem.parent / (out_stem.name + "_true_qerr.csv")
    out_csv.to_csv(csv_path, index=False)
    print(f"CSV     → {csv_path}  ({len(out_csv)} rows)")

    write_report(df, out_stem)
    plot_scatter(df, out_stem)
    plot_series(df, out_stem)

    print("Done.")


if __name__ == "__main__":
    main()

# PX1125T Precision Evaluation Rig

Measures and compares PPS (pulse-per-second) timing from a **SkyTraq PX1125T**
and a **u-blox NEO-F10T** using a **TAPR TICC** time-interval counter.  The
primary goal is to determine whether the quantization-error (qErr) values
reported by each receiver can be used to correct PPS jitter and improve
short-term stability.

## Hardware

| Device | TICC channel | Symlink | Notes |
|--------|:---:|---------|-------|
| TAPR TICC | — | `/dev/ttyTICC` | 60 ps accuracy, 10 ps resolution |
| u-blox NEO-F10T | chA | `/dev/ttyF10T` | UBX-TIM-TP qErr (reference) |
| SkyTraq PX1125T | chB | `/dev/ttyPX1125T` | $PSTI,00 qErr (under test) |

## Quick start

```bash
# Install udev rules and serial-port permissions
sudo bash setup.sh

# Install Python dependencies
pip install -r requirements.txt

# Capture 1 hour of data
python3 scripts/log_timing.py \
    --ticc /dev/ttyTICC \
    --px   /dev/ttyPX1125T \
    --f10t /dev/ttyF10T \
    --out  data/run1 \
    --duration 3600

# Analyse PPS stability (ADEV / TDEV, qErr correction)
python3 scripts/analyze_pps.py data/run1

# Derive retrospective "true" qErr from TICC and compare to logged values
python3 scripts/derive_true_qerr.py data/run1

# Export debug CSV for interactive gnuplot inspection
python3 scripts/export_qerr_debug.py data/run1
gnuplot -e "infile='data/run1_qerr_debug.csv'" scripts/plot_qerr_debug.gp
```

## Repository layout

```
.
├── scripts/
│   ├── log_timing.py          # Multi-threaded data acquisition
│   ├── analyze_pps.py         # ADEV/TDEV analysis with qErr correction
│   ├── derive_true_qerr.py   # Least-squares true qErr from TICC
│   ├── export_qerr_debug.py  # Debug CSV export for gnuplot
│   ├── plot_qerr_debug.gp    # Interactive gnuplot viewer
│   └── check_shared.py       # Drift check vs testAnt rig
├── psti_logger.py             # SkyTraq $PSTI,00 CSV logger
├── ticc_logger.py             # TICC timestamp CSV logger
├── ticc.py                    # TICC serial reader
├── timtp_logger.py            # u-blox UBX-TIM-TP CSV logger
├── probe.py                   # PX1125T baud-rate auto-detect
├── setup.sh                   # Udev rule installer
├── 99-timing-devices.rules    # Udev rules (TICC, F10T, PX1125T)
├── shared_files.toml          # Files synced with testAnt rig
├── requirements.txt
└── data/                      # Measurement runs (6 captures)
```

## Data files

Each measurement run produces a set of files named
`px1125t_<YYYYMMDDTHHMMSS>_<type>.<ext>`:

| Suffix | Contents |
|--------|----------|
| `_ticc.csv` | TICC edges: host_timestamp, ref_sec, ref_ps, channel |
| `_psti.csv` | SkyTraq: timestamp, receiver, qerr_ps, tow_ms, week |
| `_timtp.csv` | u-blox: timestamp, receiver, qerr_ps, tow_ms, week |
| `_pps_report.txt` | Analysis summary, ADEV/TDEV tables |
| `_pps_diff.png` | PPS interval-difference time series |
| `_pps_adev.png` | Allan deviation plot |
| `_pps_tdev.png` | Time deviation plot |
| `_true_qerr.csv` | Derived true qErr vs logged qErr |
| `_qerr_debug.csv` | Full debug export (14 columns) for gnuplot |

## Analysis scripts

### analyze_pps.py

Core PPS timing analysis.  Pairs TICC chA/chB edges by second, applies qErr
corrections, and computes ADEV and TDEV at &tau; = 1, 10, 100, 1000 s.
Produces before/after comparisons showing the effect of qErr correction.

### derive_true_qerr.py

Derives the actual PPS phase offset ("true qErr") from TICC timestamps using
windowed least-squares drift removal on zero-based timestamps (float64-safe
within ~1000 s windows).  Compares true qErr against logged receiver values to
detect clipping.  SkyTraq $PSTI,00 saturates at &plusmn;4200 ps; this script
reveals whether the actual phase excursions exceed that limit.

Supports both receivers:

```bash
python3 scripts/derive_true_qerr.py data/run1              # SkyTraq chB (default)
python3 scripts/derive_true_qerr.py data/run1 --timtp       # u-blox chA
```

### export_qerr_debug.py

Exports a 14-column debug CSV joining both receivers' qErr values with TICC
interval deviations and cumulative phase walk.  Designed for interactive
exploration with gnuplot.

### plot_qerr_debug.gp

Interactive gnuplot script (Qt terminal) for the debug CSV.  Right-click to
zoom, scroll to scale, `a` to autoscale, `q` to quit.

## Key findings

- **u-blox TIM-TP** qErr reliably predicts PPS phase offset (&minus;0.999
  delta-model correlation).  Applying the correction materially reduces ADEV.
- **SkyTraq $PSTI,00** qErr correlates with PPS *interval deviation* (velocity)
  at r&approx;0.5, not with phase (position).  The standard position-based
  correction formula does not apply, and cumsum-based correction diverges.
- The SkyTraq qErr exhibits a visible sawtooth whose period and direction vary
  across and within captures.  The sawtooth reflects internal receiver clock
  quantization and does not directly track PPS output timing.
- SkyTraq qErr saturates at &plusmn;4200 ps.  The 16-hour capture (T143235)
  shows only 1.2% clamping, but the shorter T171803 run shows 46% clamping.

## Shared files

Four files are synced with the `testAnt` rig via `shared_files.toml`:
`ticc.py`, `ticc_logger.py`, `timtp_logger.py`, `export_qerr_debug.py`.
Run `scripts/check_shared.py` to detect drift.  `analyze_pps.py` is an
intentional fork (different receiver labels).

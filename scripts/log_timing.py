#!/usr/bin/env python3
"""
log_timing.py — Log TICC, PX1125T ($PSTI,00), and optionally NEO-F10T
(UBX-TIM-TP) data to CSV files for later PPS timing analysis.

Three threads run concurrently:
  1. TICC reader   → <stem>_ticc.csv
  2. PX1125T NMEA  → <stem>_psti.csv     ($GNZDA for UTC, $PSTI,00 for qErr)
  3. NEO-F10T UBX  → <stem>_timtp.csv    (UBX-TIM-TP; skipped if --f10t-port absent)

Usage:
    # PX1125T + TICC only (TICC on chA=NEO-F10T, chB=PX1125T or vice versa)
    python3 scripts/log_timing.py \\
        --ticc /dev/ttyTICC \\
        --px /dev/ttyPX1125T \\
        --out data/run1 \\
        --duration 3600

    # With NEO-F10T UBX-TIM-TP (must stop onocoy-miner first):
    python3 scripts/log_timing.py \\
        --ticc /dev/ttyTICC \\
        --px /dev/ttyPX1125T \\
        --f10t /dev/ttyF10T \\
        --out data/run1 \\
        --duration 3600

Output files:
    <stem>_ticc.csv     host_timestamp, ref_sec, ref_ps, channel
    <stem>_psti.csv     timestamp, receiver, qerr_ps, tow_ms, week
    <stem>_timtp.csv    timestamp, receiver, qerr_ps, tow_ms, week  (if --f10t)

Note: Run 'python3 scripts/analyze_pps.py' afterwards to compute ADEV/TDEV.
"""

import argparse
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import serial
import pyubx2

# Ensure local modules resolve from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ticc import Ticc
from ticc_logger import TiccLogger
from psti_logger import PstiLogger, parse_gnzda, parse_psti00
from timtp_logger import TimtpLogger

_stop = threading.Event()


# ── TICC thread ───────────────────────────────────────────────────────── #

def ticc_thread(port: str, logger: TiccLogger, counters: dict) -> None:
    print(f"[TICC] Opening {port}", flush=True)
    try:
        with Ticc(port) as ticc:
            for ch, ref_sec, ref_ps in ticc:
                host_ts = datetime.now(tz=timezone.utc)
                if _stop.is_set():
                    break
                logger.write(ch, ref_sec, ref_ps, host_ts)
                counters["ticc"] += 1
    except Exception as e:
        print(f"[TICC] Error: {e}", flush=True)


# ── PX1125T NMEA thread ───────────────────────────────────────────────── #

def psti_thread(port: str, logger: PstiLogger, counters: dict) -> None:
    print(f"[PX1125T] Opening {port} @ 115200", flush=True)
    try:
        with serial.Serial(port, 115200, timeout=2) as ser:
            ser.reset_input_buffer()
            while not _stop.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="replace").strip()

                if line.startswith("$PSTI,00"):
                    qerr_ns = parse_psti00(line)
                    if qerr_ns is not None:
                        # Use host wall-clock at moment of receipt, same as
                        # F10T TIM-TP, so utc_s joins correctly with TICC host_sec.
                        host_ts = datetime.now(tz=timezone.utc)
                        logger.write(host_ts, qerr_ns)
                        counters["psti"] += 1

    except Exception as e:
        print(f"[PX1125T] Error: {e}", flush=True)


# ── NEO-F10T UBX thread ───────────────────────────────────────────────── #

def timtp_thread(port: str, logger: TimtpLogger, counters: dict) -> None:
    print(f"[NEO-F10T] Opening {port} @ 38400", flush=True)
    try:
        with serial.Serial(port, 38400, timeout=2) as ser:
            ser.reset_input_buffer()
            reader = pyubx2.UBXReader(ser, protfilter=2,
                                       quitonerror=pyubx2.ERR_IGNORE)
            while not _stop.is_set():
                try:
                    _, msg = reader.read()
                except Exception:
                    continue
                if msg is None or msg.identity != "TIM-TP":
                    continue
                utc = datetime.now(tz=timezone.utc)
                logger.write(
                    timestamp = utc,
                    receiver  = "F10T",
                    qerr_ps   = msg.qErr,
                    tow_ms    = msg.towMS,
                    week      = msg.week,
                )
                counters["timtp"] += 1

    except Exception as e:
        print(f"[NEO-F10T] Error: {e}", flush=True)


# ── status printer ────────────────────────────────────────────────────── #

def status_thread(counters: dict, interval: int = 60) -> None:
    start = time.monotonic()
    while not _stop.is_set():
        time.sleep(interval)
        if _stop.is_set():
            break
        elapsed = int(time.monotonic() - start)
        parts = [f"t={elapsed}s"]
        for k, v in counters.items():
            parts.append(f"{k}={v}")
        print("  ".join(parts), flush=True)


# ── main ──────────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Log TICC + PX1125T [+ NEO-F10T] timing data to CSV"
    )
    ap.add_argument("--ticc",     required=True, help="TICC serial port")
    ap.add_argument("--px",       required=True, help="PX1125T serial port (default: /dev/ttyPX1125T)")
    ap.add_argument("--f10t",     default=None,  help="NEO-F10T port; omit to skip TIM-TP logging")
    ap.add_argument("--out",      required=True, help="Output filename stem (e.g. data/run1)")
    ap.add_argument("--duration", type=int, default=0,
                    help="Run duration in seconds (0 = run until Ctrl-C)")
    args = ap.parse_args()

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)

    ticc_path  = stem.parent / (stem.name + "_ticc.csv")
    psti_path  = stem.parent / (stem.name + "_psti.csv")
    timtp_path = stem.parent / (stem.name + "_timtp.csv")

    counters = {"ticc": 0, "psti": 0, "timtp": 0}

    signal.signal(signal.SIGINT,  lambda *_: _stop.set())
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())

    threads: list[threading.Thread] = []

    with TiccLogger(ticc_path) as tl, PstiLogger(psti_path) as pl:
        threads.append(threading.Thread(
            target=ticc_thread, args=(args.ticc, tl, counters), daemon=True))
        threads.append(threading.Thread(
            target=psti_thread, args=(args.px, pl, counters), daemon=True))
        threads.append(threading.Thread(
            target=status_thread, args=(counters,), daemon=True))

        if args.f10t:
            with TimtpLogger(timtp_path) as tml:
                threads.append(threading.Thread(
                    target=timtp_thread, args=(args.f10t, tml, counters), daemon=True))
                for t in threads:
                    t.start()
                _wait(args.duration)
        else:
            for t in threads:
                t.start()
            _wait(args.duration)

    _stop.set()
    print(f"\nDone. TICC={counters['ticc']}  PSTI={counters['psti']}  TIMTP={counters['timtp']}")
    print(f"  {ticc_path}")
    print(f"  {psti_path}")
    if args.f10t:
        print(f"  {timtp_path}")


def _wait(duration: int) -> None:
    if duration > 0:
        print(f"Running for {duration}s (Ctrl-C to stop early)…", flush=True)
        deadline = time.monotonic() + duration
        while not _stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.5)
        _stop.set()
    else:
        print("Running until Ctrl-C…", flush=True)
        _stop.wait()


if __name__ == "__main__":
    main()

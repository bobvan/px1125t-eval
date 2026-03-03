#!/usr/bin/env python3
"""
Probe the SkyTraq PX1125T: detect baud rate and print NMEA sentences.

Usage:
    python3 probe.py [--port /dev/ttyPX1125T] [--baud 115200] [--count 40]

Auto-detection tries common baud rates until NMEA sentences appear.
"""

import argparse
import sys
import time
import serial

DEVICE   = "/dev/ttyPX1125T"
BAUDRATES = [115200, 9600, 38400, 57600, 4800]
NMEA_TIMEOUT = 3   # seconds to wait per baud rate when probing


def looks_like_nmea(line: bytes) -> bool:
    return line.startswith(b"$G") or line.startswith(b"$P")


def try_baud(port: str, baud: int, timeout: float = NMEA_TIMEOUT) -> bool:
    """Return True if NMEA sentences appear at this baud rate."""
    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            ser.reset_input_buffer()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = ser.readline()
                if looks_like_nmea(line):
                    return True
    except serial.SerialException:
        pass
    return False


def detect_baud(port: str) -> int | None:
    print(f"Probing {port} for NMEA output...")
    for baud in BAUDRATES:
        print(f"  {baud}...", end=" ", flush=True)
        if try_baud(port, baud):
            print("OK")
            return baud
        print("no NMEA")
    return None


def stream_nmea(port: str, baud: int, count: int | None):
    print(f"\nStreaming NMEA from {port} @ {baud} baud "
          f"({'∞' if count is None else count} sentences):\n")
    seen = 0
    with serial.Serial(port, baud, timeout=5) as ser:
        ser.reset_input_buffer()
        while count is None or seen < count:
            raw = ser.readline()
            if not raw:
                print("(timeout — no data)", flush=True)
                continue
            line = raw.decode("ascii", errors="replace").rstrip()
            if looks_like_nmea(raw):
                print(line, flush=True)
                seen += 1


def main():
    parser = argparse.ArgumentParser(
        description="Probe SkyTraq PX1125T and print NMEA"
    )
    parser.add_argument("--port",  default=DEVICE,
                        help=f"Serial port (default: {DEVICE})")
    parser.add_argument("--baud",  type=int, default=None,
                        help="Baud rate (default: auto-detect)")
    parser.add_argument("--count", type=int, default=40,
                        help="Number of NMEA sentences to print (default: 40, 0=∞)")
    args = parser.parse_args()

    baud = args.baud
    if baud is None:
        baud = detect_baud(args.port)
        if baud is None:
            sys.exit(f"No NMEA detected on {args.port} at any standard baud rate.")
    else:
        print(f"Using {args.port} @ {baud} baud")

    stream_nmea(args.port, baud, args.count if args.count != 0 else None)


if __name__ == "__main__":
    main()

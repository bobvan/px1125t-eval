"""
psti_logger.py — Parse $PSTI,00 sentences from SkyTraq PX1125T and log to CSV.

The PX1125T emits $PSTI,00 once per second reporting the quantization error
of the NEXT PPS edge — the same role as UBX-TIM-TP's qErr field on u-blox
receivers.

  $PSTI,00,<Mode>,<SurveyLen>,<qErr_ns>,<StdDevThresh>,<CalcStdDev>*<ck>

qErr_ns is in nanoseconds (float, e.g. -0.3).  We convert to integer
picoseconds to match timtp_logger.py's schema so that analyze_pps.py can
be used unchanged.

GPS time of week is derived from $GNZDA UTC using the current GPS–UTC leap
second offset (18 s since 2017-01-01), enabling GPS-second alignment in
analyze_pps.py without modification.
"""

from __future__ import annotations

import csv
import io
import threading
from datetime import datetime, timezone
from pathlib import Path


FIELDS = ["timestamp", "receiver", "qerr_ps", "tow_ms", "week"]

# GPS epoch and current leap-second offset (update if a new leap second is added)
_GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
_GPS_LEAP_S = 18   # GPS ahead of UTC by 18 s as of 2017-01-01


def utc_to_gps(utc: datetime) -> tuple[int, int]:
    """Return (tow_ms, week) for a UTC datetime."""
    gps_s  = (utc - _GPS_EPOCH).total_seconds() + _GPS_LEAP_S
    week   = int(gps_s // 604_800)
    tow_ms = int((gps_s % 604_800) * 1000)
    return tow_ms, week


def parse_psti00(sentence: str) -> float | None:
    """
    Parse a $PSTI,00 sentence and return the quantization error in nanoseconds.

    Returns None if the sentence is malformed or the checksum fails.
    """
    try:
        # Strip leading $ and trailing checksum
        if "*" in sentence:
            body, ck = sentence.lstrip("$").rsplit("*", 1)
            expected = 0
            for c in body:
                expected ^= ord(c)
            if int(ck.strip(), 16) != expected:
                return None
        else:
            body = sentence.lstrip("$")

        parts = body.split(",")
        # $PSTI,00,<mode>,<survey_len>,<qerr_ns>,<thresh>,<calc_std>
        if len(parts) < 5 or parts[0] != "PSTI" or parts[1] != "00":
            return None
        return float(parts[3])
    except (ValueError, IndexError):
        return None


def parse_gnzda(sentence: str) -> datetime | None:
    """
    Parse a $GNZDA (or $GPZDA) sentence and return a UTC datetime.

    $GNZDA,HHMMSS.sss,DD,MM,YYYY,tz_h,tz_m*ck
    """
    try:
        body = sentence.lstrip("$").split("*")[0]
        parts = body.split(",")
        if len(parts) < 5:
            return None
        t_str = parts[1]        # HHMMSS.sss
        day   = int(parts[2])
        month = int(parts[3])
        year  = int(parts[4])
        hh = int(t_str[0:2])
        mm = int(t_str[2:4])
        ss = int(t_str[4:6])
        us = int(round(float("0" + t_str[6:]) * 1_000_000)) if len(t_str) > 6 else 0
        return datetime(year, month, day, hh, mm, ss, us, tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


class PstiLogger:
    """
    Append PX1125T timing data ($PSTI,00) to a CSV file.

    One row per second.  Caller supplies UTC time (from $GNZDA) and the
    quantization error (from $PSTI,00).  Written under a threading lock
    so it is safe to call from a dedicated reader thread.
    """

    def __init__(self, path: Path, receiver: str = "PX1125T"):
        self.path     = path
        self.receiver = receiver
        self._file:   io.TextIOWrapper | None = None
        self._writer: csv.DictWriter   | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> "PstiLogger":
        new_file = not self.path.exists()
        self._file   = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if new_file:
            self._writer.writeheader()
        return self

    def __exit__(self, *_) -> None:
        if self._file:
            self._file.close()

    def write(self, utc: datetime, qerr_ns: float) -> None:
        tow_ms, week = utc_to_gps(utc)
        with self._lock:
            self._writer.writerow({
                "timestamp": utc.isoformat(),
                "receiver":  self.receiver,
                "qerr_ps":   int(round(qerr_ns * 1000)),
                "tow_ms":    tow_ms,
                "week":      week,
            })
            self._file.flush()

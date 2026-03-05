#!/usr/bin/env gnuplot
# plot_qerr_debug.gp — Interactive gnuplot for qErr debug CSV (pe rig)
#
# Usage (override infile on command line):
#   gnuplot -e "infile='data/px1125t_20260304T171803_qerr_debug.csv'" scripts/plot_qerr_debug.gp
#
# Interactive controls (Qt terminal):
#   Right-click drag  — zoom box
#   Scroll wheel      — zoom in/out
#   Middle-click drag — pan
#   u                 — undo zoom (restore previous view)
#   a                 — autoscale (fit all data)
#   q                 — quit
#
# Columns in the CSV:
#   1  epoch
#   2  utc_time          (string — referenced by col number only)
#   3  qerr_top_ps       F10T reported qErr (chA)
#   4  qerr_bot_ps       PX1125T reported qErr, clamped ±4200 ps (chB)
#   5  qerr_top_smooth   F10T smoothed (300-epoch median)
#   6  qerr_bot_smooth   PX1125T smoothed
#   7  qerr_diff_ps      TOP − BOT corrected differential
#   8  ticc_interval_a   TICC chA per-epoch interval deviation (ps)
#   9  ticc_interval_b   TICC chB per-epoch interval deviation (ps, "true" qErr for PX1125T)
#  10  ticc_cumphase_a   TICC chA cumulative phase (ps)
#  11  ticc_cumphase_b   TICC chB cumulative phase (ps)
#  12  ticc_cumphase_smooth_a
#  13  ticc_cumphase_smooth_b
#  14  raw_diff_ns       A−B raw interval difference (ns)

# ── default file (override with -e "infile='...'" before loading) ───────── #
if (!exists("infile")) infile = "data/px1125t_20260304T171803_qerr_debug.csv"

# ── terminal ─────────────────────────────────────────────────────────────── #
set terminal qt size 1400,900 title infile enhanced font "Sans,10"
set datafile separator ","
set datafile missing ""
set datafile columnheaders
set key outside right top

# ── multiplot layout: top = qErr, bottom = cumulative phase ─────────────── #
set multiplot layout 2,1 title infile font "Sans,11"

# ────────────────────────────────────────────────────────────────────────── #
# Panel 1: PX1125T reported qErr vs TICC retrospective (true) qErr
# ────────────────────────────────────────────────────────────────────────── #
set xlabel "Epoch"
set ylabel "qErr (ps)"
set grid
set xrange [1000:1150]
set yrange [*:*]    # auto; clamp lines will anchor visible range implicitly

# Clamp boundary markers
set arrow 1 from graph 0,first  4200 to graph 1,first  4200 nohead lc rgb "#cc4444" lw 1 dt 2
set arrow 2 from graph 0,first -4200 to graph 1,first -4200 nohead lc rgb "#cc4444" lw 1 dt 2
set label 1 "clamp +4200" at graph 0.01, first  4400 tc rgb "#cc4444" font "Sans,9"
set label 2 "clamp −4200" at graph 0.01, first -4600 tc rgb "#cc4444" font "Sans,9"

plot \
    infile using 1:4 \
        with lp lw 1 lc rgb "#aaaaaa" title "PX1125T reported (raw)", \
    infile using 1:6 \
        with lp lw 2 lc rgb "#cc4400" title "PX1125T smoothed (300-ep median)", \
    infile using 1:9 \
        with lp lw 2 lc rgb "#228800" title "TICC interval-B (true qErr)"

unset arrow 1
unset arrow 2
unset label 1
unset label 2

# ────────────────────────────────────────────────────────────────────────── #
# Panel 2: cumulative phase — smooth TICC chB vs F10T-corrected
# ────────────────────────────────────────────────────────────────────────── #
#set xlabel "Epoch"
#set ylabel "Cumulative phase (ps)"
#set yrange [*:*]
#
#plot \
#    infile using 1:11 with lines lw 1 lc rgb "#aaaaaa" title "TICC cumphase-B (raw)", \
#    infile using 1:13 with lines lw 2 lc rgb "#4488cc" title "TICC cumphase-B (smooth)", \
#    infile using 1:10 with lines lw 1 lc rgb "#ddaa00" title "TICC cumphase-A (raw)", \
#    infile using 1:12 with lines lw 2 lc rgb "#228800" title "TICC cumphase-A (smooth)"

unset multiplot
pause -1 "Press Enter (or close window) to exit"

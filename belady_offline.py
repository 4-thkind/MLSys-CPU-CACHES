"""Belady's optimal LLC hit rate on a logged ChampSim LLC access stream.

Usage: python3 belady_offline.py llc_access.bin sim_output.txt WAYS

- llc_access.bin: written by the `llclog` replacement module (same run as sim_output.txt)
- sim_output.txt: that run's ChampSim output (used to find where warmup ends)
- WAYS: LLC associativity used in that run

It first replays LRU on the log and checks it matches ChampSim's own LRU hit count
(proves the log is complete), then computes Belady (MIN, with bypass) on the same stream.
Only ROI (post-warmup) accesses are counted, like ChampSim's stats.
"""
import re
import sys

import numpy as np

path, sim_txt, ways = sys.argv[1], sys.argv[2], int(sys.argv[3])
rec = np.fromfile(path, dtype=[("line", "<u8"), ("set", "<u4"), ("type", "<u4")])
txt = open(sim_txt).read()
roi_total, roi_hit = map(int, re.search(r"cpu0->LLC TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)", txt).groups())
load_total, load_hit = map(int, re.search(r"cpu0->LLC LOAD\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)", txt).groups())
start = len(rec) - roi_total
assert start >= 0, "log shorter than ChampSim's ROI access count"
print(f"{len(rec):,} logged accesses, ROI starts at {start:,}")

lines, sets, types = rec["line"], rec["set"], rec["type"]
LOAD, WRITE = 0, 3

# next use of the same line (sets are a function of the line, so per-line is per-set)
order = np.lexsort((np.arange(len(lines)), lines))
nxt = np.full(len(lines), np.iinfo(np.int64).max, dtype=np.int64)
same = lines[order[1:]] == lines[order[:-1]]
nxt[order[:-1][same]] = order[1:][same]
# Sets are independent, so replay one set at a time (keeps memory small for big logs).
by_set = np.lexsort((np.arange(len(sets)), sets))
bounds = np.flatnonzero(np.diff(sets[by_set])) + 1
groups = np.split(by_set, bounds)


def replay(policy):
    hits = load_hits = 0
    for g in groups:
        idx_l, x_l, t_l, n_l = g.tolist(), lines[g].tolist(), types[g].tolist(), nxt[g].tolist()
        blk = {}  # line -> key (last-use time for LRU, next-use for Belady)
        for i, x, t, nu in zip(idx_l, x_l, t_l, n_l):
            roi = i >= start
            if x in blk:
                if roi:
                    hits += 1
                    load_hits += t == LOAD
                if policy == "lru":
                    if t != WRITE:  # ChampSim's LRU skips writeback hits
                        blk[x] = i
                else:
                    blk[x] = nu
                continue
            if policy == "lru":
                if len(blk) >= ways:
                    del blk[min(blk, key=blk.get)]
                blk[x] = i
            else:
                if len(blk) >= ways:
                    far = max(blk, key=blk.get)
                    if nu >= blk[far] and t != WRITE:  # optimal to bypass
                        continue
                    del blk[far]
                blk[x] = nu
    return hits, load_hits


lru_hits, lru_load = replay("lru")
match = "MATCH" if lru_hits == roi_hit else f"MISMATCH (off by {lru_hits - roi_hit:+,})"
print(f"LRU replay: {lru_hits:,} hits vs ChampSim LRU {roi_hit:,} -> {match}")
bel_hits, bel_load = replay("belady")
print(f"Belady: TOTAL hit rate {100 * bel_hits / roi_total:.2f}%, LOAD hit rate {100 * bel_load / load_total:.2f}%")
print(f"ChampSim LRU LOAD hit rate on this run: {100 * load_hit / load_total:.2f}%")

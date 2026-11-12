import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import heapq
    return heapq, mo, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # LLC ceiling model for LLM decode

    Per decode token, the CPU streams weights + KV cache + activations (footprint **F**).
    When F > LLC capacity **C**, LRU thrashes on the cyclic pattern (~0 hits), while
    Belady's optimal policy keeps ~C of it resident. So for **any** replacement policy:

    - max DRAM bytes saved per token ≈ `C / F`  (when F > C; 0 headroom when F ≤ C, since LRU already fits it)
    - max decode speedup (bandwidth-bound) ≈ `F / (F − C)`

    This is an upper bound: ignores associativity, prefetchers, compute-bound time, and L2 capacity.
    """)
    return


@app.cell
def _():
    # HF config.json fields. Verify against the model cards before the paper.
    MODELS = {
        #                    hidden layers heads kv  inter   vocab
        "SmolLM2-135M":     (576,   30,    9,   3,  1536,   49152),
        "SmolLM2-360M":     (960,   32,    15,  5,  2560,   49152),
        "Qwen2.5-0.5B":     (896,   24,    14,  2,  4864,  151936),
        "TinyLlama-1.1B":   (2048,  22,    32,  4,  5632,   32000),
        "Llama-3.2-1B":     (2048,  16,    32,  8,  8192,  128256),
        "Qwen2.5-1.5B":     (1536,  28,    12,  2,  8960,  151936),
        "Llama-3.2-3B":     (3072,  28,    24,  8,  8192,  128256),
        "Llama-2-7B":       (4096,  32,    32,  32, 11008,  32000),
        "Llama-3.1-8B":     (4096,  32,    32,  8,  14336, 128256),
    }
    # ponytail: flat bits/weight; real K-quants keep some tensors (e.g. output) at higher precision
    BPW = {"FP16": 16.0, "Q8_0": 8.5, "Q4_0": 4.5}
    KV_BYTES = 2  # F16 KV cache (llama.cpp default)
    LLC_MB = {
        "i7-13650HX (24MB)": 24,
        "Ryzen X3D (96MB)": 96,
        "EPYC Genoa (384MB)": 384,
        "EPYC Genoa-X (1152MB)": 1152,
    }
    CONTEXTS = [256, 1024, 4096]
    # Measured on the laptop (llama-bench tg128, 6 threads, 6 models): per-token time
    # fits t = bytes / DRAM_BW + T0 with residuals < 0.3 ms.
    DRAM_BW = 53.1e9   # bytes/s, effective decode bandwidth
    T0 = 0.76e-3       # s/token, fixed overhead (sync, small ops); the compute floor
    return BPW, CONTEXTS, DRAM_BW, KV_BYTES, LLC_MB, MODELS, T0


@app.cell
def _(KV_BYTES):
    MB = 2**20

    def weights_streamed_per_token(h, L, H, kv, inter, vocab):
        hd = h // H
        attn = h * (H * hd) * 2 + h * (kv * hd) * 2      # q, o + k, v
        mlp = 3 * h * inter                               # SwiGLU gate/up/down
        return L * (attn + mlp) + vocab * h               # + output head (embedding: 1 row, ignored)

    def kv_bytes_per_token(h, L, H, kv):
        return L * 2 * kv * (h // H) * KV_BYTES

    def ceiling(F, C):
        """(fraction of DRAM bytes saved, speedup) upper bound over LRU."""
        if F <= C:
            return 0.0, 1.0
        return C / F, F / (F - C)

    def ceiling_floor(F, C, bw, t0):
        """Speedup bound with the measured per-token overhead: t = DRAM bytes / bw + t0."""
        # ponytail: LLC hits treated as free; add C / LLC_bw if LLC bandwidth gets measured
        if F <= C:
            return 1.0
        return (F / bw + t0) / ((F - C) / bw + t0)
    return MB, ceiling, ceiling_floor, kv_bytes_per_token, weights_streamed_per_token


@app.cell
def _(heapq):
    # The one check: on a cyclic stream (what decode looks like), LRU gets 0 hits
    # and Belady gets ~C/N, which is the formula above.
    def lru_hits(trace, C):
        from collections import OrderedDict
        c, hits = OrderedDict(), 0
        for x in trace:
            if x in c:
                hits += 1
                c.move_to_end(x)
            else:
                c[x] = None
                if len(c) > C:
                    c.popitem(last=False)
        return hits

    def belady_hits(trace, C):
        nxt, last = [0] * len(trace), {}
        for i in range(len(trace) - 1, -1, -1):
            nxt[i] = last.get(trace[i], float("inf"))
            last[trace[i]] = i
        cache, heap, hits = {}, [], 0
        for i, x in enumerate(trace):
            if x in cache:
                hits += 1
            elif len(cache) >= C:
                while True:  # lazy-deleted max-heap on next use
                    nu, y = heapq.heappop(heap)
                    if cache.get(y) == -nu:
                        del cache[y]
                        break
            cache[x] = nxt[i]
            heapq.heappush(heap, (-nxt[i], x))
        return hits

    _N, _C, _R = 1000, 250, 20
    _trace = list(range(_N)) * _R
    _steady = _N * (_R - 1)
    _lru = lru_hits(_trace, _C) / _steady
    _bel = belady_hits(_trace, _C) / _steady
    assert _lru == 0.0
    assert abs(_bel - _C / _N) < 0.02, _bel
    check_msg = f"check ok: LRU={_lru:.3f}, Belady={_bel:.3f}, C/N={_C/_N:.3f}"
    check_msg
    return (check_msg,)


@app.cell
def _(BPW, CONTEXTS, DRAM_BW, LLC_MB, MB, MODELS, T0, ceiling, ceiling_floor, kv_bytes_per_token, pd, weights_streamed_per_token):
    _rows = []
    for _name, (_h, _L, _H, _kv, _inter, _vocab) in MODELS.items():
        _wp = weights_streamed_per_token(_h, _L, _H, _kv, _inter, _vocab)
        _kvt = kv_bytes_per_token(_h, _L, _H, _kv)
        for _q, _bpw in BPW.items():
            _W = _wp * _bpw / 8
            for _ctx in CONTEXTS:
                _KV = _kvt * _ctx
                _F = _W + _KV
                for _cpu, _llc in LLC_MB.items():
                    _saved, _sp = ceiling(_F, _llc * MB)
                    _rows.append(dict(
                        model=_name, quant=_q, ctx=_ctx, cpu=_cpu, llc_mb=_llc,
                        weights_mb=_W / MB, kv_kb_per_tok=_kvt / 1024, kv_mb=_KV / MB,
                        footprint_mb=_F / MB, F_over_C=_F / (_llc * MB),
                        max_saved_pct=100 * _saved, max_speedup=_sp,
                        max_speedup_floor=ceiling_floor(_F, _llc * MB, DRAM_BW, T0),
                    ))
    df = pd.DataFrame(_rows)
    df.to_csv("ceiling_results.csv", index=False)
    df
    return (df,)


@app.cell
def _(df, mo):
    mo.md("## Laptop view (24MB L3, ctx=1024)")
    _laptop = df[(df.llc_mb == 24) & (df.ctx == 1024)]
    _laptop.pivot_table(index="model", columns="quant", values="max_saved_pct").round(1)
    return


@app.cell
def _(df, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    for _q, _g in df[df.ctx == 1024].groupby("quant"):
        _ax.scatter(_g.F_over_C, _g.max_speedup_floor, label=_q, s=18)
    _ax.set_xscale("log")
    _ax.axvline(1, color="gray", ls="--", lw=0.8)
    _ax.set_xlabel("footprint / LLC capacity  (F/C)")
    _ax.set_ylabel("max decode speedup over LRU (with measured floor)")
    _ax.set_title("Replacement-policy headroom vs. footprint-to-LLC ratio (ctx=1024)")
    _ax.legend()
    _fig.savefig("fig_ceiling.png", dpi=200, bbox_inches="tight")
    _fig
    return


@app.cell
def _(df, mo):
    mo.md("## Sweet spot: configs with ≥ 10% headroom")
    df[df.max_saved_pct >= 10].sort_values("max_speedup_floor", ascending=False)[
        ["model", "quant", "ctx", "cpu", "footprint_mb", "F_over_C", "max_saved_pct", "max_speedup", "max_speedup_floor"]
    ].round(2)
    return


@app.cell
def _(check_msg, df, mo):
    import zipfile
    _laptop = df[(df.llc_mb == 24) & (df.ctx == 1024)].pivot_table(
        index="model", columns="quant", values="max_saved_pct").round(1)
    _sweet = df[df.max_saved_pct >= 10].sort_values("max_speedup_floor", ascending=False)
    with open("summary.txt", "w") as _f:
        _f.write(check_msg + "\n\n== Laptop (24MB, ctx=1024), max % DRAM bytes saved ==\n")
        _f.write(_laptop.to_string() + "\n\n== Sweet spot (>=10% headroom) ==\n")
        _f.write(_sweet.round(2).to_string(index=False) + "\n")
    with zipfile.ZipFile("ceiling_results.zip", "w") as _z:
        for _p in ["ceiling_results.csv", "fig_ceiling.png", "summary.txt"]:
            _z.write(_p)
    mo.download(open("ceiling_results.zip", "rb").read(), filename="ceiling_results.zip",
                label="Download ceiling_results.zip")
    return


if __name__ == "__main__":
    app.run()

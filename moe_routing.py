# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers>=5.5",
#     "accelerate",
#     "numpy",
#     "pandas",
#     "matplotlib",
# ]
# ///
import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import gc
    import json
    import os
    import re
    import shutil
    import time
    import zipfile
    from collections import OrderedDict

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    return OrderedDict, gc, json, mo, np, os, pd, plt, re, shutil, time, zipfile


@app.cell
def _(mo):
    mo.md("""
    # MoE expert routing → LLC headroom

    **Phase 1 (GPU):** run each MoE on chat / code / math prompts, record which experts every
    decode token uses in every layer (hooks on the router modules).

    **Phase 2 (CPU):** from those routes, measure expert popularity skew and reuse, then simulate an
    LLC at expert granularity: LRU, BIP (scan-resistant), a popularity-aware policy, and Belady.

    Re-running is safe: finished models are cached in `moe_out/` and skipped.

    Needs `HF_TOKEN` (read) in molab secrets, and the Gemma 4 license accepted on its model page.
    """)
    return


@app.cell
def _(os):
    MODELS = {
        "qwen3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B",
        "gemma4-26b-a4b": "google/gemma-4-26B-A4B-it",
        "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "gpt-oss-20b": "openai/gpt-oss-20b",
    }
    MAX_NEW_TOKENS = 256
    DELETE_WEIGHTS_AFTER = True  # free disk between models (~200 GB total otherwise)
    BPW = 4.5                    # bytes model: Q4_0-class quantization, bits per weight
    LLC_MB = [24, 96, 384, 1152]
    OUT = "moe_out"
    SYNTHETIC = os.environ.get("MOE_SYNTHETIC") == "1"  # offline self-test, no GPU
    os.makedirs(OUT, exist_ok=True)
    return BPW, DELETE_WEIGHTS_AFTER, LLC_MB, MAX_NEW_TOKENS, MODELS, OUT, SYNTHETIC


@app.cell
def _():
    PROMPTS = {
        "chat": [
            "What are some good habits for staying focused while studying?",
            "Explain why the sky is blue to a ten year old.",
            "Write a short, friendly email declining a meeting invitation.",
            "What should I pack for a week-long trip to the mountains?",
            "Summarize the plot of Romeo and Juliet in a few paragraphs.",
            "Give me advice on preparing for a job interview.",
            "What are the pros and cons of living in a big city?",
            "Describe how a bill becomes a law in a parliamentary system.",
            "Suggest a weekly meal plan for a vegetarian on a budget.",
            "How do vaccines train the immune system?",
            "Write a short story about a lighthouse keeper.",
            "What causes inflation, and how do central banks respond?",
        ],
        "code": [
            "Write a Python function that checks whether a string is a palindrome.",
            "Implement binary search in C++ with comments.",
            "Write a SQL query that finds the top 5 customers by total order value.",
            "Explain the difference between a process and a thread, with a code example.",
            "Write a JavaScript debounce function and explain how it works.",
            "Implement an LRU cache class in Python.",
            "Write a Rust function that reverses a linked list.",
            "Write a bash script that finds the 10 largest files in a directory tree.",
            "Implement merge sort in Java.",
            "Write a Python script that parses a CSV file and prints column averages.",
            "Explain what a race condition is and show how a mutex prevents it in Go.",
            "Write a React component for a counter with increment and reset buttons.",
        ],
        "math": [
            "Solve step by step: a train travels 300 km in 4 hours. What is its average speed in m/s?",
            "Prove that the square root of 2 is irrational.",
            "Find the derivative of x^3 * sin(x) and explain each step.",
            "How many ways can 8 people be seated at a round table?",
            "Solve the system: 2x + 3y = 12, x - y = 1.",
            "Explain Bayes' theorem with a medical testing example and compute the numbers.",
            "What is the sum of the first 100 positive integers? Show two methods.",
            "Integrate 1/(x^2 + 1) from 0 to 1 and explain the result.",
            "A bag has 5 red and 3 blue balls. Two are drawn without replacement. P(both red)?",
            "Explain eigenvalues and compute them for the matrix [[2,1],[1,2]].",
            "Show that the sum of angles in a triangle is 180 degrees.",
            "Compute 17^5 mod 13 step by step.",
        ],
    }
    return (PROMPTS,)


@app.cell
def _(np, re):
    SKIP_PARTS = ("mtp", "visual", "vision")  # only the text decoder that runs during generation

    def model_layout(model):
        """Per-token bytes structure from the model's own parameters (text decoder only)."""
        layers = {}
        for name, mod in model.named_modules():
            if any(x in name for x in SKIP_PARTS):
                continue
            m = re.search(r"layers\.(\d+)$", name)
            if m and any(type(c).__name__.endswith("Router") for c in mod.modules()):
                layers[int(m.group(1))] = mod
        assert layers, "no decoder layers with a router found"
        order = sorted(layers)
        E = None
        expert_params, dense_params = [], []
        for i in order:
            layer = layers[i]
            ex = [c for c in layer.modules() if type(c).__name__.endswith("Experts")]
            assert len(ex) == 1, f"layer {i}: expected one Experts module, got {len(ex)}"
            ex_n = sum(p.numel() for p in ex[0].parameters())
            n_exp = getattr(ex[0], "num_experts", None) or next(ex[0].parameters()).shape[0]
            E = E or n_exp
            expert_params.append(ex_n / n_exp)
            dense_params.append(sum(p.numel() for p in layer.parameters()) - ex_n)
        lm = model.get_output_embeddings()
        return dict(layer_ids=order, E=int(E), expert_params=float(np.mean(expert_params)),
                    dense_params_per_layer=[int(x) for x in dense_params],
                    lm_head_params=int(lm.weight.numel()))

    def attach_router_hooks(model):
        """Record top-k expert indices from every *Router module, per forward call."""
        calls = []
        handles = []
        for name, mod in model.named_modules():
            if not type(mod).__name__.endswith("Router") or any(x in name for x in SKIP_PARTS):
                continue
            m = re.search(r"layers\.(\d+)\.", name)
            if not m:
                continue
            li = int(m.group(1))

            def hook(_mod, _inp, out, li=li):
                outs = out if isinstance(out, (tuple, list)) else (out,)
                idx = [t for t in outs if hasattr(t, "dtype") and not t.dtype.is_floating_point and t.dim() == 2]
                assert idx, "router output has no integer top-k index tensor"
                calls.append((li, idx[-1].detach().clone()))  # stays on GPU; converted after generate

            handles.append(mod.register_forward_hook(hook))
        return calls, handles

    def calls_to_steps(calls, layer_ids):
        """Group router calls into decode steps: [T, L, k]. Prefill calls (seq > 1) are skipped."""
        L = len(layer_ids)
        pos = {li: j for j, li in enumerate(layer_ids)}
        steps, cur = [], {}
        for li, idx in calls:
            idx = idx.cpu().numpy().astype(np.int16) if hasattr(idx, "cpu") else idx
            if idx.shape[0] != 1:  # prefill
                cur = {}
                continue
            cur[pos[li]] = idx[0]
            if len(cur) == L:
                steps.append(np.stack([cur[j] for j in range(L)]))
                cur = {}
        return np.stack(steps) if steps else np.zeros((0, L, 0), np.int16)
    return attach_router_hooks, calls_to_steps, model_layout


@app.cell
def _(
    DELETE_WEIGHTS_AFTER,
    MAX_NEW_TOKENS,
    MODELS,
    OUT,
    PROMPTS,
    SYNTHETIC,
    attach_router_hooks,
    calls_to_steps,
    gc,
    json,
    model_layout,
    np,
    os,
    shutil,
    time,
):
    def synthetic_routes(E, L, k, T, rng):
        # Zipf popularity per layer + sticky reuse between consecutive tokens (self-test only)
        pop = 1.0 / np.arange(1, E + 1) ** 0.9
        routes = np.zeros((T, L, k), np.int16)
        for l in range(L):
            p = rng.permutation(pop) / pop.sum()
            prev = rng.choice(E, k, replace=False, p=p)
            for t in range(T):
                keep = prev[rng.random(k) < 0.4]
                rest = [e for e in rng.choice(E, 4 * k, p=p) if e not in keep]
                cur = list(dict.fromkeys(list(keep) + rest))[:k]
                routes[t, l] = cur
                prev = np.array(cur)
        return routes

    def capture(key, repo):
        path = os.path.join(OUT, f"{key}.npz")
        if os.path.exists(path):
            return f"{key}: cached"
        if SYNTHETIC:
            rng = np.random.default_rng(0)
            E, L, k, T = 64, 8, 4, 600
            routes = synthetic_routes(E, L, k, T, rng)
            meta = dict(repo=repo, layer_ids=list(range(L)), E=E, k=k, expert_params=3e6,
                        dense_params_per_layer=[20_000_000] * L, lm_head_params=50_000_000, tok_per_s=0)
            domains = np.repeat(np.arange(3), T // 3)
            prompt_ids = np.arange(T) // 50
            np.savez_compressed(path, routes=routes, domain=domains, prompt=prompt_ids, meta=json.dumps(meta))
            return f"{key}: synthetic"

        import torch
        import transformers as tf

        kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
        if "gpt-oss" in repo:
            kwargs["quantization_config"] = tf.Mxfp4Config(dequantize=True)
        model = None
        try:
            try:
                model = tf.AutoModelForCausalLM.from_pretrained(repo, **kwargs)
            except Exception:
                gc.collect()
                torch.cuda.empty_cache()
                model = tf.AutoModelForImageTextToText.from_pretrained(repo, **kwargs)
            model.eval()
            tok = tf.AutoTokenizer.from_pretrained(repo)
            layout = model_layout(model)
            calls, handles = attach_router_hooks(model)
            assert len(handles) == len(layout["layer_ids"]), \
                f"{len(handles)} router hooks vs {len(layout['layer_ids'])} MoE layers"

            all_routes, domains, prompt_ids = [], [], []
            n_tok, raw_prompts, t0 = 0, 0, time.time()
            for d_i, (domain, prompts) in enumerate(PROMPTS.items()):
                for p_i, prompt in enumerate(prompts):
                    msgs = [{"role": "user", "content": prompt}]
                    try:
                        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                                      return_dict=True, enable_thinking=False)
                    except Exception:
                        enc = tok(prompt, return_tensors="pt")
                        raw_prompts += 1
                    enc = {k: v.to("cuda") for k, v in enc.items() if k in ("input_ids", "attention_mask")}
                    calls.clear()
                    with torch.no_grad():
                        model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                       pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
                    steps = calls_to_steps(calls, layout["layer_ids"])
                    all_routes.append(steps)
                    domains += [d_i] * len(steps)
                    prompt_ids += [d_i * 100 + p_i] * len(steps)
                    n_tok += len(steps)
            for h in handles:
                h.remove()

            routes = np.concatenate(all_routes)
            assert routes.shape[0] > 0 and routes.shape[2] > 0, "no decode steps captured"
            meta = dict(repo=repo, k=int(routes.shape[2]), tok_per_s=n_tok / (time.time() - t0),
                        raw_prompts=raw_prompts, **layout)
            np.savez_compressed(path, routes=routes, domain=np.array(domains, np.int8),
                                prompt=np.array(prompt_ids, np.int16), meta=json.dumps(meta))
        finally:  # always free GPU memory and disk, even if this model failed
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if DELETE_WEIGHTS_AFTER:
                from huggingface_hub.constants import HF_HUB_CACHE
                shutil.rmtree(os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--")), ignore_errors=True)
        note = f", {raw_prompts} prompts without chat template" if raw_prompts else ""
        return f"{key}: {routes.shape[0]} decode tokens, {meta['tok_per_s']:.1f} tok/s{note}"

    capture_log = []
    if not SYNTHETIC:
        import torch as _torch
        import transformers as _tf
        capture_log.append(f"transformers {_tf.__version__}, torch {_torch.__version__}, "
                           f"GPU: {_torch.cuda.get_device_name(0) if _torch.cuda.is_available() else 'NONE'}")
    for _key, _repo in MODELS.items():
        try:
            capture_log.append(capture(_key, _repo))
        except Exception as e:  # keep going; one broken model shouldn't kill the run
            capture_log.append(f"{_key}: FAILED — {type(e).__name__}: {e}")
    print("\n".join(capture_log))
    return (capture_log,)


@app.cell
def _(MODELS, OUT, capture_log, json, np, os):
    _ = capture_log
    DATA = {}
    for _k in MODELS:
        _p = os.path.join(OUT, f"{_k}.npz")
        if os.path.exists(_p):
            _z = np.load(_p)
            DATA[_k] = dict(routes=_z["routes"], domain=_z["domain"], prompt=_z["prompt"],
                            meta=json.loads(str(_z["meta"])))
    list(DATA)
    return (DATA,)


@app.cell
def _(BPW, DATA, np, pd):
    def stats(d):
        r, m = d["routes"], d["meta"]
        T, L, k = r.shape
        E = m["E"]
        top10, top20, reuse = [], [], []
        for l in range(L):
            c = np.sort(np.bincount(r[:, l].ravel(), minlength=E))[::-1]
            top10.append(c[: max(1, E // 10)].sum() / c.sum())
            top20.append(c[: max(1, E // 5)].sum() / c.sum())
            a = r[:-1, l]
            b = r[1:, l]
            reuse.append(np.mean([len(np.intersect1d(a[t], b[t])) / k for t in range(T - 1)]))
        exp_b = m["expert_params"] * BPW / 8
        dense_b = (sum(m["dense_params_per_layer"]) + m["lm_head_params"]) * BPW / 8
        return dict(tokens=T, layers=L, experts=E, top_k=k,
                    expert_MB=exp_b / 2**20,
                    top10pct_share=np.mean(top10), top20pct_share=np.mean(top20),
                    next_token_reuse=np.mean(reuse), random_reuse=k / E,
                    dense_MB_per_token=dense_b / 2**20,
                    expert_MB_per_token=L * k * exp_b / 2**20)

    SKEW = pd.DataFrame({k: stats(v) for k, v in DATA.items()}).T
    SKEW.round(3)
    return (SKEW,)


@app.cell
def _(DATA, PROMPTS, np, pd):
    # Does the skew hold per task? Top-10% share and next-token reuse, per domain.
    _rows = []
    _names = list(PROMPTS)
    for _k, _d in DATA.items():
        _r, _E = _d["routes"], _d["meta"]["E"]
        for _di in np.unique(_d["domain"]):
            _sub = _r[_d["domain"] == _di]
            if len(_sub) < 2:
                continue
            _top = []
            for _l in range(_sub.shape[1]):
                _c = np.sort(np.bincount(_sub[:, _l].ravel(), minlength=_E))[::-1]
                _top.append(_c[: max(1, _E // 10)].sum() / _c.sum())
            _same = (_sub[1:, :, :, None] == _sub[:-1, :, None, :]).any(-1).mean()
            _rows.append(dict(model=_k, domain=_names[int(_di)], tokens=len(_sub),
                              top10pct_share=float(np.mean(_top)), next_token_reuse=float(_same)))
    DOMAIN_SKEW = pd.DataFrame(_rows)
    DOMAIN_SKEW.round(3)
    return (DOMAIN_SKEW,)


@app.cell
def _(mo):
    mo.md("""
    **Reading the table:** `top10pct_share` = share of expert activations going to the hottest 10% of
    experts (0.10 would mean uniform). `next_token_reuse` = fraction of a token's experts reused by the
    next token, vs `random_reuse` if routing were uniform. Big gaps from uniform = non-uniform reuse,
    which is what gives a popularity-aware policy room that scan-resistant policies can't capture.
    """)
    return


@app.cell
def _(OrderedDict, np):
    # Expert-granularity, fully-associative, byte-capacity LLC models.
    # Stream per token: for each layer, its dense weights (attention, router, shared parts), then
    # its routed experts; then the output head. Returns bytes hit / bytes accessed.
    # Assumptions (state in the paper): whole-object granularity (an expert's lines move together),
    # fully associative, decode tokens only (prefill excluded), prompts streamed back to back,
    # cold start counted identically for every policy, KV cache/activations not modeled (tiny here).

    def build_stream(routes, meta, bpw):
        T, L, k = routes.shape
        E = meta["E"]
        exp_b = meta["expert_params"] * bpw / 8
        dense_b = [p * bpw / 8 for p in meta["dense_params_per_layer"]]
        lm_b = meta["lm_head_params"] * bpw / 8
        ids, sizes, is_exp = [], [], []  # is_exp kept for analysis
        LM = L + L * E
        for t in range(T):
            for l in range(L):
                ids.append(l); sizes.append(dense_b[l]); is_exp.append(False)
                for e in routes[t, l]:
                    ids.append(L + l * E + int(e)); sizes.append(exp_b); is_exp.append(True)
            ids.append(LM); sizes.append(lm_b); is_exp.append(False)
        return ids, sizes, np.array(is_exp), T  # plain lists: much faster in the dict-based sims

    def sim_lru(ids, sizes, cap):
        cache, used, hit = OrderedDict(), 0.0, 0.0
        for x, s in zip(ids, sizes):
            if x in cache:
                hit += s
                cache.move_to_end(x)
                continue
            if s > cap:
                continue
            while used + s > cap:
                _, sz = cache.popitem(last=False)
                used -= sz
            cache[x] = s
            used += s
        return hit

    def sim_bip(ids, sizes, cap, eps=32):
        # Bimodal insertion (scan-resistant, popularity-blind): new objects go to the LRU end except
        # 1 in `eps`, which go to MRU. On cyclic streams it keeps a stable subset, like SRRIP did on
        # the dense line-level traces (~95% of the ceiling there).
        cache, used, hit, n = OrderedDict(), 0.0, 0.0, 0
        for x, s in zip(ids, sizes):
            if x in cache:
                hit += s
                cache.move_to_end(x)
                continue
            if s > cap:
                continue
            while used + s > cap:
                _, sz = cache.popitem(last=False)
                used -= sz
            cache[x] = s
            used += s
            n += 1
            if n % eps:
                cache.move_to_end(x, last=False)
        return hit

    def next_uses(ids):
        nxt = np.empty(len(ids), np.int64)
        last = {}
        for i in range(len(ids) - 1, -1, -1):
            nxt[i] = last.get(ids[i], len(ids) + 1)
            last[ids[i]] = i
        return nxt

    def sim_belady(ids, sizes, cap, nxt):
        # farthest-next-use eviction (Belady's MIN; near-optimal with near-uniform sizes)
        import heapq
        cache, heap, used, hit = {}, [], 0.0, 0.0
        for i, (x, s) in enumerate(zip(ids, sizes)):
            if x in cache:
                hit += s
            elif s <= cap and nxt[i] <= len(ids):
                while used + s > cap:
                    while True:
                        nu, y = heapq.heappop(heap)
                        if y in cache and cache[y][0] == -nu:
                            break
                    # don't evict something needed sooner than x
                    if -nu < nxt[i]:
                        heapq.heappush(heap, (nu, y))
                        break
                    used -= cache.pop(y)[1]
                if used + s > cap:
                    continue
                used += s
            else:
                continue
            cache[x] = (int(nxt[i]), s)
            heapq.heappush(heap, (-int(nxt[i]), x))
        return hit

    def sim_popularity_aware(ids, sizes, cap, tokens, half_life=64, refresh=8, pin_frac=1.0):
        # Knows which object each access belongs to (a layer's dense weights, or layer+expert ID: the
        # "hint"). Reserves up to `pin_frac` of the cache for the objects with the highest recent access
        # rate (dense weights are touched every token so they rank first, then the hottest experts);
        # whatever is left is a BIP region for everything else.
        per_tok = len(ids) // tokens
        size_of = dict(zip(ids, sizes))
        growth = 2 ** (1 / half_life)  # decayed counts via a growing increment (same ranking)
        score, target, target_bytes = {}, set(), 0.0
        pinned, pinned_bytes = set(), 0.0
        extra, extra_bytes = OrderedDict(), 0.0
        hit, n_ins = 0.0, 0
        for t in range(tokens):
            if t % refresh == 0 and score:
                target, target_bytes = set(), 0.0
                for x in sorted(score, key=score.get, reverse=True):
                    if target_bytes + size_of[x] <= pin_frac * cap:
                        target.add(x)
                        target_bytes += size_of[x]
                for x in [y for y in pinned if y not in target]:  # demote
                    pinned.discard(x); pinned_bytes -= size_of[x]
                    extra[x] = size_of[x]; extra_bytes += size_of[x]
                for x in [y for y in extra if y in target]:  # promote
                    extra_bytes -= extra.pop(x)
                    pinned.add(x); pinned_bytes += size_of[x]
                while extra and pinned_bytes + extra_bytes > cap:
                    extra_bytes -= extra.popitem(last=False)[1]
            inc = growth ** t
            for x, s in zip(ids[t * per_tok:(t + 1) * per_tok], sizes[t * per_tok:(t + 1) * per_tok]):
                score[x] = score.get(x, 0.0) + inc
                if x in pinned:
                    hit += s
                elif x in extra:
                    hit += s
                    extra.move_to_end(x)
                elif x in target:
                    pinned.add(x); pinned_bytes += s
                    while extra and pinned_bytes + extra_bytes > cap:
                        extra_bytes -= extra.popitem(last=False)[1]
                else:
                    room = cap - target_bytes
                    while extra and extra_bytes + s > room:
                        extra_bytes -= extra.popitem(last=False)[1]
                    if extra_bytes + s <= room:
                        extra[x] = s; extra_bytes += s
                        n_ins += 1
                        if n_ins % 32:  # bimodal insertion, so this region is scan-resistant too
                            extra.move_to_end(x, last=False)
        return hit
    return build_stream, next_uses, sim_belady, sim_bip, sim_lru, sim_popularity_aware


@app.cell
def _(np):
    # The one check: on a pure cyclic stream LRU gets nothing and Belady keeps ~C/F.
    def sim_check(sim_lru, sim_belady, sim_bip, next_uses):
        ids = np.tile(np.arange(100), 30)
        sizes = np.ones(len(ids))
        lru = sim_lru(ids, sizes, 25) / len(ids)
        bel = sim_belady(ids, sizes, 25, next_uses(ids)) / len(ids)
        bip = sim_bip(ids, sizes, 25) / len(ids)
        assert lru == 0.0, lru
        assert 0.2 < bel < 0.26, bel
        assert bip > 0.8 * bel, bip  # scan-resistant baseline ~ ceiling on uniform reuse (dense case)
        return f"check ok: cyclic LRU={lru:.3f}, BIP={bip:.3f}, Belady={bel:.3f} (C/F=0.25)"
    return (sim_check,)


@app.cell
def _(
    BPW,
    DATA,
    LLC_MB,
    build_stream,
    next_uses,
    pd,
    sim_belady,
    sim_bip,
    sim_check,
    sim_lru,
    sim_popularity_aware,
):
    print(sim_check(sim_lru, sim_belady, sim_bip, next_uses))
    rows = []
    for _k, _d in DATA.items():
        _ids, _sizes, _isx, _T = build_stream(_d["routes"], _d["meta"], BPW)
        _nxt = next_uses(_ids)
        _total = sum(_sizes)
        for _mb in LLC_MB:
            _cap = _mb * 2**20
            _row = dict(model=_k, llc_mb=_mb, footprint_MB_per_token=_total / _T / 2**20)
            for _name, _fn in [("lru", lambda: sim_lru(_ids, _sizes, _cap)),
                               ("bip", lambda: sim_bip(_ids, _sizes, _cap)),
                               ("popularity_aware", lambda: sim_popularity_aware(_ids, _sizes, _cap, _T)),
                               ("pop_pin50", lambda: sim_popularity_aware(_ids, _sizes, _cap, _T, pin_frac=0.5)),
                               ("pop_pin75", lambda: sim_popularity_aware(_ids, _sizes, _cap, _T, pin_frac=0.75)),
                               ("belady", lambda: sim_belady(_ids, _sizes, _cap, _nxt))]:
                _row[f"{_name}_hit_pct"] = 100 * _fn() / _total
            rows.append(_row)
    RESULTS = pd.DataFrame(rows)
    RESULTS.round(2)
    return (RESULTS,)


@app.cell
def _(RESULTS, mo):
    mo.md("""
    **Reading the results:** each column is the % of per-token weight bytes served from the LLC.
    `lru` = naive, `bip` = scan-resistant but popularity-blind (stands in for SRRIP, which reached ~95%
    of the ceiling on dense models), `belady` = ceiling. The paper question: how much of the
    `bip` → `belady` gap does `popularity_aware` close? `pop_pin50` / `pop_pin75` are the same policy
    with only 50% / 75% of the cache reserved for popular objects (sensitivity check). Watch the large-LLC
    rows: if `lru` beats every policy there, short-term reuse matters more than popularity at that size.
    """)
    RESULTS.assign(bip_of_ceiling=lambda d: d.bip_hit_pct / d.belady_hit_pct.clip(lower=1e-9),
                   popularity_of_ceiling=lambda d: d.popularity_aware_hit_pct / d.belady_hit_pct.clip(lower=1e-9)
                   )[["model", "llc_mb", "bip_of_ceiling", "popularity_of_ceiling"]].round(3)
    return


@app.cell
def _(DATA, OUT, np, os, plt):
    _fig, _axes = plt.subplots(1, max(1, len(DATA)), figsize=(4 * max(1, len(DATA)), 3.2), squeeze=False)
    for _ax, (_k, _d) in zip(_axes[0], DATA.items()):
        _r, _E = _d["routes"], _d["meta"]["E"]
        for _l in np.linspace(0, _r.shape[1] - 1, 4).astype(int):
            _c = np.sort(np.bincount(_r[:, _l].ravel(), minlength=_E))[::-1]
            _ax.plot(np.arange(1, _E + 1) / _E, np.cumsum(_c) / _c.sum(), label=f"layer {_l}")
        _ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        _ax.set_title(_k, fontsize=9)
        _ax.set_xlabel("fraction of experts (hottest first)")
    _axes[0][0].set_ylabel("share of activations")
    _axes[0][0].legend(fontsize=7)
    _fig.tight_layout()
    FIG_PATH = os.path.join(OUT, "fig_popularity.png")
    _fig.savefig(FIG_PATH, dpi=200)
    _fig
    return (FIG_PATH,)


@app.cell
def _(DOMAIN_SKEW, FIG_PATH, OUT, RESULTS, SKEW, capture_log, mo, os, zipfile):
    _ = FIG_PATH
    SKEW.to_csv(os.path.join(OUT, "skew.csv"))
    DOMAIN_SKEW.to_csv(os.path.join(OUT, "skew_by_domain.csv"), index=False)
    RESULTS.to_csv(os.path.join(OUT, "cache_sim.csv"), index=False)
    with open(os.path.join(OUT, "capture_log.txt"), "w") as _f:
        _f.write("\n".join(capture_log))
    with zipfile.ZipFile("moe_results.zip", "w", zipfile.ZIP_DEFLATED) as _z:
        for _fn in os.listdir(OUT):
            _z.write(os.path.join(OUT, _fn), _fn)
    mo.download(open("moe_results.zip", "rb").read(), filename="moe_results.zip",
                label="Download moe_results.zip")
    return


if __name__ == "__main__":
    app.run()

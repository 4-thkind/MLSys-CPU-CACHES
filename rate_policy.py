import numpy as np, json

def layout(meta, bpw=4.5):
    L, E = len(meta["layer_ids"]), meta["E"]
    n = L + L * E + 1
    size = np.empty(n)
    size[:L] = np.array(meta["dense_params_per_layer"]) * bpw / 8
    size[L:L + L * E] = meta["expert_params"] * bpw / 8
    size[-1] = meta["lm_head_params"] * bpw / 8
    is_dense = np.zeros(n, bool); is_dense[:L] = True; is_dense[-1] = True
    return L, E, size, is_dense

def token_accesses(routes, L, E):
    # object ids touched by each token (all unique within a token)
    lay = np.arange(L)
    ex = L + lay[None, :, None] * E + routes.astype(np.int64)          # [T, L, k]
    T = routes.shape[0]
    dense = np.broadcast_to(np.r_[lay, L + L * E], (T, L + 1))
    return np.concatenate([dense, ex.reshape(T, -1)], axis=1)          # [T, L+1+L*k]

def sim_rate_aware(acc, size, cap, half_life, class_prior=None):
    """Keep the objects with the highest estimated access rate per token (EMA with the given
    half-life, in tokens), filling the cache by bytes. Rate = expected hits per byte cached per token,
    so ranking by rate maximizes expected byte hits. Causal: token t uses estimates from tokens < t."""
    n = len(size)
    decay = 2.0 ** (-1.0 / half_life)
    rate = np.zeros(n) if class_prior is None else class_prior.copy()
    resident = np.zeros(n, bool)
    hit = 0.0
    for t in range(acc.shape[0]):
        order = np.argsort(-rate, kind="stable")
        target = np.zeros(n, bool)
        room, cand = cap, order
        while len(cand):  # greedy fill by rank, skipping objects too big for the room left
            cand = cand[size[cand] <= room]
            take = cand[np.cumsum(size[cand]) <= room]
            if not len(take):
                break
            target[take] = True
            room -= size[take].sum()
            cand = cand[len(take):]
        resident &= target
        a = acc[t]
        hit += size[a][resident[a]].sum()
        resident[a] |= target[a]
        rate *= decay
        rate[a] += 1 - decay
    return hit

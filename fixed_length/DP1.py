# Self-contained DP for expected BFS node counts with capped applicable-edge states.
# Numerically stable Skellam pmf via log-space; ΔO = Skellam((2k-1)λ,(2k-2)λ) − Z, Z~Poisson^+(λ).
# States clipped to [0, Omax], Omax = R*l; O=0 is absorbing (no branching).
#
# Outputs:
#   E.csv / E.npy  : (Omax+1, T+1) expected node counts (rows O=0..Omax, cols t=0..T)
#   N.csv          : expected total nodes per depth (1 × (T+1))
#   b.csv          : branching ratio N[t+1]/N[t] (1 × T)
#   delta_pmf.csv  : two-column [d, p(Δ=d)]
#   params.json    : parameters used

import math
from collections import defaultdict
from typing import Dict, Tuple, List
import numpy as np
import mpmath as mp
import json

# ----------------------- helpers -----------------------

def poisson_pmf(mu: float, n: int) -> float:
    if n < 0:
        return 0.0
    return math.exp(-mu) * (mu**n) / math.factorial(n)

def ztp_pmf(lambda_: float, eps: float = 1e-12) -> Dict[int, float]:
    """Zero-truncated Poisson pmf for z >= 1, truncated where remaining tail < eps."""
    denom = 1.0 - math.exp(-lambda_)
    pmf: Dict[int, float] = {}
    z, tail = 1, 1.0
    while True:
        p = poisson_pmf(lambda_, z) / denom
        pmf[z] = p
        tail -= p
        if tail < eps:
            break
        z += 1
        if z > 200000:  # hard safety
            break
    s = sum(pmf.values())
    if s > 0:
        for k in pmf:
            pmf[k] /= s
    return pmf

def _logsumexp(logw: List[float]) -> float:
    m = max(logw)
    if not np.isfinite(m):
        return -np.inf
    return m + math.log(sum(math.exp(x - m) for x in logw))

def skellam_pmf_range_stable(mu1: float, mu2: float, eps: float = 1e-14) -> Tuple[np.ndarray, np.ndarray]:
    """Numerically stable Skellam pmf on an adaptive integer range [nmin, nmax].
       Uses log-space with Bessel I in log.
    """
    # Degenerate edge cases
    if mu1 <= 0 and mu2 <= 0:
        return np.array([0], dtype=int), np.array([1.0], dtype=float)
    if mu1 <= 0:  # N = -Poisson(mu2)
        nmax = int(max(10, math.ceil(mu2 + 12*math.sqrt(mu2 + 1))))
        ns = np.arange(-nmax, 1, dtype=int)
        pmf = np.array([poisson_pmf(mu2, -n) if n <= 0 else 0.0 for n in ns], dtype=float)
        pmf /= pmf.sum()
        return ns, pmf
    if mu2 <= 0:  # N = +Poisson(mu1)
        nmax = int(max(10, math.ceil(mu1 + 12*math.sqrt(mu1 + 1))))
        ns = np.arange(0, nmax+1, dtype=int)
        pmf = np.array([poisson_pmf(mu1, n) for n in ns], dtype=float)
        pmf /= pmf.sum()
        return ns, pmf

    mean = mu1 - mu2
    var  = mu1 + mu2
    sigma = math.sqrt(max(var, 1e-16))
    K = 12.0
    nmin = int(math.floor(mean - K*sigma - 8))
    nmax = int(math.ceil (mean + K*sigma + 8))
    if nmax <= nmin:
        nmin -= 20; nmax += 20

    ns = np.arange(nmin, nmax+1, dtype=int)

    # log pmf: log P(N=n) = -(mu1+mu2) + (n/2)log(mu1/mu2) + log I_{|n|}(2 sqrt(mu1 mu2))
    # compute in log-space then normalize with log-sum-exp
    z = 2.0 * math.sqrt(mu1 * mu2)
    log_ratio = 0.5 * math.log(mu1 / mu2)
    base = -(mu1 + mu2)

    logp: List[float] = []
    for n in ns:
        # log I_{|n|}(z)
        lnI = float(mp.log(mp.besseli(abs(int(n)), z)))
        logp.append(base + n*log_ratio + lnI)

    # trim tails where CDF mass < eps
    # first normalize in log-space
    lse = _logsumexp(logp)
    logp = [lp - lse for lp in logp]
    p = np.array([math.exp(lp) for lp in logp], dtype=float)

    # cumulative trim
    cdf = np.cumsum(p)
    left = int(np.searchsorted(cdf, eps))
    right = int(np.searchsorted(cdf, 1 - eps))
    left = max(left, 0)
    right = min(right, len(ns) - 1)
    ns2 = ns[left:right+1]
    p2 = p[left:right+1].copy()
    p2 /= p2.sum()
    return ns2, p2

def build_delta_pmf(k: int, R: int, Sigma: int, eps: float = 1e-12) -> Dict[int, float]:
    """ΔO pmf dict: Δ = X - Z,  X~Skellam((2k-1)λ,(2k-2)λ),  Z~ZTP(λ), λ=R/Σ^k."""
    lam = R / (Sigma ** k)
    mu1 = (2 * k - 1) * lam
    mu2 = (2 * k - 2) * lam
    ns, pX = skellam_pmf_range_stable(mu1, mu2, eps=eps*1e-2)
    pZ = ztp_pmf(lam, eps=eps)

    delta = defaultdict(float)
    for z, pz in pZ.items():
        # Δ = x - z
        for x, px in zip(ns, pX):
            d = int(x - z)
            delta[d] += pz * px
    print(delta)

    total = sum(delta.values())
    if total <= 0:
        return {-1: 1.0}
    for d in list(delta.keys()):
        delta[d] /= total
        if delta[d] < eps:
            del delta[d]
    # exact renormalize
    s = sum(delta.values())
    if s != 1.0 and s > 0:
        mode_d = max(delta.items(), key=lambda kv: kv[1])[0]
        delta[mode_d] += (1.0 - s)
    return dict(sorted(delta.items()))

# ----------------------- initial & DP -----------------------

def initial_capped_poisson(R: int, l: int, Sigma: int, k: int, Omax: int, eps: float = 1e-15) -> np.ndarray:
    """Poisson(mu0) capped at Omax by aggregating tail; returns vector length Omax+1."""
    mu0 = R * l / (Sigma ** k)
    p = np.zeros(Omax + 1, dtype=float)
    tail = 1.0
    n = 0
    while n < Omax:
        pn = poisson_pmf(mu0, n)
        p[n] = pn
        tail -= pn
        if tail < eps:
            break
        n += 1
        if n > 200000:
            break
    p[Omax] = max(0.0, 1.0 - p[:Omax].sum())
    s = p.sum()
    if s > 0:
        p /= s
    return p

def propagate_expected_nodes(Sigma: int, k: int, R: int, l: int, T: int,
                             eps: float = 1e-12, store_E: bool = True) -> dict:
    """Matrix-free DP: v_{t+1} = v_t * W * P (row-vector view)."""
    Omax = R * l
    delta = build_delta_pmf(k, R, Sigma, eps=eps)
    d_vals = np.array(sorted(delta.keys()), dtype=int)
    d_probs = np.array([delta[d] for d in d_vals], dtype=float)

    v = initial_capped_poisson(R, l, Sigma, k, Omax).astype(float)
    E = np.zeros((Omax + 1, T + 1), dtype=float) if store_E else None
    if E is not None:
        E[:, 0] = v.copy()

    N = np.zeros(T + 1, dtype=float)
    N[0] = v.sum()
    b = np.zeros(T, dtype=float)
    v_next = np.zeros_like(v)

    for t in range(T):
        v_next.fill(0.0)
        nz = np.nonzero(v)[0]
        for O in nz:
            if O == 0:
                continue  # absorbing
            weight = v[O] * O
            if weight == 0.0:
                continue
            Oprime = np.clip(O + d_vals, 0, Omax)
            for idx, Op in enumerate(Oprime):
                v_next[Op] += weight * d_probs[idx]
        N[t+1] = v_next.sum()
        b[t] = (N[t+1] / N[t]) if N[t] > 0 else 0.0
        if E is not None:
            E[:, t+1] = v_next
        v, v_next = v_next, v

    return {
        "E": E,
        "N": N,
        "b": b,
        "delta_pmf": (d_vals, d_probs),
        "params": {"Sigma": Sigma, "k": k, "R": R, "l": l, "T": T, "Omax": Omax},
    }

# ----------------------- main -----------------------

if __name__ == "__main__":
    # --- PRODUCTION PARAMS ---
    Sigma = 10
    k = 3
    R = 200
    l = 20     # cyclic windows for length L=20; if non-cyclic, use l = max(L-k+1, 0)
    T = 12     # change as needed

    res = propagate_expected_nodes(Sigma, k, R, l, T, eps=1e-12, store_E=True)

    E = res["E"]; N = res["N"]; b = res["b"]
    d_vals, d_probs = res["delta_pmf"]; params = res["params"]

    # save
    np.savetxt("E.csv", E, delimiter=",")
    np.save("E.npy", E)
    np.savetxt("N.csv", N[None, :], delimiter=",")
    np.savetxt("b.csv", b[None, :], delimiter=",")
    np.savetxt("delta_pmf.csv", np.vstack([d_vals, d_probs]).T, delimiter=",")
    with open("params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print("Saved: E.csv, E.npy, N.csv, b.csv, delta_pmf.csv, params.json")
    print("Shapes:", {"E": E.shape, "N": N.shape, "b": b.shape})

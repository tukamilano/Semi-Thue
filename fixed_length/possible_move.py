# Distribution-level analysis for one-step changes using Poisson approximations.
# Cyclic-string version (windows wrap around). ΔO only. Full, self-contained cell.

import random, math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Parameters
A = 10
L = 20
k = 3
R = 200

lam0 = R / (A**k)               # expected #rules per window
r = 2*k - 1                     # affected windows per rewrite (for L >> k)

alphabet = tuple(chr(i + 65) for i in range(A))

def gen_rules_map(A, k, R):
    rules_map = defaultdict(list)
    seen = set()
    while len(seen) < R:
        b = ''.join(random.choices(alphabet, k=k))
        a = ''.join(random.choices(alphabet, k=k))
        if b != a and (b,a) not in seen:
            seen.add((b,a))
            rules_map[b].append(a)
    return rules_map

# --- Cyclic helpers ---
def cyclic_window(s, i, k):
    """Return the length-k substring of s starting at i with wrap-around."""
    i %= len(s)
    if i + k <= len(s):
        return s[i:i+k]
    t = (i + k) % len(s)
    return s[i:] + s[:t]

def replace_at_cyclic(s, i, a):
    """Replace length-k block starting at i (cyclic) with string a (len k)."""
    i %= len(s)
    s_list = list(s)
    for t in range(len(a)):
        s_list[(i + t) % len(s)] = a[t]
    return ''.join(s_list)

# --- Graph helpers adapted to cyclic windows ---
def edge_count(s, rules_map):
    total = 0
    for i in range(L):  # L distinct cyclic windows
        lst = rules_map.get(cyclic_window(s, i, k))
        if lst:
            total += len(lst)
    return total

def step_random_window(s, rules_map):
    positions = [i for i in range(L) if cyclic_window(s, i, k) in rules_map]
    if not positions:
        return s, False
    i = random.choice(positions)
    b = cyclic_window(s, i, k)
    a = random.choice(rules_map[b])
    s2 = replace_at_cyclic(s, i, a)
    return s2, True

def sample_delta_O(num_batches=30, trials_per_batch=200):
    dO = []
    for _ in range(num_batches):
        rules_map = gen_rules_map(A, k, R)
        for _ in range(trials_per_batch):
            s = ''.join(random.choices(alphabet, k=L))
            O0 = edge_count(s, rules_map)
            s2, ok = step_random_window(s, rules_map)
            if not ok:
                continue
            O1 = edge_count(s2, rules_map)
            dO.append(O1 - O0)
    return np.array(dO)

# --- Theory helpers ---
def skellam_pmf_vec(ns, mu1, mu2):
    out = []
    for n in ns:
        if mu1 == 0 and mu2 == 0:
            out.append(1.0 if n == 0 else 0.0); continue
        if mu2 == 0:
            # reduces to Poisson(mu1)
            val = math.exp(-mu1) * (mu1**n) / math.factorial(n) if n>=0 else 0.0
        elif mu1 == 0:
            # negative Poisson
            val = math.exp(-mu2) * (mu2**(-n)) / math.factorial(-n) if n<=0 else 0.0
        else:
            val = mp.e**(-(mu1+mu2)) * ( (mu1/mu2)**(n/2) ) * mp.besseli(abs(n), 2*mp.sqrt(mu1*mu2))
        out.append(float(val))
    return np.array(out)

def poi_trunc_pmf_vec(ks, lam):
    Z = 1 - math.exp(-lam)
    return np.array([ (math.exp(-lam)*lam**k/(math.factorial(k)*Z)) if k>=1 else 0.0 for k in ks ])

# --- Empirical samples (ΔO only, cyclic) ---
dO_emp = sample_delta_O(num_batches=30, trials_per_batch=200)

print(f"Samples: ΔO={len(dO_emp)}")
print(f"Empirical ΔO (cyclic): mean={dO_emp.mean():.3f}, var={dO_emp.var():.3f}, min={dO_emp.min()}, max={dO_emp.max()}")

# --- ΔO theory (Skellam - zero-truncated Poisson) ---
# Same parameters as linear model under the local-independence approximation.
mu1_O = r * lam0
mu2_O = (r - 1) * lam0

no_min, no_max = int(dO_emp.min()-8), int(dO_emp.max()+8)
ns_O = list(range(no_min, no_max+1))

pmf_S = skellam_pmf_vec(ns_O, mu1_O, mu2_O); pmf_S = pmf_S / pmf_S.sum()
Z_vals = list(range(1, 12))  # truncate Z at 11 (captures > 1 - 1e-8 for lam0≈0.2)
pmf_Z = poi_trunc_pmf_vec(Z_vals, lam0)

# Convolution for DO = S - Z  =>  P(DO=n) = sum_z P(Z=z) P(S=n+z)
pmf_DO = np.zeros(len(ns_O))
for z, pz in zip(Z_vals, pmf_Z):
    shifted = np.zeros(len(ns_O))
    # index shift by z since ns_O is a consecutive integer grid
    shifted[:len(ns_O)-z] = pmf_S[z:]
    pmf_DO += pz * shifted
pmf_DO = pmf_DO / pmf_DO.sum()

# Plot
plt.figure()
plt.hist(dO_emp, bins=range(no_min, no_max+2), density=True, alpha=0.6, label="empirical (cyclic)")
plt.plot(ns_O, pmf_DO, marker='o', label="Skellam − Poi⁺ theory")
plt.title("ΔO distribution (cyclic): empirical vs [Skellam((2k−1)λ₀,(2k−2)λ₀) − Poi⁺(λ₀)]")
plt.xlabel("ΔO"); plt.ylabel("probability")
plt.legend()
plt.show()

# Theory moments
muZ = lam0 / (1 - math.exp(-lam0))
varZ = muZ * (1 + lam0 - muZ)
mean_DO_the = (mu1_O - mu2_O) - muZ
var_DO_the = (mu1_O + mu2_O) + varZ

print(f"Poisson params: λ0={lam0:.3f}, affected windows r=2k−1={r}")
print(f"ΔO theory (Skellam − Z): mean={mean_DO_the:.3f}, var≈{var_DO_the:.3f} (Z is zero-truncated Poisson)")

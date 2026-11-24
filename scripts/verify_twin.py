#!/usr/bin/env python3
# Copyright (c) 2026 Niobium Microsystems, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent numpy recomputation of the diabetes faithful twin.

The first rung of the verification ladder, and the one that certifies the
artifact every later rung is graded against. scripts/design/faithful_twin.py
produces twin_logits.csv but needs xgboost + scikit-learn + pandas; this rebuilds
the same circuit from mlp_surrogate.json with numpy alone and checks three things
the encrypted runs then depend on:

  1. model.txt -- what the C++ actually loads -- agrees with the JSON artifact,
     allowing for OpenFHE's Chebyshev convention (see below);
  2. the committed twin_logits.csv is what this circuit computes;
  3. every pre-activation lands inside the per-unit provable domain, which is
     what makes decode safety airtight under batched packing.

  scripts/verify_twin.py <repo-root>
"""
import json, sys
import numpy as np
from numpy.polynomial import chebyshev as C

D = sys.argv[1]
S = json.load(open(f"{D}/model/surrogate/mlp_surrogate.json"))
W1 = np.array(S["W1"]); b1 = np.array(S["b1"])
W2 = np.array(S["W2"]); b2 = float(S["b2"])
lo = np.array(S["dom_lo"]); hi = np.array(S["dom_hi"])
co = np.array(S["cheb_coefs"])
TH = float(S["threshold_logit"])
DELTA = 2.0 ** 45
q = lambda x: np.round(x * DELTA) / DELTA

# ---- 1. model.txt must agree with the json (the C++ reads model.txt) ----
tok = open(f"{D}/data/inputs/model.txt").read().split()
i = 0
F, H, deg = int(tok[0]), int(tok[1]), int(tok[2]); i = 3
mW1 = np.array(tok[i:i+F*H], float).reshape(F, H); i += F*H
mb1 = np.array(tok[i:i+H], float); i += H
mW2 = np.array(tok[i:i+H], float); i += H
mb2 = float(tok[i]); i += 1
mlo, mhi, mco = np.zeros(H), np.zeros(H), np.zeros((H, deg+1))
for k in range(H):
    mlo[k] = float(tok[i]); mhi[k] = float(tok[i+1]); i += 2
    mco[k] = np.array(tok[i:i+deg+1], float); i += deg+1
assert i == len(tok), f"model.txt has {len(tok)-i} trailing tokens"
# model.txt is float32-rounded text; compare at that precision
def close(a, b, tag, tol=1e-6):
    d = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    print(f"  model.txt vs json  {tag:12s} max|diff| = {d:.3e}")
    assert d < tol, f"{tag} mismatch"
print(f"model.txt: F={F} H={H} deg={deg}")
close(mW1, W1, "W1"); close(mb1, b1, "b1"); close(mW2, W2, "W2")
close([mb2], [b2], "b2"); close(mlo, lo, "dom_lo"); close(mhi, hi, "dom_hi")
# OpenFHE evaluates c[0]/2 + sum_{k>=1} c[k] T_k(x) (ckksrns-advancedshe.cpp:727),
# numpy's chebval evaluates sum_k c[k] T_k(x). model.txt therefore stores 2*c0.
oco = co.copy(); oco[:, 0] *= 2.0
close(mco, oco, "cheb_coefs (c0 doubled for the OpenFHE convention)")

# ---- 2. recompute the twin ----
X = np.loadtxt(f"{D}/data/inputs/test_features_std.csv", delimiter=",")
print(f"features: {X.shape}")
pre = q(q(X @ W1) + b1)                 # layer 1 (1 level)
h = np.empty_like(pre)
for k in range(pre.shape[1]):           # per-unit Chebyshev on its own domain
    h[:, k] = C.chebval((2*pre[:, k] - (hi[k]+lo[k])) / (hi[k]-lo[k]), co[k])
h = q(h)
logit = q(q(h @ W2) + b2)               # layer 2 (1 level)

twin = np.loadtxt(f"{D}/data/inputs/twin_logits.csv")
labels = np.loadtxt(f"{D}/data/inputs/test_labels.csv").astype(int)
err = np.abs(logit - twin)
flips = int(np.sum((logit >= TH) != (twin >= TH)))
print("\nindependent twin vs committed twin_logits.csv:")
print(f"  n={len(twin)}  max|err|={err.max():.3e}  mean|err|={err.mean():.3e}  decision flips={flips}")

# ---- 3. pre-activation envelope must sit inside the declared domains ----
inside = np.all((pre >= lo - 1e-9) & (pre <= hi + 1e-9))
print(f"  pre-activations inside per-unit domains: {inside}  "
      f"(|pre|max={np.abs(pre).max():.3f}, widest domain={np.max(np.abs(np.r_[lo,hi])):.3f})")

# ---- 4. task metrics of the twin ----
pred = (twin >= TH).astype(int)
tp = int(((pred==1)&(labels==1)).sum()); fp = int(((pred==1)&(labels==0)).sum())
fn = int(((pred==0)&(labels==1)).sum())
prec = tp/(tp+fp); rec = tp/(tp+fn); f1 = 2*prec*rec/(prec+rec)
print(f"  twin task metrics: precision={prec:.4f} recall={rec:.4f} F1={f1:.4f}")
print(f"  decision margins: min|logit-TH|={np.abs(twin-TH).min():.3e}, "
      f"#within 1e-3 of TH = {int((np.abs(twin-TH)<1e-3).sum())}")

ok = err.max() < 1e-6 and flips == 0 and inside
print("\nTWIN VERIFY: " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)

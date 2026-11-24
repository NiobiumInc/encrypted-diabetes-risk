#!/usr/bin/env python3
# Copyright (c) 2026 Niobium Microsystems, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
cleartext_impl.py: the diabetes-risk model in the clear.

The same model the encrypted server evaluates, with no encryption, no CKKS and
no Niobium. Read this first to see what the server computes; then run
harness/run_submission.py and watch the same logits come out of ciphertexts
the server cannot read.

The model is a small MLP, 21 -> 16 -> 1:

  1. encode   the client packs feature j of 32,768 records into ciphertext j
              (column-major), so every step below runs on all records at once
              and the circuit needs no rotations
  2. layer 1  pre_k = sum_j x_j * W1[j][k] + b1[k]         16 hidden units
  3. act      h_k = sigmoid(pre_k), approximated per unit by a degree-27
              Chebyshev series on that unit's own input domain [lo_k, hi_k]
  4. layer 2  logit = sum_k h_k * W2[k] + b2                one output
  5. decide   the client decrypts the logit; diabetes risk iff logit >= TH

Every weight, bias and coefficient is read from data/inputs/model.txt,
the file the C++ server loads, so this and the encrypted run cannot drift
apart. Inputs are the standardized features in data/inputs/.

    python3 harness/cleartext_impl.py                 # walk through record 0
    python3 harness/cleartext_impl.py --record 7      # another record
    python3 harness/cleartext_impl.py --n 1024 --out logits.csv

Uses only the Python standard library.
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IO = ROOT / "data" / "inputs"


def load_model(path=IO / "model.txt"):
    tok = path.read_text().split()
    F, H, deg = int(tok[0]), int(tok[1]), int(tok[2])
    it = iter(float(t) for t in tok[3:])
    W1 = [[next(it) for _ in range(H)] for _ in range(F)]
    b1 = [next(it) for _ in range(H)]
    W2 = [next(it) for _ in range(H)]
    b2 = next(it)
    lo, hi, cheb = [], [], []
    for _ in range(H):
        lo.append(next(it)); hi.append(next(it))
        cheb.append([next(it) for _ in range(deg + 1)])
    if next(it, None) is not None:
        raise ValueError(f"trailing values in {path}")
    return dict(F=F, H=H, deg=deg, W1=W1, b1=b1, W2=W2, b2=b2, lo=lo, hi=hi, cheb=cheb)


def load_threshold(path=IO / "client_meta.txt"):
    # client_meta: F, then F lower and F upper feature bounds, then the logit threshold.
    return float(path.read_text().split()[-1])


def chebyshev(c, x, lo, hi):
    """c[0]/2 + sum_{k>=1} c[k] T_k(u), with u = x mapped from [lo, hi] onto
    [-1, 1]. Halving c[0] is OpenFHE's convention for EvalChebyshevSeries, so
    model.txt stores the doubled constant term. Clenshaw's recurrence."""
    u = (2.0 * x - (hi + lo)) / (hi - lo)
    b1 = b2 = 0.0
    for ck in reversed(c[1:]):
        b1, b2 = 2.0 * u * b1 - b2 + ck, b1
    return u * b1 - b2 + c[0] / 2.0


def forward(m, x, trace=None):
    """The encrypted circuit, one record at a time. With `trace`, record each
    intermediate for the walk-through."""
    pre = [sum(x[j] * m["W1"][j][k] for j in range(m["F"])) + m["b1"][k]
           for k in range(m["H"])]
    h = [chebyshev(m["cheb"][k], pre[k], m["lo"][k], m["hi"][k]) for k in range(m["H"])]
    logit = sum(h[k] * m["W2"][k] for k in range(m["H"])) + m["b2"]
    if trace is not None:
        trace.update(pre=pre, h=h)
    return logit


def load_features(n, path=IO / "test_features_std.csv"):
    rows = []
    with open(path) as f:
        for line in f:
            if len(rows) >= n:
                break
            if line.strip():
                rows.append([float(v) for v in line.split(",")])
    return rows


def walk_through(m, x, th, index):
    t = {}
    logit = forward(m, x, t)
    print(f"record {index} of the held-out test set")
    print(f"\n1. encode: {m['F']} standardized features; the client packs each one "
          f"into its own ciphertext")
    print("   x = [" + ", ".join(f"{v:.3f}" for v in x) + "]")
    print(f"\n2. layer 1: {m['H']} pre-activations, pre_k = x . W1[:, k] + b1[k]"
          f"   (ciphertext x plaintext multiply-adds)")
    for k, p in enumerate(t["pre"]):
        print(f"   unit {k:2d}: pre = {p:+.6f}   domain [{m['lo'][k]:+.3f}, {m['hi'][k]:+.3f}]")
    print(f"\n3. activation: degree-{m['deg']} Chebyshev sigmoid per unit "
          f"(EvalChebyshevSeries)")
    for k, (p, v) in enumerate(zip(t["pre"], t["h"])):
        print(f"   unit {k:2d}: h = {v:.6f}   (exact sigmoid {1 / (1 + math.exp(-p)):.6f})")
    print(f"\n4. layer 2: logit = h . W2 + b2 = {logit:+.9f}")
    risk = logit >= th
    print(f"\n5. decide: logit {'>=' if risk else '<'} threshold {th:+.6f} "
          f"(P = {1 / (1 + math.exp(-logit)):.4f} vs P = {1 / (1 + math.exp(-th)):.2f}) -> "
          f"{'DIABETES RISK' if risk else 'no diabetes risk'}")


def main():
    p = argparse.ArgumentParser(description="The diabetes-risk model, in the clear.")
    p.add_argument("--record", type=int, default=0, help="record to walk through (default 0)")
    p.add_argument("--n", type=int, help="score the first N records instead")
    p.add_argument("--out", help="with --n: write one logit per line here")
    a = p.parse_args()
    m, th = load_model(), load_threshold()
    if a.n is None:
        rows = load_features(a.record + 1)
        if len(rows) <= a.record:
            sys.exit(f"no record {a.record}")
        walk_through(m, rows[a.record], th, a.record)
        return 0
    rows = load_features(a.n)
    logits = [forward(m, x) for x in rows]
    if a.out:
        with open(a.out, "w") as f:
            f.writelines(f"{v:.17g}\n" for v in logits)
    pos = sum(v >= th for v in logits)
    print(f"cleartext: {len(logits)} records scored, {pos} at or above the threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

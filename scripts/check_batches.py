#!/usr/bin/env python3
# Copyright (c) 2026 Niobium Microsystems, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grade a decrypted run against the cleartext reference, per slot-batch.

The comparison is split by the app's own slot-batches, because two distinct
failures both show up as a bad aggregate number and they need different fixes:

  * a slot-batch that disagrees with the reference   -> arithmetic / depth
  * a slot-batch that scores slot-batch 0's
    records instead of its own                       -> the replay was fed
                                                        slot-batch 0's inputs

A cached trace replayed against the wrong inputs produces the second, so it
is checked by name: the slot-batch's scores are compared with the reference
for slot-batch 0's records as well as its own.

With --labels it also prints the run's precision, recall and F1 on the
diabetes class. Those are reported, not graded.

  check_batches.py <results.csv> <reference.csv> <manifest.txt> [--tol 1e-3]
                   [--labels test_labels.csv [--subsample]]
"""
import argparse
import csv
import math
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("reference", help="one cleartext logit per line")
    ap.add_argument("manifest")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="tolerance on |encrypted - reference| (default 1e-3)")
    ap.add_argument("--labels", help="test_labels.csv, to print task metrics")
    ap.add_argument("--subsample", action="store_true",
                    help="the records are a slice of the test set; label the metrics so")
    a = ap.parse_args()

    TH = math.log(0.64 / 0.36)

    with open(a.results) as f:
        rd = csv.reader(f)
        next(rd)
        enc = [float(r[0]) for r in rd]
    ref = [float(x) for x in open(a.reference).read().split()]

    tok = open(a.manifest).read().split()
    n_batches = int(tok[1])
    sizes = [int(t) for t in tok[2:2 + n_batches]]

    total = sum(sizes)
    if len(enc) != total:
        print(f"FAIL: results.csv has {len(enc)} rows, manifest declares {total}")
        return 1
    if len(ref) < total:
        print(f"FAIL: reference has {len(ref)} rows, need {total}")
        return 1

    print(f"application slot-batches: {n_batches}  sizes: {sizes}")
    print(f"{'batch':>5} {'records':>8} {'max|err|':>11} {'mean|err|':>11} {'flips':>6}")

    off = 0
    worst = 0.0
    flips_total = 0
    for b, n in enumerate(sizes):
        e = enc[off:off + n]
        t = ref[off:off + n]
        errs = [abs(p - q) for p, q in zip(e, t)]
        flips = sum((p >= TH) != (q >= TH) for p, q in zip(e, t))
        mx, mn = max(errs), sum(errs) / len(errs)
        worst = max(worst, mx)
        flips_total += flips
        print(f"{b:>5} {n:>8} {mx:>11.3e} {mn:>11.3e} {flips:>6}")
        off += n

    rc = 0
    if worst >= a.tol:
        print(f"FAIL: max|err| {worst:.3e} exceeds the tolerance {a.tol:.0e}")
        rc = 1
    if flips_total:
        print(f"FAIL: {flips_total} decision flip(s) vs the reference")
        rc = 1

    # Did a later slot-batch score slot-batch 0's records instead of its own?
    for b in range(1, n_batches):
        start = sum(sizes[:b])
        k = min(sizes[b], sizes[0])
        e = enc[start:start + k]
        own = sum(1 for i in range(k) if abs(e[i] - ref[start + i]) < a.tol)
        first = sum(1 for i in range(k) if abs(e[i] - ref[i]) < a.tol)
        if first > k / 2 and first > own:
            print(f"FAIL: slot-batch {b} scored slot-batch 0's records ({first}/{k} "
                  f"match their reference, {own} match its own) -- the replay read "
                  f"slot-batch 0's inputs; this is not an arithmetic error")
            rc = 1
        else:
            print(f"slot-batch {b} scored its own records ({own}/{k} within "
                  f"tolerance; {first} also match slot-batch 0's)")

    if a.labels:
        labels = [int(x) for x in open(a.labels).read().split()][:total]
        tp = sum(1 for e, y in zip(enc, labels) if e >= TH and y == 1)
        fp = sum(1 for e, y in zip(enc, labels) if e >= TH and y == 0)
        fn = sum(1 for e, y in zip(enc, labels) if e < TH and y == 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if a.subsample:
            # Fewer than the full 45,895 records give a noisy estimate of
            # model quality, so the numbers are labelled as such.
            print(f"task metrics on this {total}-record SUBSAMPLE of the 45,895-record "
                  f"held-out test set: precision={prec:.4f} recall={rec:.4f} F1={f1:.4f}")
            print("  (a noisy estimate -- size 1 scores the complete set)")
        else:
            print(f"encrypted-run task metrics: precision={prec:.4f} "
                  f"recall={rec:.4f} F1={f1:.4f}")

    print("BATCH CHECK: " + ("PASS" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    sys.exit(main())

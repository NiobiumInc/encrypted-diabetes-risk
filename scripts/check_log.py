#!/usr/bin/env python3
# Copyright (c) 2026 Niobium Microsystems, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check that a run's own log agrees with the configuration it was asked for.

A green exit code only says the pipeline finished. This reads the log back and
asserts what can silently differ from the request: the tier, the target, and
that every application slot-batch was evaluated exactly once. It also fails if
decryption reported that the approximation error is too high.

  check_log.py <logfile> [--target NAME] [--tier NAME] [--app-batches N]
"""
import argparse
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--target")
    ap.add_argument("--tier")
    ap.add_argument("--app-batches", type=int)
    a = ap.parse_args()

    text = open(a.log, errors="replace").read()
    fails, notes = [], []

    banners = [dict(re.findall(r"(\w+)=(\S+)", line))
               for line in text.splitlines() if line.startswith("[compute] ring=")]
    if not banners:
        fails.append("no '[compute] ring=' banner in the log -- the stage never started")
    for kv in banners:
        if a.tier and kv.get("tier") != a.tier:
            fails.append(f"banner tier={kv.get('tier')} but {a.tier} was asked for")
        if a.target and kv.get("target") != a.target:
            fails.append(f"banner target={kv.get('target')} but {a.target} was asked for")
        if a.app_batches is not None and kv.get("app_slot_batches") != str(a.app_batches):
            fails.append(f"banner app_slot_batches={kv.get('app_slot_batches')} "
                         f"but {a.app_batches} expected")
    if a.app_batches is not None and banners:
        seen = sorted(int(kv.get("batch", -1)) for kv in banners)
        if seen != list(range(a.app_batches)):
            fails.append(f"slot-batches evaluated {seen}, expected each of "
                         f"0..{a.app_batches - 1} exactly once")
        else:
            notes.append(f"slot-batches evaluated: {seen}")
    done = len(re.findall(r"^\[compute\] slot-batch \d+ .*replay", text, flags=re.M))
    notes.append(f"replays completed: {done}")
    if a.app_batches is not None and done != a.app_batches:
        fails.append(f"{done} replay(s) completed, expected {a.app_batches}")

    if "approximation error is too high" in text:
        fails.append("Decode() reported the approximation error is too high")

    for n in notes:
        print(f"  {n}")
    for f in fails:
        print(f"  FAIL: {f}")
    print("LOG CHECK: " + ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

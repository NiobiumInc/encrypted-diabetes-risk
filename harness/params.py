# Copyright (c) 2026 Niobium Microsystems, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
params.py — instance sizes for the encrypted diabetes-risk harness.

Size selects how many records of the 45,895-record held-out test set are
scored, always from its head. The crypto parameters are the same for every
size (ring 2^16, depth 11, CKKS FLEXIBLEAUTO). A slot-batch holds 32,768
records, so toy is one slot-batch and small is two.
"""

# ── Instance size enum ──────────────────────────────────────────────────
TOY = 0
SMALL = 1

_NAMES = ["toy", "small"]
_RECORDS = [1024, 45895]


def instance_name(size: int) -> str:
    """toy or small. Used for work directories and the recorded trace's key."""
    return _NAMES[size]


def instance_records(size: int) -> int:
    """How many test records the size scores."""
    return _RECORDS[size]

#!/usr/bin/env python3
# Copyright (c) 2026 Niobium Microsystems, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
run_submission.py: run the encrypted diabetes-risk workload end to end.

    keygen -> encrypt -> compute -> decrypt -> grade

The client stages (keygen, encrypt, decrypt) always run locally. --target picks
where the server's compute stage runs:

    local   the Niobium SDK, replayed in the bundled simulator. No account.
    FOG     the Niobium SDK, replayed on an FPGA on the Fog. Each compute stage
            runs inside its own Fog job (needs `fog login` once).

Any other target is forwarded to the transport server named by
NBCC_FHETCH_SERVER.

    python3 harness/run_submission.py 0                 # toy, on the simulator
    python3 harness/run_submission.py 0 --target FOG    # toy, on the Fog
    python3 harness/run_submission.py 1 --target FOG    # small, on the Fog

A run passes only if every stage exits 0 and the decrypted logits reproduce the
model computed in the clear (harness/cleartext_impl.py), recomputed for these
records on every run: max|err| < 1e-3 and zero decision flips.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleartext_impl  # noqa: E402
from params import TOY, SMALL, instance_name, instance_records  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
IO = ROOT / "data" / "inputs"
CLIENT = ROOT / "niobium-client"
FOG = CLIENT / "scripts" / "fog"
SIM = CLIENT / "build" / "vendor" / "niobium-fhetch" / "fhetch_sim"

STAGES = ("keygen", "encrypt", "decrypt", "diabetes_compute_sdk")


class StageError(RuntimeError):
    pass


def shown(arg):
    """An argument as printed: paths inside the repo relative to it."""
    a = str(arg)
    if a == sys.executable:
        return "python3"
    try:
        return str(Path(a).relative_to(ROOT)) if Path(a).is_absolute() else a
    except ValueError:
        return a


def run_stage(name, cmd, log, cwd=None, env=None):
    """Run one stage, streaming its output to the terminal and the run log.
    A non-zero exit ends the run: nothing downstream may read what it left."""
    print(f"[harness] $ {' '.join(shown(c) for c in cmd)}", flush=True)
    log.write(f"\n===== {name}: {' '.join(map(str, cmd))}\n")
    t0 = time.time()
    proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
        out.append(line)
    rc = proc.wait()
    log.flush()
    wall = time.time() - t0
    if rc != 0:
        raise StageError(f"{name} exited {rc}")
    return "".join(out), wall


def sdk_env(target):
    env = os.environ.copy()
    fwd = CLIENT / "build" / "src" / "fhetch_transport" / "nbcc_fhetch_replay"
    if target == "local":
        env["NBCC_FHETCH_SIM"] = str(SIM)
    elif fwd.exists():
        env.setdefault("NBCC_FHETCH_REPLAY_BIN", str(fwd))
        if env.get("NBCC_FHETCH_SERVER"):
            env.setdefault("NBCC_FHETCH_REPLAY", str(fwd))
    return env


def main() -> int:
    p = argparse.ArgumentParser(description="Run the encrypted diabetes-risk workload.")
    p.add_argument("size", type=int, choices=range(TOY, SMALL + 1),
                   help="instance size: 0=toy (first 1,024 test records), "
                        "1=small (all 45,895)")
    p.add_argument("--target", default="local",
                   help="local (default) or FOG; see the module docstring")
    p.add_argument("-O", "--optimization", type=int, choices=[0, 1, 2, 3], default=3,
                   help="backend opt level for SDK targets (default 3)")
    p.add_argument("--fresh", action="store_true",
                   help="delete this configuration's keys and recorded trace first")
    p.add_argument("--skip-build", action="store_true",
                   help="do not run scripts/build_task.sh when binaries are missing")
    args = p.parse_args()
    tier = instance_name(args.size)
    n = instance_records(args.size)

    missing = [s for s in STAGES if not (BUILD / s).exists()]
    if missing and not args.skip_build:
        print("[harness] building (scripts/build_task.sh) ...", flush=True)
        if subprocess.run(["bash", str(ROOT / "scripts" / "build_task.sh")]).returncode:
            p.error("scripts/build_task.sh failed; see its output above")
        missing = [s for s in STAGES if not (BUILD / s).exists()]
    if missing:
        p.error(f"missing from {BUILD}: {', '.join(missing)}; run scripts/build_task.sh")

    if args.target not in ("local", "FOG") and not os.environ.get("NBCC_FHETCH_SERVER"):
        p.error(f"--target {args.target}: use local or FOG; any other target needs "
                f"NBCC_FHETCH_SERVER pointing at your own replay server")

    # Check every prerequisite before any stage runs, so a missing piece
    # stops the run here rather than after keygen and encrypt.
    if args.target == "local" and not SIM.exists():
        p.error(f"--target local needs the simulator at {shown(SIM)}; run scripts/build_task.sh")

    fog = False
    if args.target != "local" and not os.environ.get("NBCC_FHETCH_SERVER"):
        if not FOG.exists():
            p.error(f"--target {args.target} needs the Fog CLI at {FOG} "
                    f"(git submodule update --init niobium-client) or NBCC_FHETCH_SERVER")
        fog = True

    # One directory per configuration, holding its keys and this run's recorded
    # trace, so a run at one tier, target or opt level never touches another's.
    cfg = f"{args.target}_O{args.optimization}"
    work = ROOT / "work" / tier / cfg
    if args.fresh and work.exists():
        shutil.rmtree(work)
    ch, sh = work / "client_home", work / "server_home"
    for d in (ch / "ct_in", ch / "ct_out", sh / "ct_in", sh / "ct_out"):
        d.mkdir(parents=True, exist_ok=True)
    # Nothing from an earlier run may survive into this one's answer.
    for d in (ch / "ct_in", ch / "ct_out", sh / "ct_in", sh / "ct_out"):
        for f in d.iterdir():
            f.unlink()
    (ch / "results.csv").unlink(missing_ok=True)
    # The trace carries the relinearization key it was recorded with, so each
    # run records it afresh rather than replay one made with an earlier key.
    for d in sh.iterdir():
        if d.is_dir() and d.name.startswith("diabetes_compute"):
            shutil.rmtree(d)

    logpath = work / "run.log"
    metrics = {"tier": tier, "records": n, "target": args.target,
               "opt": args.optimization, "stages": {}}
    verdict = "FAIL"
    with open(logpath, "w") as log:
        try:
            # ---- client: keys (kept across runs of this configuration) ----
            if not (ch / "context.bin").exists():
                _, w = run_stage("keygen", [BUILD / "keygen", ch], log)
                metrics["stages"]["keygen_s"] = round(w, 2)
            shutil.copy(IO / "client_meta.txt", ch)
            feats = ch / "test_features_std.csv"
            with open(IO / "test_features_std.csv") as src, open(feats, "w") as dst:
                for i, line in enumerate(src):
                    if i >= n:
                        break
                    dst.write(line)

            # ---- client: encrypt ----
            out, w = run_stage("encrypt", [BUILD / "encrypt", ch, feats, ch / "ct_in", n], log)
            metrics["stages"]["encrypt_s"] = round(w, 2)
            m = re.search(r"(\d+) records admitted, (\d+) rejected", out)
            if not m or int(m.group(1)) != n or int(m.group(2)) != 0:
                # A rejected record shifts every later row against the twin.
                raise StageError(f"encrypt admitted {m.group(1) if m else '?'} of {n} records")
            manifest = (ch / "ct_in" / "manifest.txt").read_text().split()
            n_batches = int(manifest[1])

            # ---- provision the server: context, relin key, model, ciphertexts ----
            for f in ("context.bin", "key_relin.bin"):
                shutil.copy(ch / f, sh)
            shutil.copy(IO / "model.txt", sh)
            for f in (ch / "ct_in").iterdir():
                shutil.copy(f, sh / "ct_in")
            if (sh / "key_secret.bin").exists():
                raise StageError("secret key present in the server home")

            # The server must refuse a home that holds the secret key. Plant it,
            # run the stage directly (no Fog job: it has to stop before any
            # replay), expect exit 2, then remove it.
            shutil.copy(ch / "key_secret.bin", sh)
            try:
                guard = [BUILD / "diabetes_compute_sdk", sh, sh / "ct_in", sh / "ct_out",
                         "--batch", 0, "--tier", tier, f"--target={args.target}"]
                g = subprocess.run([str(c) for c in guard], capture_output=True, text=True)
                log.write(f"\n===== secret-key guard: exit {g.returncode}\n{g.stdout}{g.stderr}")
            finally:
                (sh / "key_secret.bin").unlink()
            if g.returncode != 2 or "SERVER ABORT" not in g.stderr:
                raise StageError(f"server ran with the secret key in its home (exit {g.returncode})")
            print("[harness] secret-key guard: the server refused a home holding the secret key",
                  flush=True)

            # ---- server: compute ----
            t_compute = time.time()
            env = sdk_env(args.target)
            per_batch = []
            for b in range(n_batches):
                cmd = [BUILD / "diabetes_compute_sdk", sh, sh / "ct_in", sh / "ct_out",
                       "--batch", b, "--tier", tier, f"--target={args.target}",
                       "--opt-level", f"O{args.optimization}"]
                if fog:
                    cmd = [sys.executable, FOG, "submit", *cmd]
                # cwd is the server home, so the recorded trace lands in this
                # configuration's directory.
                _, w = run_stage(f"compute[b{b}]", cmd, log, cwd=sh, env=env)
                per_batch.append(round(w, 2))
            metrics["stages"]["compute_s"] = per_batch

            for b in range(n_batches):
                ct = sh / "ct_out" / f"ct_logit_b{b}.bin"
                if not ct.exists() or ct.stat().st_mtime < t_compute:
                    raise StageError(f"compute did not write {ct.name} in this run")
                shutil.copy(ct, ch / "ct_out")

            # ---- client: decrypt ----
            _, w = run_stage("decrypt", [BUILD / "decrypt", ch, ch / "ct_out", ch / "results.csv"], log)
            metrics["stages"]["decrypt_s"] = round(w, 2)

            # ---- grade ----
            # The reference is recomputed in the clear from model.txt for exactly
            # these records, every run. The committed twin_logits.csv must agree
            # with it; if it does not, the file has drifted from the model.
            model = cleartext_impl.load_model()
            clear = [cleartext_impl.forward(model, x)
                     for x in cleartext_impl.load_features(n)]
            ref = work / "cleartext_logits.csv"
            ref.write_text("".join(f"{v:.17g}\n" for v in clear))
            committed = [float(v) for v in (IO / "twin_logits.csv").read_text().split()[:n]]
            drift = max(abs(a - b) for a, b in zip(clear, committed))
            msg = f"cleartext reference vs committed twin: max|diff| = {drift:.3e}"
            print(f"[harness] {msg}", flush=True)
            log.write(msg + "\n")
            metrics["cleartext_vs_twin"] = drift
            if len(committed) != n or drift > 1e-6:
                raise StageError("data/inputs/twin_logits.csv disagrees with "
                                 "model.txt; see scripts/verify_twin.py")

            py = sys.executable
            check = [py, ROOT / "scripts" / "check_batches.py", ch / "results.csv", ref,
                     ch / "ct_in" / "manifest.txt", "--labels", IO / "test_labels.csv"]
            if args.size < SMALL:
                check.append("--subsample")
            run_stage("check_batches", check, log)
            with open(ch / "results.csv") as f:
                next(f)
                enc = [float(line.split(",")[0]) for line in f]
            th = cleartext_impl.load_threshold()
            metrics["max_err"] = max(abs(a - b) for a, b in zip(enc, clear))
            metrics["flips"] = sum((a >= th) != (b >= th) for a, b in zip(enc, clear))
            log.flush()
            run_stage("check_log", [py, ROOT / "scripts" / "check_log.py", logpath,
                                    "--tier", tier, "--target", args.target,
                                    "--app-batches", n_batches], log)
            verdict = "PASS"
        except StageError as e:
            metrics["error"] = str(e)
            print(f"[harness] ERROR: {e}", flush=True)

    metrics["verdict"] = verdict
    (work / "measurements.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("\n[harness] ===== summary =====")
    print(f"  size      {args.size} = {tier} ({n} records)   target {args.target}"
          + f"   opt O{args.optimization}")
    if metrics.get("max_err") is not None:
        print(f"  accuracy  max|err| {metrics['max_err']:.2e} vs the cleartext model, "
              f"{metrics['flips']} decision flips")
    if metrics.get("error"):
        print(f"  error     {metrics['error']}")
    print(f"  verdict   {verdict}   (log: {logpath.relative_to(ROOT)})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

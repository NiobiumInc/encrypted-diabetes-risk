# Encrypted Diabetes Risk Inference

**NOT A MEDICALLY APPROVED DIAGNOSTIC TOOL**

Score a patient's diabetes risk without the scoring party ever seeing the
patient's answers.

A client encrypts health-survey records from the CDC's BRFSS 2015 survey and
sends them to a server. The server evaluates a small neural network on the
ciphertext and returns encrypted risk scores. Only the client can decrypt them.
The encryption is CKKS, in OpenFHE.

The server holds the model and the evaluation keys, but never the secret key.
It refuses to start if one is in its directory. The compute stage runs on a
Niobium FPGA over the Fog, so no local accelerator is needed.

The patients' answers and their scores stay confidential throughout, while the
model is disclosed to whoever evaluates the circuit.
[Trust boundary](#trust-boundary) sets out who sees what.

Every run is graded against the same model computed in the clear, record by
record. A run passes only if every decrypted score is within 1e-3 of the
cleartext score and no record's decision flips.

## Run it

Prerequisites: a C++17 compiler, CMake 3.16 or newer, Python 3, and a few GB
of disk for the keys and traces a run generates. A run peaks at about 1.3 GB
of memory. The harness uses only the Python standard library.
`requirements.txt` lists what the plaintext XGBoost model and the design
scripts need; the encrypted workload needs none of it.

Fog access is gated. Request an account at
[console.niobium.co/request-account](https://console.niobium.co/request-account).

```bash
# one-time setup: builds OpenFHE, libnbfhetch and the four stages
git submodule update --init niobium-client
./scripts/build_task.sh

# authenticate once, then score the first 1,024 test records on a real FPGA
# over the Fog. The harness takes one Fog job per slot-batch.
niobium-client/scripts/fog login -u you@example.com
python3 harness/run_submission.py 0 --target FOG

# the same records in software, as a check on the hardware result
python3 harness/run_submission.py 0

# the whole 45,895-record held-out test set on the Fog: two slot-batches,
# two Fog jobs
python3 harness/run_submission.py 1 --target FOG
```

`python3 harness/cleartext_impl.py` runs the same forward pass in the clear, on
a fresh clone with nothing built. It prints each stage for one test record (the
16 hidden pre-activations, the Chebyshev sigmoid for each unit, the logit and
the decision), annotating each step with the encrypted operation that performs
it. The harness recomputes this reference on every run and grades against it.

The size argument selects how many held-out test records are scored: `0` (toy)
takes the first 1,024, one slot-batch, and `1` (small) takes all 45,895, two
slot-batches. A CKKS ciphertext here holds 32,768 records, one per slot. Size
`1` is the graded run: it scores the complete held-out set. `--target` defaults
to `local`, the simulator.

Each configuration gets its own directory, such as `work/toy/FOG_O3/`, with
the keys, the ciphertexts, the recorded trace, `run.log` and
`measurements.json`. Pass `--fresh` to start one over.

## What happened

```console
$ python3 harness/run_submission.py 0 --target FOG
keygen: N=65536 depth=11 scale=45
encrypt: 1024 records admitted, 0 rejected by feature box, 1 slot-batch(es)
[fog] assigned 7ce5c7da-… -> https://…/jobs/7ce5c7da-…/run
[compute] ring=65536 slots=32768 features=21 units=16 cheb_deg=27 tier=toy target=FOG …
[compute] recording trace...
[compute] replaying slot-batch 0 on target FOG...
[fog] upload complete — replaying on target FOG, waiting for the server
decrypt: 1024 results -> work/toy/FOG_O3/client_home/results.csv
batch  records    max|err|   mean|err|  flips
    0     1024   7.167e-08   1.249e-08      0
BATCH CHECK: PASS
LOG CHECK: PASS

[harness] ===== summary =====
  size      0 = toy (1024 records)   target FOG   opt O3
  accuracy  max|err| 7.17e-08 vs the cleartext model, 0 decision flips
  verdict   PASS   (log: work/toy/FOG_O3/run.log)
```

Those lines are a real run, with paths shortened. A full run prints far more:
the harness echoes each stage's command, and the SDK and the transport both log
heavily.

Four programs ran in order, on two sides of the client/server boundary. The
two sides share only ciphertexts and public material.

| Stage | Program | Side | Produces |
|---|---|---|---|
| 1 | `keygen` | client | crypto context, public key, relinearization key, secret key |
| 2 | `encrypt` | client | 21 ciphertexts per slot-batch, one per feature |
| 3 | `diabetes_compute_sdk` | server | one encrypted logit ciphertext per slot-batch |
| 4 | `decrypt` | client | `results.csv`, one logit and decision per record |

Each stage takes `--help`, which describes it and names the arguments the
harness passes, so a stage can be re-run by hand.

Stage 3 is the one that uses Niobium. It records the circuit once, then replays
it on the backend `--target` chooses; every slot-batch runs that identical
circuit, so one recording serves them all and only the ciphertexts change. Every
run checks the decrypted scores against the same model computed in the clear;
see [Checking the answer](#checking-the-answer).

## More information

- [docs/DESIGN.md](docs/DESIGN.md): the model and its distilled surrogate,
  the circuit, the parameters with their rationale, the threat model, and how
  the model and the encrypted run score on the test set.
- [NIOBIUM_INTEGRATION.md](NIOBIUM_INTEGRATION.md): the record and replay flow, and
  what the backend receives.
- [data/README.md](data/README.md): dataset provenance and licence.
- [Checking the answer](#checking-the-answer): how every run is graded.
- [Trust boundary](#trust-boundary): who sees the patients' data, and who sees
  the model.
- [Security and parameters](#security-and-parameters): the CKKS configuration.
- [Backends](#backends): where the compute stage runs.
- [Repository layout](#repository-layout): where the code lives.

### Checking the answer

After decryption the harness recomputes the model in the clear for the same
records (`harness/cleartext_impl.py`) and grades the decrypted scores against
that reference, per slot-batch, with `scripts/check_batches.py`. A run passes
only if every score is within 1e-3 of the reference and no record's decision
flips.

The same check prints the run's precision, recall and F1 against the labels.
The harness also checks that the committed plaintext twin
(`data/inputs/twin_logits.csv`) still agrees with the model, and checks the
run log against the request with `scripts/check_log.py`. A stage that exits
non-zero ends the run as a failure, and no later stage reads what it left.

The `FOG` and `local` targets both agree with the cleartext model to about
1e-07, four orders of magnitude inside the 1e-3 tolerance.

### Trust boundary

The patients' answers and their risk scores stay confidential throughout. The
secret key never leaves the client, and the server, or any backend it uses,
handles only ciphertext for the inputs and the result.

The model is the exposed part. Its weights, biases and Chebyshev coefficients
enter the circuit as plaintext constants and travel in the recorded trace, so
whatever machine evaluates the circuit can read them. Running the compute stage
on the Fog therefore discloses the model to the backend's operator.

That matches the threat model in [docs/DESIGN.md](docs/DESIGN.md#5-security),
which protects the patients' data and treats the model as a plaintext operand
rather than a protected asset, and lists what each party learns. In this demo
the model is public anyway: `data/inputs/model.txt` holds it, and
`harness/cleartext_impl.py` reads it to build its reference.

### Security and parameters

| Parameter | Value |
|---|---|
| Scheme | CKKS, `FLEXIBLEAUTO` rescaling |
| Ring dimension | 2^16, so 32,768 slots: one record per slot |
| Multiplicative depth | 11 |
| Scaling modulus / first modulus | 45 bits / 60 bits |
| Security | `HEStd_128_classic` (128-bit classical) |
| Keys | public and relinearization; no rotation keys |
| Bootstrapping | none |

The circuit is 21 features → 16 hidden units, each with its own degree-27
Chebyshev approximation of the sigmoid, → 1 logit. Records are packed
column-major, one feature per ciphertext, so the circuit needs no rotations,
no rotation keys and no bootstrapping. It uses 8 levels: one for the hidden
layer's linear map, six for the Chebyshev series, one for the output layer. The
depth of 11 adds one level for `FLEXIBLEAUTO` rescaling and two spare. OpenFHE
sizes the modulus chain so `logQ` stays inside the
[Homomorphic Encryption Standard](https://homomorphicencryption.org) bound for
`N = 2^16`, giving at least 128 bits of classical security.

The OpenFHE `CCParams`, from
[src/common.hpp](src/common.hpp):

```cpp
CCParams<CryptoContextCKKSRNS> p;
p.SetRingDim(RING_DIM);                  // 65536 = 2^16
p.SetMultiplicativeDepth(MULT_DEPTH);    // 11
p.SetScalingModSize(SCALE_BITS);         // 45
p.SetFirstModSize(FIRST_BITS);           // 60
p.SetScalingTechnique(FLEXIBLEAUTO);
p.SetSecurityLevel(HEStd_128_classic);
p.SetBatchSize(SLOTS);                   // 32768
```

CKKS is approximate arithmetic, so a decrypted result carries approximation
error; that is what the 1e-3 tolerance absorbs.
[docs/DESIGN.md](docs/DESIGN.md#4-crypto-parameters-and-depth-budget) has the
full rationale and the decode-safety check.

### Backends

| `--target` | Where the compute stage runs |
|---|---|
| `local` (default) | the Niobium SDK, replayed through the simulator the client bundles. Use it to check a hardware result. |
| `FOG` | a real Niobium FPGA, as a job on the Fog. |

`FOG` is a stable alias: the Fog resolves it to its current device, so a client
names the target and leaves the device to the Fog.
[NIOBIUM_INTEGRATION.md](NIOBIUM_INTEGRATION.md) describes the record and replay flow.

The `niobium-client` submodule supplies OpenFHE as well as the transport, and its
pin records the commit the app is built and validated against. Recording a
computation relies on hooks carried by that OpenFHE, which is why every stage is
built against it.

### The model

The encrypted circuit is an MLP surrogate distilled from an XGBoost classifier.
Both ship here.

#### XGBoost classifier (plaintext)

```bash
pip install -r requirements.txt
python3 load_model.py
```

`load_model.py` loads the model, scores the held-out test set, and prints
example predictions. To score your own records:

```python
import json, numpy as np, pandas as pd, xgboost as xgb

feature_cols = json.load(open("model/feature_cols.json"))    # the 21 features, in order
model = xgb.XGBClassifier(); model.load_model("model/xgb_diabetes_model.json")

df = pd.read_parquet("data/cdc_diabetes_test.parquet")       # or your own frame with the same columns
proba = model.predict_proba(df[feature_cols].to_numpy(np.float32))[:, 1]   # P(diabetes)
pred  = (proba >= 0.64).astype(int)                          # recommended operating threshold
```

**Inputs:** the 21 features listed in `data_dictionary.md`, in the order given
by `model/feature_cols.json`. All are numeric or ordinal. The tree model reads
the raw survey codes, so no scaling, encoding or imputation is needed.
**Output:** `predict_proba(...)[:, 1]` is the diabetes probability; threshold
it to get a label.

Bounded-depth gradient-boosted trees:

```
n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.85,
colsample_bytree=0.85, min_child_weight=5.0, reg_lambda=2.0,
objective=binary:logistic, scale_pos_weight≈5.54
```

`model/xgb_diabetes_model.json` is XGBoost's portable native format.

#### Encrypted surrogate

The FHE circuit is a 21 → 16 → 1 MLP distilled from the XGBoost model and
checked against a plaintext "faithful twin" of itself. How it was designed,
distilled and validated is in [docs/DESIGN.md](docs/DESIGN.md), and §7 there
shows how the XGBoost model, the surrogate and the encrypted run each score on
the held-out test set. The scripts that produced the surrogate are in
`scripts/design/`, and the surrogate itself is in `model/surrogate/`.

### Dataset

The records are the CDC's *Diabetes Health Indicators* (BRFSS 2015) survey: 229,474
de-identified records after removing exact duplicates, 21 features, and the label
`Diabetes_binary` (1 for diabetes or prediabetes, 0 for neither; 15.3% positive).
`data_dictionary.md` describes every feature.

The data ships in `data/` as a fixed, stratified 80/20 train/test split (183,579
and 45,895 records, seed 42) plus the combined corpus. The demo encrypts the
standardized test split in `data/inputs/`: size `0` (toy) takes its first
1,024 records and size `1` (small) all 45,895.

Full provenance and the licence are in [`data/README.md`](data/README.md).

### Repository layout

```
.
├── harness/
│   ├── run_submission.py       # the entry point
│   ├── cleartext_impl.py       # the plaintext reference
│   └── params.py               # the two sizes and their record counts
├── src/
│   ├── keygen.cpp              # stage 1
│   ├── encrypt.cpp             # stage 2
│   ├── diabetes_compute_sdk.cpp  # stage 3, recorded and replayed on a Niobium backend
│   ├── decrypt.cpp             # stage 4
│   └── common.hpp              # CKKS parameters, the model loader, serialization
├── scripts/
│   ├── build_task.sh           # builds the client, then the stages
│   ├── check_batches.py        # per-slot-batch grading and task metrics
│   ├── check_log.py            # the run log checked against the request
│   ├── verify_twin.py          # recomputes the committed twin with numpy
│   └── design/                 # how the surrogate was distilled and fitted
├── docs/DESIGN.md              # the model, the circuit, the parameters, the threat model and the results
├── model/                      # the XGBoost model; the surrogate in model/surrogate/
├── data/                       # train, test and combined splits; the encrypted app's inputs in data/inputs/
├── load_model.py               # loads the XGBoost model and scores the test set
├── data_dictionary.md          # every feature, described
├── NIOBIUM_INTEGRATION.md      # the record and replay flow, and what the backend receives
├── requirements.txt            # packages for the plaintext model and design scripts
├── CMakeLists.txt              # the build, driven by build_task.sh
├── LICENSE                     # Apache-2.0
├── NOTICE, CONTRIBUTING.md, SECURITY.md
└── niobium-client/             # submodule: OpenFHE, libnbfhetch, transport, Fog CLI
```

`build/` and `work/` are created at runtime: the built stages, and per
configuration the keys, ciphertexts, recorded trace and run log.

## Acknowledgements

Built with [hardshell.ai](https://hardshell.ai), which provided the original
XGBoost model, the prepared dataset and the model-loading code. The data is the
CDC's BRFSS 2015 survey, distributed by the UCI Machine Learning Repository, and
the FHE library is [OpenFHE](https://github.com/openfheorg/openfhe-development).

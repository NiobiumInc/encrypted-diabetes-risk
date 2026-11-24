# Data

The data is the **CDC Diabetes Health Indicators** dataset: responses from the
U.S. CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey. It
reached this project from hardshell.ai, together with the model trained on it.
It is the dataset distributed by the UCI Machine Learning Repository as dataset
891, which is the dataset Alex Teboul curated on Kaggle.

- UCI: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
  (DOI [10.24432/C53919](https://doi.org/10.24432/C53919))
- Kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
  (file `diabetes_binary_health_indicators_BRFSS2015.csv`)

## Licence

BRFSS data is a work of the U.S. federal government and is in the public
domain. CDC asks that users acknowledge it as the source. The UCI entry states
no licence of its own and defers to the Kaggle dataset, which is published
under CC0 1.0, a public domain dedication. The data is redistributed here
unchanged, apart from the steps below.

## Files

| File | Rows × columns |
|---|---|
| `cdc_diabetes_prepped.parquet` | 229,474 × 22 |
| `cdc_diabetes_train.parquet` | 183,579 × 22 |
| `cdc_diabetes_test.parquet` | 45,895 × 22 |

Each file has the 21 features in `../model/feature_cols.json` plus the label
`Diabetes_binary`: 1 for diabetes or prediabetes, 0 for neither. All columns
are integers, and `../data_dictionary.md` describes each one. The positive rate
is 15.3% in all three files.

## How they were made

Starting from UCI 891's `data.csv` (253,680 rows, an `ID` column plus the 22
columns above):

1. **Drop the `ID` column.**
2. **Drop exact duplicate rows**, keeping the first occurrence. This removes
   24,206 rows and leaves the 229,474 in `cdc_diabetes_prepped.parquet`, in
   their original order.
3. **Split 80/20, stratified on the label**, with scikit-learn:
   `train_test_split(prepped, test_size=0.2, stratify=prepped["Diabetes_binary"], random_state=42)`.
   This gives `cdc_diabetes_train.parquet` and `cdc_diabetes_test.parquet`,
   with the index reset.

We checked these steps against the source on 2026-09-28. The source `data.csv`
had SHA-256
`9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964`. Steps 1–3
reproduce all three files exactly, row for row. Without stratification, step 3
gives a different split.

## Derived inputs for the encrypted app

The encrypted pipeline reads its inputs from [`inputs/`](inputs/). Each file
there is computed from the test split and the trained models:

| File | Contents | Made by |
|---|---|---|
| `test_features_std.csv` | the 45,895 test records, standardized with the training split's mean and standard deviation | `scripts/design/faithful_twin.py` |
| `test_labels.csv` | their labels | `scripts/design/faithful_twin.py` |
| `twin_logits.csv` | the twin's logit for each record: the encrypted circuit, Chebyshev series included, computed in the clear | `scripts/design/faithful_twin.py` |
| `model.txt` | the surrogate's weights and per-unit Chebyshev coefficients | `scripts/design/export_model_txt.py` |
| `client_meta.txt` | the standardized feature bounds and the decision threshold | `scripts/design/export_model_txt.py` |

To regenerate them, from the repository root:

```bash
pip install -r requirements.txt
python3 scripts/design/finalize_surrogate.py .   # trains the surrogate, fits the polynomials
python3 scripts/design/faithful_twin.py .        # twin validation and the test inputs
python3 scripts/design/export_model_txt.py .     # model.txt and client_meta.txt
```

The harness recomputes the twin logits from `model.txt` with
`harness/cleartext_impl.py` on every run, and the run fails if the committed
file disagrees.

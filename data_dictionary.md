# Data Dictionary — CDC Diabetes Health Indicators (BRFSS 2015)

229,474 records (de-duplicated) × 22 columns (21 features + label). No missing values.

**Source:** CDC Diabetes Health Indicators (BRFSS 2015), via the UCI ML Repository (dataset 891) and Kaggle (Alex Teboul, *Diabetes Health Indicators Dataset*). Underlying BRFSS data is U.S. public-domain government work. Full attribution, the licence and links are in [`data/README.md`](data/README.md).

## Target
| Column | Meaning |
|---|---|
| `Diabetes_binary` | 1 = diabetes or prediabetes, 0 = neither |

## Binary health flags (0 = no, 1 = yes)
| Column | Meaning |
|---|---|
| `HighBP` | high blood pressure |
| `HighChol` | high cholesterol |
| `CholCheck` | cholesterol check within 5 years |
| `Smoker` | smoked ≥100 cigarettes in lifetime |
| `Stroke` | ever told they had a stroke |
| `HeartDiseaseorAttack` | coronary heart disease or myocardial infarction |
| `PhysActivity` | physical activity in past 30 days |
| `Fruits` | eats fruit ≥1×/day |
| `Veggies` | eats vegetables ≥1×/day |
| `HvyAlcoholConsump` | heavy alcohol consumption |
| `AnyHealthcare` | has any health coverage |
| `NoDocbcCost` | skipped a doctor visit due to cost (past year) |
| `DiffWalk` | serious difficulty walking or climbing stairs |
| `Sex` | 0 = female, 1 = male |

## Ordinal (coded scales)
| Column | Scale |
|---|---|
| `GenHlth` | self-rated general health: 1 excellent → 5 poor |
| `Age` | 13 brackets: 1 = 18–24, …, 13 = 80+ |
| `Education` | 1 = none → 6 = college graduate |
| `Income` | 1 = <$10k → 8 = >$75k |

## Numeric
| Column | Range |
|---|---|
| `BMI` | body-mass index, 12–98 |
| `MentHlth` | days of poor mental health in past 30 (0–30) |
| `PhysHlth` | days of poor physical health in past 30 (0–30) |

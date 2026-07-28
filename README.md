# Heart Disease Classification

A small, end-to-end machine-learning project that predicts whether a patient has heart
disease from anonymised clinical measurements. Built as a practice run to rehearse the full
workflow — exploratory analysis, modelling, hyperparameter tuning, and evaluation — and a
git/GitHub feature-branch workflow.

> **Status:** complete proof of concept. The focus is the *workflow and project
> organisation*, not squeezing out maximum accuracy.

## Problem

Binary classification: given 13 clinical features (age, resting blood pressure, cholesterol,
max heart rate, etc.), predict `target` — `1` = has heart disease, `0` = does not.

## Data

- **Source:** UCI Cleveland heart disease dataset (303 rows, 13 features).
- `data/raw/heart-disease.csv` is treated as **immutable** — it is never edited in place.
- Every column is described in [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Results

The three baseline models were tuned; **Logistic Regression** won and was tuned further with
`GridSearchCV`.

| Stage | Model | Accuracy |
|-------|-------|----------|
| Baseline | K-Nearest Neighbors | 0.69 |
| Baseline | **Logistic Regression** | **0.89** |
| Baseline | Random Forest | 0.84 |
| Tuned (`GridSearchCV`) | Logistic Regression | 0.89 (hold-out) / 0.85 (5-fold CV) |

Cross-validated metrics for the final model — note the strong **recall**, the metric that
matters most in a medical context (a false negative means missing a sick patient):

| Metric | Score |
|--------|-------|
| Accuracy | 0.85 |
| Precision | 0.82 |
| Recall | **0.93** |
| F1 | 0.87 |

The 95% accuracy target set at the start was **not** reached — ~0.85 CV accuracy is close to
the ceiling widely reported for this small dataset. The honest next step is *more/better
data*, not more model types. See the conclusion in the modelling notebook for detail.

## Project structure

```
Heart_Disease/
├── data/
│   └── raw/heart-disease.csv              # immutable source data
├── docs/
│   └── data_dictionary.md                 # feature descriptions
├── notebooks/
│   ├── 01-heart-disease-eda.ipynb         # exploratory data analysis
│   └── 02-heart-disease-modeling.ipynb    # baselines → tuning → evaluation → feature importance
├── src/
│   └── models/
│       └── evaluate.py                    # reusable fit_and_score() helper, imported by the notebook
├── requirements.txt
└── README.md
```

The split mirrors a common data-science convention: **notebooks are for thinking**
(exploration, experiments, throwaway charts), while the **`src/` package holds the decisions**
— clean, reusable code that other files import. `evaluate.py` is a small example of that
promotion: the notebook imports `fit_and_score()` from it rather than redefining it inline.

## Running it

```bash
# from the repo root
python -m pip install -r requirements.txt
jupyter notebook            # then open notebooks/ and run 01 before 02
```

Run `01-heart-disease-eda.ipynb` first (exploration), then
`02-heart-disease-modeling.ipynb` (modelling). Each notebook runs cleanly top-to-bottom with
**Restart & Run All**.

## Tech stack

Python 3.12 · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## Acknowledgements

Built while following Daniel Bourke's *Zero to Mastery* end-to-end heart disease
classification tutorial, adapted to scikit-learn 1.9 (e.g. `RocCurveDisplay` in place of the
removed `plot_roc_curve`) and organised into a small notebooks + `src/` repo.

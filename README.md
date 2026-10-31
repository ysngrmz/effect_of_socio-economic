# Explainable Machine Learning for Academic Achievement Prediction

**Reproducibility code for the study:** *Measuring the effect of socio-economic background and language learning beliefs on achievement through explainable machine learning*

This repository contains the Python analysis pipeline used to investigate how demographic characteristics, socioeconomic background, and language-learning beliefs contribute to the prediction of university students' self-reported grade point average (GPA). The workflow covers data auditing, repeated nested cross-validation, model comparison, feature-group ablation, held-out permutation importance, exploratory SHAP explanations, and reporting utilities.

> **Interpretation and reproducibility notice:** This V2 pipeline was developed after earlier model results had been examined. Accordingly, V2 model development and comparisons should be treated as **exploratory**, not as independent confirmatory validation. Repeated cross-validation on the same participants does not constitute external validation. Predictive importance is not a causal effect or a test of statistical significance.

## Repository structure

```text
.
├── README.md
├── README_TR.md                  # Original Turkish workflow notes
├── requirements.txt
├── input/
│   └── ds_v1.xlsx               # Local dataset; see Data availability below
├── scripts/
│   ├── common.py                # Data schema, preprocessing, shared utilities
│   ├── 00_selftest.py           # Basic schema / estimator smoke checks
│   ├── 01_audit.py              # Data and GPA audit; feature dictionary
│   ├── 02_train.py              # Repeated nested CV and model tuning
│   ├── 03_compare.py            # Descriptive paired model comparisons
│   ├── 04_ablation.py           # Feature-group ablation experiments
│   ├── 05_explain.py            # Held-out permutation importance
│   ├── 05_compare_explanations.py # Cross-model importance comparisons
│   ├── 05_shap.py               # Exploratory held-out SHAP explanations
│   ├── 06_morris.py             # Optional exploratory Morris-style analysis
│   ├── 07_extras.py             # Exploratory BALLI reliability diagnostics
│   ├── 08_figures.py            # Additional figures
│   ├── 09_manuscript_tables.py  # Draft manuscript tables
│   └── 10_validate.py           # Fold integrity and baseline checks
└── outputs_v2/                  # Generated results (created at runtime)
```

## Data schema

The scripts expect `input/ds_v1.xlsx` to have **44 columns in their original order**:

- Column 1: survey timestamp (metadata; excluded from model predictors).
- Column 2: self-reported GPA (prediction target, expected range 0–4).
- Columns 3–5: demographic predictors (**3** variables).
- Columns 6–10: socioeconomic-status (SES) predictors (**5** variables).
- Columns 11–44: Beliefs About Language Learning Inventory (BALLI) predictors (**34** variables).

Thus, **42 raw predictors** are modeled. The analysis expects an adjudicated dataset of **184 records**. The default GPA policy, `--gpa-policy error`, stops when missing, nonnumeric, or out-of-range GPA values are found. The alternative `--gpa-policy exclude` drops invalid records, but the recommended approach is to document and resolve exclusions before final analysis.

The preprocessing pipeline treats the questionnaire columns indexed 2–11 (zero-based) as categorical and one-hot encodes them; columns indexed 12–43 are treated as numerical Likert responses (1–5). Missing categorical values are imputed with the most frequent category, and missing numerical values with the median; numerical features are standardized. These transformations are fitted **within the training folds**, avoiding direct test-fold leakage. Treating Likert responses as numeric is a modeling assumption, not a validation of the scale's measurement properties.

## Installation

Recommended: use a dedicated Python environment (the project was developed using a Conda environment named `peerj_revision`). Install the dependencies specified in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python scripts/00_selftest.py
python scripts/01_audit.py
```

For Windows PowerShell, the original project instructions additionally recommend:

```powershell
$env:PYTHONNOUSERSITE = "1"
python -m pip install -r requirements.txt
python scripts/00_selftest.py
python scripts/01_audit.py
```

Run all commands from the **repository root**. `00_selftest.py` checks basic components; it is **not** a substitute for the full training and validation workflow. Review the audit outputs and verify the dataset schema before training.

## Reproducing the analyses

### 1. Optional quick training smoke test

```bash
python scripts/02_train.py --gpa-policy error --models Dummy Ridge GB MLP --outer-folds 2 --outer-repeats 1 --inner-folds 2 --n-iter 3 --tag smoke
python scripts/10_validate.py --tag smoke
```

### 2. Main exploratory V2 model comparison

```bash
python scripts/02_train.py --gpa-policy error --models Dummy Ridge ElasticNet BayesianRidge CatBoost ETR GB MLP RF SVR --outer-folds 5 --outer-repeats 3 --inner-folds 3 --n-iter 30 --tag main
python scripts/10_validate.py --tag main
python scripts/03_compare.py
```

This protocol uses **5 outer folds × 3 repeats = 15 outer test evaluations**, with **3-fold inner cross-validation** for hyperparameter optimization. Non-baseline models use `BayesSearchCV` with up to **30 search iterations per outer evaluation**, optimizing negative mean absolute error. The project seed is `20261021`. Preprocessing and tuning are learned from the relevant training data only.

The evaluated estimators are Dummy Regressor, Ridge, Elastic Net, Bayesian Ridge, CatBoost, Extra Tree Regressor (`ExtraTreeRegressor`, a **single tree**), Gradient Boosting, Multi-Layer Perceptron, Random Forest, and Support Vector Regression. **KTBoost is not included.** Performance summaries include MAE, RMSE, and R².

### 3. Feature-group ablation

```bash
python scripts/04_ablation.py --gpa-policy error --models Ridge GB --outer-folds 5 --outer-repeats 3 --inner-folds 3 --n-iter 30
```

The ablation configurations include each predictor group alone, all predictors, all predictors minus each group, and `All_minus_Grade`. Ablation compares predictive performance under different feature sets; it does not establish the causal effects of those features.

### 4. Held-out permutation importance and model comparisons

```bash
python scripts/05_explain.py --gpa-policy error --repeats 20
python scripts/05_compare_explanations.py --top-k 10 --plot-top 15
```

Permutation importance is calculated on saved outer-test partitions. In addition to raw-feature importance, all features within each of the three predefined groups (Demographic, SES, and BALLI) are **permuted jointly**. Reported importance is the change in held-out MAE (ΔMAE): positive values indicate worse predictions after permutation; negative values indicate lower error after permutation. Group importance is **not** the sum of individual-feature importances. Correlated features can affect the interpretation.

### 5. Exploratory SHAP analysis

For the Random Forest explanation used in the revised manuscript figures, run:

```bash
python scripts/05_shap.py --gpa-policy error --model RF --folds 3 --samples-per-fold 10 --background-size 12
```

To explore Gradient Boosting instead, change `--model RF` to `--model GB`. The original Turkish workflow demonstrates `GB`; the model must match the specific SHAP result being reproduced. SHAP uses saved outer-fold models and a model-agnostic permutation explainer on sampled held-out observations. Explaining only three folds is a computationally limited, exploratory analysis, not a summary of all 15 outer evaluations. Model selection made after examining V2 results should be disclosed as exploratory.

### 6. Auxiliary checks, figures, and manuscript-table drafts

```bash
python scripts/07_extras.py --gpa-policy error
python scripts/08_figures.py
python scripts/09_manuscript_tables.py
```

### 7. Optional exploratory Morris-style analysis

```bash
python scripts/06_morris.py --gpa-policy error --model GB --trajectories 32 --num-levels 6
```

**Do not treat the Morris output as the main explainability result.** Its interventions and rounding of Likert responses do not preserve the observed dependencies among questionnaire items, and the results should not be interpreted as causal or as a standard validated sensitivity analysis.

## Main generated files

The scripts write results under `outputs_v2/`. Important files include:

| Output | Description |
| --- | --- |
| `audit_summary.json`, `data_dictionary.csv`, `gpa_summary.csv` | Data checks and variable definitions |
| `main_settings.json`, `main_software_versions.json` | Run configuration, dataset hash, and software versions |
| `main_fold_metrics.csv`, `main_summary.csv` | Outer-fold scores and model summaries |
| `main_oof_predictions.csv` | Held-out predictions; **may contain participant-level information** |
| `main_best_params.csv` | Selected hyperparameters by model and outer fold |
| `main_validated_summary.csv`, `main_vs_dummy.csv` | Validation and Dummy-baseline comparisons |
| `paired_descriptive_comparisons.csv` | Descriptive paired MAE comparisons |
| `ablation_summary.csv` | Group-ablation results |
| `all_models_permutation_by_fold.csv`, `all_models_permutation_summary.csv` | Feature/group importance |
| `all_models_feature_importance_matrix.csv`, `all_models_group_importance_matrix.csv` | Cross-model importance matrices |
| `all_models_heatmap_labels.csv` | Label mapping for the permutation heatmap |
| `RF_shap_summary.csv`, `RF_shap_plot_labels.csv` | RF SHAP summary and plotted-label mapping, if RF is selected |
| `RF_shap_bar.png`, `RF_shap_bar.pdf` | RF SHAP visualization, if RF is selected |
| `models/main/*.joblib` | Saved outer-fold pipelines; **review before sharing** |

SHAP output filenames use the selected model prefix (`RF_`, `GB_`, etc.). Figure label mappings are **plot-specific**: the `F1`–`F15` codes from the permutation heatmap are not necessarily the same features as the `F1`–`F15` codes in the SHAP bar chart. Use each plot's corresponding `*_labels.csv` file.

## Important methodological limitations

1. **Exploratory development:** The V2 workflow was refined after earlier results were inspected; independent external validation remains necessary for stronger generalization claims.
2. **Limited sample:** The 184 analyzed records come from a convenience sample. The findings should not be generalized to all EFL learners without further evidence.
3. **Outcome definition:** The target is self-reported overall university GPA, **not** a direct measure of English-language proficiency.
4. **CV dependence:** The 15 repeated outer evaluations are not 15 independent samples. Their standard deviations are descriptive and must not be interpreted as independent-sample confidence intervals.
5. **Feature interpretation:** Predictive contribution and SHAP values do not imply causality, statistical significance, or the direction of a feature's relationship with GPA.
6. **BALLI measurement:** Raw Cronbach's alpha or exploratory item groupings do not establish validated subscales. Reverse coding and construct validity require reference to the original questionnaire and scoring documentation.
7. **Grade-level sensitivity:** Year of study may be related to cumulative GPA through the academic process; the `All_minus_Grade` ablation provides a sensitivity check.
8. **Privacy:** Do not publish participant-level survey data, held-out predictions, or serialized models without reviewing ethical approval, consent, and disclosure risks.

## Reproducibility notes

- Training runs refuse to reuse an existing `--tag` if its results or model directory already exist. Use a fresh tag for a genuinely new experiment.
- Several downstream scripts assume the main run is tagged `main`; changing the training tag alone is not sufficient to redirect every downstream step.
- Do not mix old `outputs/` model files with `outputs_v2/` artifacts.
- Record the environment with `python -m pip freeze > outputs_v2/environment_freeze.txt` if desired.
- `requirements.txt` specifies version ranges rather than an exact lockfile; `main_software_versions.json` records key installed versions for each main run.
- The provided code and dataset should be checked against the exact version used to produce the published tables before claiming byte-for-byte reproducibility. This README documents the supplied pipeline; it does not assert that the full training run was executed again during README preparation.

## Data availability and citation

**Data availability:** [Replace with the approved repository or controlled-access data statement. Do not claim public availability unless sharing has been authorized.]

**Paper citation:** [Add the final author list, publication year, journal, DOI, or preprint URL when available.]

**License:** [Add the chosen software license only after confirming rights to distribute the code and any included data.]

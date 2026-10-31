"""PeerJ V2 shared data schema and leakage-safe preprocessing.

IMPORTANT: column roles are explicit for the supplied 44-column questionnaire.
Ordinal Likert items are numerical 1..5; first 10 predictors are categorical.
This is a modelling assumption, not psychometric scale validation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'input' / 'ds_v1.xlsx'
OUT = ROOT / 'outputs_v2'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261021
GROUPS = {'Demographic': list(range(2,5)), 'SES': list(range(5,10)), 'BALLI': list(range(10,44))}
# Questionnaire column positions: 2..11 are nominal/multichoice, 12..43 Likert 1..5.
CATEGORICAL_INDICES = tuple(range(2,12))
LIKERT_INDICES = tuple(range(12,44))

def load(policy='error'):
    if not DATA.exists():
        raise FileNotFoundError(f'Copy your ADJUDICATED 184-row dataset to {DATA}')
    original = pd.read_excel(DATA)
    if original.shape[1] != 44:
        raise ValueError(f'Expected 44 columns, found {original.shape[1]}')
    original.columns = [' '.join(str(c).split()).strip() for c in original.columns]
    if not original.columns.is_unique:
        raise ValueError('Duplicate normalized column names')
    target = original.columns[1]
    y = pd.to_numeric(original[target], errors='coerce')
    invalid = y.isna() | ~np.isfinite(y) | (y < 0) | (y > 4)
    if invalid.any():
        if policy == 'error':
            raise ValueError(f'{invalid.sum()} invalid GPA records; resolve manually before final training')
        if policy != 'exclude':
            raise ValueError('policy must be error/exclude')
        original = original.loc[~invalid].copy()
        y = y.loc[~invalid]
    X = original.drop(columns=[original.columns[0], target]).reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = {k: [original.columns[i] for i in positions] for k, positions in GROUPS.items()}
    for idx in CATEGORICAL_INDICES:
        col = original.columns[idx]
        X[col] = X[col].astype('string').replace({'': pd.NA, 'nan': pd.NA}).astype(object)
        X[col] = X[col].where(pd.notna(X[col]), np.nan)
    for idx in LIKERT_INDICES:
        col = original.columns[idx]
        X[col] = pd.to_numeric(X[col], errors='coerce')
        invalid_likert = X[col].notna() & ~X[col].isin([1,2,3,4,5])
        if invalid_likert.any():
            raise ValueError(f'Invalid Likert codes in {col}; review original questionnaire')
    return X, y, groups, original.reset_index(drop=True)

def prep(X):
    # Explicit roles, not dtype guessing: numeric-coded categories stay categorical.
    # Whitespace-normalized header matching (avoids original Excel line breaks).
    original_names = [' '.join(str(c).split()).strip() for c in pd.read_excel(DATA, nrows=0).columns]
    cat = [c for c in X.columns if c in {original_names[i] for i in CATEGORICAL_INDICES}]
    num = [c for c in X.columns if c not in cat]
    transformers=[]
    if cat:
        transformers.append(('categorical', Pipeline([
            ('fill', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat))
    if num:
        transformers.append(('numeric', Pipeline([
            ('fill', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())]), num))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=0.0)

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def metric(y, pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {'mae': float(mean_absolute_error(y,pred)),
            'rmse': float(np.sqrt(mean_squared_error(y,pred))),
            'r2': float(r2_score(y,pred))}

class MLPArchitectureRegressor(RegressorMixin, BaseEstimator):
    """Architecture selected by integer; target scaling handled by TransformedTargetRegressor."""
    ARCHITECTURES = ((4,), (8,), (12,), (16,), (8,4), (16,8))
    def __init__(self, architecture_id=1, alpha=0.01, max_iter=1200,
                 max_fun=25000, random_state=SEED):
        self.architecture_id=architecture_id
        self.alpha=alpha
        self.max_iter=max_iter
        self.max_fun=max_fun
        self.random_state=random_state
    def fit(self,X,y):
        idx=int(self.architecture_id)
        if not 0 <= idx < len(self.ARCHITECTURES):
            raise ValueError('architecture_id outside supported range')
        self.model_=MLPRegressor(hidden_layer_sizes=self.ARCHITECTURES[idx],
            solver='lbfgs', activation='relu', alpha=self.alpha,
            max_iter=self.max_iter, max_fun=self.max_fun,
            random_state=self.random_state)
        self.model_.fit(X,y)
        return self
    def predict(self,X):
        return self.model_.predict(X)

"""All-model held-out permutation importance from SAVED outer-CV pipelines.

Runs no new hyperparameter searches. Jointly permutes each predeclared raw-feature
block, using the SAME permutation across all models for a given fold/block/repeat.
A delta MAE > 0 means predictive performance deteriorates on permutation.
"""
import argparse
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from common import load, OUT, SEED


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpa-policy', choices=['error', 'exclude'], default='error')
    p.add_argument('--models', nargs='+', default=None, help='Default: all saved models')
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--tag', default='main')
    args = p.parse_args()
    if args.repeats < 2:
        p.error('--repeats must be >= 2 for SD calculation')
    X, y, groups, _ = load(args.gpa_policy)
    model_dir = OUT / 'models' / args.tag
    if not model_dir.exists():
        raise FileNotFoundError(f'Missing saved models: {model_dir}. Run 02_train.py first.')
    saved = {}
    for path in sorted(model_dir.glob('*_fold*.joblib')):
        name, sep, fold_text = path.stem.rpartition('_fold')
        if sep and fold_text.isdigit():
            saved.setdefault(name, {})[int(fold_text)] = path
    names = args.models if args.models is not None else sorted(saved)
    if not names:
        raise ValueError('No saved models found.')
    missing = [m for m in names if m not in saved]
    if missing:
        raise ValueError(f'Models without saved fold pipelines: {missing}. Run 02_train.py for these models first.')
    common_folds = set(saved[names[0]])
    for name in names:
        if set(saved[name]) != common_folds:
            raise ValueError(f'Fold mismatch for {name}. Re-run 02_train.py with ALL models together.')
    if len(common_folds) < 2:
        warnings.warn('Only one outer fold found: stability cannot be assessed.', stacklevel=2)
    blocks = {col: [col] for col in X.columns}
    blocks.update({f'GROUP_{name}': cols for name, cols in groups.items()})
    # Cache fold pipelines once and validate identical test sets across models.
    models_by_fold = {}
    for fold in sorted(common_folds):
        models_by_fold[fold] = {}
        expected = None
        for name in names:
            info = joblib.load(saved[name][fold])
            test_idx = np.asarray(info['test_indices'], dtype=int)
            if np.any(test_idx < 0) or np.any(test_idx >= len(X)):
                raise ValueError(f'{name} fold {fold}: test indices do not fit the current dataset. Re-run training.')
            if list(X.columns) != list(info['features']):
                raise ValueError(f'{name} fold {fold}: raw features changed. Re-run training.')
            if expected is None:
                expected = test_idx
            elif not np.array_equal(test_idx, expected):
                raise ValueError(f'{name} fold {fold}: outer test split differs from other models.')
            models_by_fold[fold][name] = info['pipeline']
    # Stronger check against accidentally loading saved models from a different dataset.
    preds_file = OUT / f'{args.tag}_oof_predictions.csv'
    if preds_file.exists():
        recorded = pd.read_csv(preds_file)
        for name in names:
            subset = recorded[recorded['model'] == name]
            if subset.empty:
                warnings.warn(f'{name} absent from {preds_file.name}; cannot check saved target values.', stacklevel=2)
                continue
            ids = subset['row_id'].astype(int).to_numpy()
            if np.any(ids < 0) or np.any(ids >= len(y)) or not np.allclose(subset['observed'].to_numpy(dtype=float), y.iloc[ids].to_numpy(dtype=float)):
                raise ValueError('Target values/order differ from saved CV results. Re-run 02_train.py on the current dataset.')
    else:
        warnings.warn('OOF predictions absent: cannot verify saved targets against current data.', stacklevel=2)
    records = []
    for fold in sorted(common_folds):
        te = np.asarray(joblib.load(saved[names[0]][fold])['test_indices'], dtype=int)
        xx, yy = X.iloc[te].copy(), y.iloc[te]
        baselines = {name: mean_absolute_error(yy, model.predict(xx)) for name, model in models_by_fold[fold].items()}
        for block_index, (block, cols) in enumerate(blocks.items()):
            # Fixed permutations across models, per fold and raw-feature block.
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, fold, block_index]))
            shuffled_sets = []
            for _ in range(args.repeats):
                shuffled = xx.copy()
                order = rng.permutation(len(xx))
                shuffled.loc[:, cols] = xx.iloc[order][cols].to_numpy()
                shuffled_sets.append(shuffled)
            for name, model in models_by_fold[fold].items():
                deltas = [mean_absolute_error(yy, model.predict(shuffled)) - baselines[name] for shuffled in shuffled_sets]
                records.append({'fold': fold, 'model': name, 'feature_or_group': block,
                                'delta_mae_mean': float(np.mean(deltas)),
                                'delta_mae_sd': float(np.std(deltas, ddof=1)),
                                'baseline_mae': float(baselines[name]), 'n_test': len(te)})
        print(f'Finished fold {fold + 1}/{len(common_folds)}', flush=True)
    result = pd.DataFrame(records)
    result.to_csv(OUT / 'all_models_permutation_by_fold.csv', index=False)
    summary = (result.groupby(['model', 'feature_or_group'], as_index=False)
               .agg(mean_delta_mae=('delta_mae_mean', 'mean'),
                    sd_across_folds=('delta_mae_mean', 'std'),
                    n_outer_folds=('fold', 'nunique'),
                    mean_baseline_mae=('baseline_mae', 'mean')))
    summary.to_csv(OUT / 'all_models_permutation_summary.csv', index=False)
    print('Saved all_models_permutation_by_fold.csv and all_models_permutation_summary.csv')
    print('Note: outer folds overlap; SD is descriptive, not an independent-fold confidence interval.')
    print('Note: correlated predictors can mask one another; importance is predictive, not causal.')


if __name__ == '__main__':
    main()

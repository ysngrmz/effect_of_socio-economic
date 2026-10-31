"""Exploratory SHAP for ONE pre-specified model, using saved outer-fold models.

Model-agnostic SHAP permutation explainer operates on original input columns and
wraps each fold's complete preprocessing + estimator pipeline. The explanation
is descriptive: selecting a model from outer-CV scores is not an independent test.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load, OUT, SEED


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpa-policy', choices=['error', 'exclude'], default='error')
    p.add_argument('--model', required=True, help='Model selected for detailed explanation')
    p.add_argument('--folds', type=int, default=3, help='Number of saved outer folds to explain')
    p.add_argument('--samples-per-fold', type=int, default=10)
    p.add_argument('--background-size', type=int, default=12)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--tag', default='main')
    args = p.parse_args()
    if min(args.folds, args.samples_per_fold, args.background_size) < 1:
        p.error('--folds, --samples-per-fold and --background-size must be >= 1')
    try:
        import shap
    except ImportError as exc:
        raise SystemExit('SHAP is optional. Install it with: python -m pip install shap') from exc
    X, y, _, _ = load(args.gpa_policy)
    folder = OUT / 'models' / args.tag
    paths = sorted(folder.glob(f'{args.model}_fold*.joblib'), key=lambda q: int(q.stem.rsplit('_fold', 1)[1]))
    if not paths:
        raise FileNotFoundError(f'No saved pipelines for {args.model}; run 02_train.py first.')
    if args.folds > len(paths):
        p.error(f'Requested {args.folds} folds but only {len(paths)} saved for {args.model}')
    rng = np.random.default_rng(args.seed)
    rows = []
    summary_rows = []
    for path in paths[:args.folds]:
        info = joblib.load(path)
        if list(X.columns) != list(info['features']):
            raise ValueError('Features differ from saved model; rerun 02_train.py.')
        tr = np.asarray(info['train_indices'], dtype=int)
        te = np.asarray(info['test_indices'], dtype=int)
        if max(tr.max(), te.max()) >= len(X):
            raise ValueError('Saved indices do not fit dataset; rerun training.')
        pipeline = info['pipeline']
        bg_idx = rng.choice(tr, size=min(args.background_size, len(tr)), replace=False)
        ex_idx = rng.choice(te, size=min(args.samples_per_fold, len(te)), replace=False)
        # SHAP's tabular masker requires numeric arrays (np.isclose fails on strings).
        # Encode raw categorical columns with TRAINING-derived integer codes and
        # decode back to the original categories before calling the saved pipeline.
        # The codes are a masker representation, NOT an ordinal encoding for ML.
        train_frame = X.iloc[tr]
        categorical_maps = {}
        missing_codes = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                cats = list(pd.unique(train_frame[col].dropna().astype(str)))
                categorical_maps[col] = cats
            else:
                vals_train = pd.to_numeric(train_frame[col], errors='coerce').dropna()
                lo = float(vals_train.min()) if len(vals_train) else 0.0
                hi = float(vals_train.max()) if len(vals_train) else 1.0
                missing_codes[col] = hi + 1000 * max(hi-lo, 1.0) + 1.0

        def encode(frame):
            arr = np.zeros((len(frame), X.shape[1]), dtype=float)
            for j, col in enumerate(X.columns):
                if col in categorical_maps:
                    mapping = {cat: k for k, cat in enumerate(categorical_maps[col])}
                    arr[:, j] = frame[col].map(lambda v: -1 if pd.isna(v) else mapping.get(str(v), -2)).to_numpy(dtype=float)
                else:
                    arr[:, j] = pd.to_numeric(frame[col], errors='coerce').fillna(missing_codes[col]).to_numpy(dtype=float)
            return arr

        def predict_encoded(arr):
            arr = np.asarray(arr, dtype=float)
            frame = pd.DataFrame(index=range(len(arr)), columns=X.columns)
            for j, col in enumerate(X.columns):
                if col in categorical_maps:
                    cats = categorical_maps[col]
                    def decode(v):
                        k = int(round(v))
                        if k == -1: return np.nan
                        if k == -2 or k < 0 or k >= len(cats): return '__UNSEEN_CATEGORY__'
                        return cats[k]
                    frame[col] = [decode(v) for v in arr[:, j]]
                else:
                    values = arr[:, j].copy()
                    values[values == missing_codes[col]] = np.nan
                    frame[col] = values
            return np.asarray(pipeline.predict(frame), dtype=float)

        background = encode(X.iloc[bg_idx])
        examples = encode(X.iloc[ex_idx])
        explainer = shap.Explainer(predict_encoded, shap.maskers.Independent(background), algorithm='permutation')
        values = explainer(examples, max_evals=2*X.shape[1]+1)
        vals = np.asarray(values.values)
        if vals.shape != (len(examples), X.shape[1]):
            raise RuntimeError(f'Unexpected SHAP dimensions: {vals.shape}')
        fold = int(path.stem.rsplit('_fold', 1)[1])
        for sample_i, row_id in enumerate(ex_idx):
            for col_i, col in enumerate(X.columns):
                rows.append({'fold': fold, 'row_id': int(row_id), 'feature': col,
                             'shap_value': float(vals[sample_i, col_i]),
                             'abs_shap_value': float(abs(vals[sample_i, col_i]))})
        for col_i, col in enumerate(X.columns):
            summary_rows.append({'fold': fold, 'feature': col,
                                 'mean_abs_shap': float(np.mean(np.abs(vals[:, col_i])))})
        print(f'SHAP explained fold {fold} ({len(examples)} outer-test rows)', flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'{args.model}_shap_values.csv', index=False)
    fold_summary = pd.DataFrame(summary_rows)
    fold_summary.to_csv(OUT / f'{args.model}_shap_by_fold.csv', index=False)
    summary = (fold_summary.groupby('feature', as_index=False)
               .agg(mean_abs_shap=('mean_abs_shap', 'mean'),
                    sd_across_folds=('mean_abs_shap', 'std'),
                    n_folds=('fold', 'nunique'))
               .sort_values('mean_abs_shap', ascending=False))
    summary.to_csv(OUT / f'{args.model}_shap_summary.csv', index=False)
    top = summary.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, len(top)*0.38)))
    ax.barh(range(len(top)), top['mean_abs_shap'].to_numpy())
    ax.set_yticks(range(len(top)), [f'F{i+1}' for i in range(len(top))])
    ax.set_xlabel('Mean absolute SHAP value (GPA units)')
    ax.set_title(f'{args.model}: exploratory SHAP on held-out cases')
    fig.tight_layout()
    fig.savefig(OUT / f'{args.model}_shap_bar.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUT / f'{args.model}_shap_bar.pdf', bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'plot_label': [f'F{i+1}' for i in range(len(top))],
                  'original_feature': list(top.feature)}).to_csv(OUT / f'{args.model}_shap_plot_labels.csv', index=False)
    print('Saved SHAP values, fold summaries, summary and bar plots.')
    print('SHAP explains model predictions, not causal effects; correlated predictors may share attribution.')
    print('This model-specific explanation is exploratory if model choice used the same outer-CV results.')


if __name__ == '__main__':
    main()

"""Compare held-out permutation importance rankings across fitted model families."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from common import OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--plot-top', type=int, default=15)
    args = p.parse_args()
    path = OUT / 'all_models_permutation_summary.csv'
    if not path.exists():
        p.error('Run scripts/05_explain.py first')
    data = pd.read_csv(path)
    features = data[~data.feature_or_group.str.startswith('GROUP_')].copy()
    matrix = features.pivot(index='feature_or_group', columns='model', values='mean_delta_mae')
    matrix.to_csv(OUT / 'all_models_feature_importance_matrix.csv')
    groups = data[data.feature_or_group.str.startswith('GROUP_')].copy()
    groups.pivot(index='feature_or_group', columns='model', values='mean_delta_mae').to_csv(OUT / 'all_models_group_importance_matrix.csv')
    models = list(matrix.columns)
    corr_rows = []
    overlap_rows = []
    for i, a in enumerate(models):
        for b in models[i+1:]:
            valid = matrix[[a, b]].dropna()
            # Constant importances (e.g. Dummy) have no defined rank correlation.
            rho = float(spearmanr(valid[a], valid[b]).statistic) if len(valid) > 2 and valid[a].nunique() > 1 and valid[b].nunique() > 1 else np.nan
            k = min(args.top_k, len(valid))
            top_a = set(valid[a].nlargest(k).index)
            top_b = set(valid[b].nlargest(k).index)
            corr_rows.append({'model_a': a, 'model_b': b, 'spearman_rho': rho, 'n_features': len(valid)})
            overlap_rows.append({'model_a': a, 'model_b': b, 'top_k': k,
                                 'shared_top_k': len(top_a & top_b), 'overlap_fraction': len(top_a & top_b)/k if k else np.nan})
    pd.DataFrame(corr_rows).to_csv(OUT / 'all_models_rank_correlations.csv', index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / 'all_models_topk_overlap.csv', index=False)
    # Descriptive heat map: signed delta MAE, not normalized between models.
    strongest = matrix.clip(lower=0).mean(axis=1).nlargest(min(args.plot_top, len(matrix))).index
    plot_data = matrix.loc[strongest]
    fig, ax = plt.subplots(figsize=(max(8, len(models)*1.1), max(5, len(plot_data)*0.38+1)))
    values = plot_data.to_numpy(dtype=float)
    limit = max(float(np.nanmax(np.abs(values))), 1e-8)
    img = ax.imshow(values, cmap='coolwarm', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(len(models)), models, rotation=45, ha='right')
    ax.set_yticks(range(len(plot_data)), [f'F{i+1}' for i in range(len(plot_data))])
    ax.set_title('Held-out permutation importance across models (delta MAE)')
    fig.colorbar(img, ax=ax, label='Increase in MAE after permutation')
    fig.tight_layout()
    fig.savefig(OUT / 'all_models_importance_heatmap.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUT / 'all_models_importance_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'plot_label': [f'F{i+1}' for i in range(len(plot_data))],
                  'original_feature': list(plot_data.index)}).to_csv(OUT / 'all_models_heatmap_labels.csv', index=False)
    print('Saved cross-model feature/group matrices, rank correlations, top-k overlap and heatmap.')
    print('F1, F2, ... map to exact feature names in all_models_heatmap_labels.csv.')
    print('Correlations are descriptive and undefined for constant-importance models (e.g. Dummy).')


if __name__ == '__main__':
    main()

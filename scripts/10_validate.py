"""Sanity checks for final model comparisons and outer-fold predictions."""
import argparse
import numpy as np
import pandas as pd
from common import OUT,load
p=argparse.ArgumentParser()
p.add_argument('--tag',default='main')
p.add_argument('--gpa-policy',choices=['error','exclude'],default='error')
a=p.parse_args()
X,y,_,_=load(a.gpa_policy)
m=pd.read_csv(OUT/f'{a.tag}_fold_metrics.csv')
o=pd.read_csv(OUT/f'{a.tag}_oof_predictions.csv')
expected=set(range(m.fold.nunique()))
assert m.groupby('model').fold.apply(set).apply(lambda s:s==expected).all(), 'Unequal model folds'
assert not m[['mae','rmse','r2']].isna().any().any(), 'Missing metrics'
assert len(o)==int(m.test_n.sum()), 'OOF prediction count mismatch'
for model,g in o.groupby('model'):
    assert np.allclose(g.observed.to_numpy(),y.iloc[g.row_id.to_numpy(dtype=int)].to_numpy()),'Target order mismatch'
    assert g.groupby('row_id').size().eq(m.repeat.nunique()).all(),f'{model}: OOF repetition mismatch'
summary=m.groupby('model')[['mae','rmse','r2']].agg(['mean','std'])
summary.to_csv(OUT/f'{a.tag}_validated_summary.csv')
if 'Dummy' in m.model.values:
    pivot=m.pivot(index='fold',columns='model',values='mae')
    comparison=pd.DataFrame({'model':pivot.columns,'mean_MAE_delta_vs_Dummy':[(pivot[c]-pivot['Dummy']).mean() for c in pivot.columns],
       'fraction_outer_folds_better_than_Dummy':[(pivot[c]<pivot['Dummy']).mean() for c in pivot.columns]})
    comparison.to_csv(OUT/f'{a.tag}_vs_dummy.csv',index=False)
print('VALIDATED: matched folds, observed targets, and per-repeat coverage')
print(summary.round(4).to_string())

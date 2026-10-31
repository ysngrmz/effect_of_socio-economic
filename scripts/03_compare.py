"""Descriptive paired outer-fold differences, NOT independent-sample p-values/CI."""
import pandas as pd
from common import OUT
m=pd.read_csv(OUT/'main_fold_metrics.csv')
a=m.pivot(index='fold',columns='model',values='mae')
rows=[]
for name in a.columns:
 for other in a.columns:
  if name>=other:continue
  d=(a[name]-a[other]).dropna()
  rows.append({'model_A':name,'model_B':other,'mean_paired_MAE_difference_A_minus_B':d.mean(),'median_difference':d.median(),'folds_compared':len(d),'fraction_A_better':(d<0).mean()})
pd.DataFrame(rows).to_csv(OUT/'paired_descriptive_comparisons.csv',index=False)
print('Saved paired comparisons; do not treat folds as independent observations.')

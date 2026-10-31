import argparse
import pandas as pd
from common import load,OUT
from importlib import import_module
train=import_module('02_train')
a=argparse.ArgumentParser()
a.add_argument('--gpa-policy',choices=['error','exclude'],default='error')
a.add_argument('--models',nargs='+',default=['Ridge','GB'])
a.add_argument('--outer-repeats',type=int,default=3)
a.add_argument('--outer-folds',type=int,default=5)
a.add_argument('--inner-folds',type=int,default=3)
a.add_argument('--n-iter',type=int,default=20)
a.add_argument('--search',choices=['bayes'],default='bayes')
args=a.parse_args()
X,y,groups,_=load(args.gpa_policy)
# Explicit grade-drop test: avoid mechanically associating GPA with year of study.

sets={**groups,'All':list(X.columns)}
sets['All_minus_Grade']=[c for c in X if c != groups['Demographic'][2]]
for group in groups:
 sets['All_minus_'+group]=[c for c in X if c not in groups[group]]
for name,cols in sets.items():
 train.evaluate(X[cols],y,args.models,args.outer_folds,args.outer_repeats,args.inner_folds,args.n_iter,tag='ablation_'+name,save_models=False,search_method=args.search)
rows=[]
for name in sets:
 r=pd.read_csv(OUT/f'ablation_{name}_fold_metrics.csv')
 for model,g in r.groupby('model'):
  rows.append({'features':name,'model':model,'MAE_mean':g.mae.mean(),'MAE_sd':g.mae.std(),'R2_mean':g.r2.mean(),'n_features':len(sets[name])})
pd.DataFrame(rows).to_csv(OUT/'ablation_summary.csv',index=False)

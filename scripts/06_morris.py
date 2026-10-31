"""OPTIONAL EXPLORATORY Morris on BALLI numeric 1..5 items only.
Hold a REAL observed row's nominal/SES values fixed. Morris changes BALLI items
independently => possible unrealistic item combinations. Not conditional inference.
"""
import argparse,joblib,numpy as np,pandas as pd
from common import load,OUT,SEED
p=argparse.ArgumentParser();p.add_argument('--gpa-policy',choices=['error','exclude'],default='error');p.add_argument('--model',default='CatBoost');p.add_argument('--trajectories',type=int,default=32);p.add_argument('--num-levels',type=int,default=6);p.add_argument('--anchors',type=int,default=5)
a=p.parse_args()
try:from SALib.sample import morris as msample;from SALib.analyze import morris as manalyze
except ImportError as e:raise SystemExit('pip install SALib; '+str(e))
X,y,groups,_=load(a.gpa_policy)
balli=[c for c in groups['BALLI'] if pd.api.types.is_numeric_dtype(X[c])]
if not balli:raise ValueError('No numeric BALLI items')
problem={'num_vars':len(balli),'names':balli,'bounds':[[1.,5.]]*len(balli)}
samples=msample.sample(problem,N=a.trajectories,num_levels=a.num_levels,seed=SEED)
path=OUT/'models'/'main'/f'{a.model}_fold0.joblib'
pipe=joblib.load(path)['pipeline']
rng=np.random.default_rng(SEED)
anchor_ids=rng.choice(len(X),size=min(a.anchors,len(X)),replace=False)
results=[]
for anchor in anchor_ids:
 temp=pd.DataFrame([X.iloc[anchor].to_dict()]*len(samples),columns=X.columns)
 # Morris explores continuous interval; round to valid discrete Likert responses.
 for j,c in enumerate(balli):temp[c]=np.clip(np.rint(samples[:,j]),1,5).astype(int)
 # NOTE: rounding means finite differences no longer represent exact SALib step size.
 preds=pipe.predict(temp)
 s=manalyze.analyze(problem,samples,preds,num_levels=a.num_levels,print_to_console=False,seed=SEED)
 for j,c in enumerate(balli):results.append({'anchor':int(anchor),'feature':c,'mu':s['mu'][j],'mu_star':s['mu_star'][j],'sigma':s['sigma'][j]})
raw=pd.DataFrame(results);raw.to_csv(OUT/f'{a.model}_morris_exploratory.csv',index=False)
raw.groupby('feature')[['mu','mu_star','sigma']].mean().sort_values('mu_star',ascending=False).to_csv(OUT/f'{a.model}_morris_summary.csv')
print('EXPLORATORY ONLY: discrete rounding and independent perturbation depart from standard Morris assumptions; validate before scientific use.')

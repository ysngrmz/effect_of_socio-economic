import argparse
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from common import load,OUT
p=argparse.ArgumentParser();p.add_argument('--gpa-policy',choices=['error','exclude'],default='error');a=p.parse_args()
X,y,groups,_=load(a.gpa_policy)
# Internal consistency is descriptive, not a validation of the five original BALLI subscales.
balli=X[groups['BALLI']].select_dtypes(include='number')
def alpha(df):
 a=df.dropna()
 k=a.shape[1]
 if k<2 or len(a)<3:return np.nan
 denominator=a.sum(axis=1).var(ddof=1)
 return float(k/(k-1)*(1-a.var(ddof=1).sum()/denominator)) if denominator>0 else np.nan
# Note: reverse coding and construct-specific grouping must be validated beforehand.
with open(OUT/'balli_reliability_WARNING.txt','w',encoding='utf-8') as f:
 f.write(f'Raw (NOT necessarily interpretable) alpha for {len(balli.columns)} numeric BALLI items: {alpha(balli):.4f}\nReverse coding, subscale assignments and construct interpretation require instrument documentation. Do not report as validated scale reliability without that work.\n')
with open(OUT/'gpa_grade_note.txt','w',encoding='utf-8') as f:
 f.write('Grade is a categorical predictor; grade-stratified GPA descriptive values are in gpa_by_grade.csv. Do not interpret it as measured language proficiency. First-year GPA provenance requires author verification.\n')
print('Saved exploratory BALLI caution note and grade note; run grade-drop ablation separately if needed.')

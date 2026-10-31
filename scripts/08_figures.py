from pathlib import Path
import pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT,DATA,load
plt.rcParams.update({'font.size':10,'figure.dpi':130,'savefig.dpi':300})
_,y,_,_=load('error')
fig,ax=plt.subplots(figsize=(7,4));ax.hist(y.dropna(),bins=20);ax.set(xlabel='Self-reported GPA',ylabel='Students',title='GPA distribution (adjudicated data)');fig.tight_layout();fig.savefig(OUT/'gpa_histogram.png');fig.savefig(OUT/'gpa_histogram.pdf');plt.close(fig)
path=OUT/'main_summary.csv'
if path.exists():
 r=pd.read_csv(path,header=[0,1],index_col=0)
 # pandas multiindex read for reported mean scores
 means=r[('mae','mean')].sort_values()
 fig,ax=plt.subplots(figsize=(8,5));means.plot(kind='barh',ax=ax);ax.invert_yaxis();ax.set_xlabel('Mean outer-fold MAE');fig.tight_layout();fig.savefig(OUT/'model_mae.png');fig.savefig(OUT/'model_mae.pdf');plt.close(fig)
for path in OUT.glob('*_permutation_summary.csv'):
 if path.name == 'all_models_permutation_summary.csv':
  continue
 r=pd.read_csv(path,index_col=0).head(15)
 fig,ax=plt.subplots(figsize=(10,7));r['mean'].sort_values().plot.barh(ax=ax)
 ax.set_xlabel('Mean increase in held-out MAE after permutation');ax.set_ylabel('');fig.tight_layout();fig.savefig(OUT/(path.stem+'.png'));fig.savefig(OUT/(path.stem+'.pdf'));plt.close(fig)
print('Figures saved to outputs')

import pandas as pd
import numpy as np
from common import DATA,OUT,GROUPS,save_json

df=pd.read_excel(DATA)
df.columns=[' '.join(str(c).split()).strip() for c in df.columns]
gpa=pd.to_numeric(df.iloc[:,1],errors='coerce')
invalid=gpa.isna() | (gpa<0) | (gpa>4)
df.loc[invalid].to_csv(OUT/'audit_invalid_gpa.csv',index=False)
pd.DataFrame([{'index':i,'feature':c,'dtype':str(df[c].dtype),'missing':int(df[c].isna().sum()),'unique':int(df[c].nunique(dropna=True)), 'group': next((g for g,inds in GROUPS.items() if i in inds),'metadata/target')} for i,c in enumerate(df.columns)]).to_csv(OUT/'data_dictionary.csv',index=False)
gpa.describe(percentiles=[.05,.25,.5,.75,.95]).to_csv(OUT/'gpa_summary.csv')
pd.crosstab(df.loc[~invalid].iloc[:,4],columns='N').to_csv(OUT/'grade_counts.csv')
df.loc[~invalid].groupby(df.columns[4])[df.columns[1]].agg(['count','mean','std','median','min','max']).to_csv(OUT/'gpa_by_grade.csv')
metadata={'rows':len(df),'columns':len(df.columns),'feature_count':len(df.columns)-2,'invalid_gpa_count':int(invalid.sum()),'duplicate_rows_without_timestamp':int(df.iloc[:,1:].duplicated().sum()),'timestamp_min':str(df.iloc[:,0].min()),'timestamp_max':str(df.iloc[:,0].max()),'gpa_out_of_range_policy':'MANUAL REVIEW REQUIRED'}
save_json(OUT/'audit_summary.json',metadata)
print('Audit complete:',metadata)

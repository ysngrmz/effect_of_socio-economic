"""Nested CV V2: fixed protocol, inner-only hyperparameter selection, saved outer models."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from common import load, prep, metric, OUT, DATA, SEED, save_json, MLPArchitectureRegressor

DEFAULT_MODELS=['Dummy','Ridge','ElasticNet','BayesianRidge','CatBoost','ETR','GB','MLP','RF','SVR']

def model_specs(seed):
    log=lambda lo,hi: Real(lo,hi,prior='log-uniform')
    specs={
      'Dummy': (DummyRegressor(strategy='mean'), {}),
      'Ridge': (Ridge(), {'model__alpha':log(1e-3,1e4)}),
      'ElasticNet': (ElasticNet(max_iter=15000,random_state=seed),
          {'model__alpha':log(1e-4,1e2),'model__l1_ratio':Real(0.05,1.0)}),
      'BayesianRidge': (BayesianRidge(),
          {'model__alpha_1':log(1e-8,1e-3),'model__alpha_2':log(1e-8,1e-3),
           'model__lambda_1':log(1e-8,1e-3),'model__lambda_2':log(1e-8,1e-3)}),
      # ETR means ONE ExtraTreeRegressor (not ExtraTrees ensemble).
      'ETR': (ExtraTreeRegressor(random_state=seed),
          {'model__max_depth':Integer(1,8),'model__min_samples_leaf':Integer(3,25),
           'model__max_features':Real(0.35,1.0)}),
      'GB': (GradientBoostingRegressor(random_state=seed,loss='squared_error'),
          {'model__n_estimators':Integer(30,350),'model__learning_rate':log(.008,.2),
           'model__max_depth':Integer(1,3),'model__min_samples_leaf':Integer(3,25),
           'model__subsample':Real(.65,1.0),'model__max_features':Real(.4,1.0)}),
      'MLP': (TransformedTargetRegressor(regressor=MLPArchitectureRegressor(
          random_state=seed), transformer=StandardScaler()),
          {'model__regressor__architecture_id':Integer(0,5),
           'model__regressor__alpha':log(1e-3,1e2)}),
      'RF': (RandomForestRegressor(random_state=seed,n_jobs=1),
          {'model__n_estimators':Integer(100,400),'model__max_depth':Integer(2,12),
           'model__min_samples_leaf':Integer(2,20),'model__max_features':Real(.35,1.0)}),
      'SVR': (SVR(kernel='rbf'),
          {'model__C':log(.01,100),'model__epsilon':log(.005,.5),
           'model__gamma':log(1e-4,1.0)})
    }
    try:
        from catboost import CatBoostRegressor
        specs['CatBoost']=(CatBoostRegressor(verbose=False,random_seed=seed,
            thread_count=1,loss_function='RMSE',allow_writing_files=False),
            {'model__iterations':Integer(50,400),'model__depth':Integer(2,6),
             'model__learning_rate':log(.008,.2),'model__l2_leaf_reg':log(1,50),
             'model__random_strength':Real(0.1,5.0)})
    except ImportError:
        pass
    return specs

def evaluate(X,y,names,outer_folds=5,outer_repeats=3,inner_folds=3,
             n_iter=30,seed=SEED,tag='main',save_models=True,search_method='bayes'):
    if search_method!='bayes':
        raise ValueError('V2 supports --search bayes only')
    specs=model_specs(seed)
    missing=set(names)-set(specs)
    if missing:
        raise ValueError(f'Models unavailable: {sorted(missing)}; install catboost if needed')
    folder=OUT/'models'/tag
    if save_models:
        folder.mkdir(parents=True,exist_ok=True)
    folds=list(RepeatedKFold(n_splits=outer_folds,n_repeats=outer_repeats,
                            random_state=seed).split(X,y))
    records=[]; predictions=[]; parameters=[]
    for fold,(tr,te) in enumerate(folds):
        for name in names:
            estimator,space=specs[name]
            pipeline=Pipeline([('preprocess',prep(X)),('model',estimator)])
            if space:
                cv=KFold(n_splits=inner_folds,shuffle=True,random_state=seed+fold)
                search=BayesSearchCV(pipeline,space,n_iter=n_iter,
                    scoring='neg_mean_absolute_error',cv=cv,n_jobs=1,
                    random_state=seed+fold,refit=True,error_score='raise')
                search.fit(X.iloc[tr],y.iloc[tr])
                fitted=search.best_estimator_
                best=search.best_params_
                inner_mae=-float(search.best_score_)
            else:
                fitted=pipeline.fit(X.iloc[tr],y.iloc[tr]);best={};inner_mae=np.nan
            pred=np.asarray(fitted.predict(X.iloc[te]),dtype=float)
            score=metric(y.iloc[te],pred)
            records.append({'tag':tag,'fold':fold,'repeat':fold//outer_folds,
                'model':name,'train_n':len(tr),'test_n':len(te),
                'inner_selected_mae':inner_mae,**score})
            parameters.append({'fold':fold,'model':name,'best_params':json.dumps(best,default=str)})
            predictions.extend({'fold':fold,'model':name,'row_id':int(i),
                'observed':float(y.iloc[i]),'predicted':float(p)} for i,p in zip(te,pred))
            if save_models:
                joblib.dump({'pipeline':fitted,'train_indices':tr,'test_indices':te,
                    'features':list(X.columns),'target_test':y.iloc[te].to_numpy()},
                    folder/f'{name}_fold{fold}.joblib')
            print(f'{tag} fold={fold+1}/{len(folds)} model={name} MAE={score["mae"]:.4f}',flush=True)
    result=pd.DataFrame(records)
    result.to_csv(OUT/f'{tag}_fold_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/f'{tag}_oof_predictions.csv',index=False)
    pd.DataFrame(parameters).to_csv(OUT/f'{tag}_best_params.csv',index=False)
    result.groupby('model')[['mae','rmse','r2']].agg(['mean','std']).to_csv(OUT/f'{tag}_summary.csv')
    return result

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--gpa-policy',choices=['error','exclude'],default='error')
    a.add_argument('--models',nargs='+',default=DEFAULT_MODELS)
    a.add_argument('--outer-folds',type=int,default=5)
    a.add_argument('--outer-repeats',type=int,default=3)
    a.add_argument('--inner-folds',type=int,default=3)
    a.add_argument('--n-iter',type=int,default=30)
    a.add_argument('--search',choices=['bayes'],default='bayes')
    a.add_argument('--tag',default='main')
    a.add_argument('--seed',type=int,default=SEED)
    args=a.parse_args()
    X,y,_,_=load(args.gpa_policy)
    if len(y)!=184:
        print(f'WARNING: n={len(y)}, expected 184 from adjudicated dataset',flush=True)
    # prevent accidentally mixing model files and summaries from different runs
    if (OUT/f'{args.tag}_fold_metrics.csv').exists() or (OUT/'models'/args.tag).exists():
        raise FileExistsError(f'Tag {args.tag} already exists. Choose a new --tag or move the old outputs.')
    save_json(OUT/f'{args.tag}_settings.json',{**vars(args),'n':len(y),
        'sha256_dataset':hashlib.sha256(DATA.read_bytes()).hexdigest()})
    versions={}
    for package in ['numpy','pandas','scikit-learn','scipy','scikit-optimize','catboost','joblib','openpyxl']:
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:versions[package]='not installed'
    save_json(OUT/f'{args.tag}_software_versions.json',versions)
    evaluate(X,y,args.models,args.outer_folds,args.outer_repeats,args.inner_folds,
             args.n_iter,seed=args.seed,tag=args.tag,search_method=args.search)
if __name__=='__main__':main()

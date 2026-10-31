"""Preprocessing and estimator smoke test; does not require scikit-optimize."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from common import load, prep, MLPArchitectureRegressor
X,y,groups,_=load('error')
assert X.shape[1]==42
assert set(groups)=={'Demographic','SES','BALLI'}
for name,model in [
    ('Dummy',DummyRegressor()),
    ('GB',GradientBoostingRegressor(n_estimators=20,max_depth=1,min_samples_leaf=5,random_state=1)),
    ('MLP',TransformedTargetRegressor(regressor=MLPArchitectureRegressor(
         max_iter=50,alpha=1.0),transformer=StandardScaler()))]:
    pipeline=Pipeline([('preprocess',prep(X)),('model',model)])
    pipeline.fit(X.iloc[:130],y.iloc[:130])
    pred=pipeline.predict(X.iloc[130:])
    assert len(pred)==len(y)-130 and np.isfinite(pred).all()
    print(name,'OK',len(pred),'holdout predictions')
print('SELFTEST OK (this is not a model-performance evaluation)')

"""Extra Trees factory. Real-data model fitting is a later milestone."""
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from .preprocessing import make_preprocessor

SEARCH_SPACE = {
    'model__n_estimators': [100,200,500],
    'model__max_depth': [None,10,20,30],
    'model__min_samples_split': [2,5,10],
    'model__min_samples_leaf': [1,2,4],
    'model__max_features': ['sqrt','log2',None],
}


def make_model(seed=42, **parameters):
    estimator=ExtraTreesRegressor(n_estimators=100,random_state=seed,n_jobs=-1,bootstrap=False)
    pipeline=Pipeline([('preprocessing',make_preprocessor(scale=False)),('model',estimator)])
    if parameters: pipeline.set_params(**parameters)
    return pipeline

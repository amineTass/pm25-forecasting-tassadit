"""Fit this pipeline inside every training fold. Never pre-impute the entire CSV."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data import BASE_FEATURES

FEATURES = [c+'_lag1' for c in BASE_FEATURES] + [
    'wind_speed_lag1', 'dayofyear_sin', 'dayofyear_cos',
    'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'pm25_lag7',
    'pm25_past3_mean', 'pm25_past3_std', 'pm25_past7_mean', 'pm25_past7_std']
GASES = ['NO2_lag1', 'CO_lag1', 'O3_lag1']


def make_preprocessor(scale=False):
    # Explicit indicators remain present even if a gas is complete in one train fold.
    from sklearn.impute import MissingIndicator
    other = [c for c in FEATURES if c not in GASES]
    median = SimpleImputer(strategy='median', keep_empty_features=True)
    columns = ColumnTransformer([
        ('other', median, other),
        ('gases', SimpleImputer(strategy='median', keep_empty_features=True), GASES),
        ('gas_missing', MissingIndicator(features='all'), GASES)
    ], remainder='drop', verbose_feature_names_out=False)
    steps = [('columns', columns)]
    if scale:
        steps.append(('scaler', StandardScaler()))
    return Pipeline(steps)

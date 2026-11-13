"""Freeze positional folds, with row IDs as the durable file interface."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, KFold


def save_folds(frame, output, config):
    if not frame.row_id.is_unique or frame.date.isna().any():
        raise ValueError('Fold generation requires unique row IDs and valid dates.')
    if frame.date.max() >= pd.Timestamp(config['test_start']):
        raise ValueError('Future test rows must not enter development folds.')
    calendar = pd.date_range(frame.date.min(), frame.date.max(), freq='D')
    folds = {'temporal': []}
    gap = int(config['temporal_gap_days'])
    for train_days, validation_days in TimeSeriesSplit(n_splits=config['n_splits'], gap=gap).split(calendar):
        tr = np.flatnonzero(frame.date.isin(calendar[train_days]).to_numpy())
        va = np.flatnonzero(frame.date.isin(calendar[validation_days]).to_numpy())
        if not len(tr) or not len(va):
            raise ValueError('A temporal calendar block has no observations. Review the protocol explicitly.')
        if frame.iloc[tr].date.max()+pd.Timedelta(days=gap) >= frame.iloc[va].date.min():
            raise ValueError('Temporal embargo violation.')
        folds['temporal'].append((tr, va))
    folds['spatial'] = list(GroupKFold(n_splits=config['n_splits']).split(frame, groups=frame.station_id))
    folds['spatial_grid'] = list(GroupKFold(n_splits=config['n_splits']).split(frame, groups=frame.id))
    folds['standard'] = list(KFold(n_splits=config['n_splits'], shuffle=True, random_state=config['seed']).split(frame))
    overview=[]
    for scheme, parts in folds.items():
        payload=[]
        for number, (tr, va) in enumerate(parts, 1):
            train=frame.iloc[tr]; validation=frame.iloc[va]
            if set(train.row_id) & set(validation.row_id):
                raise ValueError('Training and validation row overlap.')
            grouping = 'station_id' if scheme=='spatial' else ('id' if scheme=='spatial_grid' else None)
            if grouping and set(train[grouping]) & set(validation[grouping]):
                raise ValueError('Spatial grouping violation.')
            payload.append(dict(fold=number,train_row_ids=train.row_id.tolist(),validation_row_ids=validation.row_id.tolist()))
            overview.append(dict(scheme=scheme,fold=number,train_rows=len(train),validation_rows=len(validation),
                                 train_start=str(train.date.min().date()),train_end=str(train.date.max().date()),
                                 validation_start=str(validation.date.min().date()),validation_end=str(validation.date.max().date())))
        (output/f'folds_{scheme}.json').write_text(json.dumps(payload),encoding='utf-8')
    pd.DataFrame(overview).to_csv(output/'fold_overview.csv',index=False)
    return folds

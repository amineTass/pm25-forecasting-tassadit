"""One command reproduces the first milestone: python -m src.run."""
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .data import ROOT, BASE_FEATURES, prepare
from .eda import explore
from .splits import save_folds
from .preprocessing import FEATURES, GASES, make_preprocessor
from .model import SEARCH_SPACE


def run():
    started=time.perf_counter()
    config=json.loads((ROOT/'config/project.json').read_text())
    output=ROOT/'results'/config['group'];output.mkdir(parents=True,exist_ok=True)
    (output/'figures').mkdir(exist_ok=True)
    processed=ROOT/'data/processed';processed.mkdir(parents=True,exist_ok=True)
    print('1/6 Source audit and causal feature construction',flush=True)
    data=prepare(config)
    data['stations'].to_csv(output/'station_mapping.csv',index=False)
    data['source_profile'].to_csv(output/'source_missingness.csv',index=False)
    pd.DataFrame(data['summary']['inventory']).to_csv(output/'source_inventory.csv',index=False)
    (output/'audit.json').write_text(json.dumps(data['summary'],indent=2),encoding='utf-8')
    (output/'join_sensitivity.json').write_text(json.dumps(data['summary']['join_sensitivity'],indent=2),encoding='utf-8')
    data['development'].to_csv(processed/'development_forecast.csv',index=False)
    data['holdout'].to_csv(processed/'future_test_reserved.csv',index=False)
    print('2/6 Development EDA and station-wise temporal diagnostics',flush=True)
    info=explore(data,output,config)
    print('3/6 Freeze temporal, station, grid and random partitions',flush=True)
    development=data['development']
    folds=save_folds(development,output,config)
    print('4/6 Verify source alignment and fold-fitted preprocessing',flush=True)
    checks=[]
    for number,(train,validation) in enumerate(folds['temporal'],1):
        pipeline=make_preprocessor(scale=False)
        tr=development.iloc[train];va=development.iloc[validation]
        x=pipeline.fit_transform(tr[FEATURES]);v=pipeline.transform(va[FEATURES])
        assert np.isfinite(x).all() and np.isfinite(v).all()
        assert x.shape[1]==v.shape[1]==len(FEATURES)+3
        empty=[c for c in FEATURES if tr[c].isna().all()]
        checks.append(dict(fold=number,train_rows=len(x),validation_rows=len(v),transformed_columns=x.shape[1],
                           all_finite=True,empty_training_features=';'.join(empty),scaling='disabled for Extra Trees'))
    pd.DataFrame(checks).to_csv(output/'preprocessing_fold_checks.csv',index=False)
    feature_missing=development[FEATURES].isna().agg(['sum','mean']).T.rename(columns={'sum':'missing_count','mean':'missing_fraction'})
    feature_missing.index.name='feature';feature_missing.to_csv(output/'forecast_feature_missingness.csv')
    future=data['forecast']
    assert ((future.date-future.feature_date).dt.days==1).all()
    ground=data['mapped'].set_index(['station_id','date']).pm25
    for lag in [1,2,3,7]:
        past=[]
        for step in range(1,lag+1):
            values=ground.reindex(pd.MultiIndex.from_arrays([future.station_id,future.date-pd.Timedelta(days=step)])).to_numpy()
            past.append(values)
        expected=past[-1].copy()
        expected[~np.isfinite(np.stack(past)).all(axis=0)]=np.nan
        assert np.allclose(expected,future[f'pm25_lag{lag}'],equal_nan=True)
    gee=data['gee'].set_index(['id','date'])
    key=pd.MultiIndex.from_arrays([future.id,future.feature_date])
    for feature in BASE_FEATURES:
        assert np.allclose(gee.reindex(key)[feature].to_numpy(),future[feature+'_lag1'],equal_nan=True)
    assert not set(['id','station_id','pm25','latitude','longitude']) & set(FEATURES)
    assert not set(development.row_id)&set(data['holdout'].row_id)
    validation=dict(status='PASS',checks=checks,causal_features='All atmospheric input dates equal target date minus one calendar day',
                    lag_values='Exact site/date values; each lag k additionally requires every preceding day 1..k',
                    holdout='No future-test row in development folds; no future-test performance or EDA distributions computed',
                    spatial='Primary station-ID GroupKFold; additional grid-ID grouped diagnostic',
                    supplied_provenance='All supplied rows found in raw tables, target and predictor values checked')
    (output/'validation_checks.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    (output/'extra_trees_plan.json').write_text(json.dumps(dict(model='ExtraTreesRegressor',status='Factory implemented; real-data training not run',
                   search_space=SEARCH_SPACE,seed=42,bootstrap=False,scaling=False,
                   tuning='Inner training partitions only; never outer validation or future test',max_configurations=50,
                   robustness='Keep top three distinct configurations; perturb key settings',baselines=['Training mean','Persistence on observed lag1 rows; report coverage']),indent=2),encoding='utf-8')
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','matplotlib','seaborn','statsmodels','reportlab']}
    code=hashlib.sha256()
    for source in sorted((ROOT/'src').glob('*.py')):code.update(source.name.encode());code.update(source.read_bytes())
    metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),stage='Data exploration and preprocessing',model='ExtraTreesRegressor',
                  seed=config['seed'],python=sys.version.split()[0],platform=platform.platform(),machine=platform.machine(),logical_cpus=os.cpu_count(),
                  configuration=json.dumps(config,sort_keys=True),config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                  code_sha256=code.hexdigest(),archive_sha256=data['summary']['archive_sha256'],versions=json.dumps(versions),runtime_seconds=time.perf_counter()-started)
    pd.DataFrame([metadata]).to_csv(output/'experiments.csv',index=False)
    (output/'feature_manifest.json').write_text(json.dumps(dict(features=FEATURES,feature_count_before_indicators=len(FEATURES),transformed_feature_count=len(FEATURES)+3,
                  excluded=['station_id','sensor_name','id','latitude','longitude','date','datetime_utc','feature_date','pm25'],
                  source='Atmospheric predictors from the independent GEE table at t-1; target history strictly before t',
                  target='pm25 on target day t',horizon_days=1,lag_gap_policy='Require complete daily target history back to each requested lag',
                  gas_imputation='Training-fold median plus three explicit indicators; zero fallback only for all-empty training columns',scaling=False,
                  availability_assumption='Daily t-1 atmospheric inputs and past ground observations available at issue time; publication latency not verified'),indent=2),encoding='utf-8')
    manifest=dict(source_archive_sha256=data['summary']['archive_sha256'],development_csv_sha256=hashlib.sha256((processed/'development_forecast.csv').read_bytes()).hexdigest(),
                  rows=len(development),features=FEATURES,folds={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('folds_*.json'))})
    (output/'fold_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    consolidated=ROOT/'results/consolidated';consolidated.mkdir(exist_ok=True)
    for name in ['Exploratory_Data.csv','Exploratory_temporal.csv','experiments.csv']:
        table=pd.read_csv(output/name);table.insert(0,'group',config['group']);table.to_csv(consolidated/name,index=False)
    print('5/6 Generate the English report from these results',flush=True)
    from .report import make_report
    make_report(data['summary'],info,output,config)
    print('6/6 Completed. See reports/ and results/'+config['group'],flush=True)
    print(json.dumps(data['summary'],indent=2))


if __name__=='__main__':run()

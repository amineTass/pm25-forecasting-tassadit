"""Regression tests for leakage, calendar gaps and the assigned model factory."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import target_history
from src.preprocessing import FEATURES,GASES,make_preprocessor
from src.model import make_model
from src.splits import save_folds


class ProtocolTests(unittest.TestCase):
    def ground(self):
        return pd.DataFrame({'station_id':['A']*4+['B']*4,'date':pd.to_datetime(['2024-01-01','2024-01-02','2024-01-04','2024-01-05']*2),'pm25':[1.,2.,4.,5.,101.,102.,104.,105.]})

    def test_gap_masks_lag_and_incomplete_rolling_window(self):
        f=pd.DataFrame({'station_id':['A'],'date':pd.to_datetime(['2024-01-06'])})
        x=target_history(f,self.ground()).iloc[0]
        self.assertEqual(x.pm25_lag1,5.)
        self.assertEqual(x.pm25_lag2,4.)
        self.assertTrue(np.isnan(x.pm25_lag3))
        self.assertTrue(np.isnan(x.pm25_past3_mean))

    def test_stations_never_mix(self):
        f=pd.DataFrame({'station_id':['A','B'],'date':pd.to_datetime(['2024-01-06']*2)})
        self.assertEqual(target_history(f,self.ground()).pm25_lag1.tolist(),[5.,105.])

    def test_target_at_prediction_day_does_not_change_inputs(self):
        ground=self.ground();f=pd.DataFrame({'station_id':['A'],'date':pd.to_datetime(['2024-01-05'])})
        before=target_history(f,ground).filter(regex='^pm25_')
        ground.loc[ground.date.ge(pd.Timestamp('2024-01-05')),'pm25']=400.
        after=target_history(f,ground).filter(regex='^pm25_')
        pd.testing.assert_frame_equal(before,after)

    def test_complete_history_statistics(self):
        g=pd.DataFrame({'station_id':['A']*7,'date':pd.date_range('2024-01-01',periods=7),'pm25':np.arange(1.,8.)})
        f=pd.DataFrame({'station_id':['A'],'date':[pd.Timestamp('2024-01-08')]})
        row=target_history(f,g).iloc[0]
        self.assertEqual(row.pm25_lag7,1.)
        self.assertEqual(row.pm25_past7_mean,4.)
        self.assertAlmostEqual(row.pm25_past7_std,np.std(np.arange(1.,8.),ddof=1))

    def frame(self):
        dates=pd.date_range('2024-01-01',periods=80)
        records=[dict(row_id=i*6+j,date=d,station_id=f'S{j}',id=j//2) for i,d in enumerate(dates) for j in range(6)]
        f=pd.DataFrame(records);f.index=np.arange(len(f))*3+200
        return f

    def test_folds_work_with_nondefault_index_and_respect_groups(self):
        f=self.frame();cfg=dict(test_start='2025-01-01',n_splits=3,temporal_gap_days=1,seed=42)
        with tempfile.TemporaryDirectory() as d:
            folds=save_folds(f,Path(d),cfg)
            for tr,va in folds['temporal']:
                self.assertLess(f.iloc[tr].date.max()+pd.Timedelta(days=1),f.iloc[va].date.min())
            for kind,key in [('spatial','station_id'),('spatial_grid','id')]:
                for tr,va in folds[kind]:self.assertTrue(set(f.iloc[tr][key]).isdisjoint(f.iloc[va][key]))

    def test_future_test_rows_rejected(self):
        f=self.frame();f.loc[f.index[-1],'date']=pd.Timestamp('2025-01-01')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):save_folds(f,Path(d),dict(test_start='2025-01-01',n_splits=3,temporal_gap_days=1,seed=42))

    def test_training_only_imputation_and_explicit_gas_indicators(self):
        train=pd.DataFrame({c:[1.,3.,np.nan] for c in FEATURES})
        for c in GASES:train[c]=np.nan
        prep=make_preprocessor();prep.fit(train)
        imputer=prep.named_steps['columns'].named_transformers_['other']
        self.assertTrue(np.allclose(imputer.statistics_,2.))
        validation=pd.DataFrame({c:[np.nan] for c in FEATURES})
        a=prep.transform(validation)
        self.assertEqual(a.shape,(1,len(FEATURES)+3))
        self.assertTrue(np.isfinite(a).all());self.assertEqual(a[0,-3:].tolist(),[1.,1.,1.])
        self.assertTrue(np.allclose(imputer.statistics_,2.))

    def test_extra_trees_factory_without_scaling_and_reproducible(self):
        frame=pd.DataFrame({c:np.arange(8,dtype=float) for c in FEATURES})
        y=np.arange(8,dtype=float)
        a=make_model(model__n_estimators=5,model__n_jobs=1)
        b=make_model(model__n_estimators=5,model__n_jobs=1)
        self.assertNotIn('scaler',a.named_steps['preprocessing'].named_steps)
        self.assertFalse(a.named_steps['model'].bootstrap)
        np.testing.assert_allclose(a.fit(frame,y).predict(frame),b.fit(frame,y).predict(frame))


if __name__=='__main__':unittest.main()

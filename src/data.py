"""Source validation, fixed station matching, and causal daily features."""
from pathlib import Path
import hashlib
import os
from tempfile import NamedTemporaryFile
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
BASE_FEATURES = ['AOD', 'NO2', 'CO', 'O3', 'temperature_celsius', 'pressure_mb', 'wind_u', 'wind_v']


def target_history(forecasting, mapped):
    """Exact calendar lags require an uninterrupted history back to the requested day."""
    result = forecasting.copy()
    source = mapped.set_index(['station_id', 'date']).pm25
    if not source.index.is_unique:
        raise ValueError('Target history requires unique physical-site/day keys.')
    history = {}
    for lag in range(1, 8):
        key = pd.MultiIndex.from_arrays([result.station_id, result.date-pd.Timedelta(days=lag)])
        history[lag] = source.reindex(key).to_numpy()
    block = pd.DataFrame(history, index=result.index)
    for lag in [1, 2, 3, 7]:
        uninterrupted = block[list(range(1, lag+1))].notna().all(axis=1)
        result[f'pm25_lag{lag}'] = block[lag].where(uninterrupted)
    for window in [3, 7]:
        values = block[list(range(1, window+1))]
        complete = values.notna().all(axis=1)
        result[f'pm25_past{window}_mean'] = values.mean(axis=1).where(complete)
        result[f'pm25_past{window}_std'] = values.std(axis=1).where(complete)
    return result


def verify_archive(path, expected):
    if not zipfile.is_zipfile(path):
        raise ValueError('Expected a ZIP archive. Save Kaggle dataset version 1 as data/raw/dataset.zip.')
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Source archive differs from the audited version. Replace data/raw/dataset.zip with Kaggle dataset version 1.')


def download(config):
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / 'dataset.zip'
    expected = config.get('dataset_archive_sha256')
    if path.exists():
        verify_archive(path, expected)
        return path

    url = f"https://www.kaggle.com/api/v1/datasets/download/{config['dataset']}?datasetVersionNumber={config['dataset_version']}"
    with NamedTemporaryFile(dir=raw, suffix='.download', delete=False) as temporary:
        candidate = Path(temporary.name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            candidate.write_bytes(response.read())
        verify_archive(candidate, expected)
        os.replace(candidate, path)
    except OSError as error:
        raise RuntimeError('Could not download Kaggle dataset version 1. Save it manually as data/raw/dataset.zip.') from error
    finally:
        candidate.unlink(missing_ok=True)
    return path


def prepare(config):
    if config['forecast_horizon_days'] != 1:
        raise ValueError('This verified protocol supports a one-day horizon only.')
    path = download(config)
    with zipfile.ZipFile(path) as z:
        pm = pd.read_csv(z.open('pm25.csv'))
        gee = pd.read_csv(z.open('gee_features_daily.csv'))
        supplied = pd.read_csv(z.open('combined_gee_pm25.csv'))
    inventory = []
    source_profile = []
    for name, frame in [('pm25.csv', pm), ('gee_features_daily.csv', gee), ('combined_gee_pm25.csv', supplied)]:
        inventory.append(dict(file=name, rows=len(frame), columns=len(frame.columns),
                              exact_duplicates=int(frame.duplicated().sum()),
                              date_min=frame.date.min(), date_max=frame.date.max()))
        for column in frame.columns:
            source_profile.append(dict(file=name, variable=column, dtype=str(frame[column].dtype),
                                       missing_count=int(frame[column].isna().sum()),
                                       missing_pct=float(frame[column].isna().mean()*100)))
        frame['date'] = pd.to_datetime(frame.date, errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
        if frame.date.isna().any():
            raise ValueError(f'{name}: invalid dates; explicit resolution required.')
        if 'datetime_utc' in frame:
            timestamps=pd.to_datetime(frame.datetime_utc, errors='coerce', utc=True)
            if timestamps.isna().any() or not timestamps.dt.tz_localize(None).dt.normalize().eq(frame.date).all():
                raise ValueError(f'{name}: datetime_utc disagrees with date.')
    if not pm.latitude.between(-90,90).all() or not pm.longitude.between(-180,180).all():
        raise ValueError('Invalid ground coordinates.')
    if not gee.lat.between(-90,90).all() or not gee.lon.between(-180,180).all():
        raise ValueError('Invalid GEE coordinates.')
    keys = ['sensor_name', 'latitude', 'longitude']
    if pm.duplicated(['date'] + keys).any() or gee.duplicated(['date', 'id']).any():
        raise ValueError('Duplicate station-day or GEE location-day; resolve explicitly before proceeding.')
    grid = gee[['id', 'lat', 'lon']].drop_duplicates().sort_values('id').reset_index(drop=True)
    if grid.id.duplicated().any():
        raise ValueError('A GEE ID changes coordinates; inspect the metadata.')
    stations = pm[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    # Sensor labels can be reused or coordinates can change: keep physical sites separate.
    stations['station_id'] = [f'S{i:03d}' for i in range(len(stations))]
    distances = cdist(stations[['latitude', 'longitude']], grid[['lat', 'lon']])
    nearest = distances.argmin(axis=1)
    stations['id'] = grid.iloc[nearest].id.to_numpy()
    stations['distance_degrees'] = distances.min(axis=1)
    stations['accepted'] = stations.distance_degrees.le(config['spatial_threshold_degrees'])
    # Mapping is fixed per physical station, not rematched daily according to availability.
    pm = pm.merge(stations, on=keys, validate='many_to_one')
    n_missing = int(pm.pm25.isna().sum())
    n_low = int(pm.pm25.le(config['target_min_exclusive']).sum())
    n_high = int(pm.pm25.ge(config['target_max_exclusive']).sum())
    valid = pm[pm.pm25.gt(config['target_min_exclusive']) & pm.pm25.lt(config['target_max_exclusive'])].copy()
    mapped = valid[valid.accepted].copy()
    same_day = mapped.merge(gee, on=['date', 'id'], how='inner', validate='many_to_one')
    supplied_check = supplied.merge(gee, on=['date', 'id'], suffixes=('_provided', '_raw'), validate='one_to_one')
    # Ground target and predictor checks establish provenance; this does not prove instrument validity.
    target_check = supplied.merge(pm[['date'] + keys + ['pm25']], on=['date'] + keys, suffixes=('_provided', '_raw'), validate='one_to_one')
    discrepancies = {c: int((~np.isclose(supplied_check[c+'_provided'], supplied_check[c+'_raw'], rtol=1e-9, atol=1e-12, equal_nan=True)).sum()) for c in BASE_FEATURES}
    source_checks = dict(provided_rows=len(supplied), target_rows_found=len(target_check),
                         target_mismatches=int((~np.isclose(target_check.pm25_provided, target_check.pm25_raw, rtol=1e-9, atol=1e-12)).sum()),
                         predictor_mismatches=discrepancies,
                         datetime_utc_agrees_with_date=True,
                         provided_max_grid_distance_degrees=float(np.hypot(supplied_check.latitude-supplied_check.lat,
                                                                           supplied_check.longitude-supplied_check.lon).max()))
    if len(supplied_check) != len(supplied) or len(target_check) != len(supplied) or source_checks['target_mismatches'] or any(discrepancies.values()):
        raise ValueError('Supplied combined rows disagree with raw sources. Inspect before analysis.')
    provided_valid = supplied[supplied.pm25.gt(config['target_min_exclusive']) & supplied.pm25.lt(config['target_max_exclusive'])]
    exact_overlap = provided_valid.merge(same_day[['date']+keys], on=['date']+keys, how='inner', validate='one_to_one')
    join_sensitivity = dict(provided_valid_rows=len(provided_valid), reconstructed_valid_rows=len(same_day),
                            exact_valid_rows_retained=len(exact_overlap),
                            additional_valid_same_day_rows=len(same_day)-len(exact_overlap),
                            exact_valid_rows_not_retained=len(provided_valid)-len(exact_overlap),
                            primary_rule='Assignment section 5.3.4: date and fixed nearest coordinates within 0.25 degrees')
    # Forecast target at day t; all atmospheric inputs come from day t-1.
    past_gee = gee.drop(columns=['datetime_utc', 'lat', 'lon']).rename(columns={c:c+'_lag1' for c in BASE_FEATURES})
    past_gee['feature_date'] = past_gee['date']
    past_gee['date'] += pd.Timedelta(days=config['forecast_horizon_days'])
    forecasting = mapped.merge(past_gee, on=['date', 'id'], how='inner', validate='many_to_one')
    forecasting = target_history(forecasting, mapped)
    forecasting['wind_speed_lag1'] = np.hypot(forecasting.wind_u_lag1, forecasting.wind_v_lag1)
    day = forecasting.date.dt.dayofyear
    days_in_year = np.where(forecasting.date.dt.is_leap_year, 366, 365)
    forecasting['dayofyear_sin'] = np.sin(2*np.pi*(day-1)/days_in_year)
    forecasting['dayofyear_cos'] = np.cos(2*np.pi*(day-1)/days_in_year)
    forecasting = forecasting.sort_values(['date', 'station_id']).reset_index(drop=True)
    forecasting.insert(0, 'row_id', np.arange(len(forecasting)))
    cutoff = pd.Timestamp(config['test_start'])
    development = forecasting[forecasting.date.lt(cutoff)].copy()
    holdout = forecasting[forecasting.date.ge(cutoff)].copy()
    eda = same_day[same_day.date.lt(cutoff)].copy()
    # The audit covers structural metadata globally; label-based EDA uses development only.
    summary = dict(inventory=inventory, archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   missing_target=n_missing, invalid_low=n_low, invalid_high=n_high,
                   valid_target_rows=len(valid), physical_stations=len(stations), sensor_names=int(pm.sensor_name.nunique()),
                   mapped_stations=int(stations.accepted.sum()), unmapped_valid_target_rows=int((~valid.accepted).sum()),
                   valid_mapped_target_rows=len(mapped), same_day_matched_rows=len(same_day),
                   missing_same_day_gee=len(mapped)-len(same_day), forecast_rows=len(forecasting),
                   development_rows=len(development), holdout_rows=len(holdout),
                   eda_rows=len(eda), eda_stations=int(eda.station_id.nunique()),
                   source_checks=source_checks, join_sensitivity=join_sensitivity, test_start=config['test_start'])
    return dict(pm=pm, gee=gee, supplied=supplied, stations=stations, mapped=mapped,
                same_day=same_day, eda=eda, forecast=forecasting, development=development,
                holdout=holdout, summary=summary, source_profile=pd.DataFrame(source_profile))

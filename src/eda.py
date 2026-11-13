"""Development-only descriptive analyses; no test-label exploration."""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from .data import BASE_FEATURES

sns.set_theme(style='whitegrid', context='notebook', palette=['#2377a5', '#d27c49', '#4b9e90'])
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 160, 'axes.spines.top': False, 'axes.spines.right': False})


def savefig(output, name):
    plt.tight_layout()
    plt.savefig(output/'figures'/f'{name}.png', bbox_inches='tight')
    plt.close()


def longest_run(series):
    groups = series.isna().cumsum()
    valid = series.dropna()
    if valid.empty:
        return valid
    chunks = [part for _, part in valid.groupby(groups.loc[valid.index])]
    return max(chunks, key=len)


def explore(data, output, config):
    frame = data['eda'].copy()
    nums = BASE_FEATURES + ['pm25']
    table = frame[nums].describe().T.rename(columns={'25%':'q25','50%':'median','75%':'q75'})
    table['skewness'] = frame[nums].skew()
    table['excess_kurtosis'] = frame[nums].kurt()
    table['missing_count'] = frame[nums].isna().sum()
    table['missing_pct'] = frame[nums].isna().mean()*100
    table['scope'] = 'development_same_day_matched'
    table.index.name = 'variable'
    table.reset_index().to_csv(output/'Exploratory_Data.csv', index=False)
    pd.DataFrame([{'stage':k, 'count':data['summary'][k]} for k in [
        'missing_target','invalid_low','invalid_high','valid_target_rows','unmapped_valid_target_rows',
        'valid_mapped_target_rows','missing_same_day_gee','same_day_matched_rows','forecast_rows','development_rows','holdout_rows']
    ]).to_csv(output/'cleaning_counts.csv', index=False)
    # Required summaries and counts share one canonical EDA file (long, typed rows).
    numeric = table.reset_index().assign(record_type='numeric_summary')
    counts = pd.read_csv(output/'cleaning_counts.csv').rename(columns={'stage':'variable','count':'count'}).assign(record_type='cleaning_count', scope='global_quality_audit')
    pd.concat([numeric, counts], ignore_index=True).to_csv(output/'Exploratory_Data.csv', index=False)
    gases = ['NO2','CO','O3']
    frame['month'] = frame.date.dt.to_period('M').astype(str)
    frame['year'] = frame.date.dt.year
    frame['month_number'] = frame.date.dt.month
    frame['wind_speed'] = np.hypot(frame.wind_u, frame.wind_v)
    frame['season'] = frame.month_number.map({12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'})
    frame.groupby('year').pm25.agg(count='size',median='median',q25=lambda s:s.quantile(.25),q75=lambda s:s.quantile(.75),
                                  above25_pct=lambda s:100*s.gt(25).mean(),above50_pct=lambda s:100*s.gt(50).mean()).to_csv(output/'annual_drift.csv')
    frame.groupby('month_number')[nums+['wind_speed']].agg(['count','median','mean','std']).to_csv(output/'seasonal_predictor_summary.csv')
    frame.groupby('station_id').agg(sensor_name=('sensor_name','first'),rows=('pm25','size'),first_date=('date','min'),last_date=('date','max')).to_csv(output/'station_summary.csv')
    conditional=[]
    for family,column in [('season','season'),('station','station_id')]:
        for label,part in frame.groupby(column):
            pair=part[['pm25','AOD']].dropna()
            conditional.append(dict(stratification=family,group=label,rows=len(pair),
                                    spearman=pair.pm25.corr(pair.AOD,method='spearman') if len(pair)>=30 and pair.pm25.nunique()>1 and pair.AOD.nunique()>1 else np.nan))
    for weather in ['temperature_celsius','wind_speed']:
        strata=pd.qcut(frame[weather],4,duplicates='drop')
        for label,part in frame.groupby(strata,observed=True):
            pair=part[['pm25','AOD']].dropna()
            conditional.append(dict(stratification=weather,group=str(label),rows=len(pair),
                                    spearman=pair.pm25.corr(pair.AOD,method='spearman') if len(pair)>=30 and pair.pm25.nunique()>1 and pair.AOD.nunique()>1 else np.nan))
    conditional=pd.DataFrame(conditional)
    conditional.to_csv(output/'aod_conditional_associations.csv',index=False)
    frame.groupby('month')[gases].agg(lambda x:x.isna().mean()*100).to_csv(output/'gas_missingness_by_month.csv')
    frame.groupby('station_id')[gases].agg(lambda x:x.isna().mean()*100).to_csv(output/'gas_missingness_by_station.csv')
    rows = []
    diagnostics = []
    series_by_station = {}
    # Temporal diagnostics use the full mapped ground record, not only days with satellite matches.
    observed = data['mapped'][data['mapped'].date.lt(pd.Timestamp(config['test_start']))]
    for station, part in observed.groupby('station_id'):
        series = part.set_index('date').pm25.sort_index().asfreq('D')
        run = longest_run(series)
        diffs = part.date.sort_values().diff().dt.days
        record = dict(station_id=station, sensor_name=part.sensor_name.iloc[0], observations=len(part),
                      first_date=str(series.index.min().date()), last_date=str(series.index.max().date()),
                      span_days=len(series), missing_days=int(series.isna().sum()), missing_day_pct=100*series.isna().mean(),
                      consecutive_pair_pct=100*diffs.eq(1).sum()/max(len(part)-1,1),
                      longest_contiguous_days=len(run), acf_lag1_calendar=series.autocorr(1) if len(part)>2 else np.nan,
                      acf_lag7_calendar=series.autocorr(7) if len(part)>8 else np.nan)
        if len(run)>=30 and run.nunique()>1:
            try:
                test = adfuller(run.to_numpy(), autolag='AIC')
                lb = acorr_ljungbox(run.to_numpy(), lags=[min(7,len(run)//5)], return_df=True)
                record.update(adf_statistic=test[0], adf_pvalue=test[1], adf_lags=test[2],
                              ljung_box_lag=int(lb.index[0]), ljung_box_pvalue=float(lb.lb_pvalue.iloc[0]))
            except ValueError:
                record['diagnostic_status']='insufficient_variation'
        else:
            record['diagnostic_status']='contiguous_run_under_30_days'
        rows.append(record)
        series_by_station[station] = (series, run)
    temporal = pd.DataFrame(rows).sort_values('longest_contiguous_days', ascending=False)
    temporal['scope']='development_mapped_ground_history'
    temporal.to_csv(output/'Exploratory_temporal.csv', index=False)
    drift = frame.groupby('month').pm25.agg(count='size', median='median', q25=lambda s:s.quantile(.25),
                                         q75=lambda s:s.quantile(.75), above25_pct=lambda s:100*s.gt(25).mean(),
                                         above50_pct=lambda s:100*s.gt(50).mean())
    drift.to_csv(output/'monthly_drift.csv')
    # Separate calendar-aware lead/lag associations, within each station.
    lag_rows = []
    for station, part in frame.groupby('station_id'):
        daily = part.set_index('date')[nums].sort_index().asfreq('D')
        for feature in BASE_FEATURES:
            for lag in [-3,-1,0,1,3,7]:
                shifted = daily[feature].shift(lag)
                pairs = pd.concat([daily.pm25, shifted], axis=1).dropna()
                if len(pairs)>=30 and pairs.iloc[:,0].nunique()>1 and pairs.iloc[:,1].nunique()>1:
                    lag_rows.append(dict(station_id=station,feature=feature,lag_days=lag,pairs=len(pairs),
                                         spearman=pairs.iloc[:,0].corr(pairs.iloc[:,1],method='spearman')))
    pd.DataFrame(lag_rows).to_csv(output/'lead_lag.csv', index=False)
    correlations = {}
    high = []
    for method in ['pearson','spearman']:
        correlations[method] = frame[nums].corr(method=method)
        correlations[method].to_csv(output/f'correlation_{method}.csv')
        for a,b in itertools.combinations(BASE_FEATURES,2):
            r=correlations[method].loc[a,b]
            if abs(r)>.9: high.append(dict(method=method, feature_1=a, feature_2=b, correlation=r))
    pd.DataFrame(high,columns=['method','feature_1','feature_2','correlation']).to_csv(output/'high_correlations.csv', index=False)
    high_pair_count=len(high)
    q1,q3=frame.pm25.quantile([.25,.75]); low=q1-1.5*(q3-q1);upper=q3+1.5*(q3-q1)
    outliers = dict(rule='development IQR only; flag, preserve plausible events',lower=float(low),upper=float(upper),
                    flagged_count=int((frame.pm25.lt(low)|frame.pm25.gt(upper)).sum()))
    # Visuals use the same development population, explicitly labeled in the report.
    fig,axes=plt.subplots(3,3,figsize=(12,8))
    for ax,c in zip(axes.flat,nums):
        ax.hist(frame[c].dropna(),bins=45,color='#2377a5',alpha=.85)
        ax.set_title(c,fontsize=11);ax.set_ylabel('Rows')
    savefig(output,'01_distributions')
    fig,axes=plt.subplots(3,3,figsize=(12,8))
    for ax,c in zip(axes.flat,nums):
        ax.boxplot(frame[c].dropna(),vert=False,flierprops={'markersize':2});ax.set_title(c,fontsize=11)
    savefig(output,'02_boxplots')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(pd.to_datetime(drift.index),drift['median'],color='#2377a5')
    axes[0].fill_between(pd.to_datetime(drift.index),drift.q25,drift.q75,color='#2377a5',alpha=.2)
    axes[0].set(ylabel='PM2.5 (ug/m3)',title='Monthly median and IQR; station mix varies')
    sns.boxplot(data=frame,x='month_number',y='pm25',showfliers=False,ax=axes[1],color='#96c5db')
    axes[1].set(xlabel='Calendar month',ylabel='PM2.5 (ug/m3)',title='Seasonality, pooled sites (outliers hidden)')
    savefig(output,'03_temporal_patterns')
    plt.figure(figsize=(9,7));sns.heatmap(correlations['spearman'],annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='vlag',square=True)
    plt.title('Development data: Spearman correlations');savefig(output,'04_correlations')
    missing=frame.groupby('year')[gases].agg(lambda s:s.isna().mean()*100)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    missing.plot(kind='bar',ax=axes[0]);axes[0].set(ylabel='Missing (%)',title='Gas missingness by year')
    temporal.missing_day_pct.hist(ax=axes[1],bins=20,color='#2377a5')
    axes[1].set(xlabel='Missing calendar days per site (%)',ylabel='Physical sites',title='Ground-series coverage between first/last day')
    savefig(output,'05_missingness')
    best=temporal.station_id.iloc[0];series,run=series_by_station[best]
    lag_count=min(30,len(run)//2-1)
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    axes[0].plot(run.index,run.values,color='#2377a5');axes[0].tick_params(axis='x',rotation=30)
    axes[0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0].set(title=f'{best}: longest contiguous run',ylabel='PM2.5 (ug/m3)')
    for ax,values,title in [(axes[1],acf(run,nlags=lag_count),'ACF (contiguous run)'),(axes[2],pacf(run,nlags=lag_count,method='ywm'),'PACF (contiguous run)')]:
        ax.stem(np.arange(len(values)),values);ax.set(title=title,xlabel='Calendar lag (days)')
        ax.axhline(1.96/np.sqrt(len(run)),color='gray',ls='--');ax.axhline(-1.96/np.sqrt(len(run)),color='gray',ls='--')
    savefig(output,'06_time_diagnostics')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    sites=observed.groupby('station_id').agg(latitude=('latitude','first'),longitude=('longitude','first'),n=('pm25','size'))
    sc=axes[0].scatter(sites.longitude,sites.latitude,c=sites.n,cmap='viridis',s=25)
    axes[0].set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title='Mapped ground sites; no geographic basemap')
    fig.colorbar(sc,ax=axes[0],label='Development target observations')
    observed.groupby('station_id').size().sort_values(ascending=False).head(15).plot(kind='bar',ax=axes[1],color='#2377a5')
    axes[1].set(title='15 most represented physical sites',ylabel='Development observations',xlabel='Site')
    savefig(output,'07_spatial_coverage')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    plots=[]
    for ax,c in zip(axes,['AOD','temperature_celsius','wind_u']):
        image=ax.hexbin(frame[c],frame.pm25,gridsize=35,mincnt=1,cmap='Blues',bins='log')
        plots.append(image)
        ax.set(xlabel=c,ylabel='PM2.5 (ug/m3)',title='Development associations')
    common_max=max(float(plot.get_array().max()) for plot in plots)
    for plot in plots:plot.set_clim(1,common_max)
    fig.subplots_adjust(wspace=.38,right=.86,bottom=.18)
    color_axis=fig.add_axes([.90,.20,.014,.62])
    fig.colorbar(image,cax=color_axis,label='Rows per bin (log scale)')
    fig.savefig(output/'figures/08_relationships.png',dpi=160,bbox_inches='tight');plt.close(fig)
    info = dict(target_median=float(frame.pm25.median()),target_mean=float(frame.pm25.mean()),
                target_skew=float(frame.pm25.skew()),target_p25=float(q1),target_p75=float(q3),
                above25_pct=float(100*frame.pm25.gt(25).mean()),above50_pct=float(100*frame.pm25.gt(50).mean()),
                gas_missing_pct=(frame[gases].isna().mean()*100).to_dict(),
                median_missing_day_pct=float(temporal.missing_day_pct.median()),
                longest_run_station=best,longest_run_days=len(run),
                median_calendar_lag1=float(temporal.acf_lag1_calendar.median()),
                sites_tested_adf=int(temporal.adf_pvalue.notna().sum()),outliers=outliers,
                target_correlations=correlations['spearman'].pm25.drop('pm25').to_dict(),
                high_correlation_pairs=high_pair_count,
                aod_seasonal_associations=conditional[conditional.stratification.eq('season')].to_dict('records'))
    (output/'eda_findings.json').write_text(__import__('json').dumps(info,indent=2),encoding='utf-8')
    return info

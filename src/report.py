"""An evidence-based twelve-page report for the November milestone, not final results."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.pagesizes import A4
from .data import ROOT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

def register_fonts():
    candidates=[Path("/usr/share/fonts/truetype/liberation"), Path("C:/Windows/Fonts"), Path("/System/Library/Fonts/Supplemental")]
    for folder in candidates:
        for regular,bold in [("LiberationSans-Regular.ttf","LiberationSans-Bold.ttf"),("arial.ttf","arialbd.ttf"),("Arial.ttf","Arial Bold.ttf")]:
            if (folder/regular).exists() and (folder/bold).exists():
                pdfmetrics.registerFont(TTFont("ReportRegular",str(folder/regular)))
                pdfmetrics.registerFont(TTFont("ReportBold",str(folder/bold)))
                pdfmetrics.registerFontFamily("ReportRegular",normal="ReportRegular",bold="ReportBold",italic="ReportRegular",boldItalic="ReportBold")
                return "ReportRegular","ReportBold"
    return "Helvetica","Helvetica-Bold"



def make_report(a, e, output, config):
    pdf=ROOT/'reports/PM25_Phase1_Report.pdf'
    regular,bold=register_fonts()
    team=json.loads((ROOT/'config/team.json').read_text())
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCustom',fontName=bold,fontSize=25,leading=30,textColor=colors.HexColor('#16374b'),spaceAfter=18))
    styles.add(ParagraphStyle(name='SectionCustom',fontName=bold,fontSize=17,leading=21,textColor=colors.HexColor('#16374b'),spaceAfter=13))
    styles.add(ParagraphStyle(name='BodyCustom',fontName=regular,fontSize=10,leading=14,spaceAfter=10))
    styles.add(ParagraphStyle(name='SmallCustom',fontName=regular,fontSize=8,leading=11,spaceAfter=7))
    styles.add(ParagraphStyle(name='CaptionCustom',fontName=regular,fontSize=8.5,leading=12,textColor=colors.HexColor('#475e6a'),spaceAfter=12))
    story=[]
    def p(text,style='BodyCustom'):story.append(Paragraph(text,styles[style]))
    def heading(number,text):p(f'{number}. {text}','SectionCustom')
    def image(name,width=475):
        path=output/'figures'/f'{name}.png';w,h=ImageReader(str(path)).getSize()
        story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,10))
    def table(rows,widths=None):
        rows=[[Paragraph(escape(str(c)),styles['SmallCustom']) for c in row] for row in rows]
        t=Table(rows,colWidths=widths or [240,235],hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5f0f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#2377a5')),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8fa')])]))
        story.append(t);story.append(Spacer(1,12))
    def nextpage():story.append(PageBreak())
    p('PM2.5 Forecasting','TitleCustom')
    p('Data exploration and preprocessing<br/>First milestone | 14 November 2026','SectionCustom')
    p('Master 1 mini project | Academic year 2026-2027')
    p('<b>Team:</b> '+', '.join(team['members']))
    p('<b>Assigned model:</b> Extra Trees Regressor (method group G6)')
    p('Prepared from the specified Mediterranean Basin dataset, version 1. This report covers the first scheduled milestone. Model training, performance comparisons, statistical tests and explainability belong to the subsequent milestones and are not claimed here.')
    table([['Milestone','Date in the assignment'],['Data exploration and preprocessing','14 November 2026'],['Training, validation, test and evaluation','12 December 2026'],['Statistical testing and explainability','2 January 2027']])
    p('Study objective','SectionCustom')
    p('Construct an auditable, reproducible dataset for one-day-ahead daily PM2.5 regression. The immediate questions are whether the measurements and metadata support time- and location-aware validation, how missingness and temporal gaps affect usable samples, and which transformations can be applied without using future information.')
    p('<b>Main findings.</b> The three supplied CSV files have different populations. Reconstructing a fixed nearest-grid join recovers observations omitted by the supplied exact-location join. Invalid target values are removed under the assignment rule. Missing calendar days and variable station coverage require explicit date-based lags. Daily inputs are shifted by one day for forecasting; same-day associations remain descriptive.')
    p('<b>Status:</b> Real-data exploration and preprocessing have been executed. Automated temporal, spatial and preprocessing checks pass. This submission covers the November data milestone. The technical folder G6_ExtraTrees identifies the assigned method; it does not invent a formal class group code.')
    nextpage()
    heading(1,'Research questions and Extra Trees')
    p('<b>Main question.</b> How reliably can Extra Trees forecast daily PM2.5 at later dates and locations absent from fitting? Model experiments are pending; this milestone establishes the data and evaluation design needed to answer it.')
    table([['Research question','Evidence at this milestone'],['RQ1. Is the source usable and traceable?','Audit, fixed validity rules, source-value checks and matching counts.'],['RQ2. Can histories support daily forecasting?','Station timelines, calendar gaps, contiguous-run diagnostics and strict history masks.'],['RQ3. How do atmospheric associations vary?','Pearson/Spearman, season/site stratification and weather-dependent AOD associations.'],['RQ4. How reliable and interpretable is the model?','Frozen folds and a declared experiment plan. Performance and importance remain pending.']], [210,265])
    p('<b>Assigned method.</b> Extra Trees combines randomized decision trees and averages their regression estimates. Random candidate thresholds introduce diversity. Trees can capture nonlinear relationships, but usefulness on this dataset must be measured. Complexity controls include tree depth and leaf size. Scaling is generally unnecessary. Our implementation fixes bootstrap=False and seed 42, retaining the prescribed training-fold imputation and gas indicators.')
    p('<b>Related method.</b> Geurts, Ernst and Wehenkel (2006), <i>Extremely randomized trees</i>, Machine Learning 63, 3-42, introduces this ensemble family (DOI 10.1007/s10994-006-6226-1). We use the documented scikit-learn 1.8 implementation. The original method paper is background, not evidence of PM2.5 accuracy or transfer to Algeria.')
    table([['Hyperparameter','Declared candidates from G6'],['n_estimators','100, 200, 500'],['max_depth','None, 10, 20, 30'],['min_samples_split','2, 5, 10'],['min_samples_leaf','1, 2, 4'],['max_features','sqrt, log2, None']], [210,265])
    p('src/model.py provides a tested pipeline factory. The search plan caps configurations at 50 within training partitions, retains three distinct candidates, and checks sensitivity. No real-data model has been fitted for this milestone. The factory smoke test uses synthetic numbers only.','SmallCustom')
    nextpage()
    heading(2,'Source, provenance and variables')
    p('Source: Sarah Ameziane, <i>Air Quality - Mediterranean Basin (2016-2026)</i>, Kaggle, dataset version 1. The publisher describes PM2.5 observations sourced from OpenAQ and atmospheric predictors extracted with Google Earth Engine from ERA5, Sentinel-5P and MODIS. These descriptions are provenance claims, not an independent instrument-quality certification.')
    table([['Source file','Rows / columns / date range']]+[[x['file'],f"{x['rows']:,} / {x['columns']} / {x['date_min']} to {x['date_max']}"] for x in a['inventory']])
    table([['Variable family','Units / interpretation'],['pm25','ug/m3; continuous daily surface target'],['AOD','Dimensionless; aerosol optical depth'],['NO2, CO, O3','mol/m2; atmospheric column quantities, not surface concentrations'],['temperature_celsius','Degrees Celsius; ERA5 temperature'],['pressure_mb','hPa; pressure'],['wind_u, wind_v','m/s; wind vector components'],['date, sensor_name, coordinates, id','Time and grouping/join metadata; excluded as raw predictors']], [160,315])
    p('<b>License.</b> Kaggle labels the license "Other (specified in description)". Its description attributes OpenAQ CC BY 4.0 and Copernicus/NASA open data and asks users to check individual source licenses. A single blanket redistribution license is not established. Raw source data are therefore downloaded by the reproducible code and excluded from the delivery archive.')
    p('<b>Reproducibility.</b> The archive checksum, code checksum, configuration, seed, package versions and UTC execution time are stored in experiments.csv. The initial archive SHA-256 is '+a['archive_sha256']+'.','SmallCustom')
    nextpage()
    heading(3,'Cleaning and station matching')
    table([['Quality step','Observed count'],['Raw PM2.5 rows',f"{a['inventory'][0]['rows']:,}"],['Missing / <=0 / >=500 target',f"{a['missing_target']} / {a['invalid_low']:,} / {a['invalid_high']:,}"],['Valid target rows',f"{a['valid_target_rows']:,}"],['Physical sites / sensor labels',f"{a['physical_stations']} / {a['sensor_names']}"],['Sites within 0.25 degrees',a['mapped_stations']],['Valid targets outside spatial threshold',f"{a['unmapped_valid_target_rows']:,}"],['Valid mapped targets',f"{a['valid_mapped_target_rows']:,}"],['Mapped targets without same-day GEE',f"{a['missing_same_day_gee']:,}"],['Valid reconstructed same-day rows',f"{a['same_day_matched_rows']:,}"]])
    p('A physical site is defined by sensor label and coordinates. A label alone is insufficient because labels can recur at different coordinates. Each site receives a deterministic internal station ID. Every GEE ID must have one stable coordinate pair; station-day and GEE-day keys must be unique. The source files have zero exact duplicate rows. Duplicate key conflicts trigger an explicit error instead of silent aggregation.')
    p('Each physical site is assigned once to its nearest GEE location using Euclidean latitude/longitude distance; mappings beyond 0.25 degrees are rejected. Matching on the same date follows this fixed map. This is an approximate angular threshold, not a distance in kilometres. The mapping does not switch to a different location when a daily source record is absent.')
    p(f"<b>Like-for-like join check.</b> The supplied combined file retains {a['join_sensitivity']['provided_valid_rows']:,} valid target rows after the same validity filter. The reconstructed join retains {a['join_sensitivity']['reconstructed_valid_rows']:,}; {a['join_sensitivity']['additional_valid_same_day_rows']:,} valid rows are additional and {a['join_sensitivity']['exact_valid_rows_not_retained']:,} supplied valid rows are lost. The fixed nearest-coordinate rule follows assignment section 5.3.4. It is fully recorded in join_sensitivity.json.")
    p('The supplied combined file has '+f"{a['source_checks']['provided_rows']:,}"+' rows before target filtering. Its predictor values and target values were checked against their raw sources; results are in audit.json. Rebuilding the join changes the analysis population, so report counts always identify the population used. Matching loss can cause selection bias and must not be interpreted as random missingness.')
    nextpage()
    heading(4,'Target distribution and plausible extremes')
    p(f"Descriptive EDA uses {a['eda_rows']:,} valid same-day matched development rows before {config['test_start']}, covering {a['eda_stations']} physical sites. The reserved future period contributes no distribution plots or model-selection information.")
    image('01_distributions')
    table([['Development target statistic','Value'],['Mean / median',f"{e['target_mean']:.2f} / {e['target_median']:.2f} ug/m3"],['25th / 75th percentile',f"{e['target_p25']:.2f} / {e['target_p75']:.2f} ug/m3"],['Skewness',f"{e['target_skew']:.2f}"],['Proportion >25 / >50 ug/m3',f"{e['above25_pct']:.2f}% / {e['above50_pct']:.2f}%"]])
    p('A right-skewed target means a few large errors may dominate RMSE. MAE and errors by concentration band will therefore complement RMSE later. The 25 and 50 ug/m3 thresholds are assignment reporting bands, not diagnostic health categories. MAPE will need caution at low positive concentrations.')
    nextpage()
    heading(5,'Missingness and outlier policy')
    image('05_missingness')
    p('Development same-day missingness is '+', '.join(f'{k}: {v:.2f}%' for k,v in e['gas_missing_pct'].items())+'. The temporal pattern matters: an overall percentage alone cannot show structural early-period absence. The publisher attributes gaps partly to clouds; the dataset alone does not establish a cause for every missing value.')
    p('Use training-fold median imputation for numerical features. Each gas also receives an explicit missingness indicator; gas values are never forward-filled across days. No backward fill or future-dependent interpolation is applied. If a feature is entirely absent from a training fold, the implementation retains a zero-filled column with an indicator and records the limitation; a later feature-availability sensitivity analysis is needed.')
    p(f"Development-only IQR screening flags {e['outliers']['flagged_count']:,} PM2.5 values outside [{e['outliers']['lower']:.2f}, {e['outliers']['upper']:.2f}] ug/m3. These observations remain in the dataset when they satisfy the assignment's fixed validity rule. An IQR flag alone does not demonstrate a measurement error.")
    p('All variables receive numerical summaries, skewness, excess kurtosis, missingness counts, histograms and boxplots. The separate full boxplot panel is provided in figures/02_boxplots.png. Any future learned outlier threshold must be fitted inside training folds and compared on identical held-out observations.')
    p('KNN imputation is deferred until correlation stability across training folds can justify it. PCA and feature selection are also deferred rather than fitted globally. Tree models will use unscaled inputs; scale-sensitive models will use fold-fitted StandardScaler.')
    nextpage()
    heading(6,'Temporal regularity and dependence')
    image('03_temporal_patterns')
    p(f"The median within-site missing-calendar-day rate is {e['median_missing_day_pct']:.1f}% between each site's first and last observed development day. Changes in the pooled monthly distribution can reflect station composition as well as pollution changes. Pooled trends are descriptive and do not establish a common trend for all locations.")
    image('06_time_diagnostics')
    p(f"Station-wise ACF uses exact calendar lags on reindexed daily ground series. The median lag-1 correlation is {e['median_calendar_lag1']:.3f}. The example above uses site {e['longest_run_station']}'s longest uninterrupted development run ({e['longest_run_days']} days). ADF and Ljung-Box are run only on contiguous, nonconstant segments of at least 30 days; {e['sites_tested_adf']} sites qualified.",'SmallCustom')
    p('An ADF p-value is evidence about a particular segment and specification, not proof of global stationarity. A weekly seasonal period is not assumed automatically. Lead-lag associations at negative lags are exploratory future-variable diagnostics and never become predictive inputs. Detailed per-site diagnostics and lead-lag results are provided in CSV form.','SmallCustom')
    nextpage()
    heading(7,'Spatial coverage and predictor associations')
    image('07_spatial_coverage')
    p('Station counts are unequal and several physical sites can share a satellite grid location. The analysis distinguishes sensor labels, physical sites and grid IDs. The primary GroupKFold groups by physical station as required in section 5.4. Additional grid-grouped folds keep sites sharing an atmospheric location together and test sensitivity to shared grid information.')
    image('08_relationships')
    p('Associations are descriptive and can be confounded by season, geography and meteorology. Satellite columns represent atmospheric quantities rather than direct measurements of the surface target. AOD-PM2.5 relations can change with vertical aerosol structure and weather. No causal claim follows from a correlation or scatterplot.','SmallCustom')
    p(f"Pairs with absolute Pearson or Spearman correlation above 0.9 are recorded in high_correlations.csv ({e['high_correlation_pairs']} method-pair entries). Correlated predictors can destabilize interpretation even when predictions remain adequate. Raw IDs and coordinates are excluded from the feature manifest.",'SmallCustom')
    nextpage()
    heading(8,'Seasonal and meteorological associations')
    p('We examined AOD-PM2.5 associations within seasons and individual sites, and within development-only temperature and wind-speed quartiles. These are exploratory stratifications, not fitted model features. They are never computed on the reserved future target period.')
    table([['Season','Matched rows / Spearman rho']]+[[r['group'],f"{r['rows']:,} / {r['spearman']:.3f}"] for r in e['aod_seasonal_associations']])
    conditional=pd.read_csv(output/'aod_conditional_associations.csv')
    weather=conditional[conditional.stratification.isin(['temperature_celsius','wind_speed'])]
    table([['Weather stratum','Rows / AOD-PM2.5 Spearman rho']]+[[r.stratification+' '+r.group,f'{r.rows:,} / {r.spearman:.3f}'] for r in weather.itertuples()], [260,215])
    p('Different correlations across conditions can suggest interactions worth evaluating with Extra Trees. They do not establish a causal influence, and pooled and conditional coefficients are not directly interchangeable. Weather strata also differ in season and station composition. Sample counts accompany every coefficient.')
    p('The site-level coefficients are in aod_conditional_associations.csv. Annual target medians, IQR and high-concentration proportions are in annual_drift.csv. seasonal_predictor_summary.csv covers monthly atmospheric patterns. These additions complete the requested seasonal/site and meteorological exploration.')
    p('No filter, PCA, KNN imputer or feature-selection rule has been chosen from these associations. Any later learned selection must occur inside the model training partitions.','SmallCustom')
    nextpage()
    heading(9,'Forecasting definition and preprocessing')
    p('<b>Operational question.</b> Predict PM2.5 on day t using atmospheric data dated t-1, past station PM2.5 where available, and calendar information for t. The source contains daily satellite/reanalysis observations; their actual publication latency has not been verified. This is a retrospective one-day-ahead experiment under an explicit availability assumption, not a demonstrated live forecasting service.')
    table([['Input family','Construction / safeguard'],['Atmospheric features','AOD, gases, temperature, pressure and wind from exactly t-1. Same-day columns are excluded.'],['Historical PM2.5','Lags 1, 2, 3 and 7 by exact site/date. Lag k is missing unless every preceding daily target from 1 through k exists. No lag crosses a missing intervening day.'],['Rolling history','Past-only 3- and 7-day mean/std; output is missing unless every required daily value exists.'],['Calendar','Sine/cosine of day of year; leap years handled.'],['Wind speed','sqrt(u^2 + v^2) from t-1.'],['Missing values and scaling','Fit the imputer on each training fold. Retain three gas indicators. Disable scaling for Extra Trees.'],['Excluded metadata','Sensor names, raw IDs, coordinates, timestamps and same-day target.']], [145,330])
    p(f"The forecast-ready table has {a['forecast_rows']:,} rows; {a['development_rows']:,} precede the holdout cutoff and {a['holdout_rows']:,} are reserved. It is built directly from valid mapped ground targets and the preceding day's GEE source. Its population differs from same-day matched EDA: a target need not have same-day GEE when t-1 features are present.")
    p('<b>Decision gate.</b> Temporal validation is justified by reliable dates and repeated site measurements, but sparsity limits complete lag windows. Retain missing history explicitly and use training-fold imputation. For a station with no historical PM2.5, remove target-history features in a declared cold-start variant. A spatial experiment with past target history instead evaluates locations absent from model fitting but having observed history; those are different deployment questions.')
    p('The serialized design is recorded in feature_manifest.json. Exact date alignment, historical lag values and predictor provenance are independently checked against the raw tables. No learned transformation is fitted on the final future period.')
    nextpage()
    heading(10,'Frozen validation and verification')
    overview=pd.read_csv(output/'fold_overview.csv')
    table([['Temporal fold','Training end / validation period / rows']]+[[r.fold,f'{r.train_end} / {r.validation_start} to {r.validation_end} / {r.train_rows:,} + {r.validation_rows:,}'] for r in overview[overview.scheme.eq('temporal')].itertuples()], [85,390])
    p('Temporal CV uses five expanding training windows on a complete calendar with a one-day gap. Splitting row positions directly would risk placing different stations from the same day on opposite sides of a boundary. The implementation splits dates first and maps dates to row IDs. Each validation target date follows the training dates by at least two calendar days.')
    p('Primary spatial CV uses five GroupKFold partitions grouped by physical station ID. Extra grid-grouped folds provide a stricter shared-location sensitivity check. IDs and coordinates are excluded as predictive features. Diagnostic shuffled KFold uses seed 42 and is not eligible to select the final model. All development folds exclude target dates from 2025 onward. Fold row IDs are saved once and reused.')
    p('<b>Forecast updates during validation.</b> Lagged observations before each predicted day can include earlier validation-day targets under a rolling one-day-ahead scenario. This is permitted only if those observations would have arrived before the next prediction. A fixed-origin multi-day scenario requires recursive forecasts and a different protocol; it is not claimed here.')
    p('<b>Verification performed.</b> Key uniqueness; fixed site-to-grid mapping; source-target and all predictor-value consistency; interrupted-lag regression checks; no future source date; historical values checked by exact site/date; disjoint future-test rows; nonoverlapping spatial groups; chronological gap; and finite, consistent preprocessing output across all five temporal folds. See validation_checks.json.')
    fold_checks=pd.read_csv(output/'preprocessing_fold_checks.csv').fillna('')
    p('<b>All-empty training features.</b> '+ '; '.join(f"fold {r.fold}: {r.empty_training_features or 'none'}" for r in fold_checks.itertuples())+'. Gas indicators remain explicit. A later validation value cannot retroactively teach the model a feature absent throughout training.','SmallCustom')
    p('Model selection is deferred. Future hyperparameter optimization must be nested inside each training partition and respect the evaluation scenario. Reusing outer validation scores for tuning would bias the reported comparison.')
    nextpage()
    heading(11,'Delivery checklist and next milestones')
    table([['Available for the first milestone','Location'],['Executable source and configuration','src/ and config/project.json'],['Guided notebook and instructions','notebooks/01_EDA_and_Preprocessing.ipynb; README.md'],['Required numerical / temporal summaries','Exploratory_Data.csv; Exploratory_temporal.csv'],['Missingness, drift, correlations and joining audit','results/G6_ExtraTrees/*.csv and audit.json'],['Frozen split definitions','folds_temporal.json; folds_spatial.json; folds_standard.json'],['Causal feature definition / validation evidence','feature_manifest.json; validation_checks.json'],['Charts and reproducibility metadata','figures/; experiments.csv']], [270,205])
    p('<b>Reproduction.</b> Run the automated regression checks and python -m src.run. The pinned source archive is verified before use. Windows and Linux launchers stop on errors. An executed notebook explains the same pipeline. The result directory identifies the assigned method. The nearest-coordinate rule follows section 5.3.4; the exact-match sensitivity is saved for transparent comparisons.')
    p('<b>December milestone.</b> Use the confirmed Extra Trees assignment; add persistence and training-mean baselines with explicit eligible-row coverage; run tuning with the declared search budget; save fold-level metrics, runtimes and hyperparameters; and compare training sizes. Model choice uses temporal validation, with spatial results addressing transfer limitations. Reserve the future period until the design is frozen.')
    p('<b>January milestone.</b> Add paired fold comparisons with Holm correction and explicit five-fold uncertainty; permutation importance and ranking stability; reference-model ablations; concentration-band and monthly error analysis; and a justified operational recommendation. The final technical report must meet the assignment\'s 10-15-page English requirement and the oral presentation must fit 15 minutes.')
    p('<b>Limits.</b> No validation in Algeria is established. Matching and source coverage can bias the retained sample. Daily aggregation and sensor calibration are not independently verified. Atmospheric-input latency is assumed. Early gas coverage and intermittent target history need sensitivity checks. No model performance or scientific superiority is claimed at this stage.')
    p('<b>Sources.</b> (1) Supplied assignment, updated PM2.5 Forecasting mini-project, sections 3, 5 and 9. (2) Kaggle dataset v1 and publisher metadata: https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026 (accessed 12 November 2026). (3) scikit-learn API: Pipeline, SimpleImputer, TimeSeriesSplit, GroupKFold. (4) statsmodels API: ADF, ACF/PACF and Ljung-Box. The execution environment records exact software versions. (5) Geurts, Ernst and Wehenkel (2006), DOI 10.1007/s10994-006-6226-1. (6) scikit-learn 1.8 ExtraTreesRegressor API: https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html.','SmallCustom')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#b8cedb'));canvas.line(42,39,A4[0]-42,39)
        canvas.setFont(regular,8);canvas.setFillColor(colors.HexColor('#475e6a'))
        canvas.drawString(42,26,'PM2.5 | Data exploration and preprocessing | November 2026')
        canvas.drawRightString(A4[0]-42,26,str(doc.page))
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=52,
                      title='PM2.5 Forecasting - First Milestone',author=', '.join(team['members'])).build(story,onFirstPage=footer,onLaterPages=footer)
    return pdf

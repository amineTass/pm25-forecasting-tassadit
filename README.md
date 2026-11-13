# PM2.5 forecasting - Extra Trees

**Team:** Amrane Mohamed; Bensaidane Nazim Adem; Tassadit Mohamed Amine; Benabdi Adem.

**Assigned method:** Extra Trees Regressor, G6 in the assignment's method inventory.

This submission covers **data exploration and preprocessing, due 14 November 2026**. Real-data model training/evaluation is due 12 December 2026, and statistics/explainability on 2 January 2027. Model accuracy is not claimed at this first milestone.

## Start here

- `reports/PM25_Phase1_Report.pdf`: the English technical report generated from executed results.
- `presentation/`: 15-minute PowerPoint, PDF slide copy and rehearsal/Q&A guide.
- `reports/PM25_Project_Simple_Explanation.pdf`: a short plain-language explanation.
- `notebooks/01_EDA_and_Preprocessing.ipynb`: executed analysis with explanations.
- `results/G6_ExtraTrees/`: audit, summaries, plots, frozen partitions and verification records.

`G6_ExtraTrees` is a technical folder name based on the assigned method, not an invented formal class team code. If the teacher later gives a formal submission code, set `submission_group_code` in `config/project.json` and, if required, use that code as the result-folder value `group` before rerunning. Team names and the assigned model are already included.

## Run on Windows

Use Python **3.11 or 3.12**. Double-click `run_windows.cmd`, or run these commands from the extracted project folder in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.run
```

Replace `-3.12` with `-3.11` if appropriate. The launcher stops if a command fails. Its Windows execution still needs a native computer; the core pipeline was reproduced in a clean Python 3.12 environment on Linux. No GPU is required.

## Run on Linux / macOS

```bash
bash run.sh
```

Or create a Python 3.11/3.12 virtual environment, install `requirements.txt`, run the tests and execute `python -m src.run` as above. The exact core package versions are pinned.

The first run downloads **Kaggle dataset version 1**, about 9 MB compressed, and verifies the audited archive checksum. If automatic download is unavailable, download v1 manually from the dataset page and save it as `data/raw/dataset.zip`. A checksum mismatch stops the run rather than mixing a different source with old folds. The source CSVs and generated observation-level tables are not redistributed in this archive. The code recreates them locally; all aggregate results and figures are included.

For interactive use, install `requirements-notebook.txt`, then launch JupyterLab from the same environment. Open the notebook with the project folder as its working directory. The command-line pipeline requires none of the optional notebook packages.

## Protocol

1. Audit three source CSVs, their missingness, timestamps, coordinates and unique keys. Verify all supplied combined targets and predictors against raw source values.
2. Remove the assignment's invalid target ranges: `pm25 <= 0`, `pm25 >= 500`, and missing targets. Preserve plausible pollution extremes.
3. Define a physical site using sensor label plus coordinates. Map it once to the nearest fixed GEE location within Euclidean angular distance 0.25 degrees. Match on date and grid ID. This implements **assignment section 5.3.4**. The distance is not kilometres.
4. Record the exact-match comparison after applying the same target cleaning: 40,983 valid supplied rows are retained; the reconstructed same-day join has 46,914 rows, including 5,931 additional valid matches and losing none of the supplied valid rows. See `join_sensitivity.json`. Shared cross-group comparisons must still use identical row populations and folds.
5. Keep descriptive EDA before 2025. Summarize distributions, missingness by month/site, seasonal and annual patterns, station histories, correlations and AOD associations within season/site and weather quartiles.
6. Define a retrospective forecast of day `t` from atmospheric data on `t-1`, historical PM2.5 and known calendar information. A target-history lag `k` requires **all preceding days from 1 through k**. Missing intervening days mask that lag. Rolling statistics likewise require complete past windows. No row shift bridges gaps.
7. Reserve **2025-01-01 onward** for final testing. Freeze five chronological folds on a full calendar with a one-day gap. Primary spatial GroupKFold groups by physical station ID, as section 5.4 specifies. Additional `spatial_grid` folds group sites sharing a GEE location. Shuffled folds are diagnostic only.
8. Fit medians within training partitions only and keep explicit indicators for NO2, CO and O3. Never forward-fill these gas columns. Record entirely absent training features: no median can be estimated for them, so retain a zero fallback and gas indicator. Extra Trees uses **no scaler**.

## Two populations, two purposes

Same-day EDA uses 34,832 valid matched development rows. Forecasting matches targets at `t` to GEE at `t-1`, producing 33,282 development rows and 12,101 reserved rows. These counts differ because atmospheric availability differs between dates. No future-test distribution or performance is used for model selection.

The predictor manifest contains 19 numeric inputs before three explicit gas indicators. Metadata, station names, raw IDs and coordinates are excluded from predictive features. Past target history is valid only for a station where those observations are available. An entirely unmonitored site requires a separate feature set without PM2.5 history. The experiment assumes preceding-day atmospheric products are available at issue time; actual publication latency has not been established.

During a rolling daily validation block, an earlier day's observed PM2.5 may enter the following day's history if it would be available by then. This is not a fixed-origin multiday forecast. Hyperparameter tuning must remain inside each outer training partition.

## Source code and outputs

- `src/data.py`: source verification, fixed matching and strict history features.
- `src/eda.py`: numerical, temporal, seasonal and conditional exploration.
- `src/preprocessing.py`: training-fitted imputation and explicit gas indicators.
- `src/model.py`: Extra Trees pipeline factory and declared search space. Tested with synthetic data only.
- `src/splits.py`: chronological, station, shared-grid and diagnostic random folds.
- `src/run.py`: reproducible first-stage orchestration, verification and consolidation.
- `src/report.py`: report generation from the same result tables.
- `tests/test_protocol.py`: eight leakage/gap/model-factory regression tests.

The required `Exploratory_Data.csv` combines typed numerical-summary and cleaning-count records. `Exploratory_temporal.csv` describes development station histories. Additional CSVs cover raw missingness, gas gaps, annual drift, monthly predictors, AOD conditional associations, forecast feature availability and empty training features. `results/consolidated/` contains this group's exploration files with an explicit group column; it does not pretend to consolidate other teams' results.

Frozen JSON files contain `row_id` values, not assumed pandas index labels. `fold_manifest.json` ties them to the generated development CSV checksum. Do not edit the source, features or sample and then reuse earlier folds. `experiments.csv` records source/code/configuration hashes, versions, hardware, seed and runtime. The clean-environment execution and test evidence is also included.

## Later milestones

Train a mean baseline, persistence baseline with explicit coverage, and Extra Trees. Tune the declared G6 ranges in inner training partitions, at most 50 configurations, retain three distinct candidates and test parameter sensitivity. Save real fold metrics, bootstrap uncertainty, runtime and model size. Then add training-size sensitivity, controlled ablations, paired tests with correction, permutation importance and ranking stability. Perform the final future evaluation only after freezing the design.

`results.csv`, `hyperparameters.csv`, `feature_importance.csv`, `comparisons.csv`, `ablation.csv` and `ranking_stability.csv` belong to actual later model runs. Empty placeholders are not supplied as if they were results. The presentation clearly distinguishes measured EDA findings from this plan.

## Git reproduction

A local repository snapshot is provided as `repository.bundle` in the complete delivery. Clone it with:

```bash
git clone repository.bundle pm25-extra-trees
cd pm25-extra-trees
```

Then run the platform-specific instructions. The bundle contains source and configuration, without raw data or secrets. Team profile links supplied for attribution are recorded in `config/team.json`; no remote repository or public publication is assumed.

## Sources

Dataset v1: https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026

Extra Trees: Geurts, Ernst and Wehenkel (2006), *Extremely randomized trees*, Machine Learning 63, 3-42. https://doi.org/10.1007/s10994-006-6226-1

Implementation: https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html

Publisher metadata labels the license "Other (specified in description)" and attributes OpenAQ, Copernicus and NASA. The package preserves this provenance and uses download-on-run instead of asserting a blanket redistribution license. Regional proxy data do not establish transfer performance in Algeria, and this audit does not independently certify sensor calibration or daily aggregation.

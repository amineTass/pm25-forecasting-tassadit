# Submission audit and corrections

This update covers the November exploration/preprocessing submission and its 15-minute presentation.

- Added all four team names and the confirmed Extra Trees assignment to the report/configuration, matching the slides.
- Replaced the generic result folder with the explicit technical name `G6_ExtraTrees`. No formal teacher-issued team code is invented.
- Documented nearest-coordinate matching under assignment section 5.3.4. Compared cleaned populations: all 40,983 valid supplied rows are retained, and 5,931 additional valid nearby matches are recovered.
- Tightened historical features: lag k requires every preceding daily target from 1 through k. Missing intervening days mask the lag, and incomplete rolling windows remain missing.
- Made primary spatial folds follow physical station IDs. Retained shared-grid grouped folds as an additional diagnostic, and fixed fold code to work with non-default pandas indexes.
- Added exact source-archive verification, stricter source-value consistency checks, source missingness, annual drift, monthly predictor summaries and AOD associations within season/site and weather strata.
- Made the Extra Trees preprocessing default unscaled. Recorded the three all-empty gas features in the first temporal training fold, with explicit missingness indicators and documented fallback.
- Added the assigned-model factory and G6 search plan. Its regression test uses a tiny synthetic sample; no scientific model accuracy is claimed.
- Added eight leakage/gap/reproducibility tests and ran the full pipeline in a clean Python 3.12 environment. Executed all 11 notebook code cells with rich outputs.
- Expanded the report to 12 pages with research questions, the Extra Trees method, references and conditional analyses. Embedded readable fonts and checked rendered pages.
- Updated the 14-slide main talk, questions slide, four backup slides, speaker notes, slide PDF, rehearsal guide and simple explanation.
- Added platform launchers, consolidated exploration files, source/configuration/data hashes and a local Git bundle.

The native Windows launcher and PowerPoint application have not been executed in this Linux environment. The core code was tested in an isolated environment and the slides were rendered and structurally checked. Test the PPTX on the presentation computer before speaking.

Real-data Extra Trees tuning/evaluation remains the December milestone. Statistical tests, ablations and explanation stability remain later work. These are explicit future stages, not missing November accuracy tables.

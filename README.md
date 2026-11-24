# Student Insights

A Streamlit app with three pages that share one set of filters and one model.

| Page | Question it answers | Method |
|---|---|---|
| Drivers | Which characteristics shift the outcome, others held constant? | Unpenalised logistic regression, average marginal effects, application-clustered 95% CIs |
| Segments | What kinds of applicant are there, and how does each fare? | k-means on one-hot characteristics (equal weight per variable), Wilson CIs, silhouette + resampling stability |
| Probability | What is the chance for this profile, and how reliable is it? | Same fitted model; interval on the log-odds scale; accuracy on held-back applications (AUC, Brier, calibration) |

Results are associations in observational data, not proof of cause.

## Run (Windows)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```
Use `python -m streamlit`, not `streamlit`: a copied venv keeps the old folder path baked in.
Load the parquet/CSV on the Home page; it runs data checks and reports problems in plain words.

## Layout
* `utils/ame.py` – all regression maths and data cleaning (no Streamlit). `fit_model()` fits once; `effects_table()` and `predict.py` both use that fit.
* `utils/central_filters.py` – every filter, with an audit of what each removed. "Filters in force" is printed on every page and in every download.
* `utils/predict.py`, `utils/segments.py`, `utils/narrative.py`, `utils/data_checks.py` – engines, no Streamlit.
* `utils/ui.py` – shared model-setup block for Drivers and Probability.
* `utils/model_config.py` – outcomes, characteristic columns, controls. Edit here to add a variable.
* `tests/` – `python -m pytest tests -v`. Do not delete `test_matches_l2_page_maths`; it pins the maths to the original L2 calculation.

## Design choices (and why)
* No penalty and no class weighting: coefficients are ordinary maximum-likelihood estimates and probabilities stay true probabilities, so results are reportable.
* Splits for accuracy are by application, never within one.
* Segments warn when structure is weak or unstable instead of presenting them as real types.
* Missing values are kept visible as "Not recorded" by default; small groups merge into "Other (small groups)".

## Known limits
* Tested on synthetic data only. Reconcile against a result you already trust on real data before reporting.
* A group of a handful of rows can make a model fail to converge; raise the merge threshold.
* Clustering needs `application_number`; without it intervals are too narrow (the Home page warns).

## Before changing anything
Run the tests, make the change, run them again.

SSP Scenario Prediction Code — MRIO Trade Flow Forecasting (2025 & 2030)
=========================================================================
Description:
    Trains a hybrid Transformer + XGBoost model on historical MRIO matrices
    and projects trade flow matrices for future years (e.g. 2025, 2030)
    using SSP scenario GDP and population projections.
    Predictions are balanced via iterative proportional fitting (RAS).

    The MRIO structure is non-square: source rows are indexed by
    (country × sector) and destination columns by a different
    (country × sector) set.

Inputs:
    - Annual MRIO CSV matrices (one file per year, filename must contain a
      4-digit year, e.g. Z_2015.csv)
    - Historical GDP/population CSV  (columns: country, year, Population, gdp)
    - SSP2 scenario GDP/population CSV  (same column structure, future years)
    - Static country-level feature CSV  (economic status, trade bloc, etc.)

Outputs:
    - Balanced predicted trade flow matrix per forecast year
      → <OUTPUT_TEMPLATE>_prediction_balanced_<year>.csv
    - Trained model artefacts saved to <TFRECORD_DIR>:
        best_model_unified.joblib, best_model_transformer/,
        scalers.pkl, best_params.json, metadata_ts<N>.json,
        training_history.json

Dependencies:
    pandas, numpy, scikit-learn, tensorflow, xgboost,
    scikit-optimize, optuna, joblib, tqdm

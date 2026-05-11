"""
SSP Scenario Prediction Code — MRIO Trade Flow Forecasting (2025 & 2030)
=========================================================================
Description:
    Trains a Transformer + XGBoost ensemble on historical MRIO data and
    generates trade flow matrix predictions for future years (e.g. 2025, 2030)
    using SSP2 scenario GDP and population projections.
    Supports non-square matrix structures (e.g. 49 countries × 44 sectors).
    Predictions are balanced via an iterative RAS procedure.

Inputs:
    - Annual MRIO CSV matrices (one file per year, filename must contain a
      4-digit year, e.g. Z_2015.csv)
    - Historical GDP/population CSV  (columns: country, year, Population, gdp)
    - SSP2 scenario GDP/population CSV (same column structure)
    - Static country-level feature CSV (economic_status, trade_bloc, ...)

Outputs:
    - Balanced predicted trade flow matrix per target year
      → <OUTPUT_TEMPLATE_PATH>_prediction_balanced_<year>.csv
    - Model artefacts saved to <TFRECORD_DIR>:
        best_model_unified.joblib, best_params.json,
        best_model_transformer/, scalers.pkl, metadata_ts<N>.json

Dependencies:
    pandas, numpy, scikit-learn, tensorflow, xgboost,
    scikit-optimize, optuna, joblib, tqdm
"""

import gc
import glob
import json
import os
import pickle
import random
import re
import traceback
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from tensorflow.keras.layers import (
    Attention, Dense, Dropout, Flatten, Input,
    LayerNormalization, MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tqdm import tqdm

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# User-configurable paths and settings — set before running
# ============================================================
CSV_DIR              = "path/to/mrio_csv_folder"          # Annual MRIO CSV files
HIST_CSV             = "path/to/gdp_population.csv"        # Historical GDP/population
FUTURE_CSV           = "path/to/ssp2_gdp_population.csv"   # SSP2 scenario data
COUNTRY_CSV          = "path/to/country_features.csv"      # Static country features
OUTPUT_TEMPLATE_PATH = "path/to/output/prediction_matrix"  # Output filename prefix
TFRECORD_DIR         = "path/to/cache/tfrecords"

YEARS_TO_FORECAST  = [2025, 2030]   # Target future years
TIME_STEPS         = 5              # Historical time window for feature construction
BATCH_SIZE         = 2048
OPTIMIZATION_CALLS = 15
TRAIN_EPOCHS       = 30

# Matrix dimensions — must match your MRIO data
NUM_COUNTRIES = 49
NUM_SECTORS   = 44


# ============================================================
# 1. GPU and precision setup
# ============================================================

def setup_gpu():
    """Enable memory growth, mixed precision, and select distribution strategy."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            strategy = (
                tf.distribute.MirroredStrategy() if len(gpus) > 1
                else tf.distribute.OneDeviceStrategy("/gpu:0")
            )
            print(f"Mixed precision enabled. Strategy: {'multi-GPU' if len(gpus) > 1 else 'single GPU'}.")
            return True, strategy
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    print("No GPU detected; using CPU.")
    return False, None


GPU_AVAILABLE, STRATEGY = setup_gpu()


# ============================================================
# 2. Custom loss
# ============================================================

def custom_tweedie_loss(p=1.5):
    """Tweedie deviance loss (1 < p < 2) for positive, heavy-tailed targets."""
    def loss(y_true, y_pred):
        eps    = 1e-7
        y_pred = tf.maximum(y_pred, eps)
        y_true = tf.maximum(y_true, 0.0)
        return tf.reduce_mean(
            tf.pow(y_pred, 2 - p) / (2 - p) - y_true * tf.pow(y_pred, 1 - p) / (1 - p)
        )
    return loss


# ============================================================
# 3. TFRecord utilities
# ============================================================

def _float_feature(value):
    if isinstance(value, np.ndarray):
        value = value.flatten()
    elif not isinstance(value, (list, tuple)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(features, target, time_steps, feature_length):
    feature_dict = {
        'features':       _float_feature(features.flatten()),
        'target':         _float_feature(target),
        'time_steps':     _int64_feature(time_steps),
        'feature_length': _int64_feature(feature_length),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
    ).SerializeToString()


def parse_tfrecord_function(example_proto, time_steps, feature_length):
    feature_description = {
        'features':       tf.io.FixedLenFeature([time_steps * feature_length], tf.float32),
        'target':         tf.io.FixedLenFeature([1], tf.float32),
        'time_steps':     tf.io.FixedLenFeature([1], tf.int64),
        'feature_length': tf.io.FixedLenFeature([1], tf.int64),
    }
    parsed   = tf.io.parse_single_example(example_proto, feature_description)
    features = tf.reshape(parsed['features'], [time_steps, feature_length])
    target   = tf.reshape(parsed['target'], [])
    return features, target


def create_tf_data_pipeline(tfrecord_path, time_steps, feature_length,
                             batch_size, is_training=True, buffer_size=10000):
    """Build a tf.data pipeline from a TFRecord file."""
    if not os.path.exists(tfrecord_path) or os.path.getsize(tfrecord_path) == 0:
        print(f"Warning: TFRecord file missing or empty: {tfrecord_path}")
        return None
    ds = (
        tf.data.TFRecordDataset(tfrecord_path)
        .map(lambda x: parse_tfrecord_function(x, time_steps, feature_length),
             num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
    )
    if is_training:
        ds = ds.shuffle(buffer_size)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================
# 4. Data processing
# ============================================================

class TFRecordDataProcessor:
    """
    Loads annual MRIO CSV matrices and country-level covariates,
    constructs time-series feature tensors, and writes TFRecord files.
    Supports non-square matrices (row and column country/sector counts may differ).
    """

    # Fixed one-hot encoding categories for country features
    ECONOMIC_STATUS_CATS = ['Developed', 'Developing']
    TRADE_BLOC_CATS      = ['EU', 'NAFTA', 'ASEAN', 'Other']

    def __init__(
        self,
        csv_folder_path,
        excel_file_path,
        country_csv_path,
        tfrecord_dir="./tfrecords",
        future_excel_path=None,
        num_countries=NUM_COUNTRIES,
        num_sectors=NUM_SECTORS,
    ):
        self.csv_folder_path   = csv_folder_path
        self.excel_file_path   = excel_file_path
        self.future_excel_path = future_excel_path
        self.tfrecord_dir      = tfrecord_dir
        self.num_countries     = num_countries
        self.num_sectors       = num_sectors

        self.scaler_features     = MinMaxScaler()
        self.scaler_target       = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
        self.country_year_lookup = {}
        self.yearly_stats_cache  = {}

        os.makedirs(self.tfrecord_dir, exist_ok=True)

        # Locate annual CSV files
        self.all_csv_files = sorted(glob.glob(os.path.join(csv_folder_path, "Z_*.csv")))
        if not self.all_csv_files:
            self.all_csv_files = sorted(glob.glob(os.path.join(csv_folder_path, "*.csv")))

        self.year_to_file_map = {
            int(re.search(r'(\d{4})', os.path.basename(f)).group(1)): f
            for f in self.all_csv_files
            if re.search(r'(\d{4})', os.path.basename(f))
        }

        self.full_row_labels, self.full_col_labels = self._get_original_matrix_labels()
        print(f"Matrix structure: {len(self.full_row_labels)} rows × {len(self.full_col_labels)} cols "
              f"({num_countries} countries × {num_sectors} sectors)")

        self.valid_years = self._validate_years()

        self.country_feature_lookup = self._load_country_features(country_csv_path)
        self.economic_status_cats   = len(self.ECONOMIC_STATUS_CATS)
        self.trade_bloc_cats        = len(self.TRADE_BLOC_CATS)

        # 14 base features (incl. diagonal indicator) + 4 growth + 2 × (status_cats + bloc_cats) country features
        self.feature_length = 14 + 4 + 2 * (self.economic_status_cats + self.trade_bloc_cats)
        print(f"Feature length: {self.feature_length}")

    # ----------------------------------------------------------
    # Matrix label extraction
    # ----------------------------------------------------------

    def _load_and_process_mrio_csv(self, file_path):
        """Read a raw MRIO CSV and return a labeled DataFrame."""
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']:
            try:
                df_raw = pd.read_csv(file_path, header=None, index_col=None, dtype=str, encoding=encoding)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Cannot load {os.path.basename(file_path)}")

        country_cols  = df_raw.iloc[0, 2:].ffill()
        sector_cols   = df_raw.iloc[1, 2:]
        combined_cols = [f"{str(co).strip()}_{str(se).strip()}" for co, se in zip(country_cols, sector_cols)]

        country_rows  = df_raw.iloc[2:, 0].ffill()
        sector_rows   = df_raw.iloc[2:, 1]
        combined_rows = [f"{str(co).strip()}_{str(se).strip()}" for co, se in zip(country_rows, sector_rows)]

        data = df_raw.iloc[2:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return pd.DataFrame(data.values, index=combined_rows, columns=combined_cols)

    def _get_original_matrix_labels(self):
        """Extract row and column labels from the most recent MRIO CSV."""
        if not self.all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_folder_path}")
        ref_file = self.year_to_file_map[max(self.year_to_file_map)]
        df_ref   = self._load_and_process_mrio_csv(ref_file)
        return df_ref.index.tolist(), df_ref.columns.tolist()

    def _validate_years(self):
        """Return a sorted list of years whose CSV files load without error."""
        valid = []
        for year in sorted(self.year_to_file_map):
            try:
                data = self.load_single_year_data(year, validate_only=True)
                if data is not None and not data.empty:
                    valid.append(year)
                del data
                gc.collect()
            except Exception as e:
                print(f"Year {year} failed validation: {e}")
        return valid

    # ----------------------------------------------------------
    # Country feature loading
    # ----------------------------------------------------------

    def _load_country_features(self, file_path):
        """Load and one-hot encode static country features."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Country features file not found: {file_path}")

        df = None
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise IOError(f"Cannot decode {file_path}")

        df['economic_status'] = pd.Categorical(df['economic_status'], categories=self.ECONOMIC_STATUS_CATS)
        df['trade_bloc']      = pd.Categorical(df['trade_bloc'],      categories=self.TRADE_BLOC_CATS)
        feature_df = pd.concat([
            df['country_code'],
            pd.get_dummies(df['economic_status'], prefix='status'),
            pd.get_dummies(df['trade_bloc'],      prefix='bloc'),
        ], axis=1)

        lookup = {row.country_code: np.array(row[1:], dtype=np.float32) for row in feature_df.itertuples(index=False)}
        print(f"Loaded country features for {len(lookup)} countries.")
        return lookup

    # ----------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------

    def load_single_year_data(self, year, validate_only=False):
        """Load and align the MRIO matrix for a given year."""
        if year not in self.year_to_file_map:
            return None
        df = self._load_and_process_mrio_csv(self.year_to_file_map[year])
        if validate_only:
            return df
        return df.reindex(index=self.full_row_labels, columns=self.full_col_labels).fillna(0.0).astype(np.float32)

    def load_excel_data(self):
        """Build a (country, year) → (population, GDP) lookup from historical and scenario CSVs."""
        for path in [self.excel_file_path, self.future_excel_path]:
            if not (path and os.path.exists(path)):
                continue
            df      = self._load_feature_csv(path)
            updates = dict(zip(zip(df['country'], df['year']), zip(df['Population'], df['gdp'])))
            self.country_year_lookup.update(updates)
            print(f"Loaded {len(updates)} (country, year) entries from {os.path.basename(path)}.")

    def _load_feature_csv(self, file_path):
        """Load a GDP/population CSV with encoding fallback and required-column check."""
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']:
            try:
                df = pd.read_csv(file_path, engine='c', dtype={'country': str, 'year': int}, encoding=encoding)
                df.columns = df.columns.str.strip()
                required = {'country', 'year', 'Population', 'gdp'}
                if not required.issubset(df.columns):
                    raise ValueError(f"Missing columns {required - set(df.columns)} in {file_path}")
                return df
            except UnicodeDecodeError:
                continue
        raise IOError(f"Cannot load {file_path}")

    # ----------------------------------------------------------
    # Annual statistics cache
    # ----------------------------------------------------------

    def _precompute_yearly_stats(self):
        """Cache per-row/column flow statistics for each training year."""
        print("Precomputing annual flow statistics...")
        for year in tqdm([y for y in self.valid_years if y <= 2022]):
            try:
                df = (
                    self._load_and_process_mrio_csv(self.year_to_file_map[year])
                    .reindex(index=self.full_row_labels, columns=self.full_col_labels)
                    .fillna(0.0)
                )
                total_outflow   = df.sum(axis=1)
                country_outflow = (
                    total_outflow.rename_axis('label').reset_index()
                    .assign(country=lambda x: x['label'].str.split('_').str[0])
                    .groupby('country')[0].sum()
                )
                self.yearly_stats_cache[year] = {
                    'outflow_stats':         dict(zip(df.index,   zip(df.mean(axis=1), df.std(axis=1), df.max(axis=1)))),
                    'inflow_stats':          dict(zip(df.columns, zip(df.mean(axis=0), df.std(axis=0), df.max(axis=0)))),
                    'total_outflow':         total_outflow.to_dict(),
                    'total_inflow':          df.sum(axis=0).to_dict(),
                    'country_total_outflow': country_outflow.to_dict(),
                }
                del df
                gc.collect()
            except Exception as e:
                print(f"Warning: could not compute stats for year {year}: {e}")

    # ----------------------------------------------------------
    # Feature construction
    # ----------------------------------------------------------

    def _construct_focused_feature_label(self, row_label, col_label, target_year, time_steps, yearly_data):
        """
        Build a (time_steps, feature_length) feature array and scalar target
        for cell (row_label, col_label) in the target_year matrix.

        Returns (None, None) if any required data is missing or invalid.
        """
        src_c = row_label.split('_')[0]
        dst_c = col_label.split('_')[0]

        pop_src, gdp_src = self.country_year_lookup.get((src_c, target_year), (0, 0))
        _,       gdp_dst = self.country_year_lookup.get((dst_c, target_year), (0, 0))

        pop_src = pop_src if not np.isnan(pop_src) else 0
        gdp_src = max(0, gdp_src) if not np.isnan(gdp_src) else 0
        gdp_dst = max(0, gdp_dst) if not np.isnan(gdp_dst) else 0

        default_feat = np.zeros(self.economic_status_cats + self.trade_bloc_cats, dtype=np.float32)
        src_f = self.country_feature_lookup.get(src_c, default_feat)
        dst_f = self.country_feature_lookup.get(dst_c, default_feat)

        all_features = []
        for year in range(target_year - time_steps, target_year):
            matrix = yearly_data.get(year)
            stats  = self.yearly_stats_cache.get(year)
            if matrix is None or stats is None:
                return None, None
            if row_label not in matrix.index or col_label not in matrix.columns:
                return None, None

            o_stats     = stats['outflow_stats'].get(row_label, (0, 0, 0))
            i_stats     = stats['inflow_stats'].get(col_label,  (0, 0, 0))
            total_out   = stats['total_outflow'].get(row_label,  0)
            total_in    = stats['total_inflow'].get(col_label,   0)
            country_out = stats['country_total_outflow'].get(src_c, 0)

            gdp_src_hist = self.country_year_lookup.get((src_c, year), (0, 0))[1]
            gdp_dst_hist = self.country_year_lookup.get((dst_c, year), (0, 0))[1]
            gdp_src_hist = gdp_src_hist if not np.isnan(gdp_src_hist) else 1
            gdp_dst_hist = gdp_dst_hist if not np.isnan(gdp_dst_hist) else 1

            years_ahead = float(target_year - year)
            cf_src      = np.clip(gdp_src / (gdp_src_hist + 1e-9), 0.5, 3.0)
            cf_dst      = np.clip(gdp_dst / (gdp_dst_hist + 1e-9), 0.5, 3.0)
            gr_src      = cf_src ** (1.0 / max(1, years_ahead)) - 1.0
            gr_dst      = cf_dst ** (1.0 / max(1, years_ahead)) - 1.0

            is_diagonal = 1.0 if row_label == col_label else 0.0

            base_feat = np.array([
                np.log1p(max(0, float(matrix.loc[row_label, col_label]))),
                np.log1p(max(0, pop_src)),
                np.log1p(gdp_src),
                np.log1p(max(0, o_stats[0])),
                np.log1p(max(0, o_stats[1])),
                np.log1p(max(0, o_stats[2])),
                np.log1p(max(0, i_stats[0])),
                np.log1p(max(0, i_stats[1])),
                np.log1p(max(0, i_stats[2])),
                np.log1p(max(0, total_out)),
                np.log1p(max(0, total_in)),
                np.log1p(gdp_src) * np.log1p(gdp_dst),   # gravity proxy
                total_out / (country_out + 1e-9),          # sector outflow share
                is_diagonal,                               # self-consumption indicator
            ], dtype=np.float32)

            growth_feat = np.array([years_ahead, gr_src, gr_dst, cf_src * cf_dst], dtype=np.float32)

            vec = np.nan_to_num(
                np.concatenate([base_feat, growth_feat, src_f, dst_f]),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            all_features.append(vec)

        target_mat = yearly_data.get(target_year)
        if target_mat is None or row_label not in target_mat.index or col_label not in target_mat.columns:
            return None, None

        target = max(0, float(target_mat.loc[row_label, col_label]))
        if np.isnan(target) or np.isinf(target) or len(all_features) != time_steps:
            return None, None

        return np.array(all_features, dtype=np.float32), target

    # ----------------------------------------------------------
    # Scaler fitting
    # ----------------------------------------------------------

    def prepare_scalers(self, time_steps, sample_size=50000):
        """Fit feature and target scalers on a random subsample of training data."""
        print("Fitting scalers...")
        years        = [y for y in self._find_longest_consecutive_years() if y < 2022]
        yearly_data  = {y: self.load_single_year_data(y) for y in years}
        target_years = [y for y in years if y >= years[0] + time_steps]

        all_positions = [(r, c) for r in self.full_row_labels for c in self.full_col_labels]
        positions     = random.sample(all_positions, min(sample_size, len(all_positions)))

        X_sample, y_sample = [], []
        for pos in tqdm(positions, desc="Sampling for scalers"):
            for ty in target_years:
                f, t = self._construct_focused_feature_label(pos[0], pos[1], ty, time_steps, yearly_data)
                if f is not None:
                    X_sample.append(f)
                    y_sample.append(t)
                if len(X_sample) >= sample_size:
                    break
            if len(X_sample) >= sample_size:
                break

        X_flat = np.array(X_sample).reshape(-1, self.feature_length)
        y_flat = np.array(y_sample).reshape(-1, 1)
        self.scaler_features.fit(X_flat)
        self.scaler_target.fit(y_flat)
        print(f"Scalers fitted on {len(X_flat)} samples.")

        with open(os.path.join(self.tfrecord_dir, 'scalers.pkl'), 'wb') as fh:
            pickle.dump({'scaler_features': self.scaler_features, 'scaler_target': self.scaler_target}, fh)

        del yearly_data, X_sample, y_sample
        gc.collect()

    # ----------------------------------------------------------
    # TFRecord generation
    # ----------------------------------------------------------

    def create_tfrecords(self, time_steps, train_split=0.8):
        """Generate stratified train/val TFRecord files from historical data."""
        print(f"Generating TFRecords (time_steps={time_steps})...")
        self.load_excel_data()
        self._precompute_yearly_stats()
        self.prepare_scalers(time_steps)

        years        = [y for y in self._find_longest_consecutive_years() if y < 2022]
        yearly_data  = {y: self.load_single_year_data(y) for y in years}
        target_years = [y for y in years if y >= years[0] + time_steps]
        if not target_years:
            raise ValueError("Insufficient data to generate training targets.")

        all_positions     = [(r, c) for r in self.full_row_labels for c in self.full_col_labels]
        potential_samples = [(pos, yr) for pos in all_positions for yr in target_years]

        strata = {k: [] for k in ['zero', 'tiny', 'small', 'medium', 'large', 'extra_large']}
        for pos, ty in tqdm(potential_samples, desc="Stratifying"):
            _, t = self._construct_focused_feature_label(pos[0], pos[1], ty, time_steps, yearly_data)
            if t is not None:
                if   t == 0:        strata['zero'].append((pos, ty))
                elif t <= 50:       strata['tiny'].append((pos, ty))
                elif t <= 1_000:    strata['small'].append((pos, ty))
                elif t <= 10_000:   strata['medium'].append((pos, ty))
                elif t <= 100_000:  strata['large'].append((pos, ty))
                else:               strata['extra_large'].append((pos, ty))

        # Retain all large-value cells; subsample smaller strata
        ratios = {'zero': 0.05, 'tiny': 0.20, 'small': 0.40, 'medium': 1.0, 'large': 1.0, 'extra_large': 1.0}
        all_samples = []
        for name, items in strata.items():
            n = max(1, int(len(items) * ratios[name])) if items else 0
            all_samples.extend(random.sample(items, min(n, len(items))))

        random.shuffle(all_samples)
        split = int(len(all_samples) * train_split)
        paths = {
            'train': os.path.join(self.tfrecord_dir, f'train_ts{time_steps}.tfrecord'),
            'val':   os.path.join(self.tfrecord_dir, f'val_ts{time_steps}.tfrecord'),
        }
        self._write_tfrecord(all_samples[:split], yearly_data, time_steps, paths['train'])
        self._write_tfrecord(all_samples[split:], yearly_data, time_steps, paths['val'])

        metadata = {
            'time_steps':     time_steps,
            'feature_length': self.feature_length,
            'train_samples':  split,
            'val_samples':    len(all_samples) - split,
            'train_tfrecord': paths['train'],
            'val_tfrecord':   paths['val'],
            'num_countries':  self.num_countries,
            'num_sectors':    self.num_sectors,
            'matrix_shape':   f"{len(self.full_row_labels)}x{len(self.full_col_labels)}",
        }
        with open(os.path.join(self.tfrecord_dir, f'metadata_ts{time_steps}.json'), 'w') as fh:
            json.dump(metadata, fh, indent=2)

        del yearly_data
        gc.collect()
        return metadata

    def _write_tfrecord(self, samples, yearly_data, time_steps, output_path):
        with tf.io.TFRecordWriter(output_path) as writer:
            for pos, ty in tqdm(samples, desc=f"Writing {os.path.basename(output_path)}"):
                f, t = self._construct_focused_feature_label(pos[0], pos[1], ty, time_steps, yearly_data)
                if f is not None and t is not None:
                    f_s = self.scaler_features.transform(f.reshape(-1, self.feature_length)).reshape(f.shape)
                    t_s = self.scaler_target.transform([[t]])[0][0]
                    writer.write(serialize_example(f_s, t_s, time_steps, self.feature_length))

    # ----------------------------------------------------------
    # XGBoost data preparation
    # ----------------------------------------------------------

    def prepare_xgboost_data(self, time_steps, train_split=0.8, max_positions=100_000):
        """Build flattened feature arrays for XGBoost training."""
        print("Preparing XGBoost data...")
        self.load_excel_data()
        self._precompute_yearly_stats()

        years        = [y for y in self._find_longest_consecutive_years() if y < 2022]
        yearly_data  = {y: self.load_single_year_data(y) for y in years}
        target_years = [y for y in years if y >= years[0] + time_steps]
        if not target_years:
            raise ValueError("Insufficient data for XGBoost training.")

        all_positions = [(r, c) for r in self.full_row_labels for c in self.full_col_labels]
        positions     = random.sample(all_positions, min(max_positions, len(all_positions)))

        X_all, y_all = [], []
        for pos in tqdm(positions, desc="Building XGBoost samples"):
            for ty in target_years:
                f, t = self._construct_focused_feature_label(pos[0], pos[1], ty, time_steps, yearly_data)
                if f is not None and t is not None:
                    X_all.append(f.flatten())
                    y_all.append(t)

        if not X_all:
            raise ValueError("No valid XGBoost training samples generated.")

        X = np.array(X_all, dtype=np.float32)
        y = np.log1p(np.array(y_all, dtype=np.float32))
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, train_size=train_split, random_state=42)
        print(f"XGBoost data ready — train: {len(X_tr)}, val: {len(X_val)}")
        return X_tr, X_val, y_tr, y_val

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------

    def _find_longest_consecutive_years(self):
        """Return the longest unbroken run of consecutive valid years."""
        if not self.valid_years:
            return []
        years = sorted(self.valid_years)
        best, cur = [], []
        for i, y in enumerate(years):
            if i == 0 or y == years[i - 1] + 1:
                cur.append(y)
            else:
                if len(cur) > len(best):
                    best = cur
                cur = [y]
        return cur if len(cur) > len(best) else best


# ============================================================
# 5. Transformer model components
# ============================================================

class PositionalEncoding(tf.keras.layers.Layer):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, max_position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.d_model      = d_model
        angles            = self._get_angles(
            np.arange(max_position)[:, None],
            np.arange(d_model)[None, :],
            d_model,
        )
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.constant(angles, dtype=tf.float32)

    @staticmethod
    def _get_angles(pos, i, d_model):
        return pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))

    def call(self, inputs):
        return inputs + tf.cast(self.pos_encoding[:tf.shape(inputs)[1], :], inputs.dtype)

    def get_config(self):
        return {**super().get_config(), 'max_position': self.max_position, 'd_model': self.d_model}


class TransformerBlock(tf.keras.layers.Layer):
    """Single Transformer encoder block with residual connections and layer normalisation."""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads = d_model, num_heads
        self.dff, self.dropout_rate  = dff, dropout_rate
        self.mha     = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn     = tf.keras.Sequential([Dense(dff, activation='relu'), Dense(d_model)])
        self.norm1   = LayerNormalization(epsilon=1e-6)
        self.norm2   = LayerNormalization(epsilon=1e-6)
        self.drop1   = Dropout(dropout_rate)
        self.drop2   = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.norm1(inputs + self.drop1(self.mha(inputs, inputs), training=training))
        return self.norm2(x + self.drop2(self.ffn(x), training=training))

    def get_config(self):
        return {**super().get_config(), 'd_model': self.d_model, 'num_heads': self.num_heads,
                'dff': self.dff, 'dropout_rate': self.dropout_rate}


class OptimizedTimeSeriesTransformer(tf.keras.layers.Layer):
    """Stack of Transformer encoder blocks with input projection and positional encoding."""

    def __init__(self, d_model, num_heads, num_layers, dff, max_position, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads = d_model, num_heads
        self.num_layers, self.dff    = num_layers, dff
        self.max_position            = max_position
        self.dropout_rate            = dropout_rate
        self.input_projection = Dense(d_model)
        self.pos_encoding     = PositionalEncoding(max_position, d_model)
        self.enc_layers       = [TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout          = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.input_projection(inputs)
        x = x * tf.cast(tf.math.sqrt(float(self.d_model)), x.dtype)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training)
        return x

    def get_config(self):
        return {**super().get_config(), 'd_model': self.d_model, 'num_heads': self.num_heads,
                'num_layers': self.num_layers, 'dff': self.dff,
                'max_position': self.max_position, 'dropout_rate': self.dropout_rate}


# ============================================================
# 6. Main predictor
# ============================================================

class HighPerformanceTransformerPredictor:
    """
    End-to-end pipeline:
      1. Bayesian hyperparameter search for the Transformer
      2. Optuna-optimised XGBoost training
      3. Transformer training
      4. SSP scenario-based prediction for future years with RAS balancing
    """

    def __init__(
        self,
        csv_folder_path,
        excel_file_path,
        country_csv_path,
        output_template_path,
        tfrecord_dir="./tfrecords",
        future_features_path=None,
    ):
        self.output_template_path = output_template_path
        self.tfrecord_dir         = tfrecord_dir
        self.data_processor       = TFRecordDataProcessor(
            csv_folder_path, excel_file_path, country_csv_path,
            tfrecord_dir, future_features_path,
        )
        self.feature_length = self.data_processor.feature_length
        self.best_params    = None
        self.unified_model  = None

    # ----------------------------------------------------------
    # Model construction
    # ----------------------------------------------------------

    def build_model_with_params(self, params):
        """Instantiate the Transformer model from a hyperparameter dictionary."""
        scope = STRATEGY.scope() if STRATEGY else tf.name_scope('model')
        with scope:
            inp         = Input(shape=(params['time_steps'], self.feature_length), dtype='float32')
            transformer = OptimizedTimeSeriesTransformer(
                params['d_model'], params['num_heads'], params['num_layers'],
                params['dff'], params['time_steps'], params['dropout_rate'],
            )
            x   = transformer(inp)
            x   = Attention()([x, x])
            x   = Flatten()(x)
            x   = Dropout(params['dropout_rate'])(Dense(params['dense_units'], activation=params['activation'])(x))
            x   = Dropout(params['dropout_rate'])(Dense(params['dense_units'] // 2, activation=params['activation'])(x))
            out = Dense(1, activation='linear', dtype='float32')(x)
            model = Model(inp, out)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                loss=custom_tweedie_loss(p=1.5),
                metrics=['mae'],
            )
        return model

    # ----------------------------------------------------------
    # Hyperparameter optimisation
    # ----------------------------------------------------------

    def bayesian_optimization(self, n_calls=10, time_steps=8, batch_size=2048):
        """Search Transformer hyperparameters via Bayesian optimisation."""
        n_calls  = max(n_calls, 10)
        metadata = self.data_processor.create_tfrecords(time_steps=time_steps)
        ds_tr    = create_tf_data_pipeline(metadata['train_tfrecord'], time_steps, self.feature_length, batch_size)
        ds_val   = create_tf_data_pipeline(metadata['val_tfrecord'],   time_steps, self.feature_length, batch_size, False)
        if ds_tr is None or ds_val is None:
            raise ValueError("Cannot create data pipelines for optimisation.")

        print(f"Bayesian optimisation — {n_calls} calls, time_steps={time_steps}...")
        fixed = {'batch_size': batch_size, 'activation': 'gelu', 'time_steps': time_steps}
        dims  = [
            Real(1e-5, 1e-3,  name='learning_rate', prior='log-uniform'),
            Integer(64,  128, name='d_model'),
            Integer(2,   8,   name='num_heads'),
            Integer(1,   2,   name='num_layers'),
            Integer(128, 256, name='dff'),
            Real(0.1,   0.3,  name='dropout_rate'),
            Integer(64,  128, name='dense_units'),
        ]

        @use_named_args(dims)
        def objective(**params):
            full_params = {**params, **fixed}
            try:
                model   = self.build_model_with_params(full_params)
                es      = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
                history = model.fit(ds_tr, validation_data=ds_val, epochs=15, verbose=0, callbacks=[es])
                val_loss = min(history.history.get('val_loss', [1e10]))
                del model
                gc.collect()
                tf.keras.backend.clear_session()
                return val_loss
            except Exception as e:
                print(f"Trial error: {e}")
                return 1e10

        result           = gp_minimize(objective, dims, n_calls=n_calls, random_state=42)
        self.best_params = {dim.name: val for dim, val in zip(dims, result.x)}
        self.best_params.update(fixed)

        with open(os.path.join(self.tfrecord_dir, 'best_params.json'), 'w') as fh:
            json.dump(
                {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                 for k, v in self.best_params.items()},
                fh, indent=2,
            )
        print(f"Best hyperparameters: {self.best_params}")

    def tune_xgboost_model(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Search XGBoost hyperparameters via Optuna."""
        print(f"Tuning XGBoost ({n_trials} trials)...")

        def objective(trial):
            params = {
                'objective':              'reg:tweedie',
                'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
                'tree_method':            'hist',
                'n_estimators':           trial.suggest_int('n_estimators', 500, 2000, step=100),
                'learning_rate':          trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth':              trial.suggest_int('max_depth', 4, 10),
                'subsample':              trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree':       trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight':       trial.suggest_int('min_child_weight', 1, 10),
                'gamma':                  trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'random_state': 42, 'n_jobs': -1,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=1800)
        best = study.best_params
        best.update({'objective': 'reg:tweedie', 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1})
        return best

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def train_all_models(self, train_epochs=30):
        """Train XGBoost and Transformer models using the optimised hyperparameters."""
        if self.best_params is None:
            raise ValueError("Run bayesian_optimization() before training.")

        time_steps = self.best_params['time_steps']
        batch_size = self.best_params['batch_size']

        with open(os.path.join(self.tfrecord_dir, f'metadata_ts{time_steps}.json'), 'r') as fh:
            metadata = json.load(fh)

        print("Training XGBoost model...")
        X_tr, X_val, y_tr, y_val = self.data_processor.prepare_xgboost_data(time_steps)
        best_xgb               = self.tune_xgboost_model(X_tr, y_tr, X_val, y_val)
        self.unified_model     = xgb.XGBRegressor(**best_xgb)
        self.unified_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
        joblib.dump(self.unified_model, os.path.join(self.tfrecord_dir, 'best_model_unified.joblib'))

        print(f"Training Transformer model (epochs={train_epochs})...")
        ds_tr  = create_tf_data_pipeline(metadata['train_tfrecord'], time_steps, self.feature_length, batch_size)
        ds_val = create_tf_data_pipeline(metadata['val_tfrecord'],   time_steps, self.feature_length, batch_size, False)
        if ds_tr and ds_val:
            transformer = self.build_model_with_params(self.best_params)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.tfrecord_dir, 'best_model_transformer'),
                    save_best_only=True,
                ),
            ]
            history = transformer.fit(ds_tr, validation_data=ds_val, epochs=train_epochs, callbacks=callbacks)
            with open(os.path.join(self.tfrecord_dir, 'training_history.json'), 'w') as fh:
                json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, fh, indent=2)
            return history

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    def _construct_prediction_features(self, r, c, input_years, yearly_data, pred_year):
        """
        Build a feature array for cell (r, c) using historical input_years and
        future-year macro data from the SSP scenario lookup.
        """
        src_c, dst_c = r.split('_')[0], c.split('_')[0]

        pop_src, gdp_src = self.data_processor.country_year_lookup.get((src_c, pred_year), (0, 0))
        _,       gdp_dst = self.data_processor.country_year_lookup.get((dst_c, pred_year), (0, 0))
        pop_src = pop_src if not np.isnan(pop_src) else 0
        gdp_src = max(0, gdp_src) if not np.isnan(gdp_src) else 0
        gdp_dst = max(0, gdp_dst) if not np.isnan(gdp_dst) else 0

        years_ahead  = float(pred_year - max(input_years))
        default_feat = np.zeros(
            self.data_processor.economic_status_cats + self.data_processor.trade_bloc_cats, dtype=np.float32
        )
        src_f = self.data_processor.country_feature_lookup.get(src_c, default_feat)
        dst_f = self.data_processor.country_feature_lookup.get(dst_c, default_feat)

        all_features = []
        for year in input_years:
            matrix = yearly_data.get(year)
            stats  = self.data_processor.yearly_stats_cache.get(year)
            if matrix is None or stats is None:
                return None

            o_stats     = stats['outflow_stats'].get(r,   (0, 0, 0))
            i_stats     = stats['inflow_stats'].get(c,    (0, 0, 0))
            total_out   = stats['total_outflow'].get(r,    0)
            total_in    = stats['total_inflow'].get(c,     0)
            country_out = stats['country_total_outflow'].get(src_c, 0)

            gdp_sh = self.data_processor.country_year_lookup.get((src_c, year), (0, 0))[1]
            gdp_dh = self.data_processor.country_year_lookup.get((dst_c, year), (0, 0))[1]
            gdp_sh = gdp_sh if not np.isnan(gdp_sh) else 1
            gdp_dh = gdp_dh if not np.isnan(gdp_dh) else 1

            step   = float(pred_year - year)
            cf_src = np.clip(gdp_src / (gdp_sh + 1e-9), 0.5, 3.0)
            cf_dst = np.clip(gdp_dst / (gdp_dh + 1e-9), 0.5, 3.0)
            gr_src = cf_src ** (1.0 / max(1, step)) - 1.0
            gr_dst = cf_dst ** (1.0 / max(1, step)) - 1.0

            is_diagonal = 1.0 if r == c else 0.0

            base = np.array([
                np.log1p(float(matrix.loc[r, c])),
                np.log1p(pop_src), np.log1p(gdp_src),
                np.log1p(o_stats[0]), np.log1p(o_stats[1]), np.log1p(o_stats[2]),
                np.log1p(i_stats[0]), np.log1p(i_stats[1]), np.log1p(i_stats[2]),
                np.log1p(total_out), np.log1p(total_in),
                np.log1p(gdp_src) * np.log1p(gdp_dst),
                total_out / (country_out + 1e-9),
                is_diagonal,                          # self-consumption indicator
            ], dtype=np.float32)

            growth = np.array([years_ahead, gr_src, gr_dst, cf_src * cf_dst], dtype=np.float32)
            vec    = np.nan_to_num(np.concatenate([base, growth, src_f, dst_f]), nan=0.0, posinf=0.0, neginf=0.0)
            all_features.append(vec)

        return np.array(all_features, dtype=np.float32) if len(all_features) == len(input_years) else None

    def _load_transformer(self):
        """Load the trained Transformer model from disk with custom layer registration."""
        transformer_path = os.path.join(self.tfrecord_dir, 'best_model_transformer')
        if not os.path.exists(transformer_path):
            print("Warning: Transformer model not found — non-diagonal cells will fall back to XGBoost.")
            return None
        try:
            custom_objects = {
                'OptimizedTimeSeriesTransformer': OptimizedTimeSeriesTransformer,
                'TransformerBlock':               TransformerBlock,
                'PositionalEncoding':             PositionalEncoding,
                'loss':                           custom_tweedie_loss(p=1.5),
            }
            model = tf.keras.models.load_model(transformer_path, custom_objects=custom_objects)
            print("Transformer model loaded successfully.")
            return model
        except Exception as e:
            print(f"Failed to load Transformer model: {e}")
            return None

    def _predict_one_year(self, prediction_year, historical_data, last_historical_year):
        """
        Generate raw cell-level predictions for a given future year using the hybrid model:
          - Diagonal cells (self-consumption): XGBoost regressor
          - Non-diagonal cells (inter-sector/inter-region flows): Transformer

        Both predictions are blended with a conservative historical growth baseline.
        """
        print(f"\nPredicting year {prediction_year}...")
        time_steps  = self.best_params['time_steps']
        input_years = list(range(last_historical_year - time_steps + 1, last_historical_year + 1))

        # Load scalers (needed to scale features for the Transformer)
        with open(os.path.join(self.tfrecord_dir, 'scalers.pkl'), 'rb') as fh:
            scalers         = pickle.load(fh)
            scaler_features = scalers['scaler_features']

        # Load Transformer for non-diagonal cells
        transformer_model = self._load_transformer()

        pred_df = pd.DataFrame(
            0.0,
            index=self.data_processor.full_row_labels,
            columns=self.data_processor.full_col_labels,
        )
        hist_mat      = historical_data[last_historical_year]
        years_diff    = prediction_year - last_historical_year
        annual_growth = 1.03   # conservative baseline annual growth rate

        # Partition cells into diagonal (self-consumption) and non-diagonal
        diag_X,     diag_pos     = [], []
        non_diag_X, non_diag_pos = [], []

        for r in tqdm(self.data_processor.full_row_labels, desc=f"Building features ({prediction_year})"):
            for c in self.data_processor.full_col_labels:
                X = self._construct_prediction_features(r, c, input_years, historical_data, prediction_year)
                if X is not None:
                    if r == c:
                        diag_X.append(X)
                        diag_pos.append((r, c))
                    else:
                        non_diag_X.append(X)
                        non_diag_pos.append((r, c))

        def _apply_growth_blend(preds_raw, pos_list):
            """Blend model output with historical growth baseline and apply bounds."""
            for i, (r, c) in enumerate(pos_list):
                hist_val = float(hist_mat.loc[r, c])
                expected = hist_val * np.clip(annual_growth ** years_diff, 0.8, 1.5)
                model    = max(0.0, preds_raw[i])
                if model < expected * 0.5:
                    final = expected
                elif model > expected * 3.0:
                    final = expected * 1.5
                else:
                    final = 0.7 * model + 0.3 * expected
                pred_df.loc[r, c] = np.clip(final, hist_val * 0.7, hist_val * 1.3)

        # --- XGBoost: diagonal cells ---
        if diag_X:
            preds_raw = np.expm1(
                self.unified_model.predict(np.array([x.flatten() for x in diag_X]))
            )
            _apply_growth_blend(preds_raw, diag_pos)
            print(f"Diagonal predictions complete ({len(diag_pos)} cells via XGBoost).")

        # --- Transformer: non-diagonal cells ---
        if non_diag_X:
            if transformer_model is not None:
                X_arr = np.array(non_diag_X)           # (B, time_steps, feature_length)
                B, T, F = X_arr.shape
                X_scaled  = scaler_features.transform(X_arr.reshape(-1, F)).reshape(B, T, F)
                preds_raw = np.expm1(transformer_model.predict(X_scaled, verbose=0).flatten())
                print(f"Non-diagonal predictions complete ({len(non_diag_pos)} cells via Transformer).")
            else:
                # Fallback to XGBoost when Transformer is unavailable
                print("Falling back to XGBoost for non-diagonal cells.")
                preds_raw = np.expm1(
                    self.unified_model.predict(np.array([x.flatten() for x in non_diag_X]))
                )
            _apply_growth_blend(preds_raw, non_diag_pos)

        print(f"Total: {len(diag_pos)} diagonal + {len(non_diag_pos)} non-diagonal cells.")
        pred_df.clip(lower=0.0, inplace=True)
        return pred_df

    def _ras_balance(self, initial_df, pred_year, last_historical_year, historical_data,
                     max_iter=100, tolerance=1e-4, annual_growth_rate=1.035):
        """
        Balance the predicted matrix using iterative proportional fitting (RAS).
        Target row/column marginals are derived from the last observed matrix scaled
        by a compound growth factor.

        Parameters
        ----------
        annual_growth_rate : float
            Expected annual growth applied to historical marginals (default: 3.5%).
        """
        print(f"Applying RAS balancing for {pred_year}...")
        hist_mat   = historical_data[last_historical_year]
        years_diff = pred_year - last_historical_year
        growth     = annual_growth_rate ** years_diff

        u_target = hist_mat.sum(axis=1) * growth
        v_target = hist_mat.sum(axis=0) * growth
        u_target[u_target < 1e-6] = 1e-6
        v_target[v_target < 1e-6] = 1e-6

        Z = initial_df.values.copy()
        Z[Z < 0] = 0

        # Pre-scale if initial total deviates substantially from target
        init_total   = Z.sum()
        target_total = u_target.sum()
        if init_total < target_total * 0.5 or init_total > target_total * 2.0:
            print(f"Pre-scaling matrix (initial={init_total:,.0f}, target={target_total:,.0f}).")
            Z *= target_total / (init_total + 1e-9)

        Z[Z == 0] = 1e-12
        u_np, v_np = u_target.values, v_target.values
        prev_err   = float('inf')

        for it in range(max_iter):
            r_sum = Z.sum(axis=1); r_sum[r_sum < 1e-12] = 1e-12
            Z    *= np.clip(u_np / r_sum, 0.8, 1.25)[:, None]

            c_sum = Z.sum(axis=0); c_sum[c_sum < 1e-12] = 1e-12
            Z    *= np.clip(v_np / c_sum, 0.8, 1.25)[None, :]

            err = np.abs(Z.sum(axis=1) - u_np).sum() + np.abs(Z.sum(axis=0) - v_np).sum()
            if it % 10 == 0:
                print(f"  Iteration {it + 1}: error={err:.2e}")

            if err < tolerance * Z.sum() or (it > 20 and abs(err - prev_err) / max(prev_err, 1e-9) < 0.001):
                print(f"RAS converged at iteration {it + 1}.")
                break
            prev_err = err
        else:
            print(f"RAS did not fully converge after {max_iter} iterations (final error={err:.2e}).")

        result           = pd.DataFrame(Z, index=initial_df.index, columns=initial_df.columns)
        result[result < 0] = 0
        print(f"RAS complete. Matrix total: {result.sum().sum():,.0f}")
        return result

    def _diagnose_prediction(self, pred_df, year, stage, historical_data=None):
        """Print summary statistics for a predicted matrix (quality check)."""
        print(f"\n{'=' * 50}")
        print(f"Diagnostics — {year} ({stage})")
        print(f"{'=' * 50}")
        total = pred_df.sum().sum()
        zeros = (pred_df.values == 0).sum()
        nans  = pred_df.isna().sum().sum()
        print(f"Total:  {total:,.0f}")
        print(f"Shape:  {pred_df.shape[0]} × {pred_df.shape[1]}")
        print(f"Zeros:  {zeros} ({100 * zeros / pred_df.size:.1f}%)")
        if nans:
            print(f"NaNs:   {nans}")

        if historical_data:
            last_year  = max(historical_data)
            hist_total = historical_data[last_year].sum().sum()
            if year > last_year and hist_total > 0:
                diff = year - last_year
                cagr = (total / hist_total) ** (1.0 / diff) - 1
                print(f"vs {last_year}: CAGR={cagr * 100:.2f}%, cumulative growth={100 * (total / hist_total - 1):.1f}%")

        row_sums = pred_df.sum(axis=1)
        col_sums = pred_df.sum(axis=0)
        print(f"Row sums — min: {row_sums.min():,.0f}, max: {row_sums.max():,.0f}, mean: {row_sums.mean():,.0f}")
        print(f"Col sums — min: {col_sums.min():,.0f}, max: {col_sums.max():,.0f}, mean: {col_sums.mean():,.0f}")

    # ----------------------------------------------------------
    # Forecast pipeline
    # ----------------------------------------------------------

    def forecast_future_years(self, years_to_forecast):
        """
        Generate RAS-balanced predictions for each year in years_to_forecast
        using SSP2 scenario GDP/population data loaded at initialisation.
        """
        if self.unified_model is None:
            raise ValueError("No trained model found. Run load_existing_model() or run_pipeline() first.")

        self.data_processor.load_excel_data()
        self.data_processor._precompute_yearly_stats()

        print("Loading historical matrices...")
        historical_data      = {y: self.data_processor.load_single_year_data(y)
                                for y in self.data_processor.valid_years}
        last_historical_year = max(historical_data)
        print(f"Latest historical year (baseline): {last_historical_year}")

        output_paths = []
        for year in sorted(years_to_forecast):
            raw_pred = self._predict_one_year(year, historical_data, last_historical_year)
            self._diagnose_prediction(raw_pred, year, "raw prediction", historical_data)

            balanced = self._ras_balance(raw_pred, year, last_historical_year, historical_data)
            self._diagnose_prediction(balanced, year, "after RAS", historical_data)

            out_path = self._get_output_path(year)
            self._save_matrix(out_path, balanced)
            output_paths.append(out_path)

        print(f"\nAll predictions complete: {sorted(years_to_forecast)}")
        return output_paths

    def run_pipeline(self, optimization_calls=10, train_epochs=25, time_steps=8,
                     batch_size=2048, years_to_forecast=None):
        """Full end-to-end pipeline: optimise hyperparameters → train → forecast."""
        if years_to_forecast is None:
            years_to_forecast = YEARS_TO_FORECAST
        print("=" * 70)
        print("MRIO Future Prediction Pipeline (SSP Scenario)")
        print("=" * 70)
        try:
            if not self.best_params:
                print("\nStep 1: Bayesian hyperparameter optimisation")
                self.bayesian_optimization(n_calls=optimization_calls,
                                           time_steps=time_steps, batch_size=batch_size)
            print("\nStep 2: Model training")
            self.train_all_models(train_epochs=train_epochs)
            print(f"\nStep 3: Forecasting {years_to_forecast}")
            return self.forecast_future_years(years_to_forecast)
        except Exception as e:
            print(f"Pipeline failed: {e}")
            traceback.print_exc()
            raise

    # ----------------------------------------------------------
    # Persistence helpers
    # ----------------------------------------------------------

    def _save_matrix(self, path, df):
        """Save a balanced prediction matrix to CSV."""
        df.reindex(
            index=self.data_processor.full_row_labels,
            columns=self.data_processor.full_col_labels,
            fill_value=0.0,
        ).to_csv(path, encoding='utf-8')
        print(f"Saved: {path}")

    def _get_output_path(self, year):
        base, _ = os.path.splitext(self.output_template_path)
        return f"{base}_prediction_balanced_{year}.csv"

    def load_existing_model(self):
        """Load a pre-trained XGBoost model and hyperparameters from disk."""
        params_path = os.path.join(self.tfrecord_dir, 'best_params.json')
        model_path  = os.path.join(self.tfrecord_dir, 'best_model_unified.joblib')

        if not os.path.exists(params_path):
            return False
        with open(params_path, 'r') as fh:
            self.best_params = json.load(fh)

        if os.path.exists(model_path):
            try:
                self.unified_model = joblib.load(model_path)
                print("Existing XGBoost model loaded.")
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
        return False


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    for path in [CSV_DIR, HIST_CSV, FUTURE_CSV, COUNTRY_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input not found: {path}")

    os.makedirs(os.path.dirname(OUTPUT_TEMPLATE_PATH) or '.', exist_ok=True)
    os.makedirs(TFRECORD_DIR, exist_ok=True)

    predictor = HighPerformanceTransformerPredictor(
        csv_folder_path      = CSV_DIR,
        excel_file_path      = HIST_CSV,
        country_csv_path     = COUNTRY_CSV,
        future_features_path = FUTURE_CSV,
        output_template_path = OUTPUT_TEMPLATE_PATH,
        tfrecord_dir         = TFRECORD_DIR,
    )
    tf.keras.backend.clear_session()

    if predictor.load_existing_model():
        print("Existing model found — skipping training.")
        predictor.data_processor._precompute_yearly_stats()
        predictor.forecast_future_years(years_to_forecast=YEARS_TO_FORECAST)
    else:
        print("No existing model found — running full pipeline.")
        predictor.run_pipeline(
            optimization_calls = OPTIMIZATION_CALLS,
            train_epochs       = TRAIN_EPOCHS,
            time_steps         = TIME_STEPS,
            batch_size         = BATCH_SIZE,
            years_to_forecast  = YEARS_TO_FORECAST,
        )

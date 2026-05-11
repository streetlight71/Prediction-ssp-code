"""
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
"""

import gc
import glob
import json
import os
import pickle
import random
import re
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


# ============================================================
# User-configurable paths — set these before running
# ============================================================
CSV_DIR          = "path/to/mrio_csv_folder"           # Folder of annual MRIO CSV files
HIST_CSV         = "path/to/gdp_population.csv"         # Historical GDP and population
SSP2_CSV         = "path/to/ssp_gdp_population.csv"    # SSP2 scenario GDP/population
COUNTRY_CSV      = "path/to/country_features.csv"        # Static country-level features
OUTPUT_TEMPLATE  = "path/to/output/prediction_matrix"   # Output filename stem
TFRECORD_DIR     = "path/to/cache/tfrecords"

# Forecast configuration
YEARS_TO_FORECAST = [2025, 2030]
TIME_STEPS        = 5


# ============================================================
# 1. GPU setup
# ============================================================

def setup_gpu():
    """Enable mixed-precision training and configure distribution strategy."""
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
            print(f"Mixed-precision enabled; using {'multi-GPU' if len(gpus) > 1 else 'single-GPU'} strategy.")
            return True, strategy
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found; using CPU.")
    return False, None


GPU_AVAILABLE, STRATEGY = setup_gpu()


# ============================================================
# 2. Custom loss
# ============================================================

def custom_tweedie_loss(p=1.5):
    """
    Tweedie deviance loss for non-negative, heavy-tailed targets.

    Args:
        p: Tweedie variance power (1 < p < 2).
    """
    def loss(y_true, y_pred):
        eps    = 1e-7
        y_pred = tf.maximum(y_pred, eps)
        y_true = tf.maximum(y_true, 0.0)
        return tf.reduce_mean(
            tf.pow(y_pred, 2 - p) / (2 - p)
            - y_true * tf.pow(y_pred, 1 - p) / (1 - p)
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


def create_tf_data_pipeline(
    tfrecord_path, time_steps, feature_length,
    batch_size, is_training=True, buffer_size=10000,
):
    """Build a batched tf.data pipeline from a TFRecord file."""
    if not os.path.exists(tfrecord_path) or os.path.getsize(tfrecord_path) == 0:
        print(f"Warning: TFRecord file missing or empty: {tfrecord_path}")
        return None
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(
        lambda x: parse_tfrecord_function(x, time_steps, feature_length),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.cache()
    if is_training:
        ds = ds.shuffle(buffer_size=buffer_size)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================
# 4. Data processing
# ============================================================

class TFRecordDataProcessor:
    """
    Loads non-square annual MRIO CSV matrices and country-level covariates,
    constructs time-series feature tensors, and writes TFRecord files
    for model training.

    The matrix is non-square: rows index (source country × sector) and
    columns index (destination country × sector), which may differ.
    """

    def __init__(
        self,
        csv_folder_path,
        excel_file_path,
        country_csv_path,
        tfrecord_dir="./tfrecords",
        future_excel_path=None,
    ):
        self.csv_folder_path   = csv_folder_path
        self.excel_file_path   = excel_file_path
        self.future_excel_path = future_excel_path
        self.tfrecord_dir      = tfrecord_dir

        self.scaler_features    = MinMaxScaler()
        self.scaler_target      = FunctionTransformer(
            func=np.log1p, inverse_func=np.expm1, validate=False
        )
        self.country_year_lookup  = {}
        self.yearly_stats_cache   = {}

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
        print(f"Matrix structure: {len(self.full_row_labels)} rows × {len(self.full_col_labels)} cols")

        self.valid_years = self._validate_years()

        self.country_feature_lookup = self._load_country_features(country_csv_path)
        self.economic_status_cats   = 2
        self.trade_bloc_cats        = 4

        # Feature vector length:
        # 13 base features + 4 growth features + 2 × (2 + 4) country one-hot = 29
        self.feature_length = 13 + 4 + 2 * (self.economic_status_cats + self.trade_bloc_cats)

    # ----------------------------------------------------------
    # Matrix label extraction
    # ----------------------------------------------------------

    def _get_original_matrix_labels(self):
        """Extract row and column labels from the most recent MRIO CSV."""
        if not self.all_csv_files:
            raise FileNotFoundError(f"No CSV files found in: {self.csv_folder_path}")
        ref_file = self.year_to_file_map[max(self.year_to_file_map)]
        df       = self._load_and_process_mrio_csv(ref_file)
        print(f"Row labels: {len(df.index)}, column labels: {len(df.columns)}")
        return df.index.tolist(), df.columns.tolist()

    # ----------------------------------------------------------
    # CSV loading
    # ----------------------------------------------------------

    def _load_and_process_mrio_csv(self, file_path):
        """Read a raw MRIO CSV; return a DataFrame with (country_sector) labels."""
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']:
            try:
                df_raw = pd.read_csv(file_path, header=None, dtype=str, encoding=encoding)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Cannot load: {os.path.basename(file_path)}")

        country_cols = df_raw.iloc[0, 2:].ffill()
        sector_cols  = df_raw.iloc[1, 2:]
        cols = [f"{str(co).strip()}_{str(se).strip()}" for co, se in zip(country_cols, sector_cols)]

        country_rows = df_raw.iloc[2:, 0].ffill()
        sector_rows  = df_raw.iloc[2:, 1]
        rows = [f"{str(co).strip()}_{str(se).strip()}" for co, se in zip(country_rows, sector_rows)]

        data = df_raw.iloc[2:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return pd.DataFrame(data.values, index=rows, columns=cols)

    def load_single_year_data(self, year, validate_only=False):
        """Load and align the MRIO matrix for a given year."""
        if year not in self.year_to_file_map:
            return None
        try:
            df = self._load_and_process_mrio_csv(self.year_to_file_map[year])
            if validate_only:
                return df
            return df.reindex(
                index=self.full_row_labels, columns=self.full_col_labels
            ).fillna(0.0).astype(np.float32)
        except Exception as e:
            raise IOError(f"Error loading year {year}: {e}")

    def _validate_years(self):
        """Return the list of years for which the CSV files load successfully."""
        valid = []
        for year in sorted(self.year_to_file_map):
            try:
                data = self.load_single_year_data(year, validate_only=True)
                if data is not None and not data.empty:
                    valid.append(year)
                del data
                gc.collect()
            except Exception as e:
                print(f"Year {year} validation failed: {e}")
        return valid

    # ----------------------------------------------------------
    # Country and economic feature loading
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
            raise IOError(f"Cannot decode: {file_path}")

        df['economic_status'] = pd.Categorical(
            df['economic_status'], categories=['Developed', 'Developing'], ordered=False
        )
        df['trade_bloc'] = pd.Categorical(
            df['trade_bloc'], categories=['EU', 'NAFTA', 'ASEAN', 'Other'], ordered=False
        )
        feature_df = pd.concat(
            [df['country_code'],
             pd.get_dummies(df['economic_status'], prefix='status'),
             pd.get_dummies(df['trade_bloc'],       prefix='bloc')],
            axis=1,
        )
        lookup = {
            row.country_code: np.array(row[1:], dtype=np.float32)
            for row in feature_df.itertuples(index=False)
        }
        print(f"Country features loaded for {len(lookup)} countries.")
        return lookup

    # ----------------------------------------------------------
    # GDP/population lookup
    # ----------------------------------------------------------

    def load_excel_data(self):
        """Build a (country, year) → (population, GDP) lookup from historical and SSP CSVs."""
        if not os.path.exists(self.excel_file_path):
            raise FileNotFoundError(f"Historical features file not found: {self.excel_file_path}")

        df_hist = self._load_feature_csv(self.excel_file_path)
        self.country_year_lookup = dict(
            zip(zip(df_hist['country'], df_hist['year']),
                zip(df_hist['Population'], df_hist['gdp']))
        )
        print(f"Loaded {len(self.country_year_lookup)} historical (country, year) entries.")

        if self.future_excel_path and os.path.exists(self.future_excel_path):
            df_future    = self._load_feature_csv(self.future_excel_path)
            future_lookup = dict(
                zip(zip(df_future['country'], df_future['year']),
                    zip(df_future['Population'], df_future['gdp']))
            )
            self.country_year_lookup.update(future_lookup)
            print(f"Merged {len(future_lookup)} SSP scenario (country, year) entries.")

    def _load_feature_csv(self, file_path):
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']:
            try:
                df = pd.read_csv(
                    file_path, engine='c',
                    dtype={'country': str, 'year': int}, encoding=encoding,
                )
                df.columns = df.columns.str.strip()
                required = {'country', 'year', 'Population', 'gdp'}
                if not required.issubset(df.columns):
                    raise ValueError(f"Missing columns in {file_path}: {required - set(df.columns)}")
                return df
            except UnicodeDecodeError:
                continue
        raise IOError(f"Cannot load: {file_path}")

    # ----------------------------------------------------------
    # Annual statistics cache
    # ----------------------------------------------------------

    def _precompute_yearly_stats(self):
        """Cache outflow/inflow statistics and country-level totals for each training year."""
        print("Precomputing annual flow statistics...")
        for year in tqdm([y for y in self.valid_years if y <= 2022]):
            try:
                df = self._load_and_process_mrio_csv(self.year_to_file_map[year])
                df = df.reindex(index=self.full_row_labels, columns=self.full_col_labels).fillna(0.0)

                total_outflow = df.sum(axis=1)
                total_outflow_df = total_outflow.reset_index()
                total_outflow_df.columns = ['label', 'sector_outflow']
                total_outflow_df['country'] = total_outflow_df['label'].str.split('_').str[0]
                country_total = total_outflow_df.groupby('country')['sector_outflow'].sum()

                self.yearly_stats_cache[year] = {
                    'outflow_stats':        dict(zip(df.index,   zip(df.mean(axis=1), df.std(axis=1), df.max(axis=1)))),
                    'inflow_stats':         dict(zip(df.columns, zip(df.mean(axis=0), df.std(axis=0), df.max(axis=0)))),
                    'total_outflow':        total_outflow.to_dict(),
                    'total_inflow':         df.sum(axis=0).to_dict(),
                    'country_total_outflow': country_total.to_dict(),
                }
                del df, total_outflow_df, country_total
                gc.collect()
            except Exception as e:
                print(f"Warning: could not compute stats for year {year}: {e}")

    # ----------------------------------------------------------
    # Feature construction
    # ----------------------------------------------------------

    def _safe_scalar(self, val, fallback=0.0):
        """Return val if finite, else fallback."""
        try:
            v = float(val)
            return v if np.isfinite(v) else fallback
        except (TypeError, ValueError):
            return fallback

    def _construct_focused_feature_label(self, row_label, col_label, target_year, time_steps, yearly_data):
        """
        Build a (time_steps, feature_length) feature array and scalar target
        for the cell (row_label, col_label) in the target_year MRIO matrix.

        Returns (None, None) if required data is missing.
        """
        src_c = row_label.split('_')[0]
        dst_c = col_label.split('_')[0]

        pop_src, gdp_src = self.country_year_lookup.get((src_c, target_year), (0, 0))
        _,       gdp_dst = self.country_year_lookup.get((dst_c, target_year), (0, 0))
        gdp_src = self._safe_scalar(gdp_src)
        gdp_dst = self._safe_scalar(gdp_dst)
        pop_src = self._safe_scalar(pop_src)

        all_features = []
        for year in range(target_year - time_steps, target_year):
            matrix = yearly_data.get(year)
            stats  = self.yearly_stats_cache.get(year)
            if matrix is None or stats is None:
                return None, None
            if row_label not in matrix.index or col_label not in matrix.columns:
                return None, None

            cell_val      = float(matrix.loc[row_label, col_label])
            o_stats       = stats['outflow_stats'].get(row_label, (0, 0, 0))
            i_stats       = stats['inflow_stats'].get(col_label, (0, 0, 0))
            total_out     = stats['total_outflow'].get(row_label, 0)
            total_in      = stats['total_inflow'].get(col_label, 0)
            country_out   = stats['country_total_outflow'].get(src_c, 0)
            outflow_ratio = total_out / (country_out + 1e-9)

            _, gdp_src_h = self.country_year_lookup.get((src_c, year), (0, 0))
            _, gdp_dst_h = self.country_year_lookup.get((dst_c, year), (0, 0))
            gdp_src_h = self._safe_scalar(gdp_src_h, fallback=1.0)
            gdp_dst_h = self._safe_scalar(gdp_dst_h, fallback=1.0)

            years_ahead = float(target_year - year)
            cf_src = np.clip(gdp_src / (gdp_src_h + 1e-9), 0.5, 3.0)
            cf_dst = np.clip(gdp_dst / (gdp_dst_h + 1e-9), 0.5, 3.0)
            gr_src = cf_src ** (1.0 / max(1, years_ahead)) - 1.0
            gr_dst = cf_dst ** (1.0 / max(1, years_ahead)) - 1.0

            base_feat = np.array([
                np.log1p(max(0, cell_val)),
                np.log1p(pop_src),
                np.log1p(gdp_src),
                np.log1p(max(0, o_stats[0])),
                np.log1p(max(0, o_stats[1])),
                np.log1p(max(0, o_stats[2])),
                np.log1p(max(0, i_stats[0])),
                np.log1p(max(0, i_stats[1])),
                np.log1p(max(0, i_stats[2])),
                np.log1p(max(0, total_out)),
                np.log1p(max(0, total_in)),
                np.log1p(gdp_src) * np.log1p(gdp_dst),   # gravity interaction
                outflow_ratio,
            ], dtype=np.float32)

            growth_feat = np.array([years_ahead, gr_src, gr_dst, cf_src * cf_dst], dtype=np.float32)

            default_vec = np.zeros(self.economic_status_cats + self.trade_bloc_cats, dtype=np.float32)
            src_f = self.country_feature_lookup.get(src_c, default_vec)
            dst_f = self.country_feature_lookup.get(dst_c, default_vec)

            feat = np.concatenate([base_feat, growth_feat, src_f, dst_f])
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            all_features.append(feat)

        target_mat = yearly_data.get(target_year)
        if target_mat is None or row_label not in target_mat.index or col_label not in target_mat.columns:
            return None, None
        target = float(target_mat.loc[row_label, col_label])
        if not np.isfinite(target) or len(all_features) != time_steps:
            return None, None

        return np.array(all_features, dtype=np.float32), max(0.0, target)

    # ----------------------------------------------------------
    # Consecutive year detection
    # ----------------------------------------------------------

    def _find_longest_consecutive_years(self):
        """Return the longest run of consecutive valid years."""
        if not self.valid_years:
            return []
        years   = sorted(self.valid_years)
        longest, current = [], []
        for i, y in enumerate(years):
            if i == 0 or y == years[i - 1] + 1:
                current.append(y)
            else:
                if len(current) > len(longest):
                    longest = current
                current = [y]
        return current if len(current) > len(longest) else longest

    # ----------------------------------------------------------
    # Scaler fitting
    # ----------------------------------------------------------

    def prepare_scalers(self, time_steps, sample_size=50000):
        """Fit feature and target scalers on a random subsample of training cells."""
        print("Fitting scalers...")
        training_years = [y for y in self._find_longest_consecutive_years() if y < 2022]
        if len(training_years) < time_steps + 1:
            raise ValueError(f"Need at least {time_steps + 1} consecutive years before 2022.")

        yearly_data  = {y: self.load_single_year_data(y) for y in training_years}
        target_years = [y for y in training_years if y >= training_years[0] + time_steps]
        all_positions = [(r, c) for r in self.full_row_labels for c in self.full_col_labels]
        sample_pos    = random.sample(all_positions, min(sample_size, len(all_positions)))

        X_sample, y_sample = [], []
        for pos in tqdm(sample_pos, desc="Sampling for scalers"):
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
        y_arr  = np.array(y_sample).reshape(-1, 1)
        self.scaler_features.fit(X_flat)
        self.scaler_target.fit(y_arr)
        print(f"Scalers fitted on {len(X_flat)} samples.")

        with open(os.path.join(self.tfrecord_dir, 'scalers.pkl'), 'wb') as fh:
            pickle.dump({'scaler_features': self.scaler_features, 'scaler_target': self.scaler_target}, fh)

        del yearly_data, X_sample, y_sample
        gc.collect()

    # ----------------------------------------------------------
    # TFRecord generation
    # ----------------------------------------------------------

    def create_tfrecords(self, time_steps, train_split=0.8):
        """Generate stratified train/val TFRecord files for Transformer training."""
        print(f"Generating TFRecords (time_steps={time_steps})...")
        self.load_excel_data()
        self._precompute_yearly_stats()
        self.prepare_scalers(time_steps)

        training_years = [y for y in self._find_longest_consecutive_years() if y < 2022]
        yearly_data    = {y: self.load_single_year_data(y) for y in training_years}
        target_years   = [y for y in training_years if y >= training_years[0] + time_steps]
        if not target_years:
            raise ValueError("Not enough data to generate training targets before 2022.")

        all_positions    = [(r, c) for r in self.full_row_labels for c in self.full_col_labels]
        potential_samples = [(pos, yr) for pos in all_positions for yr in target_years]

        # Stratified sampling by value magnitude
        strata = {k: [] for k in ["zero", "tiny", "small", "medium", "large", "extra_large"]}
        for sample in tqdm(potential_samples, desc="Stratifying samples"):
            pos, ty = sample
            _, t = self._construct_focused_feature_label(pos[0], pos[1], ty, time_steps, yearly_data)
            if t is None:
                continue
            if   t == 0:              strata["zero"].append(sample)
            elif t <= 50:             strata["tiny"].append(sample)
            elif t <= 1_000:          strata["small"].append(sample)
            elif t <= 10_000:         strata["medium"].append(sample)
            elif t <= 100_000:        strata["large"].append(sample)
            else:                     strata["extra_large"].append(sample)

        ratios = {"zero": 0.05, "tiny": 0.20, "small": 0.40,
                  "medium": 1.0, "large": 1.0, "extra_large": 1.0}
        all_samples = []
        for name, pool in strata.items():
            n = max(1, int(len(pool) * ratios[name])) if pool else 0
            all_samples.extend(random.sample(pool, min(n, len(pool))))

        random.shuffle(all_samples)
        split = int(len(all_samples) * train_split)
        train_path = os.path.join(self.tfrecord_dir, f'train_ts{time_steps}.tfrecord')
        val_path   = os.path.join(self.tfrecord_dir, f'val_ts{time_steps}.tfrecord')
        self._write_tfrecord(all_samples[:split], yearly_data, time_steps, train_path)
        self._write_tfrecord(all_samples[split:], yearly_data, time_steps, val_path)

        meta = {
            'time_steps':     time_steps,
            'feature_length': self.feature_length,
            'train_samples':  split,
            'val_samples':    len(all_samples) - split,
            'train_tfrecord': train_path,
            'val_tfrecord':   val_path,
            'matrix_shape':   f"{len(self.full_row_labels)}x{len(self.full_col_labels)}",
        }
        with open(os.path.join(self.tfrecord_dir, f'metadata_ts{time_steps}.json'), 'w') as fh:
            json.dump(meta, fh, indent=2)
        return meta

    def _write_tfrecord(self, samples, yearly_data, time_steps, output_path):
        with tf.io.TFRecordWriter(output_path) as writer:
            for (pos, ty) in tqdm(samples, desc=f"Writing {os.path.basename(output_path)}"):
                f, t = self._construct_focused_feature_label(pos[0], pos[1], ty, time_steps, yearly_data)
                if f is not None and t is not None:
                    f_s = self.scaler_features.transform(
                        f.reshape(-1, self.feature_length)
                    ).reshape(f.shape)
                    t_s = self.scaler_target.transform([[t]])[0][0]
                    writer.write(serialize_example(f_s, t_s, time_steps, self.feature_length))

    # ----------------------------------------------------------
    # XGBoost data preparation
    # ----------------------------------------------------------

    def prepare_xgboost_data(self, time_steps, train_split=0.8):
        """Build flattened feature arrays for XGBoost training."""
        print("Preparing XGBoost training data...")
        self.load_excel_data()
        self._precompute_yearly_stats()

        training_years = [y for y in self._find_longest_consecutive_years() if y < 2022]
        yearly_data    = {y: self.load_single_year_data(y) for y in training_years}
        target_years   = [y for y in training_years if y >= training_years[0] + time_steps]
        if not target_years:
            raise ValueError("Not enough data to generate XGBoost training targets.")

        all_positions = [(r, c) for r in self.full_row_labels for c in self.full_col_labels]
        sampled_pos   = random.sample(all_positions, min(100000, len(all_positions)))

        X_all, y_all = [], []
        for pos in tqdm(sampled_pos, desc="Building XGBoost samples"):
            for ty in target_years:
                f, t = self._construct_focused_feature_label(pos[0], pos[1], ty, time_steps, yearly_data)
                if f is not None and t is not None:
                    X_all.append(f.flatten())
                    y_all.append(t)

        if not X_all:
            raise ValueError("No valid training samples generated for XGBoost.")

        X = np.array(X_all, dtype=np.float32)
        y = np.log1p(np.array(y_all, dtype=np.float32))
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, train_size=train_split, random_state=42)
        print(f"XGBoost data ready — train: {len(X_tr)}, val: {len(X_val)}")
        return X_tr, X_val, y_tr, y_val


# ============================================================
# 5. Transformer architecture
# ============================================================

class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding for the time-step dimension."""

    def __init__(self, max_position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.d_model      = d_model
        pos   = np.arange(max_position)[:, np.newaxis]
        i     = np.arange(d_model)[np.newaxis, :]
        angle = pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])
        self.pos_encoding = tf.constant(angle, dtype=tf.float32)

    def call(self, inputs):
        return inputs + tf.cast(self.pos_encoding[:tf.shape(inputs)[1], :], inputs.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_position": self.max_position, "d_model": self.d_model})
        return cfg


class TransformerBlock(tf.keras.layers.Layer):
    """Single Transformer encoder block: multi-head attention + feed-forward."""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads = d_model, num_heads
        self.dff, self.dropout_rate  = dff, dropout_rate
        self.mha       = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn       = tf.keras.Sequential([Dense(dff, activation='relu'), Dense(d_model)])
        self.norm1     = LayerNormalization(epsilon=1e-6)
        self.norm2     = LayerNormalization(epsilon=1e-6)
        self.drop1     = Dropout(dropout_rate)
        self.drop2     = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x   = self.norm1(inputs + self.drop1(self.mha(inputs, inputs), training=training))
        return self.norm2(x + self.drop2(self.ffn(x), training=training))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads,
                    "dff": self.dff, "dropout_rate": self.dropout_rate})
        return cfg


class OptimizedTimeSeriesTransformer(tf.keras.layers.Layer):
    """Stack of Transformer blocks with positional encoding for time-series input."""

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
        x  = self.input_projection(inputs)
        x *= tf.cast(tf.math.sqrt(float(self.d_model)), x.dtype)
        x  = self.dropout(self.pos_encoding(x), training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads,
                    "num_layers": self.num_layers, "dff": self.dff,
                    "max_position": self.max_position, "dropout_rate": self.dropout_rate})
        return cfg


# ============================================================
# 6. Predictor
# ============================================================

class HighPerformanceTransformerPredictor:
    """
    Orchestrates hyperparameter optimisation, model training, SSP-based
    future forecasting, and RAS balancing for non-square MRIO matrices.
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
            tfrecord_dir, future_excel_path=future_features_path,
        )
        self.best_params    = None
        self.best_score     = float('inf')
        self.feature_length = self.data_processor.feature_length
        self.unified_model  = None

    # ----------------------------------------------------------
    # Model construction
    # ----------------------------------------------------------

    def build_model_with_params(self, params):
        """Instantiate and compile the Transformer regression model."""
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
        """Run Bayesian optimisation to select Transformer hyperparameters."""
        n_calls  = max(n_calls, 10)
        metadata = self.data_processor.create_tfrecords(time_steps=time_steps)
        ds_tr    = create_tf_data_pipeline(metadata['train_tfrecord'], time_steps, self.feature_length, batch_size)
        ds_val   = create_tf_data_pipeline(metadata['val_tfrecord'],   time_steps, self.feature_length, batch_size, False)
        if ds_tr is None or ds_val is None:
            raise ValueError("Cannot build data pipeline for optimisation.")

        dimensions = [
            Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform'),
            Integer(64,  128, name='d_model'),
            Integer(2,   8,   name='num_heads'),
            Integer(1,   2,   name='num_layers'),
            Integer(128, 256, name='dff'),
            Real(0.1,  0.3,   name='dropout_rate'),
            Integer(64, 128,  name='dense_units'),
        ]
        fixed = {'batch_size': batch_size, 'activation': 'gelu', 'time_steps': time_steps}

        @use_named_args(dimensions)
        def objective(**params):
            full_params = {**params, **fixed}
            try:
                model = self.build_model_with_params(full_params)
                h = model.fit(
                    ds_tr, validation_data=ds_val, epochs=15, verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                )
                val_loss = min(h.history['val_loss'])
                del model
                gc.collect()
                tf.keras.backend.clear_session()
                return val_loss
            except Exception as e:
                print(f"Optimisation trial error: {e}")
                return 1e10

        result = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=42)
        self.best_params = {dim.name: val for dim, val in zip(dimensions, result.x)}
        self.best_params.update(fixed)
        self.best_score = result.fun

        with open(os.path.join(self.tfrecord_dir, 'best_params.json'), 'w') as fh:
            json.dump(
                {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                 for k, v in self.best_params.items()},
                fh, indent=2,
            )

    def tune_xgboost_model(self, X_tr, y_tr, X_val, y_val, n_trials=50):
        """Run Optuna hyperparameter search for the XGBoost regressor."""
        print(f"Tuning XGBoost (n_trials={n_trials})...")

        def objective(trial):
            params = {
                'objective':               'reg:tweedie',
                'tweedie_variance_power':  trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
                'tree_method':             'hist',
                'n_estimators':            trial.suggest_int('n_estimators', 500, 2000, step=100),
                'learning_rate':           trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth':               trial.suggest_int('max_depth', 4, 10),
                'subsample':               trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree':        trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight':        trial.suggest_int('min_child_weight', 1, 10),
                'gamma':                   trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'random_state': 42, 'n_jobs': -1,
            }
            m = xgb.XGBRegressor(**params)
            m.fit(X_tr, y_tr, verbose=False)
            return np.sqrt(mean_squared_error(y_val, m.predict(X_val)))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=1800)
        best = study.best_params
        best.update({'objective': 'reg:tweedie', 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1})
        return best

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def train_all_models(self, train_epochs=30):
        """Train the unified XGBoost model and the Transformer model."""
        if self.best_params is None:
            raise ValueError("Run bayesian_optimization() before training.")

        time_steps = self.best_params['time_steps']
        batch_size = self.best_params['batch_size']

        with open(os.path.join(self.tfrecord_dir, f'metadata_ts{time_steps}.json')) as fh:
            meta = json.load(fh)

        # XGBoost
        print("Training XGBoost model...")
        X_tr, X_val, y_tr, y_val = self.data_processor.prepare_xgboost_data(time_steps)
        best_xgb = self.tune_xgboost_model(X_tr, y_tr, X_val, y_val, n_trials=50)
        self.unified_model = xgb.XGBRegressor(**best_xgb)
        self.unified_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
        joblib.dump(self.unified_model, os.path.join(self.tfrecord_dir, 'best_model_unified.joblib'))

        # Transformer
        print(f"Training Transformer model (epochs={train_epochs})...")
        ds_tr  = create_tf_data_pipeline(meta['train_tfrecord'], time_steps, self.feature_length, batch_size)
        ds_val = create_tf_data_pipeline(meta['val_tfrecord'],   time_steps, self.feature_length, batch_size, False)
        if ds_tr and ds_val:
            model = self.build_model_with_params(self.best_params)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.tfrecord_dir, 'best_model_transformer'),
                    save_best_only=True,
                ),
            ]
            history = model.fit(ds_tr, validation_data=ds_val, epochs=train_epochs, callbacks=callbacks)
            with open(os.path.join(self.tfrecord_dir, 'training_history.json'), 'w') as fh:
                json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, fh, indent=2)
            return history

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    def _construct_prediction_features(self, r, c, input_years, yearly_data, pred_year):
        """
        Build a feature array for cell (r, c) using input_years as history
        and pred_year SSP GDP/population as the target context.
        """
        src_c, dst_c = r.split('_')[0], c.split('_')[0]
        pop_src, gdp_src = self.data_processor.country_year_lookup.get((src_c, pred_year), (0, 0))
        _,       gdp_dst = self.data_processor.country_year_lookup.get((dst_c, pred_year), (0, 0))
        gdp_src = 0 if np.isnan(gdp_src) else gdp_src
        gdp_dst = 0 if np.isnan(gdp_dst) else gdp_dst
        pop_src = 0 if np.isnan(pop_src) else pop_src

        years_ahead = float(pred_year - max(input_years))
        all_features = []

        for year in input_years:
            matrix = yearly_data.get(year)
            stats  = self.data_processor.yearly_stats_cache.get(year)
            if matrix is None or stats is None:
                return None

            o_stats     = stats['outflow_stats'].get(r, (0, 0, 0))
            i_stats     = stats['inflow_stats'].get(c, (0, 0, 0))
            total_out   = stats['total_outflow'].get(r, 0)
            total_in    = stats['total_inflow'].get(c, 0)
            country_out = stats['country_total_outflow'].get(src_c, 0)

            _, gdp_src_h = self.data_processor.country_year_lookup.get((src_c, year), (0, 0))
            _, gdp_dst_h = self.data_processor.country_year_lookup.get((dst_c, year), (0, 0))
            gdp_src_h = 1.0 if np.isnan(gdp_src_h) else gdp_src_h
            gdp_dst_h = 1.0 if np.isnan(gdp_dst_h) else gdp_dst_h

            cf_src = np.clip(gdp_src / (gdp_src_h + 1e-9), 0.5, 3.0)
            cf_dst = np.clip(gdp_dst / (gdp_dst_h + 1e-9), 0.5, 3.0)
            span   = max(1, pred_year - year)
            gr_src = cf_src ** (1.0 / span) - 1.0
            gr_dst = cf_dst ** (1.0 / span) - 1.0

            base_feat = np.array([
                np.log1p(matrix.loc[r, c]),
                np.log1p(pop_src), np.log1p(gdp_src),
                np.log1p(o_stats[0]), np.log1p(o_stats[1]), np.log1p(o_stats[2]),
                np.log1p(i_stats[0]), np.log1p(i_stats[1]), np.log1p(i_stats[2]),
                np.log1p(total_out), np.log1p(total_in),
                np.log1p(gdp_src) * np.log1p(gdp_dst),
                total_out / (country_out + 1e-9),
            ], dtype=np.float32)

            growth_feat = np.array([years_ahead, gr_src, gr_dst, cf_src * cf_dst], dtype=np.float32)

            default_vec = np.zeros(
                self.data_processor.economic_status_cats + self.data_processor.trade_bloc_cats,
                dtype=np.float32,
            )
            feat = np.concatenate([
                base_feat, growth_feat,
                self.data_processor.country_feature_lookup.get(src_c, default_vec),
                self.data_processor.country_feature_lookup.get(dst_c, default_vec),
            ])
            all_features.append(np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0))

        return np.array(all_features, dtype=np.float32) if len(all_features) == len(input_years) else None

    def _predict_one_year(self, prediction_year, historical_data, last_historical_year):
        """Generate raw model predictions for all cells in the given forecast year."""
        print(f"\nPredicting year {prediction_year}...")
        time_steps  = self.best_params['time_steps']
        input_years = list(range(last_historical_year - time_steps + 1, last_historical_year + 1))

        prediction_df = pd.DataFrame(
            0.0,
            index=self.data_processor.full_row_labels,
            columns=self.data_processor.full_col_labels,
        )
        hist_mat  = historical_data[last_historical_year]
        years_diff = prediction_year - last_historical_year

        batch_X, batch_pos = [], []
        for r in tqdm(self.data_processor.full_row_labels, desc=f"Building features ({prediction_year})"):
            for c in self.data_processor.full_col_labels:
                X = self._construct_prediction_features(r, c, input_years, historical_data, prediction_year)
                if X is not None:
                    batch_X.append(X)
                    batch_pos.append((r, c))

        if batch_X:
            preds_log = self.unified_model.predict(
                np.array([x.flatten() for x in batch_X])
            )
            preds_raw = np.expm1(preds_log)

            conservative_growth = 1.03
            for i, (r, c) in enumerate(batch_pos):
                hist_val = hist_mat.loc[r, c]
                growth   = np.clip(conservative_growth ** years_diff, 0.8, 1.5)
                expected = hist_val * growth
                model    = max(0.0, preds_raw[i])

                # Blend model prediction with conservative historical growth
                if model < expected * 0.5:
                    final = expected
                elif model > expected * 3.0:
                    final = expected * 1.5
                else:
                    final = 0.7 * model + 0.3 * expected

                # Enforce bounds relative to last observed value
                final = np.clip(final, hist_val * 0.7, hist_val * 1.3)
                prediction_df.loc[r, c] = final

            print(f"Predictions generated for {len(batch_pos)} cells.")

        prediction_df.clip(lower=0.0, inplace=True)
        return prediction_df

    # ----------------------------------------------------------
    # RAS balancing
    # ----------------------------------------------------------

    def _balance_matrix_ras(self, initial_df, pred_year, last_historical_year,
                            historical_data, max_iter=100, tolerance=1e-4):
        """
        Balance the predicted matrix using iterative proportional fitting (RAS).

        Row and column targets are derived from the last observed year
        scaled by a compound annual growth factor.
        """
        print(f"Applying RAS balancing (target year: {pred_year})...")
        hist_mat   = historical_data[last_historical_year]
        years_diff = pred_year - last_historical_year
        growth     = 1.035 ** years_diff

        u_target = (hist_mat.sum(axis=1) * growth).clip(lower=1e-6)
        v_target = (hist_mat.sum(axis=0) * growth).clip(lower=1e-6)

        Z = initial_df.values.copy().clip(min=0)
        total_tgt = u_target.sum()
        if Z.sum() < total_tgt * 0.5 or Z.sum() > total_tgt * 2.0:
            Z = Z * (total_tgt / (Z.sum() + 1e-9))
        Z[Z == 0] = 1e-12

        u_np, v_np = u_target.values, v_target.values
        prev_err = float('inf')

        for it in range(max_iter):
            r_sum = Z.sum(axis=1).clip(min=1e-12)
            Z *= np.clip(u_np / r_sum, 0.8, 1.25)[:, None]

            c_sum = Z.sum(axis=0).clip(min=1e-12)
            Z *= np.clip(v_np / c_sum, 0.8, 1.25)[None, :]

            err = np.abs(Z.sum(axis=1) - u_np).sum() + np.abs(Z.sum(axis=0) - v_np).sum()
            if it % 10 == 0:
                print(f"  Iteration {it + 1}: error = {err:.2e}")
            if err < tolerance * Z.sum() or (it > 20 and abs(err - prev_err) / max(prev_err, 1e-9) < 0.001):
                print(f"RAS converged at iteration {it + 1}.")
                break
            prev_err = err

        result = pd.DataFrame(Z.clip(min=0), index=initial_df.index, columns=initial_df.columns)
        print(f"RAS complete — final sum: {result.sum().sum():,.0f}")
        return result

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------

    def _diagnose_prediction(self, pred_df, year, stage, historical_data=None):
        """Print summary statistics for a predicted matrix."""
        print(f"\n{'=' * 50}\nDiagnostics — {year} ({stage})\n{'=' * 50}")
        total   = pred_df.sum().sum()
        n_cells = pred_df.size
        n_zero  = (pred_df.values == 0).sum()
        n_na    = pred_df.isna().sum().sum()

        print(f"Matrix shape: {pred_df.shape[0]} × {pred_df.shape[1]}")
        print(f"Total sum:    {total:,.0f}")
        print(f"Zero cells:   {n_zero}/{n_cells} ({100*n_zero/n_cells:.1f}%)")
        if n_na:
            print(f"WARNING: {n_na} NA values detected.")

        if historical_data:
            last = max(historical_data)
            hist_total = historical_data[last].sum().sum()
            if year > last and hist_total > 0:
                span = year - last
                ann_growth = (total / hist_total) ** (1.0 / span) - 1
                print(f"vs {last}: cumulative growth {(total/hist_total - 1)*100:.1f}%, "
                      f"annualised {ann_growth*100:.2f}%")

        row_s, col_s = pred_df.sum(axis=1), pred_df.sum(axis=0)
        for label, s in [("Row sums", row_s), ("Col sums", col_s)]:
            print(f"{label}: min={s.min():,.0f}, max={s.max():,.0f}, mean={s.mean():,.0f}")
        print("=" * 50)

    # ----------------------------------------------------------
    # Forecasting pipeline
    # ----------------------------------------------------------

    def forecast_future_years(self, years_to_forecast):
        """
        Generate and balance MRIO predictions for each forecast year.

        Steps per year:
          1. Raw model prediction using SSP scenario GDP/population
          2. RAS balancing against historically projected marginals
          3. Save the balanced matrix to CSV
        """
        if self.unified_model is None:
            raise ValueError("No trained model found. Run train_all_models() or load_existing_model() first.")

        self.data_processor.load_excel_data()
        historical_data   = {y: self.data_processor.load_single_year_data(y)
                             for y in self.data_processor.valid_years}
        last_historical   = max(historical_data)
        print(f"Last observed year (baseline): {last_historical}")

        output_paths = []
        for year in sorted(years_to_forecast):
            raw = self._predict_one_year(year, historical_data, last_historical)
            self._diagnose_prediction(raw, year, "raw prediction", historical_data)

            balanced = self._balance_matrix_ras(raw, year, last_historical, historical_data)
            self._diagnose_prediction(balanced, year, "after RAS", historical_data)

            path = self._save_prediction(balanced, year)
            output_paths.append(path)

        print(f"\nAll forecast years {years_to_forecast} completed.")
        return output_paths

    def _save_prediction(self, df, year):
        """Save a balanced prediction matrix to CSV."""
        base, _ = os.path.splitext(self.output_template_path)
        path    = f"{base}_prediction_balanced_{year}.csv"
        df.reindex(
            index=self.data_processor.full_row_labels,
            columns=self.data_processor.full_col_labels,
            fill_value=0.0,
        ).to_csv(path, encoding='utf-8')
        print(f"Saved: {path}")
        return path

    def load_existing_model(self):
        """Load previously trained XGBoost model and hyperparameters from disk."""
        params_path = os.path.join(self.tfrecord_dir, 'best_params.json')
        model_path  = os.path.join(self.tfrecord_dir, 'best_model_unified.joblib')
        if not os.path.exists(params_path):
            return False
        with open(params_path) as fh:
            self.best_params = json.load(fh)
        if os.path.exists(model_path):
            try:
                self.unified_model = joblib.load(model_path)
                print("Unified XGBoost model loaded successfully.")
                return True
            except Exception as e:
                print(f"Model load failed: {e}")
        return False

    def run_pipeline(self, optimization_calls=10, train_epochs=25, time_steps=8,
                     batch_size=2048, years_to_forecast=None):
        """
        End-to-end pipeline:
          1. Bayesian hyperparameter optimisation
          2. Model training
          3. Future year forecasting
        """
        if years_to_forecast is None:
            years_to_forecast = [2025, 2030]
        try:
            if not self.best_params:
                print("Step 1: Bayesian hyperparameter optimisation")
                self.bayesian_optimization(n_calls=optimization_calls,
                                           time_steps=time_steps, batch_size=batch_size)
            print("Step 2: Training models")
            self.train_all_models(train_epochs=train_epochs)
            print(f"Step 3: Forecasting {years_to_forecast}")
            return self.forecast_future_years(years_to_forecast)
        except Exception as e:
            import traceback
            print(f"Pipeline failed: {e}")
            traceback.print_exc()
            raise


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    for path in [CSV_DIR, HIST_CSV, SSP2_CSV, COUNTRY_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file/folder not found: {path}")

    os.makedirs(os.path.dirname(OUTPUT_TEMPLATE) or ".", exist_ok=True)
    os.makedirs(TFRECORD_DIR, exist_ok=True)

    predictor = HighPerformanceTransformerPredictor(
        csv_folder_path      = CSV_DIR,
        excel_file_path      = HIST_CSV,
        country_csv_path     = COUNTRY_CSV,
        future_features_path = SSP2_CSV,
        output_template_path = OUTPUT_TEMPLATE,
        tfrecord_dir         = TFRECORD_DIR,
    )
    tf.keras.backend.clear_session()

    if predictor.load_existing_model():
        print("Trained model found — skipping training and forecasting directly.")
        predictor.forecast_future_years(years_to_forecast=YEARS_TO_FORECAST)
    else:
        print("No trained model found — running full training and forecasting pipeline.")
        predictor.run_pipeline(
            optimization_calls  = 15,
            train_epochs        = 30,
            time_steps          = TIME_STEPS,
            batch_size          = 2048,
            years_to_forecast   = YEARS_TO_FORECAST,
        )
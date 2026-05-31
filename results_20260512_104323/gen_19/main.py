# evolve/initial.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (hyperparameters)
XGB_PARAMS = {
    "n_estimators": 420,
    "max_depth": 4,
    "min_child_weight": 2,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.8,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "learning_rate": 0.06,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 40,
}

CLIP_PERCENTILE = 97
MIN_PRED_STD_RATIO = 0.18
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# EVOLVE-BLOCK-START  (feature engineering)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stationary micro-regime feature bank:
    - short lookbacks only
    - rolling z-score / rank normalization
    - rate-of-change of market character
    - compact interaction terms
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    eps = 1e-9
    log_ret = np.log(close / close.shift(1))
    ret1 = close.pct_change(1)

    # --- short return stack ---
    for lag in [1, 2, 3, 5, 8, 12, 20]:
        feat[f"ret_{lag}"] = close.pct_change(lag)
        feat[f"logret_{lag}"] = np.log(close / close.shift(lag))

    # --- standardized momentum / reversal ---
    for w in [5, 10, 20]:
        r = close.pct_change(1)
        r_mean = r.rolling(w).mean()
        r_std = r.rolling(w).std()
        feat[f"ret_z_{w}"] = (r - r_mean) / (r_std + eps)

        mom = close.pct_change(w)
        feat[f"mom_z_{w}"] = (mom - mom.rolling(w * 2).mean()) / (mom.rolling(w * 2).std() + eps)

    # --- volatility surface: level + trend + rank ---
    for w in [5, 10, 20]:
        vol = log_ret.rolling(w).std()
        feat[f"vol_{w}"] = vol
        feat[f"vol_z_{w}"] = (vol - vol.rolling(50).mean()) / (vol.rolling(50).std() + eps)
        feat[f"vol_rank_{w}"] = vol.rolling(60).rank(pct=True)
        feat[f"vol_trend_{w}"] = vol.diff(max(1, w // 2)) / (vol.shift(max(1, w // 2)) + eps)

    vol_fast = log_ret.rolling(5).std()
    vol_mid = log_ret.rolling(10).std()
    vol_slow = log_ret.rolling(20).std()
    feat["vol_fast_mid"] = vol_fast / (vol_mid + eps)
    feat["vol_mid_slow"] = vol_mid / (vol_slow + eps)
    feat["vol_regime_delta"] = feat["vol_fast_mid"] - feat["vol_mid_slow"]

    # --- trend consistency / efficiency ---
    for w in [5, 10, 20]:
        path = close.diff().abs().rolling(w).sum()
        net = (close - close.shift(w)).abs()
        eff = net / (path + eps)
        feat[f"eff_{w}"] = eff
        feat[f"eff_trend_{w}"] = eff.diff(max(1, w // 2))

    # --- short-range structure ---
    for w in [5, 10, 20]:
        sma = close.rolling(w).mean()
        std = close.rolling(w).std()
        z = (close - sma) / (std + eps)
        feat[f"price_z_{w}"] = z
        feat[f"price_dev_{w}"] = (close - sma) / (sma + eps)
        feat[f"sma_slope_{w}"] = sma.diff(max(1, w // 2)) / (sma.shift(max(1, w // 2)) + eps)

    feat["sma_cross_5_20"] = (close.rolling(5).mean() - close.rolling(20).mean()) / (close.rolling(20).mean() + eps)
    feat["price_vs_hl_mid_10"] = (close - ((high.rolling(10).max() + low.rolling(10).min()) / 2.0)) / (close + eps)

    # --- candle anatomy ---
    bar_range = (high - low) + eps
    feat["hl_range"] = (high - low) / close
    feat["body_ratio"] = (close - open_).abs() / bar_range
    feat["upper_wick"] = (high - np.maximum(open_, close)) / bar_range
    feat["lower_wick"] = (np.minimum(open_, close) - low) / bar_range
    feat["close_loc"] = (close - low) / bar_range
    feat["range_z_20"] = (feat["hl_range"] - feat["hl_range"].rolling(20).mean()) / (feat["hl_range"].rolling(20).std() + eps)

    # --- volume as stationary signal ---
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    feat["vol_ratio_5"] = volume / (vol_ma_5 + eps)
    feat["vol_ratio_20"] = volume / (vol_ma_20 + eps)
    feat["vol_z_20"] = (volume - vol_ma_20) / (vol_std_20 + eps)
    feat["vol_z_60"] = (volume - volume.rolling(60).mean()) / (volume.rolling(60).std() + eps)
    feat["vol_rank_20"] = volume.rolling(40).rank(pct=True)

    # volume dynamics
    feat["vol_chg_1"] = volume.pct_change(1)
    feat["vol_chg_3"] = volume.pct_change(3)
    feat["vol_trend_5"] = volume.rolling(5).mean().diff(3) / (volume.rolling(5).mean().shift(3) + eps)

    # --- price/volume interactions ---
    feat["mom_x_vol"] = feat["ret_z_5"] * feat["vol_z_20"]
    feat["mom_x_volreg"] = feat["ret_3"] * feat["vol_fast_mid"]
    feat["eff_x_vol"] = feat["eff_10"] * feat["vol_regime_delta"]
    feat["price_x_vol"] = feat["price_z_10"] * feat["vol_ratio_20"]

    # --- autocorr / mean reversion proxy ---
    for w in [5, 10, 20]:
        r = log_ret
        feat[f"acf1_{w}"] = r.rolling(w).corr(r.shift(1))
        feat[f"acf2_{w}"] = r.rolling(w).corr(r.shift(2))

    # --- regime-change features ---
    feat["vol_of_vol_20"] = log_ret.rolling(20).std().rolling(20).std()
    feat["vol_of_vol_z"] = (feat["vol_of_vol_20"] - feat["vol_of_vol_20"].rolling(60).mean()) / (feat["vol_of_vol_20"].rolling(60).std() + eps)

    feat["momentum_change"] = feat["ret_5"].diff(3)
    feat["momentum_accel"] = ret1.diff(3)
    feat["eff_change"] = feat["eff_10"].diff(3)
    feat["vol_change"] = feat["vol_10"].diff(3)

    # --- rolling ranks for robust regime invariance ---
    feat["ret_rank_20"] = ret1.rolling(20).rank(pct=True)
    feat["ret_rank_60"] = ret1.rolling(60).rank(pct=True)
    feat["price_rank_20"] = close.rolling(20).rank(pct=True)
    feat["range_rank_20"] = feat["hl_range"].rolling(20).rank(pct=True)

    # --- session/time effects ---
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    # --- order book microstructure if available ---
    if "bids" in df.columns and "asks" in df.columns:
        def obi(row, tau=1):
            try:
                bids = np.asarray(row["bids"], dtype=float)
                asks = np.asarray(row["asks"], dtype=float)
                if bids.ndim != 2 or asks.ndim != 2 or bids.shape[1] < 2 or asks.shape[1] < 2:
                    return 0.0
                bid_w = np.exp(-tau * np.arange(len(bids)))
                ask_w = np.exp(-tau * np.arange(len(asks)))
                bid_vol = np.sum(bids[:, 1] * bid_w)
                ask_vol = np.sum(asks[:, 1] * ask_w)
                return (bid_vol - ask_vol) / (bid_vol + ask_vol + eps)
            except Exception:
                return 0.0

        feat["obi_tau1"] = df.apply(lambda r: obi(r, 1), axis=1)
        feat["obi_tau3"] = df.apply(lambda r: obi(r, 3), axis=1)
        feat["obi_change"] = feat["obi_tau1"].diff(3)

    # --- funding if available ---
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        feat["funding"] = fr
        feat["funding_z_40"] = (fr - fr.rolling(40).mean()) / (fr.rolling(40).std() + eps)
        feat["funding_mom_4"] = fr.diff(4)
        feat["funding_mom_8"] = fr.diff(8)

        # funding interacting with local regime
        feat["funding_x_vol"] = feat["funding_z_40"] * feat["vol_z_20"]
        feat["funding_x_mom"] = feat["funding_z_40"] * feat["ret_z_5"]

    # --- compact composite features ---
    feat["composite_trend"] = feat["ret_z_5"] + 0.5 * feat["ret_z_10"] + 0.25 * feat["ret_z_20"]
    feat["composite_revert"] = -feat["price_z_5"] + 0.5 * (-feat["price_z_10"])
    feat["composite_regime"] = feat["vol_z_10"] + feat["vol_of_vol_z"] + feat["vol_trend_10"]

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()

    # keep feature count comfortably under 80
    return feat
# EVOLVE-BLOCK-END


# ─────────────────────────────────────────────
# DO NOT MODIFY BELOW THIS LINE
# run_experiment is the fixed interface called by evaluate.py

def run_experiment(
    X_train, y_train,
    X_test,  y_test,
    X_val,   y_val,
):
    """
    Train an XGBoost model and return raw predictions for all three splits.
    Called by evaluate.py via shinka.core.run_shinka_eval.
    """
    # Clip targets
    clip_val = np.percentile(np.abs(y_train), CLIP_PERCENTILE)
    y_train_c = y_train.clip(-clip_val, clip_val)

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    X_va = scaler.transform(X_val)

    params = {k: v for k, v in XGB_PARAMS.items()
              if k != 'early_stopping_rounds'}
    early = XGB_PARAMS.get('early_stopping_rounds', 30)

    model = xgb.XGBRegressor(**params, early_stopping_rounds=early)
    model.fit(
        X_tr, y_train_c,
        eval_set=[(X_te, y_test)],
        verbose=False,
    )

    return {
        'train': {'pred': model.predict(X_tr), 'actual': y_train.values},
        'test':  {'pred': model.predict(X_te), 'actual': y_test.values},
        'val':   {'pred': model.predict(X_va), 'actual': y_val.values},
        'model': model,
    }
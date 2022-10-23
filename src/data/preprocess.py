import pandas as pd
import numpy as np
import src.config as cfg
from src.config import TARGET_COLS

def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')
    return df

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    df = cast_types(df)
    df = drop_columns(df)
    df = df.dropna()
    return df

def extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[cfg.TARGET_COLS]
    return df, target

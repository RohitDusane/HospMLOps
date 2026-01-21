# --------------------------
# Custom helper functions
# --------------------------
import pandas as pd
import numpy as np
import os
import json
from readmit.configuration import paths_config
from readmit.components.logger import logging

from yaml import safe_load

def short_path(path: str) -> str:
    try:
        rel = os.path.relpath(path, paths_config.PROJECT_ROOT)
        return rel.replace(os.sep, "/")
    except Exception:
        return path

def read_yaml(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"YAML file not found: {filepath}")

    with open(filepath, 'r') as yaml_file:
        config = safe_load(yaml_file)  # Fixed method
        logging.info(f"✅ Successfully read YAML file")
        return config


# def handle_outliers_iqr(df: pd.DataFrame, cols=None):
#     """
#     Handle outliers using the IQR method.
#     Caps values at Q1-1.5*IQR and Q3+1.5*IQR.
#     """
#     df = df.copy()
#     if cols is None:
#         cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     for col in cols:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
#         df[col] = np.clip(df[col], lower, upper)
#     return df

# def fill_missing(df: pd.DataFrame, strategy="ffill"):
#     """
#     Fill missing values in a DataFrame.
#     Default is forward fill.
#     """
#     df = df.copy()
#     if strategy == "ffill":
#         df.fillna(method="ffill", inplace=True)
#     elif strategy == "bfill":
#         df.fillna(method="bfill", inplace=True)
#     elif strategy == "mean":
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#     return df



# ======== UTILS FOR TRANSFORMATION ========
def fill_missing(df):
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def handle_outliers(df):
    """Cap numeric columns at IQR boundaries"""
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df

def scale_numeric(df, numeric_cols, stats=None):
    """Standard scaling"""
    if stats is None:
        stats = {}
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std() if df[col].std() > 0 else 1.0
            df[col] = (df[col] - mean) / std
            stats[col] = {'mean': mean, 'std': std}
        return df, stats
    else:
        for col in numeric_cols:
            mean, std = stats[col]['mean'], stats[col]['std']
            df[col] = (df[col] - mean) / std
        return df

def encode_categorical(df, cat_cols, mappings=None):
    """One-hot encoding with saved mappings"""
    if mappings is None:
        mappings = {}
        for col in cat_cols:
            df[col] = df[col].astype(str)
            mappings[col] = df[col].unique().tolist()
            for val in mappings[col]:
                df[f"{col}_{val}"] = (df[col] == val).astype(int)
        df.drop(columns=cat_cols, inplace=True)
        return df, mappings
    else:
        for col in cat_cols:
            df[col] = df[col].astype(str)
            for val in mappings[col]:
                df[f"{col}_{val}"] = (df[col] == val).astype(int)
        df.drop(columns=cat_cols, inplace=True)
        return df
    
def save_markdown_and_csv(self, df: pd.DataFrame, name: str):
        csv_path = os.path.join(self.output_dir, f"{name}.csv")
        md_path = os.path.join(self.output_dir, f"{name}.md")
        df.to_csv(csv_path, index=False)
        df.to_markdown(md_path, index=False)
        logging.info(f"Saved table: {short_path(csv_path)}")
        logging.info(f"Saved table: {short_path(md_path)}")

def make_binary_readmission(df: pd.DataFrame, source_col: str = "readmitted"):
    """
    Convert readmitted column to binary:
    <30 -> 1
    NO, >30 -> 0
    """
    df = df.copy()
    df["readmitted_bin"] = np.where(df[source_col] == "<30", 1, 0)
    return df



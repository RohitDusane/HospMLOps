import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_figure(fig, output_dir: str, base_name: str, suffix: str = None) -> str:
    """
    Save matplotlib figure with timestamp to avoid overwriting.
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}"
    if suffix:
        filename += f"_{suffix}"
    filename += f"_{timestamp}.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path

def save_table(df: pd.DataFrame, output_dir: str, base_name: str, suffix: str = None, fmt: str = "csv") -> str:
    """
    Save pandas DataFrame as CSV or Parquet with timestamp.
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}"
    if suffix:
        filename += f"_{suffix}"
    if fmt == "csv":
        filename += f"_{timestamp}.csv"
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
    elif fmt in ["parquet", "pq"]:
        filename += f"_{timestamp}.parquet"
        path = os.path.join(output_dir, filename)
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return path

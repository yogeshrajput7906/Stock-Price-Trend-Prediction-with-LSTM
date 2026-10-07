"""Data ingestion and validation module."""

from pathlib import Path
from typing import Union

import pandas as pd


def load_stock_data(
    filepath: Union[str, Path],
    date_col: str = "Date",
    price_col: str = "Close",
) -> pd.DataFrame:
    """Load, validate, and chronologically sort stock price CSV data.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the CSV file.
    date_col : str, optional
        Name of the date column, default is 'Date'.
    price_col : str, optional
        Name of the target price column, default is 'Close'.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame sorted chronologically in ascending order,
        containing at least `date_col` and `price_col`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the required columns are missing or the file is empty.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Stock data file not found at: {path.resolve()}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"The dataset at {path} is empty.")

    # Standardize column names (strip whitespace and surrounding quotes)
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]

    # Handle dataset aliases (e.g. some datasets name closing price 'Price')
    if price_col not in df.columns and "Price" in df.columns:
        df = df.rename(columns={"Price": price_col})

    # Validate required columns
    missing_cols = [col for col in [date_col, price_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required column(s) in CSV: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Clean price column if it contains strings (e.g. commas or symbols)
    if df[price_col].dtype == object:
        df[price_col] = (
            df[price_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop missing values in essential columns
    df = df.dropna(subset=[date_col, price_col])

    # Sort strictly chronologically (earliest to latest)
    df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)

    # Drop exact date duplicates, keeping the first occurrence
    df = df.drop_duplicates(subset=[date_col]).reset_index(drop=True)

    return df

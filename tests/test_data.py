"""Unit tests for the data ingestion and validation module."""

from pathlib import Path

import pytest

from stock_prediction.data import load_stock_data


def test_load_stock_data_file_not_found():
    """Verify that a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_stock_data("non_existent_file.csv")


def test_load_stock_data_missing_columns(tmp_path: Path):
    """Verify that missing required columns raise ValueError."""
    dummy_csv = tmp_path / "bad_data.csv"
    dummy_csv.write_text("Date,Volume\n2023-01-01,1000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required column"):
        load_stock_data(dummy_csv, date_col="Date", price_col="Close")


def test_load_stock_data_empty_file(tmp_path: Path):
    """Verify that an empty CSV raises ValueError."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("Date,Close\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_stock_data(empty_csv)


def test_load_stock_data_chronological_sorting(tmp_path: Path):
    """Verify that data is sorted in ascending chronological order."""
    csv_file = tmp_path / "unsorted.csv"
    csv_content = (
        "Date,Close\n"
        "2023-01-05,150.0\n"
        "2023-01-02,140.0\n"
        "2023-01-04,145.0\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    df = load_stock_data(csv_file)
    dates = df["Date"].tolist()
    assert dates == sorted(dates)
    assert len(df) == 3
    assert df.iloc[0]["Close"] == 140.0
    assert df.iloc[-1]["Close"] == 150.0


def test_load_stock_data_price_alias(tmp_path: Path):
    """Verify that 'Price' column is automatically mapped to 'Close'."""
    csv_file = tmp_path / "alias.csv"
    csv_content = (
        "Date,Price\n"
        "2023-01-01,100.5\n"
        "2023-01-02,102.0\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    df = load_stock_data(csv_file)
    assert "Close" in df.columns
    assert df["Close"].iloc[0] == 100.5

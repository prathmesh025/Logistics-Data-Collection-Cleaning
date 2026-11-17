"""
data_exploration.py
-------------------
Module 2: Data Exploration

Responsibilities:
  * Show dimensions, column names, data types and sample records.
  * Calculate descriptive statistics.
  * Identify missing values and duplicate records.
  * Flag date columns that are still stored as text.
  * Return summary tables that can be saved and compared before/after cleaning.
"""

import sys

import pandas as pd

try:                                    # when run from main.py (project root)
    from src.data_collection import REPORTS_DIR, create_project_directories, load_all_datasets
except ModuleNotFoundError:             # when run directly: python src\data_exploration.py
    from data_collection import REPORTS_DIR, create_project_directories, load_all_datasets

# Make console output readable for wide tables
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 40)


def print_header(title):
    """Print a section header so the console output is easy to read."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def missing_values_table(df):
    """
    Return a table of columns that contain missing values, with the
    count and percentage of missing entries (empty table if none).
    """
    missing = df.isnull().sum()
    percent = (missing / len(df) * 100).round(2) if len(df) else missing * 0
    table = pd.DataFrame({"missing_count": missing, "missing_percent": percent})
    table = table[table["missing_count"] > 0]
    return table.sort_values("missing_count", ascending=False)


def count_duplicates(df):
    """Count rows that are exact copies of an earlier row."""
    return int(df.duplicated().sum())

def text_columns(df):
    """
    Return the names of text columns.
    Works on pandas 2 (dtype object) and pandas 3 (dtype str).
    """
    return [col for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_datetime64_any_dtype(df[col])]

def find_date_like_columns(df):
    """
    Find columns whose names suggest dates/times but that are still stored
    as text. These must be converted during cleaning.
    """
    keywords = ("date", "timestamp")
    return [col for col in text_columns(df)
        if any(k in col.lower() for k in keywords) or col.lower().endswith("_at")]


def explore_dataset(df, name, show_sample=True):
    """
    Print a full exploration report for one DataFrame.
    Returns (summary_dict, missing_values_table).
    """
    print_header(f"DATASET: {name}")

    rows, cols = df.shape
    print(f"Dimensions: {rows:,} rows x {cols} columns")

    print("\nColumn names and data types:")
    print(df.dtypes.to_string())

    if show_sample:
        print("\nSample records (first 5 rows):")
        print(df.head())

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print("\nDescriptive statistics (numeric columns):")
        print(numeric.describe().T.round(2))

    text = df[text_columns(df)]
    if not text.empty:
        print("\nSummary of text columns (count, unique, most frequent):")
        print(text.describe().T)

    # Show value counts for text columns with few distinct values
    # (useful for spotting inconsistent categories later).
    for col in text.columns:
        if text[col].nunique() <= 15:
            print(f"\nValue counts for '{col}':")
            print(text[col].value_counts(dropna=False).to_string())

    date_like = find_date_like_columns(df)
    if date_like:
        print(f"\nDate-like columns stored as text: {date_like}")

    missing = missing_values_table(df)
    print("\nMissing values:")
    print(missing if not missing.empty else "  none")

    duplicates = count_duplicates(df)
    print(f"\nDuplicate rows (exact copies): {duplicates:,}")

    total_cells = rows * cols
    missing_cells = int(df.isnull().sum().sum())
    summary = {
        "table": name,
        "rows": rows,
        "columns": cols,
        "missing_cells": missing_cells,
        "missing_cells_percent": round(missing_cells / total_cells * 100, 3) if total_cells else 0,
        "columns_with_missing": len(missing),
        "duplicate_rows": duplicates,
        "date_columns_as_text": len(date_like),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
    }
    return summary, missing


def explore_all(datasets, show_sample=True):
    """
    Explore every table in the dictionary.
    Returns (summary_df, missing_df) covering all tables.
    """
    summaries, missing_tables = [], []
    for name, df in datasets.items():
        summary, missing = explore_dataset(df, name, show_sample)
        summaries.append(summary)
        if not missing.empty:
            missing = missing.reset_index().rename(columns={"index": "column"})
            missing.insert(0, "table", name)
            missing_tables.append(missing)

    summary_df = pd.DataFrame(summaries)
    missing_df = (pd.concat(missing_tables, ignore_index=True) if missing_tables
                  else pd.DataFrame(columns=["table", "column", "missing_count", "missing_percent"]))

    print_header("OVERALL EXPLORATION SUMMARY")
    print(summary_df.to_string(index=False))
    return summary_df, missing_df


def save_exploration_results(summary_df, missing_df, prefix="raw"):
    """Save the summary tables to outputs/reports/ (prefix = 'raw' or 'clean')."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / f"{prefix}_exploration_summary.csv"
    missing_path = REPORTS_DIR / f"{prefix}_missing_values.csv"
    summary_df.to_csv(summary_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {missing_path}")


if __name__ == "__main__":
    # Standalone test:  python src\data_exploration.py
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # safe for Portuguese text / redirection

    create_project_directories()
    data = load_all_datasets()
    if data is not None:
        summary, missing = explore_all(data)
        save_exploration_results(summary, missing, prefix="raw")
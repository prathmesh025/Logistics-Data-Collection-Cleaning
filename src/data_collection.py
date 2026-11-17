"""
data_collection.py
------------------
Module 1: Data Collection

Responsibilities:
  * Define where raw and processed files live (project paths).
  * Check that the Olist CSV files exist before loading.
  * Load each CSV safely into a Pandas DataFrame.
  * Give clear instructions if files are missing.

Data source:
  Olist Brazilian E-Commerce Public Dataset (Kaggle)
  https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project paths (built from this file's location, so they work on any machine)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VISUALIZATION_DIR = PROJECT_ROOT / "outputs" / "visualizations"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

KAGGLE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"

# Short name -> file name inside data/raw/
DATASET_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def create_project_directories():
    """Create the output folders if they do not exist yet."""
    for folder in (RAW_DIR, PROCESSED_DIR, VISUALIZATION_DIR, REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def find_missing_files(raw_dir=RAW_DIR):
    """Return a list of expected CSV file names that are not in raw_dir."""
    return [name for name in DATASET_FILES.values()
            if not (Path(raw_dir) / name).is_file()]


def print_download_instructions(missing_files):
    """Explain the manual download steps when files are missing."""
    print("\n" + "=" * 70)
    print("DATASET FILES NOT FOUND")
    print("=" * 70)
    print("Missing files:")
    for name in missing_files:
        print(f"  - {name}")
    print("\nManual download steps (no code needed):")
    print(f"  1. Log in to Kaggle and open: {KAGGLE_URL}")
    print("  2. Click 'Download' and extract the zip file.")
    print(f"  3. Copy ALL the extracted CSV files into:\n     {RAW_DIR}")
    print("  4. Run the project again.")
    print("=" * 70 + "\n")


def load_csv(file_path):
    """
    Load one CSV file into a DataFrame.
    Returns None (and prints the reason) if the file cannot be read.
    """
    file_path = Path(file_path)
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"[ERROR] File is empty: {file_path}")
    except pd.errors.ParserError as error:
        print(f"[ERROR] Could not parse {file_path.name}: {error}")
    except UnicodeDecodeError:
        print(f"[ERROR] Encoding problem in {file_path.name}")
    return None


def load_all_datasets(raw_dir=RAW_DIR):
    """
    Load every Olist CSV listed in DATASET_FILES.

    Returns a dictionary {short_name: DataFrame}, or None if any file is
    missing or unreadable (instructions are printed in that case).
    """
    missing = find_missing_files(raw_dir)
    if missing:
        print_download_instructions(missing)
        return None

    datasets = {}
    print("Loading datasets from:", raw_dir)
    for short_name, file_name in DATASET_FILES.items():
        df = load_csv(Path(raw_dir) / file_name)
        if df is None:
            return None
        datasets[short_name] = df
        print(f"  loaded {file_name:<45} rows={df.shape[0]:>8,}  cols={df.shape[1]}")
    return datasets


if __name__ == "__main__":
    # Quick standalone test:  python src\data_collection.py
    create_project_directories()
    data = load_all_datasets()
    if data is not None:
        print("\nAll datasets loaded successfully.")
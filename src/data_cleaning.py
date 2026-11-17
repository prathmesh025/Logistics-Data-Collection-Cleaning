"""
data_cleaning.py
----------------
Module 3: Data Cleaning

Cleans every Olist table. Principles:
  * Raw data is never modified (cleaned copies go to data/processed/).
  * Suspicious records are FLAGGED instead of deleted wherever possible.
  * Every cleaning step is recorded in a cleaning log.
"""

from cmath import log
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

try:                                    # run from main.py (project root)
    from src.data_collection import (PROCESSED_DIR, REPORTS_DIR,
                                     create_project_directories, load_all_datasets)
except ModuleNotFoundError:             # run directly: python src\data_cleaning.py
    from data_collection import (PROCESSED_DIR, REPORTS_DIR,
                                 create_project_directories, load_all_datasets)

# Approximate bounding box of Brazil, used to validate coordinates
LAT_RANGE = (-34.0, 6.0)
LNG_RANGE = (-74.0, -34.0)

# The raw dataset misspells "length" in two column names
COLUMN_FIXES = {
    "product_name_lenght": "product_name_length",
    "product_description_lenght": "product_description_length",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def add_log(log, table, step, affected, detail=""):
    """Record one cleaning step (what was done and how many values it touched)."""
    log.append({"table": table, "step": step,
                "affected_count": int(affected), "detail": detail})


def standardize_column_names(df):
    """lower_snake_case column names, and fix known spelling errors."""
    df = df.copy()
    names = []
    for col in df.columns:
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip().lower()).strip("_")
        names.append(COLUMN_FIXES.get(name, name))
    df.columns = names
    return df


def drop_exact_duplicates(df, table, log, step="remove exact duplicate rows"):
    """Remove rows that are identical in every column."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    add_log(log, table, step, before - len(df), "rows identical in every column")
    return df


def _clean_text(value, remove_accents):
    value = re.sub(r"\s+", " ", str(value)).strip().lower()
    if remove_accents:
        value = "".join(ch for ch in unicodedata.normalize("NFKD", value)
                        if not unicodedata.combining(ch))
    return value


def normalize_text(series, remove_accents=False):
    """Trim, lower-case, collapse spaces (optionally remove accents).
    Works on unique values only, so it is fast even for 1M rows."""
    mapping = {v: _clean_text(v, remove_accents) for v in series.dropna().unique()}
    return series.map(mapping)


def normalize_column(df, col, table, log, remove_accents=False):
    """Normalize one text column and log how many values changed."""
    cleaned = normalize_text(df[col], remove_accents)
    changed = (cleaned.fillna("") != df[col].fillna("")).sum()
    df[col] = cleaned
    detail = "trim, lower-case, collapse spaces" + (", remove accents" if remove_accents else "")
    add_log(log, table, f"normalize text in {col}", changed, detail)
    return df


def normalize_state(df, col, table, log):
    """State codes: trim and upper-case (e.g. ' sp' -> 'SP')."""
    cleaned = df[col].str.strip().str.upper()
    changed = (cleaned.fillna("") != df[col].fillna("")).sum()
    df[col] = cleaned
    add_log(log, table, f"normalize state codes in {col}", changed, "trim, upper-case")
    return df


def standardize_zip(series):
    """Zip prefixes must be 5-character text; pandas read them as integers
    and lost leading zeros (e.g. 1046 -> '01046')."""
    numbers = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numbers.astype("string").str.zfill(5)


def convert_dates(df, columns, table, log):
    """Convert text columns to datetime; unparseable values become NaT."""
    for col in columns:
        missing_before = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        newly_missing = df[col].isna().sum() - missing_before
        add_log(log, table, f"convert {col} to datetime", df[col].notna().sum(),
                f"{int(newly_missing)} values could not be parsed (set to NaT)")
    return df


# ---------------------------------------------------------------------------
# Table-specific cleaning
# ---------------------------------------------------------------------------
def fix_zip_column(df, col, table, log):
    """Convert a zip prefix column to 5-character text; log how many needed padding."""
    padded = int((pd.to_numeric(df[col], errors="coerce") < 10000).sum())
    df[col] = standardize_zip(df[col])
    add_log(log, table, f"zip prefix {col} to 5-character text", padded,
            "values below 10000 had leading zeros restored")
    return df
def clean_orders(df, log):
    df = standardize_column_names(df)
    df = drop_exact_duplicates(df, "orders", log)
    df = normalize_column(df, "order_status", "orders", log)

    date_cols = ["order_purchase_timestamp", "order_approved_at",
                 "order_delivered_carrier_date", "order_delivered_customer_date",
                 "order_estimated_delivery_date"]
    df = convert_dates(df, date_cols, "orders", log)

    # Evidence for the missing-delivery-date strategy: is it explained by status?
    check = pd.crosstab(df["order_status"], df["order_delivered_customer_date"].isna(),
                        rownames=["order_status"], colnames=["delivered_date_missing"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    check.to_csv(REPORTS_DIR / "orders_status_vs_missing_delivery.csv")
    print("\nOrder status vs missing delivery date:")
    print(check)

    # Validate date consistency (flag, do not delete)
    p = df["order_purchase_timestamp"]
    a = df["order_approved_at"]
    c = df["order_delivered_carrier_date"]
    d = df["order_delivered_customer_date"]
    e = df["order_estimated_delivery_date"]
    df["flag_date_inconsistent"] = (a < p) | (c < p) | (d < c) | (d < p) | (e < p)
    df["flag_delivered_without_date"] = (df["order_status"] == "delivered") & d.isna()
    df["flag_status_date_conflict"] = (df["order_status"] != "delivered") & d.notna()
    add_log(log, "orders", "flag inconsistent date order", df["flag_date_inconsistent"].sum(),
            "approval/carrier/delivery/estimate earlier than preceding event")
    add_log(log, "orders", "flag delivered orders without delivery date",
            df["flag_delivered_without_date"].sum(), "kept, not deleted")
    add_log(log, "orders", "flag non-delivered orders that have a delivery date",
            df["flag_status_date_conflict"].sum(), "kept, not deleted")
    return df


def clean_order_items(df, log):
    df = standardize_column_names(df)
    df = drop_exact_duplicates(df, "order_items", log)
    df = convert_dates(df, ["shipping_limit_date"], "order_items", log)
    for col in ("price", "freight_value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["flag_invalid_price"] = df["price"].isna() | (df["price"] <= 0)
    df["flag_invalid_freight"] = df["freight_value"].isna() | (df["freight_value"] < 0)
    add_log(log, "order_items", "flag invalid price (missing or <= 0)",
            df["flag_invalid_price"].sum(), "kept, not deleted")
    add_log(log, "order_items", "flag invalid freight (missing or < 0)",
            df["flag_invalid_freight"].sum(), "zero freight is valid (free shipping)")
    return df


def clean_customers(df, log):
    df = standardize_column_names(df)
    df = drop_exact_duplicates(df, "customers", log)
    df = fix_zip_column(df, "customer_zip_code_prefix", "customers", log) 
    df = normalize_column(df, "customer_city", "customers", log, remove_accents=True)
    df = normalize_state(df, "customer_state", "customers", log)
    return df


def clean_sellers(df, log):
    df = standardize_column_names(df)
    df = drop_exact_duplicates(df, "sellers", log)
    df = fix_zip_column(df, "seller_zip_code_prefix", "sellers", log)
    df = normalize_column(df, "seller_city", "sellers", log, remove_accents=True)
    df = normalize_state(df, "seller_state", "sellers", log)
    return df


def clean_category_translation(df, log):
    df = standardize_column_names(df)
    df = drop_exact_duplicates(df, "category_translation", log)
    df = normalize_column(df, "product_category_name", "category_translation", log)
    df = normalize_column(df, "product_category_name_english", "category_translation", log)
    return df.drop_duplicates(subset="product_category_name").reset_index(drop=True)


def clean_products(df, translation, log):
    df = standardize_column_names(df)
    df = drop_exact_duplicates(df, "products", log)
    df = normalize_column(df, "product_category_name", "products", log)

    # Missing listing metadata: check whether the same rows are affected
    meta_cols = ["product_category_name", "product_name_length",
                 "product_description_length", "product_photos_qty"]
    missing_any = df[meta_cols].isna().any(axis=1)
    missing_all = df[meta_cols].isna().all(axis=1)
    df["flag_missing_listing_metadata"] = missing_any
    add_log(log, "products", "flag rows with missing listing metadata", missing_any.sum(),
            f"{int(missing_all.sum())} of these rows miss all four metadata fields")

    n_missing_cat = df["product_category_name"].isna().sum()
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    add_log(log, "products", "fill missing category with 'unknown'", n_missing_cat,
            "rows kept for analysis")

    # Weight and dimensions: non-positive values are invalid -> treat as missing
    size_cols = ["product_weight_g", "product_length_cm",
                 "product_height_cm", "product_width_cm"]
    invalid = 0
    for col in size_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = df[col] <= 0
        invalid += int(bad.sum())
        df[col] = df[col].mask(bad)
    add_log(log, "products", "treat non-positive weight/dimensions as missing", invalid)

    # Impute with the category median (fallback: overall median) and flag it
    df["flag_size_imputed"] = df[size_cols].isna().any(axis=1)
    for col in size_cols:
        category_median = df.groupby("product_category_name")[col].transform("median")
        df[col] = df[col].fillna(category_median).fillna(df[col].median())
    add_log(log, "products", "impute weight/dimensions with category median",
            df["flag_size_imputed"].sum(), "flag_size_imputed marks these rows")

    # Add English category names; fall back to the Portuguese name if no translation
    rows_before = len(df)
    df = df.merge(translation, on="product_category_name", how="left")
    assert len(df) == rows_before, "translation merge changed the row count"
    no_translation = df["product_category_name_english"].isna() & (df["product_category_name"] != "unknown")
    df["product_category_name_english"] = df["product_category_name_english"].fillna(df["product_category_name"])
    add_log(log, "products", "product rows without English translation", no_translation.sum(),
            "Portuguese name used as fallback")
    return df


def clean_payments(df, log):
    df = standardize_column_names(df)
    df = drop_exact_duplicates(df, "payments", log)
    df = normalize_column(df, "payment_type", "payments", log)
    df["flag_invalid_payment_value"] = df["payment_value"].isna() | (df["payment_value"] < 0)
    df["flag_invalid_installments"] = df["payment_installments"].isna() | (df["payment_installments"] < 1)
    add_log(log, "payments", "flag invalid payment value (missing or < 0)",
            df["flag_invalid_payment_value"].sum(), "kept, not deleted")
    add_log(log, "payments", "flag invalid installments (missing or < 1)",
            df["flag_invalid_installments"].sum(), "kept, not deleted")
    return df


def clean_reviews(df, log):
    df = standardize_column_names(df)
    df = drop_exact_duplicates(df, "reviews", log)
    df = convert_dates(df, ["review_creation_date", "review_answer_timestamp"], "reviews", log)
    # Comment title/message are optional free text: missing is NOT an error
    df["has_comment_message"] = df["review_comment_message"].notna()
    add_log(log, "reviews", "add has_comment_message flag", df["has_comment_message"].sum(),
            "missing comments left as missing (optional field)")
    df["flag_invalid_score"] = ~df["review_score"].between(1, 5)
    add_log(log, "reviews", "flag review_score outside 1-5", df["flag_invalid_score"].sum())
    return df


def clean_geolocation(df, log):
    df = standardize_column_names(df)
    df = drop_exact_duplicates(df, "geolocation", log)
    df = fix_zip_column(df, "geolocation_zip_code_prefix", "geolocation", log)
    df = normalize_column(df, "geolocation_city", "geolocation", log, remove_accents=True)
    df = normalize_state(df, "geolocation_state", "geolocation", log)
    df = drop_exact_duplicates(df, "geolocation", log,
                               step="remove duplicates exposed by normalization")
    lat, lng = df["geolocation_lat"], df["geolocation_lng"]

    df["flag_outside_brazil"] = (~lat.between(*LAT_RANGE)) | (~lng.between(*LNG_RANGE))
    add_log(log, "geolocation", "flag coordinates outside Brazil bounding box",
            df["flag_outside_brazil"].sum(), "kept in file, excluded later when aggregating")
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def clean_all(datasets):
    """Clean every table. Returns (dict of cleaned DataFrames, cleaning-log DataFrame)."""
    log = []
    cleaned = {}
    cleaned["category_translation"] = clean_category_translation(datasets["category_translation"], log)
    cleaned["orders"] = clean_orders(datasets["orders"], log)
    cleaned["order_items"] = clean_order_items(datasets["order_items"], log)
    cleaned["customers"] = clean_customers(datasets["customers"], log)
    cleaned["sellers"] = clean_sellers(datasets["sellers"], log)
    cleaned["products"] = clean_products(datasets["products"], cleaned["category_translation"], log)
    cleaned["payments"] = clean_payments(datasets["payments"], log)
    cleaned["reviews"] = clean_reviews(datasets["reviews"], log)
    cleaned["geolocation"] = clean_geolocation(datasets["geolocation"], log)
    return cleaned, pd.DataFrame(log)


def save_cleaned_datasets(cleaned, log_df):
    """Write cleaned tables to data/processed/ and the log to outputs/reports/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in cleaned.items():
        path = PROCESSED_DIR / f"clean_{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"Saved: {path}  ({len(df):,} rows)")
    log_path = REPORTS_DIR / "cleaning_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Saved: {log_path}")


if __name__ == "__main__":
    # Standalone test:  python src\data_cleaning.py
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    create_project_directories()
    raw = load_all_datasets()
    if raw is not None:
        cleaned_data, cleaning_log = clean_all(raw)
        print("\nCLEANING LOG")
        print(cleaning_log.to_string(index=False))
        save_cleaned_datasets(cleaned_data, cleaning_log)
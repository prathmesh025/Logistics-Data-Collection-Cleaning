"""
data_validation.py
------------------
Module 4: Data Validation

Checks the CLEANED tables and produces a data-quality summary:
  * primary-key uniqueness and nulls
  * references between tables (orphan records)
  * missing values remaining after cleaning
  * duplicate rows after cleaning
  * date consistency
  * numeric field validity and text formats
Status meanings:  PASS = no problems | FLAGGED = problems found (kept & flagged)
                  INFO = expected or informational, not an error
"""

import sys

import pandas as pd

try:
    from src.data_collection import PROCESSED_DIR, REPORTS_DIR, create_project_directories
    from src.data_exploration import count_duplicates, missing_values_table
except ModuleNotFoundError:
    from data_collection import PROCESSED_DIR, REPORTS_DIR, create_project_directories
    from data_exploration import count_duplicates, missing_values_table

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)

CLEAN_TABLES = ["orders", "order_items", "customers", "sellers", "products",
                "payments", "reviews", "geolocation", "category_translation"]

# Cleaned CSVs lose type information, so we restore it when reading them back
ZIP_COLUMNS = {
    "customers": ["customer_zip_code_prefix"],
    "sellers": ["seller_zip_code_prefix"],
    "geolocation": ["geolocation_zip_code_prefix"],
}
DATE_COLUMNS = {
    "orders": ["order_purchase_timestamp", "order_approved_at",
               "order_delivered_carrier_date", "order_delivered_customer_date",
               "order_estimated_delivery_date"],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}

# (table, key columns, label)
PRIMARY_KEYS = [
    ("orders", ["order_id"], "order_id"),
    ("customers", ["customer_id"], "customer_id"),
    ("sellers", ["seller_id"], "seller_id"),
    ("products", ["product_id"], "product_id"),
    ("order_items", ["order_id", "order_item_id"], "order_id + order_item_id"),
    ("payments", ["order_id", "payment_sequential"], "order_id + payment_sequential"),
    ("reviews", ["review_id"], "review_id"),
    ("reviews", ["review_id", "order_id"], "review_id + order_id"),
    ("category_translation", ["product_category_name"], "product_category_name"),
]

# (child table, child column, parent table, parent column)
REFERENCES = [
    ("orders", "customer_id", "customers", "customer_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("order_items", "seller_id", "sellers", "seller_id"),
    ("payments", "order_id", "orders", "order_id"),
    ("reviews", "order_id", "orders", "order_id"),
]

# Missing values that are expected, with the reason (everything else is FLAGGED)
EXPECTED_MISSING = {
    ("orders", "order_approved_at"): "order not yet approved (depends on order status)",
    ("orders", "order_delivered_carrier_date"): "not yet handed to carrier (depends on order status)",
    ("orders", "order_delivered_customer_date"): "not delivered (depends on order status)",
    ("reviews", "review_comment_title"): "optional free-text field",
    ("reviews", "review_comment_message"): "optional free-text field",
    ("products", "product_name_length"): "listing metadata not recorded; not imputed, see flag_missing_listing_metadata",
    ("products", "product_description_length"): "listing metadata not recorded; not imputed, see flag_missing_listing_metadata",
    ("products", "product_photos_qty"): "listing metadata not recorded; not imputed, see flag_missing_listing_metadata",
}


def add_check(results, category, table, check, violations, total, note="", info=False):
    """Append one validation result."""
    violations = int(violations)
    status = "PASS" if violations == 0 else ("INFO" if info else "FLAGGED")
    results.append({"category": category, "table": table, "check": check,
                    "records_checked": int(total), "violations": violations,
                    "status": status, "note": note})


def load_cleaned_datasets():
    """Read data/processed/clean_*.csv back with correct types."""
    datasets = {}
    for name in CLEAN_TABLES:
        path = PROCESSED_DIR / f"clean_{name}.csv"
        if not path.is_file():
            print(f"[ERROR] {path.name} not found. Run data_cleaning.py first.")
            return None
        zip_types = {c: "string" for c in ZIP_COLUMNS.get(name, [])}
        datasets[name] = pd.read_csv(path, dtype=zip_types,
                                     parse_dates=DATE_COLUMNS.get(name, []))
    return datasets


def table_summary(df, name):
    """Same metrics as data_exploration, so raw and clean can be compared."""
    rows, cols = df.shape
    missing = missing_values_table(df)
    missing_cells = int(df.isnull().sum().sum())
    date_text = [c for c in df.columns
                 if ("date" in c.lower() or "timestamp" in c.lower())
                 and not c.startswith("flag_")
                 and not pd.api.types.is_datetime64_any_dtype(df[c])]
    summary = {
        "table": name, "rows": rows, "columns": cols,
        "missing_cells": missing_cells,
        "missing_cells_percent": round(missing_cells / (rows * cols) * 100, 3) if rows * cols else 0,
        "columns_with_missing": len(missing),
        "duplicate_rows": count_duplicates(df),
        "date_columns_as_text": len(date_text),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
    }
    return summary, missing


def check_primary_keys(tables, results):
    for table, keys, label in PRIMARY_KEYS:
        df = tables[table]
        add_check(results, "Primary key", table, f"{label} not null",
                  df[keys].isna().any(axis=1).sum(), len(df))
        extra = df.duplicated(subset=keys, keep="first").sum()
        known_repeat = (table == "reviews" and keys == ["review_id"])
        note = ("review_id repeats across orders; see the composite key check below"
                if known_repeat else "number of surplus rows sharing a key")
        add_check(results, "Primary key", table, f"{label} unique", extra, len(df),
                  note, info=known_repeat)


def check_references(tables, results):
    for child, ccol, parent, pcol in REFERENCES:
        orphans = (~tables[child][ccol].isin(tables[parent][pcol])).sum()
        add_check(results, "Referential integrity", child,
                  f"{ccol} exists in {parent}.{pcol}", orphans, len(tables[child]))


def check_missing(tables, results):
    for table, df in tables.items():
        missing = missing_values_table(df)
        if missing.empty:
            add_check(results, "Missing values", table, "no missing values", 0, len(df))
            continue
        for col, row in missing.iterrows():
            reason = EXPECTED_MISSING.get((table, col), "")
            add_check(results, "Missing values", table, f"{col} missing",
                      row["missing_count"], len(df), reason, info=bool(reason))


def check_duplicates(tables, results):
    for table, df in tables.items():
        add_check(results, "Duplicate rows", table, "exact duplicate rows",
                  count_duplicates(df), len(df))


def date_rule(results, df, table, early, late, label):
    both = df[early].notna() & df[late].notna()
    add_check(results, "Date consistency", table, label,
              (df[late] < df[early]).sum(), both.sum())


def check_dates(tables, results):
    o = tables["orders"]
    date_rule(results, o, "orders", "order_purchase_timestamp", "order_approved_at", "approval not before purchase")
    date_rule(results, o, "orders", "order_purchase_timestamp", "order_delivered_carrier_date", "carrier handover not before purchase")
    date_rule(results, o, "orders", "order_delivered_carrier_date", "order_delivered_customer_date", "customer delivery not before carrier handover")
    date_rule(results, o, "orders", "order_purchase_timestamp", "order_delivered_customer_date", "customer delivery not before purchase")
    date_rule(results, o, "orders", "order_purchase_timestamp", "order_estimated_delivery_date", "estimated delivery not before purchase")
    r = tables["reviews"]
    date_rule(results, r, "reviews", "review_creation_date", "review_answer_timestamp", "review answer not before review creation")
    add_check(results, "Date consistency", "orders", "any date-order flag (flag_date_inconsistent)",
              o["flag_date_inconsistent"].sum(), len(o), "rows kept and flagged; one row can break several rules")


def numeric_rule(results, df, table, col, bad, label, info=False, note=""):
    add_check(results, "Numeric validity", table, label, bad.sum(), df[col].notna().sum(), note, info)


def check_numeric(tables, results):
    items = tables["order_items"]
    numeric_rule(results, items, "order_items", "price", items["price"] <= 0, "price > 0")
    numeric_rule(results, items, "order_items", "freight_value", items["freight_value"] < 0, "freight_value >= 0")
    numeric_rule(results, items, "order_items", "price", items["freight_value"] > items["price"],
                 "freight_value not above item price", info=True,
                 note="informational: high freight on cheap items is plausible")
    pay = tables["payments"]
    numeric_rule(results, pay, "payments", "payment_value", pay["payment_value"] < 0, "payment_value >= 0")
    numeric_rule(results, pay, "payments", "payment_installments", pay["payment_installments"] < 1, "payment_installments >= 1")
    rev = tables["reviews"]
    numeric_rule(results, rev, "reviews", "review_score", ~rev["review_score"].between(1, 5), "review_score between 1 and 5")
    prod = tables["products"]
    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        numeric_rule(results, prod, "products", col, prod[col] <= 0, f"{col} > 0")
    geo = tables["geolocation"]
    numeric_rule(results, geo, "geolocation", "geolocation_lat", geo["flag_outside_brazil"].astype(bool),
                 "coordinates inside Brazil bounding box")


def check_formats(tables, results):
    for table, cols in ZIP_COLUMNS.items():
        for col in cols:
            ok = tables[table][col].str.fullmatch(r"\d{5}").fillna(False).astype(bool)
            add_check(results, "Format", table, f"{col} is 5 digits", (~ok).sum(), len(ok))
    for table, col in [("customers", "customer_state"), ("sellers", "seller_state"),
                       ("geolocation", "geolocation_state")]:
        ok = tables[table][col].str.fullmatch(r"[A-Z]{2}").fillna(False).astype(bool)
        add_check(results, "Format", table, f"{col} is 2 capital letters", (~ok).sum(), len(ok))


def validate_all(tables):
    """Run every check. Returns (results_df, clean_summary_df, clean_missing_df)."""
    results = []
    check_primary_keys(tables, results)
    check_references(tables, results)
    check_missing(tables, results)
    check_duplicates(tables, results)
    check_dates(tables, results)
    check_numeric(tables, results)
    check_formats(tables, results)

    summaries, missing_frames = [], []
    for name, df in tables.items():
        summary, missing = table_summary(df, name)
        summaries.append(summary)
        if not missing.empty:
            m = missing.reset_index().rename(columns={"index": "column"})
            m.insert(0, "table", name)
            missing_frames.append(m)
    missing_df = (pd.concat(missing_frames, ignore_index=True) if missing_frames
                  else pd.DataFrame(columns=["table", "column", "missing_count", "missing_percent"]))
    return pd.DataFrame(results), pd.DataFrame(summaries), missing_df


def print_and_save(results_df, summary_df, missing_df):
    print("\nDATA QUALITY SUMMARY (all checks)")
    print(results_df.to_string(index=False))
    print("\nStatus counts:")
    print(results_df["status"].value_counts().to_string())
    print("\nCLEANED DATA: TABLE SUMMARY")
    print(summary_df.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "data_quality_summary.csv", index=False)
    summary_df.to_csv(REPORTS_DIR / "clean_exploration_summary.csv", index=False)
    missing_df.to_csv(REPORTS_DIR / "clean_missing_values.csv", index=False)
    print(f"\nSaved reports to: {REPORTS_DIR}")


if __name__ == "__main__":
    # Standalone test:  python src\data_validation.py
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    create_project_directories()
    cleaned = load_cleaned_datasets()
    if cleaned is not None:
        print_and_save(*validate_all(cleaned))
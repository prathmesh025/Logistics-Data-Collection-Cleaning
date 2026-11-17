"""
main.py
-------
Runs the complete Logistics Data Collection & Cleaning pipeline:
  1. Collect  2. Explore (raw)  3. Clean  4. Validate  5. Transform  6. Visualize

Usage (from the project root):   python main.py
"""

import contextlib
import sys

import pandas as pd

from src import data_cleaning as cleaning
from src import data_collection as collection
from src import data_exploration as exploration
from src import data_transformation as transformation
from src import data_validation as validation
from src import data_visualization as visualization


class Tee:
    """Write console output to the screen and to a log file at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def build_before_after(raw_summary, clean_summary):
    """Compare raw and cleaned tables using the same exploration metrics."""
    cols = ["rows", "columns", "missing_cells", "duplicate_rows", "date_columns_as_text"]
    before = raw_summary.set_index("table")[cols].add_suffix("_before")
    after = clean_summary.set_index("table")[cols].add_suffix("_after")
    comparison = before.join(after).reset_index()
    comparison.to_csv(collection.REPORTS_DIR / "before_after_summary.csv", index=False)
    return comparison


def run_pipeline():
    print("STEP 1: DATA COLLECTION")
    collection.create_project_directories()
    datasets = collection.load_all_datasets()
    if datasets is None:
        sys.exit(1)

    print("\nSTEP 2: DATA EXPLORATION (raw data)")
    log_path = collection.REPORTS_DIR / "raw_exploration_log.txt"
    with open(log_path, "w", encoding="utf-8") as log_file, contextlib.redirect_stdout(log_file):
        raw_summary, raw_missing = exploration.explore_all(datasets)
    exploration.save_exploration_results(raw_summary, raw_missing, prefix="raw")
    print(f"Full exploration log: {log_path}")
    print(raw_summary.to_string(index=False))

    print("\nSTEP 3: DATA CLEANING")
    cleaned, cleaning_log = cleaning.clean_all(datasets)
    print("\nCLEANING LOG")
    print(cleaning_log.to_string(index=False))
    cleaning.save_cleaned_datasets(cleaned, cleaning_log)

    print("\nSTEP 4: DATA VALIDATION (reading cleaned files back from disk)")
    reloaded = validation.load_cleaned_datasets()
    if reloaded is None:
        sys.exit(1)
    quality_results, clean_summary, clean_missing = validation.validate_all(reloaded)
    validation.print_and_save(quality_results, clean_summary, clean_missing)
    before_after = build_before_after(raw_summary, clean_summary)
    print("\nBEFORE vs AFTER CLEANING")
    print(before_after.to_string(index=False))

    print("\nSTEP 5: DATA TRANSFORMATION")
    order_level, item_level, centroids = transformation.transform_all(reloaded)
    transformation.print_transformation_summary(order_level, item_level)
    print("\nStatus of orders that have no items:")
    print(order_level[order_level["n_items"].isna()]["order_status"].value_counts().to_string())
    transformation.save_analysis_datasets(order_level, item_level, centroids)

    print("\nSTEP 6: VISUALIZATION")
    visualization.create_all_visualizations(raw_missing, quality_results, before_after,
                                            order_level, item_level)
    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    collection.create_project_directories()
    run_log = collection.REPORTS_DIR / "run_log.txt"
    with open(run_log, "w", encoding="utf-8") as run_file, \
            contextlib.redirect_stdout(Tee(sys.stdout, run_file)):
        run_pipeline()
    print(f"\nRun log saved to: {run_log}")
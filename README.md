# Logistics Data Collection and Cleaning Strategies Using Python

Week 1 project of the Logistics Data Science Analyst Internship. A modular, beginner-friendly Python pipeline that collects, explores, cleans, validates and transforms the Olist Brazilian E-Commerce dataset into analysis-ready logistics data.

**Author:** Prathmesh Ghangare

## Dataset

Brazilian E-Commerce Public Dataset by Olist (Kaggle): https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

It contains 9 CSV files covering orders, items, customers, sellers, products, payments, reviews, geolocation and a category translation table. The raw data is **not** included in this repository. Please check the license on the Kaggle page before reusing it.

## Project structure

```
Logistics-Data-Collection-Cleaning/
├── data/raw/              original CSVs (you download these)
├── data/processed/        cleaned and analysis-ready CSVs (created by the pipeline)
├── notebooks/             optional notebooks
├── src/
│   ├── data_collection.py     paths, file checks, safe CSV loading
│   ├── data_exploration.py    shape, dtypes, statistics, missing values, duplicates
│   ├── data_cleaning.py       per-table cleaning with a step-by-step log
│   ├── data_validation.py     post-cleaning checks and data-quality summary
│   ├── data_transformation.py derived logistics columns, analysis-ready datasets
│   └── data_visualization.py  11 Matplotlib/Seaborn charts
├── outputs/visualizations/    PNG charts
├── outputs/reports/           logs and summary CSVs
├── requirements.txt
├── README.md
└── main.py                    runs the whole pipeline
```

## Installation (Windows)

Requires Python 3.9 or newer. Tested with pandas 3.1.1, NumPy 2.5.5, Matplotlib 3.11.2 and Seaborn 0.13.2.

```bat
cd Logistics-Data-Collection-Cleaning
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

In PowerShell, if activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` or use Command Prompt.

## Getting the data

Manual download (no code needed):

1. Log in to Kaggle and open the dataset page above.
2. Click **Download** and extract the zip.
3. Copy all 9 CSV files into `data\raw\`.

Optional: install the Kaggle API, place `kaggle.json` in `C:\Users\<you>\.kaggle\`, and run `kaggle datasets download -d olistbr/brazilian-ecommerce`. If any file is missing, the pipeline prints which files and how to get them.

## Running

```bat
python main.py
```

Individual modules can also be run alone, for example `python src\data_cleaning.py`. Close any Excel windows that have output CSVs open, or Windows will block the pipeline from overwriting them.

## What the pipeline does

| Step | Module | Main actions |
|---|---|---|
| 1 | data_collection | check files, load CSVs with error handling |
| 2 | data_exploration | dimensions, dtypes, samples, statistics, missing values, duplicates |
| 3 | data_cleaning | remove duplicates, standardize names, normalize text, convert dates, pad zip codes, handle missing values, flag invalid records |
| 4 | data_validation | key uniqueness, references between tables, missing values, duplicates, date order, numeric ranges, formats |
| 5 | data_transformation | delivery times, delay and late flag, freight ratio, shipping distance, order-level and item-level datasets |
| 6 | data_visualization | missing values, before/after, validation results, delivery and freight patterns |

Cleaning principles: raw data is never modified, suspicious records are flagged rather than deleted, and every step is logged in `outputs/reports/cleaning_log.csv`.

## Outputs

- `data/processed/clean_*.csv`: cleaned tables
- `data/processed/analysis_orders.csv`, `analysis_order_items.csv`, `geolocation_zip_centroids.csv`: analysis-ready datasets
- `outputs/reports/data_quality_summary.csv`: all validation checks
- `outputs/reports/before_after_summary.csv`: before and after cleaning comparison
- `outputs/reports/cleaning_log.csv`, `derived_columns_dictionary.csv`, `run_log.txt`
- `outputs/visualizations/*.png`: charts

## Results snapshot (from the verified run)

- Geolocation: 1,000,163 to 720,496 rows (261,831 exact duplicates, then 17,836 more exposed by normalization).
- Products: missing cells 2,448 to 1,830 (610 categories set to "unknown", 8 weight/dimension values imputed).
- Orders: 5 text date columns converted to datetime. Missing delivery dates are kept because they reflect order status.
- 189 orders have inconsistent date order (flagged, not deleted).
- Validation: 71 checks, 56 PASS, 10 INFO, 5 FLAGGED.
- 7,823 of 96,282 measurable orders (8.1%) were delivered late.

## Limitations

- Shipping distance uses zip-code centroids, not real routes.
- Flagged records are kept for the analyst to decide on in later analysis.

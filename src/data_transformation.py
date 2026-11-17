"""
data_transformation.py
----------------------
Module 5: Data Transformation

Builds analysis-ready datasets from the cleaned tables and adds derived
logistics columns. Saves them to data/processed/.
"""

import sys

import numpy as np
import pandas as pd

try:
    from src.data_collection import PROCESSED_DIR, REPORTS_DIR, create_project_directories
except ModuleNotFoundError:
    from data_collection import PROCESSED_DIR, REPORTS_DIR, create_project_directories

EARTH_RADIUS_KM = 6371.0

# Description of every derived column (saved for the report)
DERIVED_COLUMNS = [
    ("analysis_order_items", "product_volume_cm3", "length x height x width of the product"),
    ("analysis_order_items", "freight_ratio", "freight_value / price"),
    ("analysis_order_items", "freight_per_kg", "freight_value / product weight in kg"),
    ("analysis_order_items", "is_cross_state", "True when seller state differs from customer state"),
    ("analysis_order_items", "shipping_distance_km", "great-circle distance between seller and customer zip centroids"),
    ("analysis_orders", "n_items / n_sellers / n_products", "number of items, distinct sellers and distinct products in the order"),
    ("analysis_orders", "total_price / total_freight / total_weight_g", "sums over the order's items"),
    ("analysis_orders", "freight_ratio", "total_freight / total_price"),
    ("analysis_orders", "approval_time_hours", "order approval time minus purchase time"),
    ("analysis_orders", "handling_time_days", "carrier handover minus purchase"),
    ("analysis_orders", "carrier_transit_days", "customer delivery minus carrier handover"),
    ("analysis_orders", "delivery_time_days", "customer delivery minus purchase"),
    ("analysis_orders", "estimated_delivery_days", "estimated delivery date minus purchase"),
    ("analysis_orders", "delivery_delay_days", "actual minus estimated delivery (positive = late)"),
    ("analysis_orders", "is_late", "1 if delivery_delay_days > 0, 0 if not, blank if not measurable"),
    ("analysis_orders", "purchase_year_month / purchase_weekday", "calendar features from the purchase timestamp"),
    ("analysis_orders", "avg_review_score / n_reviews", "review score average and count per order"),
    ("analysis_orders", "total_payment_value / max_installments", "payment sum and highest installment count per order"),
]


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km between two points (vectorised)."""
    lat1, lng1, lat2, lng2 = (np.radians(x) for x in (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def build_zip_centroids(geolocation):
    """One average coordinate per zip prefix (points outside Brazil excluded)."""
    inside = geolocation[~geolocation["flag_outside_brazil"].astype(bool)]
    centroids = (inside.groupby("geolocation_zip_code_prefix")
                 .agg(zip_lat=("geolocation_lat", "mean"),
                      zip_lng=("geolocation_lng", "mean"),
                      n_points=("geolocation_lat", "size"))
                 .reset_index()
                 .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"}))
    return centroids


def build_item_level(c, centroids):
    """One row per order item, enriched with order, customer, seller and product data."""
    items = c["order_items"][["order_id", "order_item_id", "product_id", "seller_id",
                              "price", "freight_value"]]
    orders = c["orders"][["order_id", "customer_id", "order_status"]]
    customers = c["customers"][["customer_id", "customer_unique_id",
                                "customer_zip_code_prefix", "customer_city", "customer_state"]]
    sellers = c["sellers"][["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]]
    products = c["products"][["product_id", "product_category_name_english", "product_weight_g",
                              "product_length_cm", "product_height_cm", "product_width_cm"]]

    # validate="many_to_one" makes pandas raise an error if a join would duplicate rows
    df = (items.merge(orders, on="order_id", how="left", validate="many_to_one")
               .merge(customers, on="customer_id", how="left", validate="many_to_one")
               .merge(sellers, on="seller_id", how="left", validate="many_to_one")
               .merge(products, on="product_id", how="left", validate="many_to_one"))
    assert len(df) == len(items), "a join changed the number of order items"

    df["product_volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    df["freight_ratio"] = df["freight_value"] / df["price"]            # price > 0 was validated
    df["freight_per_kg"] = df["freight_value"] / (df["product_weight_g"] / 1000)
    df["is_cross_state"] = df["customer_state"] != df["seller_state"]

    lat = centroids.set_index("zip_code_prefix")["zip_lat"]
    lng = centroids.set_index("zip_code_prefix")["zip_lng"]
    df["shipping_distance_km"] = haversine_km(
        df["seller_zip_code_prefix"].map(lat), df["seller_zip_code_prefix"].map(lng),
        df["customer_zip_code_prefix"].map(lat), df["customer_zip_code_prefix"].map(lng))
    return df


def days_between(df, later, earlier, ok):
    """Difference in days between two datetime columns; blank where not ok."""
    return ((df[later] - df[earlier]).dt.total_seconds() / 86400).where(ok)


def build_order_level(c, item_level):
    """One row per order with aggregated items, payments, reviews and delivery metrics."""
    orders = c["orders"].copy()
    customers = c["customers"][["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                                "customer_city", "customer_state"]]

    item_agg = (item_level.groupby("order_id")
                .agg(n_items=("order_item_id", "count"),
                     n_sellers=("seller_id", "nunique"),
                     n_products=("product_id", "nunique"),
                     total_price=("price", "sum"),
                     total_freight=("freight_value", "sum"),
                     total_weight_g=("product_weight_g", "sum"),
                     is_cross_state=("is_cross_state", "max"),
                     avg_shipping_distance_km=("shipping_distance_km", "mean"))
                .reset_index())
    pay_agg = (c["payments"].groupby("order_id")
               .agg(total_payment_value=("payment_value", "sum"),
                    n_payment_records=("payment_sequential", "count"),
                    max_installments=("payment_installments", "max"))
               .reset_index())
    rev_agg = (c["reviews"].groupby("order_id")
               .agg(avg_review_score=("review_score", "mean"),
                    n_reviews=("review_id", "count"))
               .reset_index())

    df = (orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
                .merge(item_agg, on="order_id", how="left", validate="one_to_one")
                .merge(pay_agg, on="order_id", how="left", validate="one_to_one")
                .merge(rev_agg, on="order_id", how="left", validate="one_to_one"))
    assert len(df) == len(orders), "a join changed the number of orders"

    df["freight_ratio"] = df["total_freight"] / df["total_price"]

    # Time metrics: blank for date-inconsistent orders; delivery metrics need status 'delivered'
    consistent = ~df["flag_date_inconsistent"].astype(bool)
    delivered = consistent & (df["order_status"] == "delivered")
    purchase = "order_purchase_timestamp"
    df["approval_time_hours"] = ((df["order_approved_at"] - df[purchase])
                                 .dt.total_seconds() / 3600).where(consistent)
    df["handling_time_days"] = days_between(df, "order_delivered_carrier_date", purchase, consistent)
    df["carrier_transit_days"] = days_between(df, "order_delivered_customer_date",
                                              "order_delivered_carrier_date", delivered)
    df["delivery_time_days"] = days_between(df, "order_delivered_customer_date", purchase, delivered)
    df["estimated_delivery_days"] = days_between(df, "order_estimated_delivery_date", purchase, consistent)
    df["delivery_delay_days"] = days_between(df, "order_delivered_customer_date",
                                             "order_estimated_delivery_date", delivered)
    df["is_late"] = (df["delivery_delay_days"] > 0).astype("Int64").where(df["delivery_delay_days"].notna())

    df["purchase_year_month"] = df[purchase].dt.to_period("M").astype(str)
    df["purchase_weekday"] = df[purchase].dt.day_name()
    return df


def save_analysis_datasets(order_level, item_level, centroids):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in (("analysis_orders", order_level),
                     ("analysis_order_items", item_level),
                     ("geolocation_zip_centroids", centroids)):
        path = PROCESSED_DIR / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"Saved: {path}  ({df.shape[0]:,} rows x {df.shape[1]} columns)")
    dictionary = pd.DataFrame(DERIVED_COLUMNS, columns=["dataset", "column", "description"])
    dictionary.to_csv(REPORTS_DIR / "derived_columns_dictionary.csv", index=False)


def transform_all(cleaned):
    """Build all analysis-ready datasets. Returns (order_level, item_level, centroids)."""
    centroids = build_zip_centroids(cleaned["geolocation"])
    item_level = build_item_level(cleaned, centroids)
    order_level = build_order_level(cleaned, item_level)
    return order_level, item_level, centroids


def print_transformation_summary(order_level, item_level):
    print("\nTRANSFORMATION SUMMARY")
    print(f"analysis_orders:      {order_level.shape[0]:,} rows x {order_level.shape[1]} columns")
    print(f"analysis_order_items: {item_level.shape[0]:,} rows x {item_level.shape[1]} columns")
    cols = ["approval_time_hours", "handling_time_days", "carrier_transit_days",
            "delivery_time_days", "estimated_delivery_days", "delivery_delay_days",
            "freight_ratio", "avg_shipping_distance_km", "avg_review_score"]
    print("\nDerived order-level columns (count, mean, min, max):")
    print(order_level[cols].describe().T[["count", "mean", "min", "max"]].round(2))
    print("\nOrders marked late (is_late):")
    print(order_level["is_late"].value_counts(dropna=False).to_string())
    print("\nOrders with no items:", int(order_level["n_items"].isna().sum()))
    print("Items with unknown shipping distance:", int(item_level["shipping_distance_km"].isna().sum()))


if __name__ == "__main__":
    # Standalone test:  python src\data_transformation.py  (run data_cleaning.py first)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        from src.data_validation import load_cleaned_datasets
    except ModuleNotFoundError:
        from data_validation import load_cleaned_datasets
    create_project_directories()
    cleaned_tables = load_cleaned_datasets()
    if cleaned_tables is not None:
        orders_df, items_df, centroids_df = transform_all(cleaned_tables)
        print_transformation_summary(orders_df, items_df)
        save_analysis_datasets(orders_df, items_df, centroids_df)
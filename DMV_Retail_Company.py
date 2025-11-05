import os
import pandas as pd
import matplotlib.pyplot as plt

IN = "retail_sales_dataset.csv"
OUT_DIR = "sales_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load & quick explore
df = pd.read_csv(IN, low_memory=False)
print("Loaded rows,cols:", df.shape)
print("Cols:", list(df.columns))
print("Sample:\n", df.head(3).to_string(index=False))

# 2. Ensure columns consistent (normalize names)
df.columns = [c.strip() for c in df.columns]

# 3. Ensure numeric totals exist
if "Total Amount" not in df.columns or df["Total Amount"].isnull().any():
    if "Quantity" in df.columns and "Price per Unit" in df.columns:
        df["Total Amount"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0) * \
                             pd.to_numeric(df["Price per Unit"], errors="coerce").fillna(0)
df["Total Amount"] = pd.to_numeric(df["Total Amount"], errors="coerce").fillna(0)

# 4. Date parsing and month column for time aggregation
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

# Decide aggregation key: Region if present, else Product Category
if "Region" in df.columns:
    key = "Region"
else:
    key = "Product Category"
print(f"\nAggregating by: {key}")

# 5. Total sales by key (task 4)
sales_by_key = df.groupby(key)["Total Amount"].sum().sort_values(ascending=False)
print("\nTotal sales by", key, ":\n", sales_by_key.head(10).round(2))

sales_by_key.head(10).plot(kind="bar", figsize=(8,4), color="coral")
plt.title(f"Total Sales by {key} (Top 10)")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

sales_by_key.plot(kind="pie", figsize=(6,6), autopct="%1.1f%%", startangle=90, cmap="Pastel1")
plt.ylabel("")
plt.title(f"Sales Share by {key}")
plt.tight_layout()
plt.show()


top5 = sales_by_key.head(5)
print("\nTop 5", key, "by sales:\n", top5.round(2))

# 8. Grouped/stacked bars by month and category
if key == "Region" and "Product Category" in df.columns:
    combo = df.groupby([key, "Product Category"])["Total Amount"].sum().unstack(fill_value=0)
    combo.plot(kind="bar", stacked=True, figsize=(10,5))
    plt.title("Sales by Region and Product Category (Stacked)")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()
elif key == "Product Category" and "Month" in df.columns:
    top_cats = df.groupby("Product Category")["Total Amount"].sum().nlargest(6).index
    combo_top = df[df["Product Category"].isin(top_cats)].groupby(["Month", "Product Category"])["Total Amount"].sum().unstack(fill_value=0)
    combo_top.plot(kind="bar", figsize=(10,5))
    plt.title("Monthly Sales for Top Product Categories")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()

# 9. Simple summary stats
print("\nSales Summary Statistics:")
print(df["Total Amount"].describe().round(2))

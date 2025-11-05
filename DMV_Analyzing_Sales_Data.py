import os, sys
import pandas as pd
import matplotlib.pyplot as plt

CSV = "sales_data_sample.csv"
XLSX = "sales data.xlsx"
JSON = "users.json"

def read_csv_try(path):
    if not os.path.exists(path): return pd.DataFrame()
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding="latin1", errors="replace")

def read_excel(path):
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_excel(path)

def read_json(path):
    if not os.path.exists(path): return pd.DataFrame()
    try:
        return pd.read_json(path)
    except Exception:
        import json
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        return pd.json_normalize(data) if isinstance(data, (list, dict)) else pd.DataFrame(data)
    
# Load files
df_csv = read_csv_try(CSV)
df_xl  = read_excel(XLSX)
df_js  = read_json(JSON)   # optional customer info

# Normalize column names
def norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

df_csv, df_xl, df_js = map(norm_cols, (df_csv, df_xl, df_js))

# Combine CSV + Excel as sales sources
frames = [d for d in (df_csv, df_xl) if not d.empty]
if not frames:
    sys.exit("No CSV or Excel sales files found. Put files in same folder or edit filenames.")

df = pd.concat(frames, ignore_index=True, sort=False)
df = norm_cols(df)

# Convert common columns and compute total
# Known CSV columns seen: ordernumber, quantityordered, priceeach, sales, orderdate, productline
if "orderdate" in df.columns:
    df["orderdate"] = pd.to_datetime(df["orderdate"], errors="coerce")

for c in ("quantityordered", "priceeach", "sales", "msrp"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
# Compute total: prefer 'sales' if present, else quantityordered * priceeach
if "sales" in df.columns and df["sales"].notna().any():
    df["total"] = df["sales"].fillna(0)
else:
    q = df.get("quantityordered", pd.Series(1, index=df.index))
    p = df.get("priceeach", pd.Series(0.0, index=df.index))
    df["total"] = pd.to_numeric(q, errors="coerce").fillna(1) * pd.to_numeric(p, errors="coerce").fillna(0)
    
# Basic cleaning: drop rows without productline (if exists) or without total
if "productline" in df.columns:
    df = df.dropna(subset=["productline"])
df = df[df["total"].notna()]

# --- Analysis prints ---
print("\n=== Simple Sales Analysis ===")
print("Rows:", len(df))
print("Total sales amount:", round(df["total"].sum(), 2))

# Average order value
order_col = "ordernumber" if "ordernumber" in df.columns else ("order_number" if "order_number" in df.columns else None)
if order_col:
    avg_order = df.groupby(order_col)["total"].sum().mean()
    print("Average order value (by order id):", round(avg_order, 2))
else:
    print("Average sale per row:", round(df["total"].mean(), 2))
    
# Top product lines (if available)
if "productline" in df.columns:
    top_lines = df.groupby("productline")["total"].sum().sort_values(ascending=False)
    print("\nTop product lines (by sales):")
    print(top_lines.head(8))
    
# Descriptive stats
print("\nNumeric summary:")
print(df[["quantityordered","priceeach","total"]].describe().transpose().loc[:,["count","mean","std","min","max"]].dropna(how="all"))

plt.figure(figsize=(8,5))
if "productline" in df.columns:
    top = df.groupby("productline")["total"].sum().sort_values(ascending=False).head(8)
    ax = top.plot(kind="bar", legend=False)
    ax.set_ylabel("Total sales")
    ax.set_title("Top product lines by sales")
    plt.tight_layout()
    plt.show()

if "productline" in df.columns:
    s = df.groupby("productline")["total"].sum()
    if len(s) > 1:
        plt.figure(figsize=(6,6))
        s.plot(kind="pie", autopct="%1.1f%%")
        plt.ylabel("")
        plt.title("Sales share by product line")
        plt.tight_layout()
        plt.show()
        
if order_col:
    orders = df.groupby(order_col)["total"].sum().dropna()
    if len(orders) > 1:
        plt.figure(figsize=(6,4))
        plt.boxplot(orders)
        plt.title("Order value distribution")
        plt.ylabel("Order total")
        plt.tight_layout()
        plt.show()
        

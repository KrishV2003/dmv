import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

IN = "Real estate.csv"
OUT = "realestate_clean"

# 1. Load & clean column names
df = pd.read_csv(IN, low_memory=False)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df = df.rename(columns={
    "no":"id",
    "x1_transaction_date":"transaction_date",
    "x2_house_age":"house_age",
    "x3_distance_to_the_nearest_mrt_station":"dist_mrt",
    "x4_number_of_convenience_stores":"n_convenience",
    "x5_latitude":"lat",
    "x6_longitude":"lon",
    "y_house_price_of_unit_area":"price_per_unit"
})

# 2. Quick explore (small prints)
print("Rows,Cols:", df.shape)
print("Cols:", list(df.columns))

# 3+6. Convert types & handle missing
num = ["transaction_date","house_age","dist_mrt","n_convenience","lat","lon","price_per_unit"]
for c in num:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# fill numeric missing with median — assign back (no inplace on a slice)
for c in df.select_dtypes(include=[np.number]).columns:
    median_val = df[c].median()
    df[c] = df[c].fillna(median_val)

# fill object missing (assign back)
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].fillna("")
    
# 4. Remove duplicates (safe)
df = df.drop_duplicates()

# 5. Standardize (strip strings) - minimal needed here
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip()
    
# 7. Outlier handling (IQR cap for key numerics)
def cap_iqr(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

for c in ["price_per_unit","dist_mrt"]:
    if c in df.columns:
        df[c] = cap_iqr(df[c])
        
        
# 8. Feature engineering
if "house_age" in df.columns:
    df["age_group"] = pd.cut(df["house_age"], bins=[-1,5,15,30,100], labels=["0-5","6-15","16-30","30+"])
if "dist_mrt" in df.columns:
    df["near_mrt"] = (df["dist_mrt"] <= df["dist_mrt"].median()).map({True:"near", False:"far"})
if "price_per_unit" in df.columns:
    df["price_per_100"] = df["price_per_unit"] * 100

# 5 (encoding) - one-hot small categories
to_dummify = [col for col in ["age_group","near_mrt"] if col in df.columns]
if to_dummify:
    df = pd.get_dummies(df, columns=to_dummify, drop_first=True)
    
# 6 (aggregation example)
print("\nAvg price by #convenience stores:")
if "n_convenience" in df.columns and "price_per_unit" in df.columns:
    print(df.groupby("n_convenience")["price_per_unit"].mean().round(2).head())
    
# 9. Scale numeric features (use .loc to assign)
scale_cols = [c for c in ["house_age","dist_mrt","n_convenience","price_per_unit","price_per_100"] if c in df.columns]
if scale_cols:
    scaled_names = [c + "_scaled" for c in scale_cols]
    df.loc[:, scaled_names] = MinMaxScaler().fit_transform(df[scale_cols])
    
# 4 (filter example)
if "dist_mrt" in df.columns and "price_per_unit" in df.columns:
    subset = df[(df["dist_mrt"] <= 500) & (df["price_per_unit"] > df["price_per_unit"].median())]
    print("Subset (near MRT & above median price):", len(subset))
else:
    print("Subset (near MRT & above median price): 0 (missing columns)")
    
# 10+11. Split and export
df = df.drop(columns=["id"], errors="ignore")
train, test = train_test_split(df, test_size=0.2, random_state=42)

df.to_csv(OUT + "_cleaned.csv", index=False)
train.to_csv(OUT + "_train.csv", index=False)
test.to_csv(OUT + "_test.csv", index=False)

print("\nSaved:", OUT + "_cleaned.csv,", OUT + "_train.csv,", OUT + "_test.csv")
print("\nDone.")


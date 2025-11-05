import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

INFILE = "Telcom_Customer_Churn.csv"  # change if needed
OUT_PREFIX = "telecom_clean"

# 1. Load
df = pd.read_csv(INFILE, low_memory=False)
print("Loaded rows,cols:", df.shape)

# 2. Quick explore
print("Columns:", list(df.columns))
print(df.head(2).to_string(index=False))
print("Missing per column:\n", df.isna().sum())

# 3. Standardize text and 6. Convert types
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()
# TotalCharges sometimes string -> numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
for n in ["tenure", "MonthlyCharges", "TotalCharges"]:
    if n in df.columns:
        df[n] = pd.to_numeric(df[n], errors="coerce")
        
        
# 4. Handle missing values
# If TotalCharges missing but tenure & MonthlyCharges present, estimate it
if "TotalCharges" in df.columns:
    miss = df["TotalCharges"].isna()
    df.loc[miss, "TotalCharges"] = (df.loc[miss, "MonthlyCharges"] * df.loc[miss, "tenure"]).fillna(0)
# Fill remaining numeric missings with median, categorical with "No"
for c in df.columns:
    if df[c].isna().any():
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c].fillna(df[c].median(), inplace=True)
        else:
            df[c].fillna("No", inplace=True)
            
# 5. Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print("Duplicates removed:", before - len(df))

# 6. Standardize inconsistent values (simple)
if "InternetService" in df.columns:
    df["InternetService"] = df["InternetService"].replace({"Fiber": "Fiber optic"})
for b in ["Yes","No"]:
    pass  # kept simple; string strip above handles most variants

# 7. Handle outliers (IQR capping for numeric columns)
nums = ["tenure", "MonthlyCharges", "TotalCharges"]
for c in nums:
    if c in df.columns:
        q1,q3 = df[c].quantile([0.25,0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        df[c] = df[c].clip(lower, upper)
        
# 8. Feature engineering (simple useful features)
# tenure group
if "tenure" in df.columns:
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1,12,24,48,60,999],
                                labels=["0-12","13-24","25-48","49-60","60+"])
# number of services (count Yes across common service cols)
service_cols = [c for c in ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                            "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"] if c in df.columns]
if service_cols:
    df["num_services"] = df[service_cols].apply(lambda r: sum(x=="Yes" for x in r), axis=1)
# avg monthly per tenure (avoid divide by zero)
df["avg_monthly_per_tenure"] = (df["TotalCharges"] / df["tenure"]).replace([np.inf,-np.inf],0).fillna(0)


# 9. Normalize / scale numeric features
scale_cols = [c for c in ["tenure","MonthlyCharges","TotalCharges","num_services","avg_monthly_per_tenure"] if c in df.columns]
if scale_cols:
    scaler = MinMaxScaler()
    df[[c + "_scaled" for c in scale_cols]] = scaler.fit_transform(df[scale_cols])
    
    
# 10. Split into train/test
if "Churn" in df.columns:
    df["Churn_bin"] = df["Churn"].map({"Yes":1,"No":0})
    X = df.drop(columns=["Churn","Churn_bin","customerID"], errors="ignore")
    y = df["Churn_bin"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    train = X_train.copy(); train["Churn"] = y_train.values
    test = X_test.copy(); test["Churn"] = y_test.values
else:
    train, test = train_test_split(df, test_size=0.2, random_state=42)


# 11. Export cleaned & split files
df.to_csv(OUT_PREFIX + "_cleaned.csv", index=False)
train.to_csv(OUT_PREFIX + "_train.csv", index=False)
test.to_csv(OUT_PREFIX + "_test.csv", index=False)
print("Exported:", OUT_PREFIX + "_cleaned.csv, *_train.csv, *_test.csv")

# Short summary prints
print("\nSummary after cleaning:")
print(df[["tenure","MonthlyCharges","TotalCharges"]].describe().round(2))
if "Churn" in df.columns:
    print("Churn distribution:\n", df["Churn"].value_counts())
    

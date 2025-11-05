import pandas as pd
import matplotlib.pyplot as plt
import os

INFILE = "City_Air_Quality.csv"
out_dir = "aqi_plots"
os.makedirs(out_dir, exist_ok=True)

# 1-2. Load & quick explore
df = pd.read_csv(INFILE, low_memory=False)
print("Rows,Cols:", df.shape)
print("Columns:", list(df.columns))
print("Head:\n", df.head(2).to_string(index=False))

# 3. Prepare datetime and relevant columns
df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
df = df.sort_values("datetime").reset_index(drop=True)
cols_pollutants = ["PM2.5", "PM10", "CO"]
for c in cols_pollutants + ["Temperature (Celsius)", "Humidity (%)", "AQI"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
# Basic missing handling: drop rows without datetime or AQI
df = df.dropna(subset=["datetime", "AQI"])

# 4. Time-series: AQI over time
plt.figure(figsize=(10,4))
plt.plot(df["datetime"], df["AQI"], marker=".", linewidth=1)
plt.title("AQI over Time")
plt.xlabel("Date & Time")
plt.ylabel("AQI")
plt.tight_layout()
plt.show()

# 5. Pollutant trends (PM2.5, PM10, CO)
plt.figure(figsize=(10,4))
for c in cols_pollutants:
    if c in df.columns:
        plt.plot(df["datetime"], df[c], label=c, linewidth=1)
plt.legend()
plt.title("Pollutant Levels Over Time")
plt.xlabel("Date & Time")
plt.ylabel("Concentration")
plt.tight_layout()
plt.show()

# 6. Bar plot: Daily average AQI
daily = df.set_index("datetime").resample("D").agg({"AQI":"mean"}).dropna()
plt.figure(figsize=(8,4))
daily["AQI"].plot(kind="bar", width=0.8, color="steelblue")
plt.title("Daily Average AQI")
plt.ylabel("Average AQI")
plt.tight_layout()
plt.show()

# 7. Box plots: Distribution of AQI and pollutants
plot_cols = [c for c in ["AQI"] + cols_pollutants if c in df.columns]
plt.figure(figsize=(8,4))
df[plot_cols].boxplot()
plt.title("Distribution: AQI & Pollutants")
plt.ylabel("Values")
plt.tight_layout()
plt.show()

# 8. Scatter / Bubble plot: AQI vs Temperature (size = Humidity)
if "Temperature (Celsius)" in df.columns and "Humidity (%)" in df.columns:
    plt.figure(figsize=(7,5))
    sizes = (df["Humidity (%)"].fillna(0).astype(float) + 1) * 5
    plt.scatter(df["Temperature (Celsius)"], df["AQI"], s=sizes, alpha=0.6, color="orange", edgecolors="k")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("AQI")
    plt.title("AQI vs Temperature (Bubble Size = Humidity)")
    plt.tight_layout()
    plt.show()
    
# 9. Correlation Heatmap
numeric = df[["AQI"] + [c for c in cols_pollutants if c in df.columns] + 
             [c for c in ["Temperature (Celsius)", "Humidity (%)"] if c in df.columns]]
corr = numeric.corr()

plt.figure(figsize=(6,5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
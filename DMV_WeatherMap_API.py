import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

API_KEY = "224b2be7c81d44e03a4fd88a077a296c"   # Your API Key
city = input("Enter city name: ") or "Mumbai"
units = "metric"  # Celsius

url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units={units}"
response = requests.get(url)
data = response.json()

if response.status_code != 200:
    print("Error fetching data:", data.get("message", "Unknown error"))
    exit()
    
records = []
for item in data["list"]:
    main = item["main"]
    wind = item["wind"]
    rain = item.get("rain", {}).get("3h", 0)
    records.append({
        "datetime": item["dt_txt"],
        "temperature": main["temp"],
        "humidity": main["humidity"],
        "wind_speed": wind["speed"],
        "rain": rain
    })

df = pd.DataFrame(records)

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.fillna(0)
df["date"] = df["datetime"].dt.date


print("\n=== Weather Summary ===")
print(f"City: {city}")
print("Average Temperature:", round(df["temperature"].mean(), 2), "°C")
print("Max Temperature:", round(df["temperature"].max(), 2), "°C")
print("Min Temperature:", round(df["temperature"].min(), 2), "°C")
print("Average Humidity:", round(df["humidity"].mean(), 2), "%")
print("Average Wind Speed:", round(df["wind_speed"].mean(), 2), "m/s")
print("Total Rainfall:", round(df["rain"].sum(), 2), "mm")

plt.figure(figsize=(8,4))
plt.plot(df["datetime"], df["temperature"], color='orange', marker='o')
plt.title(f"Temperature Trend in {city}")
plt.xlabel("Date & Time")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
daily_rain = df.groupby("date")["rain"].sum()
plt.bar(daily_rain.index, daily_rain.values, color='skyblue')
plt.title(f"Daily Rainfall in {city}")
plt.ylabel("Rain (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["temperature"], df["humidity"], alpha=0.7, color='green')
plt.title("Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.tight_layout()
plt.show()

daily_summary = df.groupby("date").agg({
    "temperature": "mean",
    "humidity": "mean",
    "rain": "sum"
}).round(2)
print("\nDaily Summary:\n", daily_summary)

coord = data["city"]["coord"]
print("\nLocation coordinates:")
print(f"Latitude: {coord['lat']}  |  Longitude: {coord['lon']}")

plt.figure(figsize=(4,3))
plt.scatter(coord["lon"], coord["lat"], color="red", s=80)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Map location of {city}")
plt.tight_layout()
plt.show()


corr = df[["temperature", "humidity", "wind_speed", "rain"]].corr()
print("\nCorrelation between Weather Attributes:\n", corr.round(2))

plt.figure(figsize=(5,4))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(label='Correlation')
plt.tight_layout()
plt.show()


"""
Export clean data to formats suitable for Jamovi analysis.
Run: python export_for_jamovi.py
"""
import pandas as pd
from analysis import load_data, monthly_resample

# Load
df = load_data()
monthly = monthly_resample(df)

# --- 1. Daily data for Jamovi (CSV) ---
daily_export = df.copy()
daily_export["change_pct"] = daily_export["change %"] * 100  # percentage
daily_export.drop(columns=["change %"], inplace=True)
daily_export.to_csv("datasets/xauusd_daily_jamovi.csv", index=True)
print(f"✓ Exported daily data: datasets/xauusd_daily_jamovi.csv ({len(daily_export)} rows)")

# --- 2. Monthly data for Jamovi (CSV) ---
monthly_export = monthly.copy()
monthly_export["monthly_change_pct"] = monthly_export["monthly_change"] * 100
monthly_export.drop(columns=["monthly_change"], inplace=True)
monthly_export.to_csv("datasets/xauusd_monthly_jamovi.csv", index=True)
print(f"✓ Exported monthly data: datasets/xauusd_monthly_jamovi.csv ({len(monthly_export)} rows)")

# --- 3. Yearly summary for Jamovi ---
yearly = df.groupby(df.index.year)["price"].agg(["mean", "min", "max", "std", "count"])
yearly.columns = ["mean_price", "min_price", "max_price", "std_price", "trading_days"]
yearly.index.name = "year"
yearly.to_csv("datasets/xauusd_yearly_jamovi.csv")
print(f"✓ Exported yearly data: datasets/xauusd_yearly_jamovi.csv ({len(yearly)} rows)")

print("\n→ เปิดไฟล์ CSV เหล่านี้ใน Jamovi ได้เลย!")

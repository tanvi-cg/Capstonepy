import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, timedelta

DATE_COL = "DATE"
TEMP_COL = "TEMPERATURE_C"
RAIN_COL = "RAINFALL_MM"
HUMIDITY_COL = "HUMIDITY_PER"
CLEANED_CSV_PATH = (r"/Users/tanvichoudhary/Documents/Capstone1/cleaned_weather.csv","a")

def make_fake_data(days=365) -> pd.DataFrame:
    print(f"Making up data for {days} days...")
    
    start = '2024-01-01'
    dates = [date.fromisoformat(start) + timedelta(days=i) for i in range(days)]
    
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    temp_trend = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365)
    temps = np.round(temp_trend + np.random.normal(0, 3, days), 1)
    
    humid_trend = 65 - 10 * np.sin(2 * np.pi * day_of_year / 365)
    humids = np.clip(np.round(humid_trend + np.random.normal(0, 5, days)), 40, 95).astype(int)
    
    rain = np.random.choice([0.0, 0.0, 0.0, 0.1, 0.5, 2.0, 5.0, 10.0], days, 
                            p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
    
    temps[50:55] = np.nan
    rain[200] = np.nan
    
    data = pd.DataFrame({
        DATE_COL: dates,
        TEMP_COL: temps,
        RAIN_COL: rain,
        HUMIDITY_COL: humids
    })
    
    data.to_csv('mock_raw_data.csv', index=False)
    print("Mock data saved. Use your real CSV later!")
    return data

if __name__ == "__main__":
    
    raw_df = make_fake_data(days=365)
    
    print("\n--- Task 1: Data Check (First few rows) ---")
    print(raw_df.head())
    
    df = raw_df.copy()
    
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.set_index(DATE_COL).sort_index()
    
    df = df[[TEMP_COL, RAIN_COL, HUMIDITY_COL]]
    
    print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")
    
    df[TEMP_COL].fillna(df[TEMP_COL].rolling(window=7, min_periods=1, center=True).mean(), inplace=True)
    
    df.dropna(inplace=True)
    
    print(f"Missing values after cleaning:\n{df.isnull().sum()}")
    
    if df.empty:
        print("\nCleaned data is empty. Cannot continue.")
    else:
        
        print("\n--- Task 3: Daily Statistics (NumPy and Pandas) ---")
        temp_array = df[TEMP_COL].to_numpy()
        print(f"Highest Temperature Ever: {np.max(temp_array):.1f}°C")
        print(f"Lowest Temperature Ever: {np.min(temp_array):.1f}°C")
        
        monthly_summary = df.groupby(df.index.to_period('M')).agg({
            TEMP_COL: ['mean', 'max'],
            RAIN_COL: 'sum',
        }).reset_index()
        monthly_summary['Month'] = monthly_summary[DATE_COL].astype(str)
        
        print("\n--- Monthly Summary Table ---")
        print(monthly_summary.to_string(index=False))

        def determine_season(month):
            if month in [12, 1, 2]: return 'Winter'
            elif month in [3, 4, 5]: return 'Spring'
            elif month in [6, 7, 8]: return 'Summer'
            else: return 'Autumn'
        
        df['SEASON'] = df.index.month.map(determine_season)
        
        seasonal_results = df.groupby('SEASON').agg({
            TEMP_COL: 'mean',
            RAIN_COL: 'sum',
        }).reindex(['Spring', 'Summer', 'Autumn', 'Winter'])

        print("\n--- Seasonal Summary Table ---")
        print(seasonal_results.to_string())

        os.makedirs('plots', exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[TEMP_COL], color='red', linewidth=1.5)
        plt.title('Daily Temperature Over the Year')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, linestyle='--')
        plt.savefig('plots/1_daily_temperature.png')
        plt.close()
        print("Saved '1_daily_temperature.png'")

        plt.figure(figsize=(10, 6))
        monthly_rain_totals = monthly_summary[RAIN_COL, 'sum']
        months = monthly_summary['Month']
        plt.bar(months, monthly_rain_totals, color='blue')
        plt.title('Monthly Total Rainfall')
        plt.xlabel('Month')
        plt.ylabel('Rainfall (mm)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/2_monthly_rainfall.png')
        plt.close()
        print("Saved '2_monthly_rainfall.png'")

        plt.figure(figsize=(8, 6))
        plt.scatter(df[TEMP_COL], df[HUMIDITY_COL], alpha=0.6, color='green')
        plt.title('How Temperature Affects Humidity')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Humidity (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/3_temp_vs_humidity.png')
        plt.close()
        print("Saved '3_temp_vs_humidity.png'")
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Temperature (°C)', color='red')
        ax1.plot(df.index, df[TEMP_COL], color='red', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='red')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Rainfall (mm)', color='blue')
        ax2.bar(df.index, df[RAIN_COL], color='blue', alpha=0.3, width=1.0)
        ax2.tick_params(axis='y', labelcolor='blue')
        
        fig.suptitle('Daily Temperature and Rainfall Together')
        fig.tight_layout()
        plt.savefig('plots/4_combined_chart.png')
        plt.close()
        print("Saved '4_combined_chart.png'")
        
        df.to_csv(r"/Users/tanvichoudhary/Documents/Capstone1/cleaned_weather.csv","a")
        print(f"\nCleaned data exported to '{CLEANED_CSV_PATH}'.")
        
        report_content = f"""
# Weather Data Analysis Report Summary

## Key Numbers
- The average temperature for the whole period was: {df[TEMP_COL].mean():.1f}°C
- The total rainfall recorded was: {df[RAIN_COL].sum():.1f} mm

## Seasonal Overview
{seasonal_results.to_string()}

**What I found:**
Summer is the hottest season, of course, but it looks like the most rain happens in [Check your monthly_summary result here] when the temperatures start to go down a little. The scatter plot shows that when it's super hot, the humidity is usually lower.

This data would be helpful for the campus to plan where to collect rainwater!
"""
        with open('final_report_summary.txt', 'w') as f:
            f.write(report_content)
            
        print("Summary report saved to 'final_report_summary.txt'.")
        print("\n--- ALL TASKS COMPLETE ---")
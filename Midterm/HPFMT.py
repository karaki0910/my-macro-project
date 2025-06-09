from fredapi import Fred
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# Replace with your FRED API key
# You can get a key from https://fred.stlouisfed.org/docs/api/api_key.html
fred_api_key = '67cefbfc31f28a06917dbb02c35ba738' # Replace with your actual API key

if fred_api_key == 'YOUR_FRED_API_KEY':
    print("Please replace 'YOUR_FRED_API_KEY' with your actual FRED API key.")
    # You might want to stop execution here if the key is not provided
    # exit() or raise Exception("FRED API key not provided.")

fred = Fred(api_key=fred_api_key)

# --- 1. Choose a country and retrieve its real GDP data from FRED ---
# You need to find the FRED series ID for the real GDP of your chosen country.
# FRED uses specific IDs for each data series.
# You can search for series on the FRED website: https://fred.stlouisfed.org/
# Common series IDs for real GDP include:
# - 'GDPC96': US Real Gross Domestic Product
# - 'JPNRGDP': Japan Real Gross Domestic Product
# - 'DEURGDP': Germany Real Gross Domestic Product (using Euro Area)
# - 'GBRRGDP': UK Real Gross Domestic Product
# - 'CNRGDP': China Real Gross Domestic Product (Note: Availability and definition might vary)

# Example: Choose Germany
country_name = 'China'
country_gdp_series_id = 'MKTGDPCNA646NWDB' # Replace with the series ID for your chosen country

try:
    country_gdp = fred.get_series(country_gdp_series_id)
    japan_gdp = fred.get_series('JPNRGDPEXP')

    # Drop any NaN values and align data by date
    country_gdp = country_gdp.dropna()
    japan_gdp = japan_gdp.dropna()

    # Ensure both series cover the same time period
    start_date = max(country_gdp.index.min(), japan_gdp.index.min())
    end_date = min(country_gdp.index.max(), japan_gdp.index.max())

    country_gdp = country_gdp.loc[start_date:end_date]
    japan_gdp = japan_gdp.loc[start_date:end_date]

    # --- 2. Apply HP-filter to chosen country's log real GDP ---
    country_log_gdp = np.log(country_gdp)
    country_cycle, country_trend = sm.tsa.filters.hpfilter(country_log_gdp, lamb=1600) # lambda=1600 is common for quarterly data

    # --- 3. Apply HP-filter to Japan's log real GDP ---
    japan_log_gdp = np.log(japan_gdp)
    japan_cycle, japan_trend = sm.tsa.filters.hpfilter(japan_log_gdp, lamb=1600) # lambda=1600 is common for quarterly data

    # --- 4. Calculate standard deviations and correlation ---
    country_cycle_std = country_cycle.std()
    japan_cycle_std = japan_cycle.std()

    print(f"\nStandard Deviation of {country_name} GDP Cycle Component: {country_cycle_std:.4f}")
    print(f"Standard Deviation of Japan GDP Cycle Component: {japan_cycle_std:.4f}")

    # Calculate correlation only for the overlapping period
    correlation = country_cycle.corr(japan_cycle)
    print(f"Correlation between {country_name} and Japan GDP Cycle Components: {correlation:.4f}")

    # --- 5. Plot cycle components ---
    plt.figure(figsize=(12, 6))
    plt.plot(country_cycle.index, country_cycle, label=f'{country_name} Cycle')
    plt.plot(japan_cycle.index, japan_cycle, label='Japan Cycle')
    plt.title(f'GDP Cycle Components: {country_name} vs Japan (HP Filter)')
    plt.xlabel('Date')
    plt.ylabel('Log GDP Cycle Component')
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check the FRED series IDs and your API key.")
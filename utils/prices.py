import pandas as pd
import os

def analyze_household_electricity_prices():
    """
    Analyzes household electricity prices for specified countries and years,
    calculating annual averages from half-yearly data for a specific consumption band.
    """
    try:
        current_script_path = os.path.abspath(__file__)      
        project_root_folder = os.path.dirname(os.path.dirname(current_script_path))
        datasets_folder = os.path.join(project_root_folder, 'datasets')
        prices_file_name = 'estat_nrg_pc_204$defaultview_filtered_en.csv' 
        prices_file_path = os.path.join(datasets_folder, prices_file_name)
        df_prices = pd.read_csv(prices_file_path)
    except Exception as e:
        print(f"error occurred while reading the file: {e}")
        return None

    # ANALYSIS PARAMETERS 
    TARGET_COUNTRIES_GEO_NAMES = ["Germany", "France"] 
    START_YEAR = 2020
    END_YEAR = 2023
    HOUSEHOLD_CONSUMPTION_BAND = "Consumption from 2 500 kWh to 4 999 kWh - band DC" 
    TARGET_PRODUCT = "Electrical energy"
    EXPECTED_TAX_STATUS = "All taxes and levies included"
    EXPECTED_CURRENCY_VALUE = "Euro"

    print("\nAnalysis parameters:")
    print(f"  Target Countries: {', '.join(TARGET_COUNTRIES_GEO_NAMES)}") 
    print(f"  Year Range: {START_YEAR} - {END_YEAR}")
    print(f"  Target Product: {TARGET_PRODUCT}")
    print(f"  Household Consumption Band for Analysis: '{HOUSEHOLD_CONSUMPTION_BAND}'")
    print(f"  Expected Tax Status: '{EXPECTED_TAX_STATUS}'")
    print(f"  Expected Currency: '{EXPECTED_CURRENCY_VALUE}'")

    required_cols = ['geo', 'TIME_PERIOD', 'nrg_cons', 'product', 'unit', 'tax', 'currency', 'OBS_VALUE']
    missing_cols = [col for col in required_cols if col not in df_prices.columns]
    if missing_cols:
        print(f"\nError: Missing columns in the prices CSV file: {', '.join(missing_cols)}")
        return None

    df_filtered = df_prices.copy()

    # Filter by countries
    df_filtered = df_filtered[df_filtered['geo'].isin(TARGET_COUNTRIES_GEO_NAMES)] 

    # Filter by product
    df_filtered = df_filtered[df_filtered['product'] == TARGET_PRODUCT]
    
    # Filter by tax status
    df_filtered = df_filtered[df_filtered['tax'] == EXPECTED_TAX_STATUS]

    # Filter by currency
    df_filtered = df_filtered[df_filtered['currency'] == EXPECTED_CURRENCY_VALUE]
    

    # Filter by the selected household consumption band
    df_filtered = df_filtered[df_filtered['nrg_cons'] == HOUSEHOLD_CONSUMPTION_BAND]

    df_filtered.loc[:, 'YEAR'] = df_filtered['TIME_PERIOD'].astype(str).str.split('-').str[0]
    df_filtered.loc[:, 'YEAR'] = pd.to_numeric(df_filtered['YEAR'], errors='coerce')
    df_filtered.loc[:, 'PRICE_VALUE'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    df_filtered.dropna(subset=['YEAR', 'PRICE_VALUE'], inplace=True)
    df_filtered.loc[:, 'YEAR'] = df_filtered['YEAR'].astype(int)

    df_filtered = df_filtered[
        (df_filtered['YEAR'] >= START_YEAR) &
        (df_filtered['YEAR'] <= END_YEAR)
    ]
    
    annual_avg_prices = df_filtered.groupby(
        ['YEAR', 'geo', 'nrg_cons', 'product', 'unit', 'tax', 'currency']
    )['PRICE_VALUE'].mean().reset_index()

    
    
    display_unit = annual_avg_prices['unit'].iloc[0] if not annual_avg_prices.empty else 'N/A'

    if not annual_avg_prices.empty:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.float_format', '{:.4f}'.format)

        try:
            # Pivot to show years as index, countries as columns, and price as values
            pivot_annual_prices = annual_avg_prices.pivot_table(
                index='YEAR',
                columns='geo',    
                values='PRICE_VALUE'
            )
            pivot_annual_prices.rename_axis(f"Avg Price ({EXPECTED_CURRENCY_VALUE}/{display_unit})", axis='columns', inplace=True)

            print(pivot_annual_prices)
            return pivot_annual_prices
        except Exception as e:
            print(f"Error creating pivot table for prices: {e}")
            return annual_avg_prices


if __name__ == "__main__":
    print("Starting Household Electricity Price Analysis...")
    result_prices_table = analyze_household_electricity_prices()


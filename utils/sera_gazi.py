import pandas as pd
import os

def analyze_carbon_dioxide_emissions():
    """
    Analyzes Carbon Dioxide emissions for specified countries (Germany, France) 
    in a specific sector for the specified years and presents it as a pivot table.
    TARGET_GAS (Carbon dioxide) will be analyzed.
    """
    df_emissions = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'estat_env_ac_ainah_r2$defaultview_filtered_en.csv'))


    #ANALYSIS PARAMETERS 
    TARGET_GAS = "Carbon dioxide" 
    TARGET_COUNTRY_GEO_NAMES = ["Germany", "France"] 
    TARGET_NACE_SECTOR = "Electricity, gas, steam and air conditioning supply"
    START_YEAR = 2020
    END_YEAR = 2023
    required_cols = ['geo', 'TIME_PERIOD', 'nace_r2', 'airpol', 'OBS_VALUE', 'unit']
    missing_cols = [col for col in required_cols if col not in df_emissions.columns]
    if missing_cols:
        print(f"\nError: Missing columns in the emissions CSV file: {', '.join(missing_cols)}")
        return None

    df_filtered = df_emissions.copy()
    df_filtered = df_filtered[df_filtered['airpol'] == TARGET_GAS]

    # Filter by countries
    df_filtered = df_filtered[df_filtered['geo'].isin(TARGET_COUNTRY_GEO_NAMES)]

    # Filter by NACE sector
    df_filtered = df_filtered[df_filtered['nace_r2'] == TARGET_NACE_SECTOR]

    # Convert TIME_PERIOD to YEAR and OBS_VALUE to numeric EMISSIONS_VALUE
    df_filtered.loc[:, 'YEAR'] = pd.to_numeric(df_filtered['TIME_PERIOD'], errors='coerce')
    df_filtered.loc[:, 'EMISSIONS_VALUE'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    df_filtered.dropna(subset=['YEAR', 'EMISSIONS_VALUE'], inplace=True)
    df_filtered.loc[:, 'YEAR'] = df_filtered['YEAR'].astype(int)
    
    df_filtered = df_filtered[
        (df_filtered['YEAR'] >= START_YEAR) &
        (df_filtered['YEAR'] <= END_YEAR)
    ]


    # Aggregate emissions. Since we've filtered for a single TARGET_GAS,
    aggregated_emissions = df_filtered.groupby(
        ['geo', 'YEAR', 'unit', 'nace_r2'] 
    )['EMISSIONS_VALUE'].sum().reset_index() 


    try:
        # Create pivot table: Countries and Sector as index, Years as columns..
        pivot_table = aggregated_emissions.pivot_table(
            index=['geo', 'nace_r2', 'unit'], 
            columns='YEAR',
            values='EMISSIONS_VALUE'
        )
    except Exception as e:
        print(f"\nError creating pivot table: {e}")
        return None

    if not pivot_table.empty:
        pivot_table.columns = pivot_table.columns.astype(int) 
        pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1) 

    #DISPLAY RESULTS 
    print(f"\n--- {TARGET_GAS.upper()} EMISSIONS FOR '{TARGET_NACE_SECTOR}' ({START_YEAR}-{END_YEAR}) ---")
    if not pivot_table.empty:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200) 
        pd.set_option('display.expand_frame_repr', False) 
        pd.set_option('display.float_format', '{:,.2f}'.format) 
        
        print(pivot_table)
        
    
    return pivot_table

if __name__ == "__main__":
    print("Starting Greenhouse Gas Emission Analysis...")
    result_pivot_table = analyze_carbon_dioxide_emissions()
    
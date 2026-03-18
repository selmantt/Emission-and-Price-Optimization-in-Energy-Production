import pandas as pd
import os

#Relative Imports

from utils.sera_gazi import analyze_carbon_dioxide_emissions 
from utils.yillik_fiyat import analyze_household_electricity_prices
from utils.enerji_uretim import analyze_energy_production
    

def combine_all_data(country_codes=["DE", "FR"], start_year=2020, end_year=2023, remove_zero_production_sources=True):
    """
    Combines data from emissions, electricity prices, and energy production analyses
    for specified countries and year range into a single DataFrame.

    Args:
        country_codes (list): A list of geo codes for the countries (e.g., ["DE", "FR"]).
        start_year (int): The starting year for analysis.
        end_year (int): The ending year for analysis.
        remove_zero_production_sources (bool): If True, energy sources with zero production
                                               across all analyzed years for a country will be removed.

    Returns:
        pandas.DataFrame: A combined DataFrame ready for optimization.
    """
    print(f"\nStarting combined data generation for countries {', '.join(country_codes)}, years {start_year}-{end_year}...")

    all_countries_data = [] 

    country_code_to_name_map = {
        "DE": "Germany",
        "FR": "France"
    }

   
    df_emissions_pivot_multi_country = analyze_carbon_dioxide_emissions()
   
    df_prices_pivot_multi_country = analyze_household_electricity_prices()

    for country_code in country_codes:
        country_name = country_code_to_name_map.get(country_code.upper(), country_code.upper())

        # Initialize DataFrame for the current country for this iteration
        years_index_current_country = pd.Index(range(start_year, end_year + 1), name='YEAR')
        current_country_combined_df = pd.DataFrame(index=years_index_current_country)


        #Process Emissions Data for the current country
        df_emissions_processed_cc = pd.DataFrame(index=years_index_current_country) # Default empty
        if df_emissions_pivot_multi_country is not None and not df_emissions_pivot_multi_country.empty:
            TARGET_NACE_SECTOR_CONST = "Electricity, gas, steam and air conditioning supply" 
            
            if (country_name in df_emissions_pivot_multi_country.index.get_level_values('geo') and
                TARGET_NACE_SECTOR_CONST in df_emissions_pivot_multi_country.index.get_level_values('nace_r2')):
                df_country_sector_emissions = df_emissions_pivot_multi_country.loc[(country_name, TARGET_NACE_SECTOR_CONST)]
                if not df_country_sector_emissions.empty:
                    df_emissions_transposed = df_country_sector_emissions.iloc[[0]].T 
                    df_emissions_transposed.index.name = 'YEAR'
                    unit_col_name = df_emissions_transposed.columns[0]
                    emissions_col_final_name = f"CO2_Emissions_{TARGET_NACE_SECTOR_CONST.split(',')[0].replace(' ','_')}_{str(unit_col_name).replace(' ','_')}"
                    df_emissions_temp_cc = df_emissions_transposed.rename(columns={unit_col_name: emissions_col_final_name})
                    df_emissions_temp_cc = df_emissions_temp_cc[
                        (df_emissions_temp_cc.index >= start_year) & (df_emissions_temp_cc.index <= end_year)]
                    if not df_emissions_temp_cc.empty:
                        df_emissions_processed_cc = df_emissions_temp_cc
                        print(f"  GHG emissions data processed for {country_name}.")
             


        #Process Electricity Price Data for the current country
        df_prices_processed_cc = pd.DataFrame(index=years_index_current_country) 
        if df_prices_pivot_multi_country is not None and not df_prices_pivot_multi_country.empty:
            if country_name in df_prices_pivot_multi_country.columns:
                df_prices_temp_cc = df_prices_pivot_multi_country[[country_name]].copy()
                price_col_final_name = 'AvgHouseholdPrice_Euro_per_kWh'
                df_prices_temp_cc.columns = [price_col_final_name]
                df_prices_temp_cc = df_prices_temp_cc[
                    (df_prices_temp_cc.index >= start_year) & (df_prices_temp_cc.index <= end_year)]
                if not df_prices_temp_cc.empty:
                    df_prices_processed_cc = df_prices_temp_cc
                    print(f"  Electricity price data processed for {country_name}.")
          

        #Get Energy Production Data for the current_country
        summary_production_df_multi_idx, production_shares_pivot_multi_idx = analyze_energy_production(
            target_country_codes=[country_code], 
            start_year=start_year,
            end_year=end_year
        )
        
        df_production_summary_processed_cc = pd.DataFrame(index=years_index_current_country)
        if summary_production_df_multi_idx is not None and not summary_production_df_multi_idx.empty:
            if country_code in summary_production_df_multi_idx.index.get_level_values('geo'):
                df_prod_sum_temp_cc = summary_production_df_multi_idx.xs(country_code, level='geo')
                df_prod_sum_temp_cc.rename(columns={
                    'CalculatedTotalProduction_GWH': 'TotalProduction_GWh',
                    'TotalRenewableProduction_GWH': 'RenewableProduction_GWh',
                    'TotalRenewableShare (%)': 'RenewableShare_Percent'
                }, inplace=True)
                if not df_prod_sum_temp_cc.empty:
                     df_production_summary_processed_cc = df_prod_sum_temp_cc
                     print(f"  Energy production summary data processed for {country_code}.")

        df_production_shares_processed_cc = pd.DataFrame(index=years_index_current_country)
        if production_shares_pivot_multi_idx is not None and not production_shares_pivot_multi_idx.empty:
            if country_code in production_shares_pivot_multi_idx.index.get_level_values('geo'):
                df_prod_shares_temp_cc = production_shares_pivot_multi_idx.xs(country_code, level='geo')
                if remove_zero_production_sources and not df_prod_shares_temp_cc.empty:
                    non_zero_sources = df_prod_shares_temp_cc.sum(axis=0) > 1e-6
                    df_prod_shares_temp_cc = df_prod_shares_temp_cc.loc[:, non_zero_sources]
                    print(f"    Removed energy sources with zero production for {country_code}.")
                if not df_prod_shares_temp_cc.empty:
                    df_prod_shares_temp_cc.columns = [
                        f"{str(col).replace(' (%)', '').replace(' ', '_').replace('-', '_').replace('.', '')}_Share_Percent" 
                        for col in df_prod_shares_temp_cc.columns
                    ]
                    df_production_shares_processed_cc = df_prod_shares_temp_cc
                    print(f"  Energy production shares data processed for {country_code}.")
            

        # Merge data for the current country
        if not df_emissions_processed_cc.empty:
            current_country_combined_df = current_country_combined_df.merge(df_emissions_processed_cc, on='YEAR', how='left')
        if not df_prices_processed_cc.empty:
            current_country_combined_df = current_country_combined_df.merge(df_prices_processed_cc, on='YEAR', how='left')
        if not df_production_summary_processed_cc.empty:
            current_country_combined_df = current_country_combined_df.merge(df_production_summary_processed_cc, on='YEAR', how='left')
        if not df_production_shares_processed_cc.empty:
            current_country_combined_df = current_country_combined_df.merge(df_production_shares_processed_cc, on='YEAR', how='left')
        
        if current_country_combined_df.shape[1] > 0: 
            current_country_combined_df['country_code'] = country_code 
            current_country_combined_df.reset_index(inplace=True) 
            all_countries_data.append(current_country_combined_df)
            print(f"  Finished processing for {country_name}. Columns: {current_country_combined_df.shape[1]-1}") 
        
    
    # Concatenate all country DataFrames

    final_combined_df = pd.concat(all_countries_data, ignore_index=True)
    # Set YEAR and country_code as multi-index if preferred, or keep them as columns
    final_combined_df.set_index(['country_code', 'YEAR'], inplace=True)
    
    print("\nOverall data combination complete.")
    return final_combined_df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.2f}".format)

    final_combined_data_all = combine_all_data(
        country_codes=["DE", "FR"], start_year=2020, end_year=2022, remove_zero_production_sources=True
    )

    if final_combined_data_all is not None and not final_combined_data_all.empty:
        print("\n--- FINAL COMBINED DATAFRAME (DE & FR) ---")
        print(final_combined_data_all)

        script_dir_main = os.path.dirname(os.path.abspath(__file__))
        project_root_main = os.path.dirname(script_dir_main)
        output_data_folder_main = os.path.join(project_root_main, "output_data")
        os.makedirs(output_data_folder_main, exist_ok=True)
        output_file_path_main = os.path.join(output_data_folder_main, "combined_data_DE_FR_2020_2022.csv")
      
        final_combined_data_all.to_csv(output_file_path_main)
       

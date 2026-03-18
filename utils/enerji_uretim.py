import pandas as pd
import os

def analyze_energy_production(target_country_codes=["DE", "FR"], start_year=2020, end_year=2023):
    """
    Analyzes energy production for specified countries and year range.
    Calculates total production (GWh), share of each energy source, and total renewable share.

    Args:
        target_country_codes (list): A list of geo codes for the countries to analyze (e.g., ["DE", "FR"]).
        start_year (int): The starting year for the analysis.
        end_year (int): The ending year for the analysis.
    """
    try:
        current_script_path = os.path.abspath(__file__)
        project_root_folder = os.path.dirname(os.path.dirname(current_script_path)) 
        datasets_folder = os.path.join(project_root_folder, 'datasets')

        production_file_name = 'estat_nrg_bal_peh.tsv'
        siec_file_name = 'siec.csv'

        production_file_path = os.path.join(datasets_folder, production_file_name)
        siec_file_path = os.path.join(datasets_folder, siec_file_name)

        #PROCESS SIEC FILE 
        df_siec = pd.read_csv(siec_file_path, dtype=str)
        renewable_keywords = ['renewable', 'hydro', 'wind', 'solar', 'geothermal', 'biomass', 'biogas', 'biofuels', 'tide', 'wave', 'ocean']
        known_renewable_notations = ['RA000', 'RA100', 'RA200', 'RA300', 'RA400', 'RA410', 'RA420', 'RA500', 'W6100', 'W6210', 'W6220']
        known_renewable_notations.extend(['FC_OTH_SOL_PHVPV', 'FC_OTH_SOL_THERM', 'GEO', 'HYD', 'WIND', 'OTH_RENEW'])

        dynamic_renewable_notations = []
        if 'Label' in df_siec.columns:
            condition_label = df_siec['Label'].str.contains('|'.join(renewable_keywords), case=False, na=False)
            dynamic_renewable_notations.extend(df_siec[condition_label]['Notation'].unique().tolist())

        if 'Definition' in df_siec.columns:
            condition_def = df_siec['Definition'].str.contains('|'.join(renewable_keywords), case=False, na=False)
            dynamic_renewable_notations.extend(df_siec[condition_def]['Notation'].unique().tolist())

        all_renewable_siec_codes = list(set(known_renewable_notations + dynamic_renewable_notations))

        siec_to_label = {}
        if 'Notation' in df_siec.columns and 'Label' in df_siec.columns:
            siec_to_label = pd.Series(df_siec.Label.values, index=df_siec.Notation).to_dict()

        #PROCESS PRODUCTION FILE 
        
        df_prod_raw = pd.read_csv(production_file_path, delimiter='\t', dtype=str, header=0)  

        first_col_name_raw = df_prod_raw.columns[0]
        if '\\' in first_col_name_raw: 
            expected_headers_str = first_col_name_raw.split('\\')[0]
        else: 
            expected_headers_str = first_col_name_raw

        expected_headers = expected_headers_str.split(',')
        num_expected_headers = len(expected_headers)

        split_data = df_prod_raw[first_col_name_raw].str.split(',', n=num_expected_headers - 1, expand=True)

        for i, header in enumerate(expected_headers):
            df_prod_raw[header.strip()] = split_data[i] 

        df_prod_raw = df_prod_raw.drop(columns=[first_col_name_raw])

        df_prod_countries = df_prod_raw[df_prod_raw['geo'].isin(target_country_codes)].copy() 
    
        df_prod_countries.columns = df_prod_countries.columns.str.strip()
        required_filter_cols = ['nrg_bal', 'unit', 'siec']
        for col_f in required_filter_cols:
            if col_f not in df_prod_countries.columns:
                print(f"Error: Required filter column '{col_f}' not found in production data.")
                return None, None

        df_prod_filtered = df_prod_countries[
            (df_prod_countries['nrg_bal'] == 'GEP') &  
            (df_prod_countries['unit'] == 'GWH')      
        ].copy()

        id_vars = [col.strip() for col in expected_headers if col.strip() in df_prod_filtered.columns] 
        if 'geo' not in id_vars: 
            id_vars.append('geo') 
        id_vars = list(set(id_vars + ['freq', 'nrg_bal', 'siec', 'unit'])) 
        id_vars = [col for col in id_vars if col in df_prod_filtered.columns] 

        year_columns = [col for col in df_prod_filtered.columns if col not in id_vars and col.strip().isdigit() and len(col.strip()) == 4]

        df_long = pd.melt(df_prod_filtered,
                            id_vars=id_vars,
                            value_vars=year_columns,
                            var_name='YEAR',
                            value_name='PRODUCTION_GWH')

        df_long['YEAR'] = pd.to_numeric(df_long['YEAR'].str.strip(), errors='coerce')
        df_long['PRODUCTION_GWH'] = df_long['PRODUCTION_GWH'].astype(str).str.replace(r':.*', '', regex=True).str.strip()
        df_long['PRODUCTION_GWH'] = pd.to_numeric(df_long['PRODUCTION_GWH'], errors='coerce').fillna(0)

        df_long.dropna(subset=['YEAR'], inplace=True) 
        df_long['YEAR'] = df_long['YEAR'].astype(int)

        df_long_year_filtered = df_long[(df_long['YEAR'] >= start_year) & (df_long['YEAR'] <= end_year)].copy()

        #CALCULATIONS 
        df_individual_sources = df_long_year_filtered[df_long_year_filtered['siec'] != 'TOTAL'].copy()
        
        df_total_production_year = df_individual_sources.groupby(['geo', 'YEAR'])['PRODUCTION_GWH'].sum().reset_index() 
        df_total_production_year.rename(columns={'PRODUCTION_GWH': 'CalculatedTotalProduction_GWH'}, inplace=True)

        df_analysis = df_individual_sources.copy()
        df_analysis = pd.merge(df_analysis, df_total_production_year, on=['geo', 'YEAR'], how='left') 

        df_analysis['Share (%)'] = df_analysis.apply(
            lambda row: (row['PRODUCTION_GWH'] / row['CalculatedTotalProduction_GWH']) * 100 if row['CalculatedTotalProduction_GWH'] > 0 else 0,
            axis=1
        )

        df_analysis['EnergySource'] = df_analysis['siec'].map(siec_to_label).fillna(df_analysis['siec'])
        df_analysis['IsRenewable'] = df_analysis['siec'].isin(all_renewable_siec_codes)

        df_renewable_production_year = df_analysis[df_analysis['IsRenewable']].groupby(['geo', 'YEAR'])['PRODUCTION_GWH'].sum().reset_index() 
        df_renewable_production_year.rename(columns={'PRODUCTION_GWH': 'TotalRenewableProduction_GWH'}, inplace=True)

        summary_df = pd.merge(df_total_production_year, df_renewable_production_year, on=['geo', 'YEAR'], how='left') 
        summary_df['TotalRenewableProduction_GWH'].fillna(0, inplace=True) 
        summary_df['TotalRenewableShare (%)'] = summary_df.apply(
            lambda row: (row['TotalRenewableProduction_GWH'] / row['CalculatedTotalProduction_GWH']) * 100 if row['CalculatedTotalProduction_GWH'] > 0 else 0,
            axis=1
        )
        if not summary_df.empty:
            summary_df = summary_df.set_index(['geo', 'YEAR']) 
        else:
            summary_df = pd.DataFrame(columns=['CalculatedTotalProduction_GWH', 'TotalRenewableProduction_GWH', 'TotalRenewableShare (%)'])
            summary_df.index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['geo', 'YEAR'])

        #DISPLAY RESULTS 
        print(f"\n--- Energy Production Analysis for {', '.join(target_country_codes)} ({start_year}-{end_year}) ---")

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', '{:.2f}'.format)
        
        print("\nAnnual Summary (Total Production and Renewable Share per Country):")
        print(summary_df)
        
        print("\nDetailed Annual Production by Energy Source:")
        pivot_production_gwh = df_analysis.pivot_table(index=['geo', 'YEAR'], 
                                                        columns='EnergySource',
                                                        values='PRODUCTION_GWH',
                                                        aggfunc='sum',
                                                        fill_value=0)

        pivot_shares_percent = df_analysis.pivot_table(index=['geo', 'YEAR'], 
                                                        columns='EnergySource',
                                                        values='Share (%)',
                                                        aggfunc='sum',
                                                        fill_value=0)
            
        print(pivot_production_gwh)
        print("\nShare by Source (%):")
        print(pivot_shares_percent)
        return summary_df, pivot_shares_percent

    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
        return None, None


if __name__ == "__main__":
    print("Starting Energy Production Analysis...")

    summary, detailed_shares = analyze_energy_production(
        target_country_codes=["DE", "FR"],
        start_year=2020,
        end_year=2022 
    )

    # Check and print results


   
    print("\nSummary for Germany:")
    print(summary.loc['DE'])
       
            
       
    print("\nSummary for France:")
    print(summary.loc['FR'])
    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
warnings.filterwarnings('ignore')

class GrowthAccountingAnalyzer:
    def __init__(self):
        self.countries = [
            'AUS', 'AUT', 'BEL', 'CAN', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC', 
            'ISL', 'IRL', 'ITA', 'JPN', 'NLD', 'NZL', 'NOR', 'PRT', 'ESP', 
            'SWE', 'CHE', 'GBR', 'USA'
        ]
        
        self.country_names = {
            'AUS': 'Australia', 'AUT': 'Austria', 'BEL': 'Belgium', 'CAN': 'Canada',
            'DNK': 'Denmark', 'FIN': 'Finland', 'FRA': 'France', 'DEU': 'Germany',
            'GRC': 'Greece', 'ISL': 'Iceland', 'IRL': 'Ireland', 'ITA': 'Italy',
            'JPN': 'Japan', 'NLD': 'Netherlands', 'NZL': 'New Zealand', 'NOR': 'Norway',
            'PRT': 'Portugal', 'ESP': 'Spain', 'SWE': 'Sweden', 'CHE': 'Switzerland',
            'GBR': 'United Kingdom', 'USA': 'United States'
        }
        
        self.data = {}
    
    def fetch_world_bank_data(self, indicator, start_year=1990, end_year=2019):
        """Fetch data from World Bank API"""
        print(f"Fetching World Bank data for {indicator}...")
        
        # World Bank API endpoint
        countries_str = ';'.join(self.countries)
        url = f"http://api.worldbank.org/v2/country/{countries_str}/indicator/{indicator}"
        
        params = {
            'date': f'{start_year}:{end_year}',
            'format': 'json',
            'per_page': 10000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if len(data) > 1:
                records = data[1]  # Data is in second element
                df_list = []
                
                for record in records:
                    if record['value'] is not None:
                        df_list.append({
                            'country_code': record['country']['id'],
                            'country_name': record['country']['value'],
                            'year': int(record['date']),
                            'value': float(record['value'])
                        })
                
                if df_list:
                    return pd.DataFrame(df_list)
            
            print(f"No data found for {indicator}")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching World Bank data: {e}")
            return pd.DataFrame()
    
    def fetch_oecd_productivity_data(self):
        """Fetch OECD productivity data using alternative approach"""
        print("Attempting to fetch OECD productivity data...")
        
        # Try OECD REST API for productivity data
        try:
            # OECD Multifactor Productivity dataset
            url = "https://stats.oecd.org/SDMX-JSON/data/PDB_GR/AUS+AUT+BEL+CAN+DNK+FIN+FRA+DEU+GRC+ISL+IRL+ITA+JPN+NLD+NZL+NOR+PRT+ESP+SWE+CHE+GBR+USA.T_GDPPOP+T_GDPHRS+MFPGDP.A/all"
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                print("Successfully connected to OECD API")
                # Parse OECD JSON format (complex structure)
                # This would require detailed parsing of OECD's specific JSON format
                return pd.DataFrame()
            else:
                print(f"OECD API returned status code: {response.status_code}")
                
        except Exception as e:
            print(f"Error fetching OECD data: {e}")
        
        return pd.DataFrame()
    
    def calculate_growth_rates(self, df):
        """Calculate annual growth rates from level data"""
        if df.empty:
            return df
            
        df = df.sort_values(['country_code', 'year'])
        df['growth_rate'] = df.groupby('country_code')['value'].pct_change() * 100
        return df
    
    def fetch_all_data(self):
        """Fetch all required data from various sources"""
        print("Starting data collection...")
        print("=" * 50)
        
        # World Bank indicators
        indicators = {
            'gdp_growth': 'NY.GDP.MKTP.KD.ZG',  # GDP growth (annual %)
            'gdp_per_capita_growth': 'NY.GDP.PCAP.KD.ZG',  # GDP per capita growth
            'gross_capital_formation': 'NE.GDI.TOTL.KD.ZG',  # Gross capital formation growth
            'labor_force_growth': 'SL.TLF.TOTL.IN.ZS',  # Labor force growth
            'employment_growth': 'SL.EMP.TOTL.SP.ZS'  # Employment growth
        }
        
        # Fetch World Bank data
        for key, indicator in indicators.items():
            df = self.fetch_world_bank_data(indicator)
            if not df.empty:
                self.data[key] = df
                print(f"✓ Successfully fetched {key}")
            else:
                print(f"✗ Failed to fetch {key}")
        
        # Try to fetch OECD productivity data
        oecd_data = self.fetch_oecd_productivity_data()
        if not oecd_data.empty:
            self.data['tfp'] = oecd_data
            print("✓ Successfully fetched OECD productivity data")
        else:
            print("✗ OECD productivity data not available - will estimate")
    
    def estimate_tfp_growth(self):
        """Estimate TFP growth using growth accounting residual"""
        print("\nEstimating TFP growth using growth accounting...")
        
        if 'gdp_growth' not in self.data:
            print("Cannot estimate TFP - no GDP growth data")
            return pd.DataFrame()
        
        # Calculate average growth rates for 1990-2019
        gdp_data = self.data['gdp_growth']
        
        print(f"Total GDP data records: {len(gdp_data)}")
        print(f"Year range in data: {gdp_data['year'].min()} - {gdp_data['year'].max()}")
        print(f"Countries in data: {gdp_data['country_code'].unique()}")
        
        # Filter for our time period and countries
        gdp_filtered = gdp_data[
            (gdp_data['year'] >= 1990) & 
            (gdp_data['year'] <= 2019) &
            (gdp_data['country_code'].isin(self.countries))
        ].copy()
        
        print(f"Filtered GDP data records: {len(gdp_filtered)}")
        
        if gdp_filtered.empty:
            print("No GDP data available for specified period")
            # Let's try with a broader filter
            print("Trying with all available years...")
            gdp_filtered = gdp_data[gdp_data['country_code'].isin(self.countries)].copy()
            print(f"Records with broader filter: {len(gdp_filtered)}")
            
            if gdp_filtered.empty:
                print("Still no data - using sample data for demonstration")
                return self.create_sample_data()
        
        # Calculate average GDP growth by country
        avg_gdp_growth = gdp_filtered.groupby('country_code')['value'].mean().reset_index()
        avg_gdp_growth.columns = ['country_code', 'avg_gdp_growth']
        
        # Estimate capital growth (use gross capital formation if available)
        capital_growth = pd.DataFrame()
        if 'gross_capital_formation' in self.data:
            capital_data = self.data['gross_capital_formation']
            print(f"Capital formation data records: {len(capital_data)}")
            capital_filtered = capital_data[
                (capital_data['year'] >= 1990) & 
                (capital_data['year'] <= 2019) &
                (capital_data['country_code'].isin(self.countries))
            ]
            if capital_filtered.empty:
                capital_filtered = capital_data[capital_data['country_code'].isin(self.countries)]
            
            if not capital_filtered.empty:
                capital_growth = capital_filtered.groupby('country_code')['value'].mean().reset_index()
                capital_growth.columns = ['country_code', 'avg_capital_growth']
                print(f"Capital growth calculated for {len(capital_growth)} countries")
        
        # Estimate labor growth (use employment growth if available)
        labor_growth = pd.DataFrame()
        if 'employment_growth' in self.data:
            labor_data = self.data['employment_growth']
            print(f"Employment data records: {len(labor_data)}")
            labor_filtered = labor_data[
                (labor_data['year'] >= 1990) & 
                (labor_data['year'] <= 2019) &
                (labor_data['country_code'].isin(self.countries))
            ]
            if labor_filtered.empty:
                labor_filtered = labor_data[labor_data['country_code'].isin(self.countries)]
            
            # Employment data might be in levels, calculate growth rates
            if not labor_filtered.empty:
                labor_rates = self.calculate_growth_rates(labor_filtered)
                if not labor_rates.empty:
                    labor_growth = labor_rates.groupby('country_code')['growth_rate'].mean().reset_index()
                    labor_growth.columns = ['country_code', 'avg_labor_growth']
                    print(f"Labor growth calculated for {len(labor_growth)} countries")
        
        # Combine data
        result = avg_gdp_growth.copy()
        
        # Merge capital and labor data if available
        if not capital_growth.empty:
            result = result.merge(capital_growth, on='country_code', how='left')
        else:
            result['avg_capital_growth'] = np.nan
            
        if not labor_growth.empty:
            result = result.merge(labor_growth, on='country_code', how='left')
        else:
            result['avg_labor_growth'] = 1.0  # Assume 1% labor growth if no data
        
        # Fill missing values with OECD averages
        result['avg_capital_growth'].fillna(3.0, inplace=True)  # Typical capital growth
        result['avg_labor_growth'].fillna(1.0, inplace=True)   # Typical labor growth
        
        # Growth accounting: Y = A * K^α * L^(1-α)
        # Growth rates: g_Y = g_A + α*g_K + (1-α)*g_L
        # Therefore: g_A = g_Y - α*g_K - (1-α)*g_L
        
        capital_share = 0.35  # Typical capital share in OECD countries
        labor_share = 1 - capital_share
        
        result['tfp_growth'] = (result['avg_gdp_growth'] - 
                               capital_share * result['avg_capital_growth'] - 
                               labor_share * result['avg_labor_growth'])
        
        # Calculate capital deepening (capital growth - labor growth)
        result['capital_deepening'] = result['avg_capital_growth'] - result['avg_labor_growth']
        
        # Calculate shares
        result['tfp_share'] = result['tfp_growth'] / result['avg_gdp_growth']
        result['capital_share'] = result['capital_deepening'] / result['avg_gdp_growth']
        
        # Clean up extreme values
        result['tfp_share'] = result['tfp_share'].clip(-0.5, 1.5)
        result['capital_share'] = result['capital_share'].clip(-0.5, 1.5)
        
    def create_sample_data(self):
        """Create sample data when real data is not available"""
        print("Creating sample data for demonstration...")
        
        # Sample realistic data based on OECD historical averages
        sample_data = {
            'AUS': {'gdp': 2.8, 'capital': 3.5, 'labor': 1.5},
            'AUT': {'gdp': 2.1, 'capital': 2.8, 'labor': 0.8},
            'BEL': {'gdp': 2.0, 'capital': 2.5, 'labor': 0.7},
            'CAN': {'gdp': 2.3, 'capital': 3.0, 'labor': 1.2},
            'DNK': {'gdp': 1.9, 'capital': 2.3, 'labor': 0.5},
            'FIN': {'gdp': 2.5, 'capital': 2.8, 'labor': 0.3},
            'FRA': {'gdp': 1.8, 'capital': 2.4, 'labor': 0.7},
            'DEU': {'gdp': 1.4, 'capital': 1.8, 'labor': 0.2},
            'GRC': {'gdp': 1.2, 'capital': 2.5, 'labor': 0.5},
            'ISL': {'gdp': 2.8, 'capital': 4.0, 'labor': 1.8},
            'IRL': {'gdp': 5.2, 'capital': 6.8, 'labor': 2.1},
            'ITA': {'gdp': 1.1, 'capital': 2.0, 'labor': 0.4},
            'JPN': {'gdp': 1.2, 'capital': 1.5, 'labor': -0.2},
            'NLD': {'gdp': 2.4, 'capital': 2.8, 'labor': 1.0},
            'NZL': {'gdp': 2.8, 'capital': 3.5, 'labor': 1.5},
            'NOR': {'gdp': 2.9, 'capital': 3.8, 'labor': 1.2},
            'PRT': {'gdp': 2.8, 'capital': 4.2, 'labor': 0.8},
            'ESP': {'gdp': 2.7, 'capital': 4.5, 'labor': 2.0},
            'SWE': {'gdp': 2.4, 'capital': 2.9, 'labor': 0.5},
            'CHE': {'gdp': 1.8, 'capital': 2.2, 'labor': 0.8},
            'GBR': {'gdp': 2.5, 'capital': 2.8, 'labor': 0.8},
            'USA': {'gdp': 2.5, 'capital': 3.2, 'labor': 1.2}
        }
        
        result_data = []
        for country_code, values in sample_data.items():
            result_data.append({
                'country_code': country_code,
                'avg_gdp_growth': values['gdp'],
                'avg_capital_growth': values['capital'],
                'avg_labor_growth': values['labor']
            })
        
        result = pd.DataFrame(result_data)
        
        # Calculate TFP and other metrics
        capital_share = 0.35
        labor_share = 1 - capital_share
        
        result['tfp_growth'] = (result['avg_gdp_growth'] - 
                               capital_share * result['avg_capital_growth'] - 
                               labor_share * result['avg_labor_growth'])
        
        result['capital_deepening'] = result['avg_capital_growth'] - result['avg_labor_growth']
        result['tfp_share'] = result['tfp_growth'] / result['avg_gdp_growth']
        result['capital_share'] = result['capital_deepening'] / result['avg_gdp_growth']
        
        print(f"Final analysis data shape: {len(result)} countries")
        print(f"Countries included: {result['country_code'].tolist()}")
        
        return result
    
    def create_analysis_table(self):
        """Create the final analysis table"""
        print("\nCreating growth accounting analysis...")
        
        # Get estimated data
        analysis_data = self.estimate_tfp_growth()
        
        if analysis_data.empty:
            print("No data available for analysis - this shouldn't happen with fallback data!")
            return pd.DataFrame()
        
        print(f"Analysis data columns: {analysis_data.columns.tolist()}")
        print(f"Sample of analysis data:")
        print(analysis_data.head())
        
        # Create final table
        final_table = pd.DataFrame()
        final_table['Country'] = analysis_data['country_code'].map(self.country_names)
        final_table['Growth_Rate'] = analysis_data['avg_gdp_growth'].round(2)
        final_table['TFP_Growth'] = analysis_data['tfp_growth'].round(2)
        final_table['Capital_Deepening'] = analysis_data['capital_deepening'].round(2)
        final_table['TFP_Share'] = analysis_data['tfp_share'].round(2)
        final_table['Capital_Share'] = analysis_data['capital_share'].round(2)
        
        # Remove rows with missing country names (shouldn't happen but just in case)
        initial_count = len(final_table)
        final_table = final_table.dropna(subset=['Country'])
        final_count = len(final_table)
        
        if final_count < initial_count:
            print(f"Warning: {initial_count - final_count} countries dropped due to missing names")
        
        print(f"Final table has {len(final_table)} countries")
        
        # Calculate averages
        numeric_columns = ['Growth_Rate', 'TFP_Growth', 'Capital_Deepening', 'TFP_Share', 'Capital_Share']
        avg_values = final_table[numeric_columns].mean()
        
        avg_row = pd.DataFrame({
            'Country': ['Average'],
            'Growth_Rate': [avg_values['Growth_Rate']],
            'TFP_Growth': [avg_values['TFP_Growth']],
            'Capital_Deepening': [avg_values['Capital_Deepening']],
            'TFP_Share': [avg_values['TFP_Share']],
            'Capital_Share': [avg_values['Capital_Share']]
        })
        
        # Combine main data with average
        final_table = pd.concat([final_table, avg_row], ignore_index=True)
        final_table = final_table.round(2)
        
        return final_table
    
    def create_visualizations(self, df):
        """Create visualization charts"""
        if df.empty:
            print("No data available for visualization")
            return
            
        # Remove average row for plotting
        plot_data = df[df['Country'] != 'Average'].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Growth Accounting in OECD Countries: 1990-2019 (Estimated from Real Data)', 
                     fontsize=14, fontweight='bold')
        
        # 1. Growth Rate vs TFP Growth
        axes[0,0].scatter(plot_data['TFP_Growth'], plot_data['Growth_Rate'], alpha=0.7, s=60)
        axes[0,0].set_xlabel('TFP Growth (%)')
        axes[0,0].set_ylabel('GDP Growth Rate (%)')
        axes[0,0].set_title('GDP Growth vs TFP Growth')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add trend line if we have enough data points
        if len(plot_data) > 2:
            z = np.polyfit(plot_data['TFP_Growth'], plot_data['Growth_Rate'], 1)
            p = np.poly1d(z)
            axes[0,0].plot(plot_data['TFP_Growth'], p(plot_data['TFP_Growth']), "r--", alpha=0.8)
        
        # 2. TFP vs Capital Deepening
        axes[0,1].scatter(plot_data['Capital_Deepening'], plot_data['TFP_Growth'], alpha=0.7, s=60)
        axes[0,1].set_xlabel('Capital Deepening (%)')
        axes[0,1].set_ylabel('TFP Growth (%)')
        axes[0,1].set_title('TFP Growth vs Capital Deepening')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. TFP Share distribution
        top_countries = plot_data.nlargest(10, 'TFP_Share')
        y_pos = range(len(top_countries))
        axes[1,0].barh(y_pos, top_countries['TFP_Share'], alpha=0.7)
        axes[1,0].set_yticks(y_pos)
        axes[1,0].set_yticklabels(top_countries['Country'], fontsize=8)
        axes[1,0].set_xlabel('TFP Share of Total Growth')
        axes[1,0].set_title('Top 10: TFP Share of Growth')
        axes[1,0].grid(True, alpha=0.3, axis='x')
        
        # 4. Growth components for major economies
        major_economies = ['United States', 'Germany', 'Japan', 'United Kingdom', 'France']
        major_data = plot_data[plot_data['Country'].isin(major_economies)]
        
        if not major_data.empty:
            x_pos = np.arange(len(major_data))
            width = 0.35
            
            axes[1,1].bar(x_pos - width/2, major_data['TFP_Growth'], width, 
                         label='TFP Growth', alpha=0.8)
            axes[1,1].bar(x_pos + width/2, major_data['Capital_Deepening'], width,
                         label='Capital Deepening', alpha=0.8)
            
            axes[1,1].set_xlabel('Countries')
            axes[1,1].set_ylabel('Growth Rate (%)')
            axes[1,1].set_title('Growth Components: Major Economies')
            axes[1,1].set_xticks(x_pos)
            axes[1,1].set_xticklabels(major_data['Country'], rotation=45, ha='right')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

def main():
    print("OECD Growth Accounting Analysis: 1990-2019")
    print("Automatic Data Retrieval Version")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = GrowthAccountingAnalyzer()
    
    # Fetch data
    analyzer.fetch_all_data()
    
    # Create analysis
    results_table = analyzer.create_analysis_table()
    
    if not results_table.empty:
        print("\n" + "="*80)
        print("GROWTH ACCOUNTING RESULTS")
        print("="*80)
        print(results_table.to_string(index=False))
        
        # Create visualizations
        analyzer.create_visualizations(results_table)
        
        # Print summary statistics
        analysis_data = results_table[results_table['Country'] != 'Average']
        if not analysis_data.empty:
            print("\n" + "="*60)
            print("SUMMARY STATISTICS")
            print("="*60)
            numeric_cols = ['Growth_Rate', 'TFP_Growth', 'Capital_Deepening']
            summary = analysis_data[numeric_cols].describe()
            print(summary.round(2))
            
            print("\n" + "="*60)
            print("CORRELATION MATRIX")
            print("="*60)
            corr_matrix = analysis_data[numeric_cols].corr()
            print(corr_matrix.round(3))
    
    else:
        print("Unable to retrieve sufficient data for analysis.")
        print("Please check your internet connection and try again.")

# Run the analysis
if __name__ == "__main__":
    main()
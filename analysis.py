# To be used in notebook for analysis

# Libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
import calendar
import math
import itertools 
import os 
import matplotlib.pyplot as plt 

from typing import Tuple

import warnings
warnings.filterwarnings('ignore')

########################################################################################
########################################################################################
# Function to import and consolidate data
def import_datasets(path: str = '/datasets/') -> pd.DataFrame:
    """
       Merge all asset datasets into one.
    """
    def imports(path):
        # Import all csv files and store them in a list
        dfs = [pd.read_csv(path + csv).set_index(['Date']) for csv in os.listdir(path)]
        # Get all asset names
        assets = [csv.split('_')[0].lower() for csv in os.listdir(path)]
        # Edit column names in each df to indicate which asset they belong to
        n = 0
        while n < len(assets):
            for df in dfs:
                df.columns = [col + '_' + assets[n] for col in df.columns]
                n += 1 
        
        print(f"Directory contains {len(assets)} assets: {', '.join(assets)}")

        return dfs
    
    def consolidate(path):
        # Concatenate all dfs stored in the list to produce the final dataframe
        return pd.concat(imports(path), axis=1).dropna().reset_index() # Remove dates not found in all datasets
    
    # Call
    final_df = consolidate(path)
    print(f"After combining, there are {final_df.shape[0]} rows and {final_df.shape[1]} columns.")
    return final_df

########################################################################################
########################################################################################
# Second function to calculate correlations
def correlate(
    df: pd.DataFrame,
    price: str = 'close',
    start_ymd: str = '1000-01-01',
    end_ymd: str = '3000-01-01',
    interval: Tuple[str, int] = ('m', 1)
    ):
    """
       Calculates pearson's correlation scores between all asset pairs within
       specified time frame and broken down by specified time interval
       
       Input:
          df: pd.DataFrame - dataframe from import_datasets()
          price: str - choose from open, high, low, close
          start_ymd: str - start date with '%Y-%m-%d' format
          end_ymd: str - end date with '%Y-%m-%d' format
          interval: Tuple[str, int] - break down correlations by specified intervals
             - str: year, month, day, hour - 'y', 'm', 'd', 'h', '-'
             - int: one year, two years, one month, two months etc - 1, 2, 3
    """
    def preprocess(df, price, start_ymd, end_ymd, interval):
        # Filter dates
        df = df[(df['Date'] >= start_ymd) & (df['Date'] <= end_ymd)]
        # Get all assets from df columns
        assets = list(set([i[-1] for i in df.columns.str.split('_') if len(i)==2]))
        # Get correct price columns based on price parameter
        price_cols = [price.capitalize() + '_' + i for i in assets]
        # Aggregate prices based on specified interval
        # If not specified, skip
        if interval[0] != '-':
            df['date_interval'] = pd.to_datetime(df['Date']).dt.to_period(interval[0])
            date_intervals= list(df['date_interval'].astype(str).unique())
            date_intervals.append('3000-01-01')
        else:
            date_intervals = None

        return df, assets, price_cols, date_intervals
        
    def consolidate_corr(df, price, start_ymd, end_ymd, interval):
        df, assets, price_cols, date_intervals = preprocess(df, price, start_ymd, end_ymd, interval)
        if date_intervals == None:
            # If we only want to statically calculate one correlation score for one time period
            corr = {}
            for first_asset, second_asset in itertools.combinations(price_cols, 2):
               key, value = calculate_corr(df, first_asset, second_asset)
               corr[key] = value
            
            results = pd.DataFrame(corr, index=['>= ' + start_ymd + ' <= ' + end_ymd])
            
            return results

        else:
            # If we want to dynamically track the correlations over the specified time period and interval
            dates = []
            corr = []
            pairs = []

            start = 0
            mid = interval[1]
            end = len(date_intervals) - 1

            while mid <= end:
                subdf = df[(df['Date'] >= date_intervals[start]) & (df['Date'] < date_intervals[mid])]
                if len(subdf) > 1:
                    for first_asset, second_asset in itertools.combinations(price_cols, 2):
                        key, value = calculate_corr(subdf, first_asset, second_asset)
                        dates.append('>= ' + date_intervals[start] + ' < ' + date_intervals[mid])
                        corr.append(value)
                        pairs.append(key)
                    start += interval[1]
                    mid += interval[1]
                else:
                    start += interval[1]
                    mid += interval[1]

            results = pd.DataFrame(list(zip(dates, pairs, corr)), columns=['interval','pairs','correlation'])
            results = results.set_index(['interval','pairs']).unstack().reset_index().droplevel(level=0, axis=1).rename(columns={'':'interval'})
            
            return results

    def calculate_corr(df, first_asset, second_asset):
        ratios_first_asset = np.log(df[first_asset]/df[first_asset].shift()).tolist()[1:]
        ratios_second_asset = np.log(df[second_asset]/df[second_asset].shift()).tolist()[1:]

        return first_asset + '-' + second_asset, round(pearsonr(ratios_first_asset, ratios_second_asset)[0], 5)

    def visualise(df, price, start_ymd, end_ymd, interval):
        final_df = consolidate_corr(df, price, start_ymd, end_ymd, interval)
        if interval[0] != '-':
            columns_to_plot = final_df.columns.tolist()[1:]

            plt.figure(figsize=(12,6))
            for cols in columns_to_plot:
                plt.plot(final_df['interval'], final_df[cols], label=cols)
            plt.title(f"Correlation scores over time by {interval[1]} {interval[0]}")
            plt.legend()
            plt.ylim(0,1)
            plt.ylabel('Correlation scores')
            plt.xlabel('Time')
            plt.xticks('')
            plt.show()

        return final_df
        
    return visualise(df, price, start_ymd, end_ymd, interval)

########################################################################################
########################################################################################
# Third function for regression analysis
def ols_regressor(
    df: pd.DataFrame,
    price: str = 'close',
    start_ymd: str = '1000-01-01',
    end_ymd: str = '3000-01-01',
    ):
    """
       Produces OLS Regression Summary Table
    """
    def preprocess(df, price, start_ymd, end_ymd):
        # Filter dates
        df = df[(df['Date'] >= start_ymd) & (df['Date'] <= end_ymd)]
        # Get all assets from df columns
        assets = list(set([i[-1] for i in df.columns.str.split('_') if len(i)==2]))
        # Provide options to select variables
        print(f"1) Choose ONE asset that you want as dependent variable from: {', ' .join(assets)}")
        dependent = input() # in string format
        print('\n')
        print(f"2) Choose ONE OR MORE asset that you want as independent variable(s) from: {', '.join(assets)}")
        independent = input() # in string format
        
        return df, dependent, [i.strip() for i in independent.split(',')]

    # return preprocess(df, price, start_ymd, end_ymd)

    def ols_summary_table(df, price, start_ymd, end_ymd):
        df, dependent_var, independent_var = preprocess(df, price, start_ymd, end_ymd)
        dependent_var_price_col = 'Close_' + dependent_var
        independent_var_price_col = ' + '.join([i+j for j in independent_var for i in ['Close_']])
        # Instantiate and fit OLS regression model
        model = smf.ols(f"{dependent_var_price_col} ~ {independent_var_price_col}", data=df)
        
        return model.fit().summary()
    
    print('\n')
    print(ols_summary_table(df, price, start_ymd, end_ymd))
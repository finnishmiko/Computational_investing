'''
Homework 5 for Coursera Computational investing course
wiki.quantsoftware.org

Modify simulate function to calculate Bollinger Bands
Save Band values to a file
'''

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def simulate(dt_start, dt_end, ls_symbols, lf_weights):
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo')

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    
    # Filling the data for NAN
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values
    #print "Close prices:\n", na_price
    
    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = na_price / na_price[0, :]

    # Copy the normalized prices to a new ndarry to find returns.
    na_rets = na_normalized_price.copy()
    #print "normalized returns:\n", na_rets

    # Calculate Bollinger value from 20 day rolling mean and std
    na_moving_avg = pd.rolling_mean(na_price, 20)
    #print na_moving_avg
    na_moving_std = pd.rolling_std(na_price, 20)
    #print na_moving_std
    bollinger_val = (na_price - na_moving_avg) / na_moving_std
    #print bollinger_val

    ### Print Bollinger values to a csv file ###
    filename = 'BollingerFile.csv'
    j=0
    import csv
    writer = csv.writer(open(filename, 'wb'), delimiter=',')
    for s_sym in ls_symbols:
        for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
            # Write findings to CSV-file
            row_to_enter = [ldt_timestamps[i].year, ldt_timestamps[i].month, ldt_timestamps[i].day, s_sym, bollinger_val[i,j]]
            writer.writerow(row_to_enter)
            #print ldt_timestamps[i], s_sym, bollinger_val[i,j]
        j+=1

    #return df_events

    
    # Calculate value of portfolio
    portfolio_daily_value = np.sum(na_rets * lf_weights, axis=1)
    #print "Portfolio daily value: ", portfolio_daily_value
    
    # Cumulative return is the last day's value
    cum_ret = portfolio_daily_value[-1]

    # Calculate the daily returns of the prices. (Inplace calculation)
    # returnize0 works on ndarray and not dataframes.
    tsu.returnize0(portfolio_daily_value)
    #print "daily returns \n", portfolio_daily_value

    # Calculate volatility as portfolio's standart deviation
    vol = np.std(portfolio_daily_value)
    daily_ret = np.mean(portfolio_daily_value, axis=0)

    # Calculate Sharpe ratio with assumption that risk free rate is 0
    # and year has 252 trading days 
    risk_free_rate=0
    sharpe = (daily_ret - risk_free_rate)/vol*np.sqrt(252)#len(portfolio_daily_value))

    #ls_symbols.remove('SPY')
    #lf_weights.remove(0)
    return vol, daily_ret, sharpe, cum_ret
 

def main():
    ''' Main Function'''
    # Start and End date of the charts
    dt_start = dt.datetime(2010, 1, 1)
    dt_end = dt.datetime(2010, 12, 31)
    #dt_start = dt.datetime(2011, 1, 1)
    #dt_end = dt.datetime(2011, 12, 31)
    #dt_end = dt.datetime(2011, 1, 9)

    # List of symbols. Last one is for reference.
    ls_symbols = ["AAPL", "GOOG", "IBM", "MSFT"]
    #ls_symbols = ["GOOG"]

    # List of equity weights in portfolio
    lf_weights = [1.0,0.0,0.0,0.0]
    #lf_weights = [0.0,0.0,0.0,1.0]

    # Run simulate-function with optimal allocation
    vol, daily_ret, sharpe, cum_ret = simulate(dt_start, dt_end, ls_symbols, lf_weights)
    
    # Print the return values
    print "Start Date: ", dt_start
    print "End Date: ", dt_end
    print "Symbols: ", ls_symbols
    print "Allocation: ", lf_weights
    
    print "Sharpe Ratio: ", sharpe
    print "Volatility (stdev of daily returns): ", vol
    print "Average Daily Return: ", daily_ret
    print "Cumulative Return: ", cum_ret
    
    
if __name__ == '__main__':
    main()

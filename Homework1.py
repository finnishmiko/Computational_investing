'''
Homework 1 for Coursera Computational investing course
wiki.quantsoftware.org

Function simulate to assess the performance of a 4 stock portfolio.
Find optimal allocation that has best Sharpe ratio
Function reference to compare portfolio to SPY
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

#print "Pandas Version", pd.__version__


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

    # Calculate daily return
    daily_ret = np.mean(portfolio_daily_value, axis=0)

    # Calculate Sharpe ratio with assumption that risk free rate is 0
    # and year has 252 trading days 
    risk_free_rate=0
    sharpe = (daily_ret - risk_free_rate)/vol*np.sqrt(252)#len(portfolio_daily_value))

    # Return calculated values
    return vol, daily_ret, sharpe, cum_ret


def reference(dt_start, dt_end, ls_symbols, lf_weights):
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
    
    # Calculate value of portfolio
    portfolio_daily_value = np.sum(na_rets * lf_weights, axis=1)
      
    # Plot portfolio with reference
    na_price = d_data['close'].values
    plt.clf()
    #fig=plt.figure()
    #fig.add_subplot(111)
    plt.plot(ldt_timestamps, portfolio_daily_value)# na_normalized_price)
    plt.plot(ldt_timestamps, na_normalized_price[:,4])
    
    plt.legend(["portfolio","SPY"])#ls_symbols)
    plt.ylabel('Normalized Close')
    plt.xlabel('Date')
    plt.savefig('close.pdf', format='pdf')
    

def main():
    ''' Main Function'''
    # Start and End date of the charts
    #dt_start = dt.datetime(2010, 1, 1)
    #dt_end = dt.datetime(2010, 12, 31)
    dt_start = dt.datetime(2011, 1, 1)
    dt_end = dt.datetime(2011, 12, 31)
    #dt_end = dt.datetime(2011, 1, 9)


    # List of symbols
    ls_symbols = ["AAPL", "GLD", "GOOG", "XOM"]
    #ls_symbols = ["AXP", "HPQ", "IBM", "HNZ"]
    #ls_symbols = ["AAPL", "GOOG", "IBM", "MSFT"]
    #ls_symbols = ["BRCM", "ADBE", "AMD", "ADI"]
    #ls_symbols = ["BRCM", "TXN", "IBM", "HNZ"]
    #ls_symbols = ["C", "GS", "IBM", "HNZ"]
    

    # List of equity weights in portfolio
    lf_weights = [0.4,0.4,0.0,0.2]
    #lf_weights = [0.0,0.0,0.0,1.0]

    # Find optimal allocations with Brute-force
    # No cash position i.e. 100% is in equities
    # Works with 4 equities only
    step = 10 # [%]
    weights = []
    opt_weights = []
    max_sharpe=-5.0
    for a in range(0,101, step):
        for b in range(0, 101, step):
            for c in range(0, 101, step):
                for d in range(0, 101, step):
                    if a+b+c+d == 100:
                        weights = [a/100.0, b/100.0, c/100.0, d/100.0]
                        vol, daily_ret, sharpe, cum_ret = simulate(dt_start, dt_end, ls_symbols, weights)
                        if sharpe > max_sharpe:
                            max_sharpe = sharpe
                            opt_weights = weights
                            #print max_sharpe, opt_weights
            
    # Run simulate-function again with optimal allocation
    vol, daily_ret, sharpe, cum_ret = simulate(dt_start, dt_end, ls_symbols, opt_weights)
    
    # Print the return values
    print "Start Date: ", dt_start
    print "End Date: ", dt_end
    print "Symbols: ", ls_symbols
    print "Allocation: ", opt_weights
    
    print "Sharpe Ratio: ", sharpe
    print "Volatility (stdev of daily returns): ", vol
    print "Average Daily Return: ", daily_ret
    print "Cumulative Return: ", cum_ret
    
    # compare optimal portfolio to reference
    ref="SPY"
    ls_symbols.append(ref)
    opt_weights.append(0.0)
    lf_weights = opt_weights[:]
    print ls_symbols, lf_weights
    reference(dt_start, dt_end, ls_symbols, opt_weights)
    
if __name__ == '__main__':
    main()



''' Part 2.5: Trian run results compared to given values --> works ok
======== RESTART:  ========
Start Date:  2011-01-01 00:00:00
End Date:  2011-12-31 00:00:00
Symbols:  ['AAPL', 'GLD', 'GOOG', 'XOM']
Allocation:  [0.4, 0.4, 0.0, 0.2]
Sharpe Ratio:  1.02828403099
Volatility (stdev of daily returns):  0.0101467067654
Average Daily Return:  0.000657261102001
Cumulative Return:  1.16487261965
>>> 
======== RESTART:  ========
Start Date:  2010-01-01 00:00:00
End Date:  2010-12-31 00:00:00
Symbols:  ['AXP', 'HPQ', 'IBM', 'HNZ']
Allocation:  [0.0, 0.0, 0.0, 1.0]
Sharpe Ratio:  1.29889334008
Volatility (stdev of daily returns):  0.00924299255937
Average Daily Return:  0.000756285585593
Cumulative Return:  1.1960583568
>>>

Part 3: Optimize
======== RESTART:  ========
Start Date:  2011-01-01 00:00:00
End Date:  2011-12-31 00:00:00
Symbols:  ['AAPL', 'GLD', 'GOOG', 'XOM']
Allocation:  [0.4, 0.4, 0.0, 0.2]
Sharpe Ratio:  1.02828403099
Volatility (stdev of daily returns):  0.0101467067654
Average Daily Return:  0.000657261102001
Cumulative Return:  1.16487261965
'''

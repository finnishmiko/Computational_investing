'''
Homework 7 for Coursera Computational investing course
wiki.quantsoftware.org

Run homework 6 Bollinger band event study trades with market simulator
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



def reference(dt_start, dt_end, ls_symbols):
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
    portfolio_daily_value = np.sum(na_rets * 1, axis=1)
    
    # Calculate the daily returns of the prices. (Inplace calculation)
    # returnize0 works on ndarray and not dataframes.
    tsu.returnize0(portfolio_daily_value)
    #print "daily returns \n", portfolio_daily_value

    daily_ret = np.mean(portfolio_daily_value, axis=0)
    vol = np.std(portfolio_daily_value)
    risk_free_rate=0
    sharpe = (daily_ret - risk_free_rate)/vol*np.sqrt(252)#len(portfolio_daily_value))
    print "Sharpe of $SPX: ", sharpe
    print "Total return of $SPX: ", na_normalized_price[-1,-1]
    print "Standard dev of $SPX: ", vol #np.std(na_normalized_price[:,-1])
    print "Avg raily ret of $SPX: ", daily_ret #np.mean(na_normalized_price[:,-1])
    
    
def main():
    # "python marketsim.py 1000000 orders.csv values.csv"
    
    dates = []
    previousDate = []
    holdingsWithDubl = []
    cashStart = 100000.0
    #orders = "orders-short.csv"
    orders = "ordersFile.csv" # for homework4
    filename='fundValue.csv'
    tradeCSV = []

    # Read trades from the CSV-file
    import csv
    reader = csv.reader(open(orders, 'rU'), delimiter=',')
    for row in reader:
            # Date is in 1st 3 columns
            currentDate = dt.datetime(int(row[0]), int(row[1]), int(row[2]))
            dates.append(currentDate)
            # 4th column is stock symbol
            holdingsWithDubl.append(row[3])
             #### Extra line
            #tradeCSV.append([row[0], row[1], row[2], row[3], row[4], row[5]])
    # Last item is not neccessary the last date!
    print "First date: ", min(dates)
    dt_start = min(dates)
    print "Last date: ", max(dates)
    dt_end = max(dates) + dt.timedelta(days=1)

    #dates=list(set(dates)) #### Extra line
    #tradeCSV = np.sort(tradeCSV, axis=0)
    #print tradeCSV #### Extra line
    dates = np.sort(dates)
    #print dates #### Extra line
    #print "holdings\n", holdingsWithDubl
    ls_symbols=list(set(holdingsWithDubl))
    print "Unique holdings:\n", ls_symbols
        
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
    #print ldt_timestamps

    # Create dataframe for dates and symbols
    df_trade = pd.DataFrame(0, index=ldt_timestamps, columns=ls_symbols)
    '''
        |SYM1|SYM2|SYM3|
    D1	| 0  | 0  | 0  |
    D2	| 0  | 0  | 0  |
    
    '''
    # Create a trade matrix
    dateIndex = []
    quantity = 0
    reader = csv.reader(open(orders, 'rU'), delimiter=',')
    '''
    Year | Month | Day | SYM | Buy/Sell | Amount
    '''
    counter=0
    for row in reader:
        print "csv for: ", row
        dateIndex = dt.datetime(int(row[0]), int(row[1]), int(row[2])) + dt_timeofday
        for dt_date in ldt_timestamps:
            #print dt_date
            if dt_date == dateIndex:
                if row[4] == "Sell":
                    quantity = -int(row[5])
                else:
                    quantity = int(row[5])
                #print quantity
                #counter+=1
                #print dateIndex
                #print row[3]
                # df_trade[SYM].dateIndex
                # Add value to the cell since there could be buying and selling during same day
                df_trade[row[3]].ix[dateIndex] = df_trade[row[3]].ix[dateIndex] + quantity
        #print "for loop df_trade:\n ", df_trade
    #print "Counter", counter
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
    # append price data
    na_price = np.insert(na_price, len(ls_symbols), 1.0, axis=1)
    print "Close prices:\n", na_price
    

    # Cash timeseries
    ts_cash = pd.TimeSeries(0.0, index=ldt_timestamps)
    ts_cash[0] = cashStart
    #print "Cash type: ", type(ts_cash)

    i = 0
    for index, row in df_trade.iterrows():
        j = 0
        #print "dataframe for loop: ", i#row[j]
        for s_sym in ls_symbols:
            ts_cash[i] = ts_cash[i] - na_price[i,j] * int(df_trade[s_sym].ix[index])
            #print na_price[i,j] * int(df_trade[s_sym].ix[index])
            #print "price", na_price[i,j]
            j+=1
        i+=1
    ts_cash = np.cumsum(ts_cash)
    print "ts_cash:\n", ts_cash
    
    # Update trade dataframe ownings per each day
    df_trade = df_trade.cumsum()
    df_trade['_CASH'] = ts_cash 
    print df_trade
    df_price = df_trade.copy()
    df_price = df_trade * na_price
    print df_price

    ts_fund = pd.TimeSeries(0, index=ldt_timestamps)
    df_price = df_price.cumsum(axis=1)
    
    ts_fund = df_price['_CASH']
    print "ts_fund\n", ts_fund #df_price.cumsum(axis=1)

    import csv
    writer = csv.writer(open(filename, 'wb'), delimiter=',')
    for row_index in ts_fund.index:
        #print row_index
        #print ts_fund[row_index]
        row_to_enter = [row_index, ts_fund[row_index]] #['JUNK', 'DATA']
        #print row_to_enter
        writer.writerow(row_to_enter)
    print "end writing"    

    
    # Normalizing the prices to start at 1 and see relative returns
    #na_normalized_price = na_price / na_price[0, :]
    ts_fund_normalized = ts_fund / ts_fund[0]
    print "Normalize", ts_fund_normalized

    # Copy the normalized prices to a new ndarry to find returns.
    na_rets = ts_fund_normalized.copy()
    
    # Calculate value of portfolio
    portfolio_daily_value = ts_fund.copy() #np.sum(na_rets * lf_weights, axis=1)
    #print "Portfolio daily value: ", portfolio_daily_value
    #print "End value: ", portfolio_daily_value[-1]
    
    # Cumulative return is the last day's value
    cum_ret = ts_fund_normalized[-1] #portfolio_daily_value[-1]
    #print "check cum ret", cum_ret

    # Calculate the daily returns of the prices. (Inplace calculation)
    # returnize0 works on ndarray and not dataframes.
    tsu.returnize0(portfolio_daily_value)
    #print "daily returns \n", portfolio_daily_value

    vol = np.std(portfolio_daily_value)
    daily_ret = np.mean(portfolio_daily_value, axis=0)
    risk_free_rate=0
    sharpe = (daily_ret - risk_free_rate)/vol*np.sqrt(252)#len(portfolio_daily_value))
    
    # Print the return values
    print "Start Date: ", dt_start
    print "End Date: ", dt_end - dt.timedelta(days=1)
    print "Symbols: ", ls_symbols
    #print "Allocation: ", lf_weights

    print "\nEnd value: ", ts_fund[-1]
    print "Sharpe Ratio: ", sharpe
    print "Cumulative Return: ", cum_ret
    print "Volatility (stdev of daily returns): ", vol
    print "Average Daily Return: ", daily_ret
    
    
    # compare optimal portfolio to reference
    ls_symbols = []
    ref="$SPX"
    print "\nReference: ", ref
    ls_symbols.append(ref)
    #weights.append(0.0)
    #lf_weights = opt_weights[:]
    #print ls_symbols, lf_weights
    reference(dt_start, dt_end, ls_symbols)
    
if __name__ == '__main__':
    main()



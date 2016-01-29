'''
Homework 3 for Coursera Computational investing course
wiki.quantsoftware.org

Market simulator to calculate portfolio daily value.
Orders are read from file and portfolio value is saved to a file
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
    # Read orders from orders(.csv)
    # and save results to filename(.csv)
    dates = []
    previousDate = []
    holdingsWithDubl = []
    cashStart = 1000000.0
    #orders = "orders-short.csv"
    orders = "orders.csv"
    #orders = "orders01.csv"
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
            tradeCSV.append([row[0], row[1], row[2], row[3], row[4], row[5]])
    # Last item is not neccessary the last date!
    print "\nFirst date: ", min(dates)
    dt_start = min(dates)
    print "\nLast date: ", max(dates)
    dt_end = max(dates) + dt.timedelta(days=1)

    dates=list(set(dates))
    print tradeCSV[3]
    
    #print "holdings\n", holdingsWithDubl
    ls_symbols=list(set(holdingsWithDubl))
    print "Unique holdings:\n", ls_symbols
        
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Create dataframe for dates and symbols
    df_trade = pd.DataFrame(0, index=ldt_timestamps, columns=ls_symbols)
    
    # Create a trade matrix
    dateIndex = []
    quantity = 0
    reader = csv.reader(open(orders, 'rU'), delimiter=',')
    for row in reader:
        print "csv for: ", row
        dateIndex = dt.datetime(int(row[0]), int(row[1]), int(row[2])) + dt_timeofday
        #print dateIndex
        for dt_date in ldt_timestamps:
            #print dt_date
            if dt_date == dateIndex:
                if row[4] == "Sell":
                    quantity = -int(row[5])
                else:
                    quantity = int(row[5])
                df_trade[row[3]].ix[dateIndex] = df_trade[row[3]].ix[dateIndex] + quantity
    print "for loop df_trade:\n ", df_trade
    
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
            #print "osakkeita", int(df_trade[s_sym].ix[index])
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
        row_to_enter = [row_index, ts_fund[row_index]] 
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
    
    print "Sharpe Ratio: ", sharpe
    print "Volatility (stdev of daily returns): ", vol
    print "Average Daily Return: ", daily_ret
    print "Cumulative Return: ", cum_ret

    
    # compare optimal portfolio to reference
    #ref="SPY"
    #ls_symbols.append(ref)
    #opt_weights.append(0.0)
    #lf_weights = opt_weights[:]
    #print ls_symbols, lf_weights
    #reference(dt_start, dt_end, ls_symbols, opt_weights)
    
if __name__ == '__main__':
    main()



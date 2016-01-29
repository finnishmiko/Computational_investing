'''
Homework 6 for Coursera Computational investing course
wiki.quantsoftware.org

Add Bollinger band calculation to the event profiler
and save trades to a orders(.csv) file.
Use adjusted closing price with this indicator.
'''


import pandas as pd
import numpy as np
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep

"""
Accepts a list of symbols along with start and end date
Returns the Event Matrix which is a pandas Datamatrix
Event matrix has the following structure :
    |IBM |GOOG|XOM |MSFT| GS | JP |
(d1)|nan |nan | 1  |nan |nan | 1  |
(d2)|nan | 1  |nan |nan |nan |nan |
(d3)| 1  |nan | 1  |nan | 1  |nan |
(d4)|nan |  1 |nan | 1  |nan |nan |
...................................
...................................
Also, d1 = start date
nan = no information about any event.
1 = status bit(positively confirms the event occurence)
"""


def find_events(ls_symbols, d_data):
    ''' Finding the event dataframe '''
    #df_close = d_data['actual_close'] # actual close
    df_close = d_data['close']#.values # adjusted close with Bollinger
    
    ts_market = df_close['SPY']

    ###### Bollinger
        
    # Calculate Bollinger value from 20 day rolling mean and std
    na_moving_avg = pd.rolling_mean(df_close, 20)
    #print na_moving_avg
    na_moving_std = pd.rolling_std(df_close, 20)
    #print na_moving_std
    bollinger_val = (df_close - na_moving_avg) / na_moving_std
    print bollinger_val
    ######
    
    df_close = bollinger_val
    ts_market = bollinger_val['SPY']

    print "Finding Events"

    # Creating an empty dataframe
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN

    # Time stamps for the event range
    ldt_timestamps = df_close.index

    # Write findings to CSV-file
    filename = 'ordersFile.csv'
    import csv
    writer = csv.writer(open(filename, 'wb'), delimiter=',')
                
    for s_sym in ls_symbols:
        for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
            f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
            f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
            f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
            f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
            f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
            f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) - 1

            # Event is found if the symbol is down more then 2 while the
            # market is up more then 1
            if f_symprice_today < -2 and f_symprice_yest >= -2 and f_marketprice_today >= 1:
                df_events[s_sym].ix[ldt_timestamps[i]] = 1

                # Write findings to CSV-file
                row_to_enter = [ldt_timestamps[i].year, ldt_timestamps[i].month, ldt_timestamps[i].day, s_sym, 'Buy', 100] #['JUNK', 'DATA']
                if ldt_timestamps[i]<ldt_timestamps[-5]:
                    sell_date = ldt_timestamps[i+5]
                else:
                    sell_date = ldt_timestamps[-1]
                row2_to_enter = [sell_date.year, sell_date.month, sell_date.day, s_sym, 'Sell', 100] #['JUNK', 'DATA']
                writer.writerow(row_to_enter)
                writer.writerow(row2_to_enter)
                print ldt_timestamps[i], s_sym, sell_date#, ldt_timestamps[-1]

    return df_events


if __name__ == '__main__':
    dt_start = dt.datetime(2008, 1, 1)
    dt_end = dt.datetime(2009, 12, 31)# + dt.timedelta(days=1)
    #dt_end = dt.datetime(2008, 1, 31) #+ dt.timedelta(days=1)
    print dt_start
    
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))
    #print ldt_timestamps
    dataobj = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list('sp5002012')
    #ls_symbols = dataobj.get_symbols_from_list('sp5002008')
    ls_symbols.append('SPY')

    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    #print d_data
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    df_events = find_events(ls_symbols, d_data)
    print df_events
    
    print "Creating Study"
    ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20,
                s_filename='my_event_study.pdf', b_market_neutral=True, b_errorbars=True,
                s_market_sym='SPY')

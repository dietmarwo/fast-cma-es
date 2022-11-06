# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/CryptoTrading.adoc for a detailed description.

# do
# pip install yfinance
# pip install finplot

import yfinance as yf
import finplot as fplt
import pandas as pd
import numpy as np
import math, time
from pathlib import Path

from fcmaes import retry, modecpp
from fcmaes.optimizer import logger, Bite_cpp, De_cpp, dtime
from scipy.optimize import Bounds
import ctypes as ct
import multiprocessing as mp 
from numba import njit
from numba.typed import List

START_CASH = 1000000.0

# download price history of a ticker and cache result into a csv file
def get_history(ticker, start, end):
    p = Path('ticker_cache')
    p.mkdir(exist_ok=True)
    fname = f'history_{ticker}_{start}_{end}.xz'
    files = p.glob(fname)
    for file in files: # if cached just load the csv
        df = pd.read_csv(file, compression='xz')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date') # restore date index
        return df
    df = yf.download(ticker, start=start, end=end, auto_adjust = True)
    df.to_csv('ticker_cache/' + fname, compression='xz')  # save history
    return df

# compute the exponential mean average of a time series of prices
def get_ema(prices, period):
    return prices.ewm(span=period,min_periods=period,adjust=False,ignore_na=False).mean().to_numpy()

# compute the simple mean average of a time series of prices
def get_sma(prices, period):
    return prices.rolling(period).mean().to_numpy()

# use numba to speed up the trading simulation

@njit(fastmath=True)
def buy_all(cash, num_coins, price, date, logs):
    num = int(cash / price)
    if not date is None:
        logs.append(date + " cash " + str(int(cash)) # no string formatting in numba :-(
                    + " buy " + str(num)
                    + " num_coins " + str(num_coins)
                    + " price " + str(int(100*price))+ " ct") # see https://github.com/numba/numba/issues/4758
    return cash - num*price, num_coins + num

@njit(fastmath=True)    
def sell_all(cash, num_coins, price, date, logs):
    if not date is None:
        logs.append(date + " cash " + str(int(cash)) 
                    + " sell " + str(num_coins) 
                    + " num_coins " + str(num_coins)
                    + " price " + str(int(100*price)) + " ct")
    return cash + num_coins*price, 0

# simulate HODL strategy
@njit(fastmath=True)        
def hodl(close, start_cash):
    cash, num_coins = buy_all(start_cash, 0, close[0], None, None) # buy first day
    cash, num_coins = sell_all(cash, num_coins, close[-1], None, None) # sell last day
    return cash / start_cash # return cash gain factor

# simulate a simple crossing EMA/SMA trading strategy, return the resulting cash factor
@njit(fastmath=True)        
def strategy(close, start_cash, ema, sma, wait_buy, wait_sell, dates):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    logs = List(); logs.append("") # needs to be String typed
    num_trades = 0
    # for all trading days
    for i in range(len(close)):
        c_ema = ema[i]; c_sma = sma[i]; price = close[i]  
        date = dates[i] if not dates is None else None       
        if np.isnan(c_ema) or np.isnan(c_sma):
            continue
        # buy if current ema > current sma and > wait_buy days since last trade
        if num_coins == 0 and c_ema > c_sma and i > last_trade + wait_buy:
            cash, num_coins = buy_all(cash, num_coins, price, date, logs)
            last_trade = i
            num_trades += 1
        # sell if current ema < current sma and > wait_sell days since last trade
        elif num_coins > 0 and c_ema < c_sma and i > last_trade + wait_sell:
            cash, num_coins = sell_all(cash, num_coins, price, date, logs)
            last_trade = i
            num_trades += 1

    cash, num_coins = sell_all(cash, num_coins, price, date, logs)
    # return cash gain factor compared with HODL strategy
    return cash / start_cash / hodl(close, start_cash), num_trades, logs

def simulate(prices, ema_period, sma_period, wait_buy, wait_sell, dates=None):
    close = prices.to_numpy()
    ema = get_ema(prices, ema_period)
    sma = get_sma(prices, sma_period) 
    return strategy(close, START_CASH, ema, sma, wait_buy, wait_sell, dates)

class fitness(object):

    def __init__(self, tickers, start, end, max_trades = None):
        self.evals = mp.RawValue(ct.c_int, 0) 
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.t0 = time.perf_counter()
        self.tickers = tickers
        self.max_trades = max_trades
        self.histories = {}
        self.closes = {}
        self.dates = {}
        self.hodls = {}
        for ticker in tickers:
            self.histories[ticker] = get_history(ticker, start=start, end=end)
            self.closes[ticker] = self.histories[ticker].Close
            self.dates[ticker] = np.array([d.strftime('%Y.%m.%d') for d in self.histories[ticker].index])
            self.hodls[ticker] = hodl(self.closes[ticker].to_numpy(), START_CASH)   
        hodls = list(self.hodls.values())                    
        logger().info("hodl = {0:.3f} {1:s}"
                .format(np.prod(hodls) ** (1.0/len(hodls)), str([round(fi,1) for fi in hodls])))
        
    def fun(self, x):
        # simulate the EMS/SMA strategy for all tickers
        factors = []
        num_trades = []
        for ticker in self.tickers:    
            # convert the optimization variables into integers and use them to configure the simulation
            f, num, _ = simulate(self.closes[ticker], int(x[0]), int(x[1]), int(x[2]), int(x[3]))
            factors.append(f)  
            num_trades.append(num)          
        factor = np.prod(factors) ** (1.0/len(factors)) # normalize the accumulated factor
        y = -factor # our optimization algorithm minimizes the resulting value, we maximize factor
        # book keeping and logging for parallel optimization
        self.evals.value += 1
        if y < self.best_y.value:
            self.best_y.value = y       
            logger().info("nsim = {0}: time = {1:.1f} fac = {2:.3f} {3:s} ntr = {4:s} x = {5:s}"
                .format(self.evals.value, dtime(self.t0), -y, 
                        str([round(fi,1) for fi in factors]),
                        str([int(ntr) for ntr in num_trades]),  
                        str([int(xi) for xi in x])))
        return y, factors, num_trades      
           
    def __call__(self, x):  
        y, _, _ = self.fun(x)
        return y   
    
    def mofun(self, x):
        _, factors, num_trades = self.fun(x)
        ys = [-f for f in factors] # higher factor is better
        constraints = [ntr - self.max_trades for ntr in num_trades] # at most max_trades trades
        return np.array(ys + constraints)
     
    def get_trades(self, ticker, x):
        _, _, log = simulate(self.closes[ticker], int(x[0]), int(x[1]), int(x[2]), int(x[3]), self.dates[ticker])
        trades = []
        for i in range(1,len(log)):
            l = log[i].split()
            trade = {'date': l[0], 
                'cash': int(l[2]), 
                'type': l[3],
                'traded': int(l[4]), 
                'num_coins': int(l[6]), 
                'price': int(l[8])/1000}
            trades.append(trade)
        logger().info('\n' + ticker)
        for l in log: logger().info(l)
        return trades

    def get_values(self, ticker, x):
        trades = self.get_trades(ticker, x)   
        cash = START_CASH
        num_coins = 0
        closes = self.closes[ticker]
        dates = self.dates[ticker]
        values = []
        i = 0
        for date, price in zip(dates,closes):
            trade = trades[i]
            tdate = trade['date']
            if date == tdate:
                i += 1
                traded = trade['traded']
                if trade['type'] == 'buy':
                    num_coins, cash = num_coins + traded, cash - traded*price
                else:
                    num_coins, cash = num_coins - traded, cash + traded*price   
                # print(trade)  
            values.append(cash + num_coins*price)
        return np.array(np.fromiter((v/START_CASH*closes[0] for v in values), dtype=float)) # adjust to initial stock price        
            
    def plot(self, x):
        axs = {}
        ema_period = int(x[0])
        sma_period = int(x[1])        
        for ticker in self.tickers:
            axs[ticker] = fplt.create_plot(ticker, maximize=False) 
            history = self.histories[ticker]
            candles = history[['Open','Close','High','Low']]
            fplt.candlestick_ochl(candles, ax=axs[ticker])
            volumes = history[['Open','Close','Volume']]
            fplt.volume_ocv(volumes, ax=axs[ticker].overlay())
            fplt.plot(history.Close.rolling(sma_period).mean(), ax=axs[ticker])
            fplt.plot(history.Close.ewm(span=ema_period,min_periods=ema_period,
                                        adjust=False,ignore_na=False).mean(), ax=axs[ticker])
            values = self.get_values(ticker, x)
            history['Value'] = values
            fplt.plot(values, ax=axs[ticker])
            closes = history.Close.to_numpy()
            fplt.plot(closes, ax=axs[ticker])            
            history.to_csv(ticker + '.csv')
        fplt.show()
    
def optimize(tickers, start, end):
    bounds = Bounds([20,50,10,10], [50,100,200,200])
    fit = fitness(tickers, start, end) 
    ret = retry.minimize(fit, bounds, logger = None, 
                          num_retries=32, optimizer=Bite_cpp(2000))
    fit.plot(ret.x)
    
def optimize_mo(tickers, start, end, nsga_update = True):
    nobj = len(tickers) # number of objectives
    ncon = nobj # number of constraints
    max_trades = 8
    fit = fitness(tickers, start, end, max_trades) 
    bounds = Bounds([20,50,10,10], [50,100,200,200])
    xs, front = modecpp.retry(fit.mofun, len(tickers), ncon, bounds, num_retries=32, popsize = 48, 
                  max_evaluations = 16000, nsga_update = nsga_update, logger = logger(), workers=32)
    for y, x in zip(front, xs):
        print("fac " + str([-round(yi,2) for yi in y[:nobj]]) +  
              " trades " + str([int(max_trades+ci) for ci in y[nobj:]]) + 
              " x = " + str([int(xi) for xi in x]))

if __name__ == '__main__':
    
    # ticker names: https://finance.yahoo.com/lookup
    tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD']
    #tickers = ['BTC-USD']
    
    start="2019-01-01"
    end="2030-04-30" 
    
    optimize(tickers, start, end)
    
    #optimize_mo(tickers, start, end)
    
    # fit = fitness(tickers, start, end) 
    # fit.plot([20,60,10,10])

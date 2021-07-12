from datetime import datetime
from pandas_datareader import data as pdr

import helper_fcn
import pandas as pd
import numpy as np

equity_name = ['XLY', 'XLI', 'XLF', 'XLV', 'XLK', 'XLP']
credit_name = ['EMB', 'HYG', 'LQD', 'MBB']
hedge_name = ['GLD', 'IEF']
pe_name = ['PSP', 'IGF', 'VNQ']
alter_name = ['MNA']

stocks = equity_name + credit_name + pe_name + hedge_name + alter_name + ['SPY', 'CAD=X', '^IRX']

price_data = pdr.get_data_yahoo(stocks, start=datetime(2010, 1, 1), end=datetime(2020, 12, 12))
adj_price = price_data['Adj Close'].dropna(how='any').drop(columns=['CAD=X', '^IRX'])

risk_free = adj_price.iloc[1:, -1] / 252 / 100
currency_cad = adj_price.iloc[1:, -2] / 252
cash_value = (1 + risk_free).cumprod().fillna(method='bfill')

log_return = np.log(adj_price).diff().dropna()

col = ['EQ1', 'EQ2', 'EQ3', 'EQ4', 'EQ5', 'EQ6', 'CR1', 'CR2', 'CR3', 'CR4', 'Gold', 'Bond', 'PE', 'Inf', 'REITs', 'HF', 'SPY']
data_index = adj_price.index.values
price_to_name = pd.DataFrame(adj_price.values, index=data_index, columns=col).dropna()
return_to_name = price_to_name.pct_change().dropna()

benchmark_return = (log_return[['SPY', 'LQD']] * np.array([0.6, 0.4])).sum(axis=1)

benchmark_return = benchmark_return.loc[pd.to_datetime('2015-01-01'):]
benchmark_net_value = np.exp(benchmark_return.cumsum())
benchmark_sharpe = benchmark_return.mean() / benchmark_return.std() * 16

ERC_equity_return, ERC_equity_net_value, ERC_equity_weights = helper_fcn.equity_attributes(adj_price, equity_name, 1000)
ERC_credit_return, ERC_credit_net_value, ERC_credit_weights = helper_fcn.equity_attributes(adj_price, credit_name, 1000)
ERC_pe_return, ERC_pe_net_value, ERC_pe_weights = helper_fcn.equity_attributes(adj_price, pe_name, 1000)

df = pd.DataFrame(columns=['Equity', 'Credit', 'PE'], index=ERC_credit_net_value.index)
df.Equity = ERC_equity_net_value.values
df.Credit = ERC_credit_net_value.values
df.PE = ERC_pe_net_value.values
df_return, df_net_value, df_weight = helper_fcn.equity_attributes(df, df.columns, 1000)

weights_total = pd.concat(
    [(ERC_equity_weights.T * df_weight.Equity.values).T, (ERC_credit_weights.T * df_weight.Credit.values).T, 
     (ERC_pe_weights.T * df_weight.PE.values).T],
    axis=1)
weights_total *= 0.8

for i in hedge_name:
    weights_total[i] = 0.075

weights_total['MNA'] = 0.05

all_names = equity_name + credit_name + pe_name + hedge_name + alter_name
total_log_return = (weights_total * log_return.loc[weights_total.index[1:], all_names]).sum(axis=1)

total_sharpe = total_log_return.mean() / total_log_return.std() * 16

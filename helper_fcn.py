from portfolio import CvarPortfolio
from portfolio import ErcPortfolio

from pandas_datareader import data as pdr
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

np.random.seed(0)
sns.set()


def equity_attributes(price, equity_name, num):
    erc_equity = ErcPortfolio()
    equity_weights = pd.DataFrame(erc_equity.get_allocations(price[equity_name], num),
                                  columns=equity_name,
                                  index=price.index)
    equity_weights = equity_weights.shift(1)
    equity_weights.replace(np.nan, 1 / len(equity_name), inplace=True)
    equity_return = (equity_weights * np.log(price[equity_name]).diff()).sum(axis=1)
    equity_net_value = np.exp(equity_return.cumsum())
    equity_sharpe = equity_return.mean() / equity_return.std() * np.sqrt(252)

    print('Return:', round(equity_return.mean() * 252, 3))
    print('Std.:  ', round(equity_return.std() * 16, 3))
    print('Sharpe:', round(equity_sharpe, 3))

    return [equity_return, equity_net_value, equity_weights.loc[equity_return.index]]


def mix_attributes(rf, df_mix, num):
    mix_portfolio = CvarPortfolio(rf.loc[df_mix.index])
    weights_mix = pd.DataFrame(mix_portfolio.get_allocations(df_mix.values, num), 
                               columns=df_mix.columns, 
                               index=df_mix.index)
    weights_mix = weights_mix.shift(1)
    weights_mix.replace(np.nan, 1 / df_mix.shape[1], inplace=True)
    return_mix = (weights_mix * np.log(df_mix).diff()).sum(axis=1)
    net_value_mix = np.exp(return_mix.cumsum())
    sharpe_mix = (return_mix - rf.loc[df_mix.index]).mean() / return_mix.std() * np.sqrt(252)

    print('Return:', round(return_mix.mean() * 252, 3))
    print('Std.:  ', round(return_mix.std() * 16, 3))
    print('Sharpe:', round(sharpe_mix, 3))

    return [return_mix, net_value_mix, weights_mix.loc[return_mix.index]]


def get_price_return_data(stocks):
    price_data = pdr.get_data_yahoo(stocks, start=datetime(2010, 1, 1), end=datetime(2021, 6, 1))
    price_data = price_data["Adj Close"]

    price_data = price_data.drop(columns=['CAD=X', '^IRX']).dropna()
    return_data = np.log(price_data).diff().dropna()
    return price_data, return_data


def get_total_rtn_nv_w(price, equity_name, credit_name, pe_name):
    equity_return, equity_net_value, equity_weights = equity_attributes(price, equity_name, 1000)
    credit_return, credit_net_value, credit_weights = equity_attributes(price, credit_name, 1000)
    pe_return, pe_net_value, pe_weights = equity_attributes(price, pe_name, 1000)

    df_mix = pd.DataFrame(columns=['Equity', 'Credit'], index=credit_net_value.index)
    df_mix.Equity = equity_net_value.values
    df_mix.Credit = credit_net_value.values

    return_mix, net_value_mix, weights_mix = equity_attributes(df_mix, df_mix.columns, 1000)
    weights_mix = weights_mix * 0.9
    weights_mix['PE'] = 0.1

    weights_total = pd.concat([(equity_weights.T * weights_mix.Equity.values).T,
                               (credit_weights.T * weights_mix.Credit.values).T,
                               (pe_weights.T * 0.1).T],
                              axis=1)
    weights_total = weights_total.dropna()

    all_names = equity_name + credit_name + pe_name

    log_return = np.log(price.dropna()).diff().dropna()
    total_return = (weights_total * log_return.loc[weights_total.index[1:], all_names]).sum(axis=1)

    total_net_value = np.exp(total_return.cumsum()).plot()
    plt.show()

    total_sharpe = total_return.mean() / total_return.std() * 16

    print('Return:', round(total_return.mean() * 252, 3))
    print('Std.:  ', round(total_return.std() * 16, 3))
    print('Sharpe:', round(total_sharpe, 3))

    return total_return, total_net_value, weights_total

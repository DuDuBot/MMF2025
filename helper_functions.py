import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from portfolio_strategies import ERC
from portfolio_strategies import MomentumERC

np.random.seed(0)
sns.set()


def fit_rp(price, ticker, num, momentum=False):
    if momentum:
        portfolio = MomentumERC()
    else:
        portfolio = ERC()

    allocation = portfolio.get_allocations(price[ticker], num)
    weight = pd.DataFrame(allocation, columns=ticker, index=price.index)
    weight = weight.shift(1)
    weight.replace(np.nan, 1 / len(ticker), inplace=True)

    rtn = (weight * np.log(price[ticker]).diff()).sum(axis=1)
    value = np.exp(rtn.cumsum())

    return [rtn, value, weight.loc[rtn.index]]


def equal_weighted_portfolio(price, equity, credit, alts, momentum=False):
    rtn_equity, value_equity, weight_equity = fit_rp(price, equity, 1000, momentum)
    rtn_credit, value_credit, weight_credit = fit_rp(price, credit, 1000, momentum)
    rtn_alts, _, weight_alts = fit_rp(price, alts, 1000, momentum)

    eq_cr = pd.DataFrame(columns=['Equity', 'Credit'], index=value_credit.index)
    eq_cr.Equity = value_equity.values
    eq_cr.Credit = value_credit.values
    _, _, weight_add = fit_rp(eq_cr, eq_cr.columns, 1000)
    weight_add = weight_add * 0.9
    weight_add['Alts'] = 0.1

    all_weights = pd.concat([(weight_equity.T * weight_add.Equity.values).T,
                            (weight_credit.T * weight_add.Credit.values).T,
                            (weight_alts.T * weight_add.Alts.values).T], axis=1)
    all_weights = all_weights.dropna()

    names = equity + credit + alts
    rtn = np.log(price.dropna()).diff().dropna()
    rtn_total = (all_weights * rtn.loc[all_weights.index[1:], names]).sum(axis=1)
    value_total = np.exp(rtn_total.cumsum())
    sharpe_ratio = rtn_total.mean() / rtn_total.std() * np.sqrt(252)

    print('Return: ', round(rtn_total.mean() * 252, 3))
    print('Std.: ', round(rtn_total.std() * np.sqrt(252), 3))
    print('Sharpe Ratio: ', round(sharpe_ratio, 3))

    value_total.index = pd.to_datetime(value_total.index)
    plt.plot(value_total)
    plt.show()

    return rtn_total, value_total, all_weights, [rtn_equity, rtn_credit, rtn_alts]


def display_fcn(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

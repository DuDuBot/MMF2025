from pandas_datareader import data as pdr
from tqdm import tqdm
from datetime import datetime
from datetime import date, timedelta
from hmmlearn import hmm

import quantstats as qs
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import helper_fcn
import os

import regimeDetection as rgd

np.random.seed(0)
sns.set()
os.getcwd()


def market_econ_regression(value, is_port=False):
    value = value.copy()
    value['Value'] = value.sum(axis=1)
    value['Return_temp'] = value['Value'].pct_change().dropna()
    if is_port:
        value.loc[
            list(value.loc[value.index.isin(rebalance_date)][1:].index),
            'Return_temp'
        ] = (value.loc[list(value.loc[value.index.isin(rebalance_date)][1:].index), 'Value']) \
            / ((value.shift(1).loc[list(value.loc[value.index.isin(rebalance_date)][1:].index), 'Value']) + 10000) - 1

        risk_returns = value['Return_temp']

    else:
        value.loc[
            list(value.loc[value.index.isin(rebalance_date)][1:].index), 'Return_temp'
        ] = np.nan
        risk_returns = value['Return_temp'].fillna(method='ffill')

    risk_returns.index = risk_returns.index.map(lambda x: pd.to_datetime(str(x)))

    monthly_returns = (risk_returns + 1).groupby([risk_returns.index.year, risk_returns.index.month]).prod() - 1
    monthly_returns.index.names = ['Year', 'Month']
    monthly_returns = monthly_returns.reset_index(level=[0, 1])

    indices = []
    for i in range(len(monthly_returns)):
        indices.append(date(int(monthly_returns.iloc[i].Year), int(monthly_returns.iloc[i].Month), 1))

    monthly_returns.index = indices
    monthly_returns.drop(['Year', 'Month'], axis=1, inplace=True)
    monthly_returns = monthly_returns.set_index(pd.DatetimeIndex(monthly_returns.index))
    monthly_returns.columns = ['Returns']

    mdl = sm.OLS(monthly_returns.loc['2016-04-01':'2021-06-01'],
                 sm.add_constant(risk_data.loc['2016-04-01':'2021-06-01'][risk_data.columns])).fit()
    return mdl.params


def senario_analysis(econ_data, scenario):
    if scenario == 'down':
        pool = risk_data.sort_values('Unemploy', ascending=0).iloc[:10, :]
        rd1 = list(np.random.choice(pool.iloc[:, 0], 3))
        rd2 = list(np.random.choice(pool.iloc[:, 1], 3))
        rd3 = list(np.random.choice(pool.iloc[:, 2], 3))
        rd4 = list(np.random.choice(pool.iloc[:, 3], 3))
        rd5 = list(np.random.choice(pool.iloc[:, 4], 3))
        rd6 = list(np.random.choice(pool.iloc[:, 5], 3))

    if scenario == 'up':
        pool = risk_data.sort_values('Unemploy').iloc[:10, :]
        rd1 = list(np.random.choice(pool.iloc[:, 0], 3))
        rd2 = list(np.random.choice(pool.iloc[:, 1], 3))
        rd3 = list(np.random.choice(pool.iloc[:, 2], 3))
        rd4 = list(np.random.choice(pool.iloc[:, 3], 3))
        rd5 = list(np.random.choice(pool.iloc[:, 4], 3))
        rd6 = list(np.random.choice(pool.iloc[:, 5], 3))

    simulation = pd.DataFrame([rd1, rd2, rd3, rd4, rd5, rd6]).T
    simulation.columns = econ_data.columns
    return simulation


def calculate_exposure(value, chosen_date):
    weights = value.loc[chosen_date][:-7] / (value.loc[chosen_date][:-7].sum())
    eq_weight = weights[equity_name].sum()
    cr_weight = weights[credit_name].sum()
    alt_weight = weights[alt_name].sum()
    hedge_weight = weights[hedge_name].sum()
    eq_weight_cad = weights[equity_name_cad].sum()
    cr_weight_cad = weights[credit_name_cad].sum()
    alt_weight_cad = weights[alt_name_cad].sum()
    hedge_weight_cad = weights[hedge_name_cad].sum()
    cash_amount = weights['Cash']

    attributes = [eq_weight, cr_weight, alt_weight, hedge_weight,
                  eq_weight_cad, cr_weight_cad, alt_weight_cad, hedge_weight_cad,
                  cash_amount]
    return pd.DataFrame(attributes,
                        index=['EQ_USD', 'CR_USD', 'Alt_USD', 'Hedge_USD', 'EQ_CAD', 'CR_CAD', 'Alt_CAD', 'Hedge_CAD',
                               'Cash'], columns=['Weight'])


def get_return(value, chosen_date):
    value = value.copy()
    value['Value'] = value.sum(axis=1)
    value['Return_temp'] = value['Value'].pct_change().dropna()
    value.loc[list(value.loc[value.index.isin(rebalance_date)][1:].index), 'Return_temp'] = np.nan
    risk_return = value['Return_temp'].fillna(method='ffill')
    return risk_return.loc[:chosen_date]


def get_attribution(value, chosen_date):
    equity = get_return(value[equity_name], chosen_date)[-1]
    credit = get_return(value[credit_name], chosen_date)[-1]
    alternatives = get_return(value[alt_name], chosen_date)[-1]

    equity_cad = get_return(value[equity_name_cad], chosen_date)[-1]
    credit_cad = get_return(value[credit_name_cad], chosen_date)[-1]
    alternatives_cad = get_return(value[alt_name_cad], chosen_date)[-1]

    return_data = [equity, credit, alternatives, equity_cad, credit_cad, alternatives_cad]
    sum_of_returns = np.array(return_data).sum()
    eq_weight = equity / sum_of_returns
    cr_weight = credit / sum_of_returns
    alt_weight = alternatives / sum_of_returns

    eq_weight_cad = equity_cad / sum_of_returns
    cr_weight_cad = credit_cad / sum_of_returns
    alt_weight_cad = alternatives_cad / sum_of_returns

    attributions = [eq_weight, cr_weight, alt_weight, eq_weight_cad, cr_weight_cad, alt_weight_cad]
    return pd.DataFrame({'Returns': return_data, 'Returns Attribution': attributions,
                       'Name': ['EQ_USD', 'CR_USD', 'Alt_USD', 'EQ_CAD', 'CR_CAD', 'Alt_CAD']}).set_index('Name')


def risk_attribution(value, chosen_date):
    exposure = calculate_exposure(value, chosen_date)
    eq = get_return(value[equity_name], chosen_date)
    cr = get_return(value[credit_name], chosen_date)
    alt = get_return(value[alt_name], chosen_date)

    eq_cad = get_return(value[equity_name_cad], chosen_date)
    cr_cad = get_return(value[credit_name_cad], chosen_date)
    alt_cad = get_return(value[alt_name_cad], chosen_date)

    all_returns = [eq, cr, alt, eq_cad, cr_cad, alt_cad]
    col_name = ['EQ_USD', 'CR_USD', 'Alt_USD', 'EQ_CAD', 'CR_CAD', 'Alt_CAD']
    attr_on_exp = exposure.loc[col_name] / exposure.loc[col_name].sum()
    rtn_data = pd.DataFrame(all_returns).T.dropna()
    rtn_data.columns = col_name

    attr = np.dot(np.array(attr_on_exp.T), np.array(rtn_data.cov()))
    risk = pd.DataFrame(attr, columns=col_name, index=[chosen_date])
    risk_attr = risk / risk.sum(axis=1)[0]
    return risk_attr.T


stocks = ['SCO', 'SPY', 'GLD', 'VWO', 'IEF', 'EMB', 'lqd', 'VNQ', 'MNA', 'CAD=X', '^IRX']
counter = datetime(2010, 1, 1)
end = datetime(2021, 6, 1)

equity_name = ['XLY', 'XLI', 'XLF', 'XLV', 'XLK', 'XLP']
equity_industry_usd = ['Consumer Discretionary', 'Industrial', 'Financial', 'Health Care', 'Technology',
                       'Consumer Staples']
equity_name_cad = ['XMD.TO', 'XFN.TO', 'ZUH.TO', 'XIT.TO', 'ZDJ.TO']
equity_industry_cad = ['Mid_Small', 'Financial', 'Health Care', 'Information Technology', 'DJI']
credit_name = ['EMB', 'HYG', 'LQD', 'MBB']
credit_industry_usd = ['Emerging Markets', 'High Yield', 'Investment Grade', 'Mortgage Backed Securities']
credit_name_cad = ['ZEF.TO', 'XHY.TO', 'ZCS.TO', 'XQB.TO']
credit_industry_cad = ['Emerging Markets', 'High Yield', 'Corporate Bonds', 'Investment Grade']

data = pdr.get_data_yahoo(stocks, start=counter, end=end)
data = data['Adj Close']

rf = data.iloc[1:, -1] / 252
cad = data.iloc[1:, -2] / 252
data = data.iloc[1:, :-2]
returns = data.pct_change().dropna()
cols = ['Oil', 'SPX', 'Gold', 'EM EQ', 'US Bond', 'EMD', 'US HY', 'REIT', 'Hedge Fund']
data_to_name = pd.DataFrame(data.values, index=data.index.values, columns=cols).dropna()
return_to_name = data_to_name.pct_change().dropna()

# Use when regime changes
#hedge_name = ['IEF']
#hedge_industry_name = ['US_Treasury']
#hedge_name_cad = ['CGL.TO']
#hedge_industry_cad = ['Gold_CAD']

#alt_name = ['PSP', 'IGF', 'VNQ', 'MNA']
#alt_industry_usd = ['PE', 'Infra', 'REITs', 'HF']
#alt_name_cad = ['CGR.TO', 'CIF.TO']
#alt_industry_cad = ['REITs', 'Infra']

#stocks = equity_name + credit_name + alt_name + hedge_name + ['SPY', 'CAD=X', '^IRX']
#stocks_cad = equity_name_cad + credit_name_cad + alt_name_cad + hedge_name_cad + ['SPY', 'CAD=X', '^IRX']

#stock_price, stock_return = helper_fcn.get_price_return_data(stocks)
#price_cad, return_cad = helper_fcn.get_price_return_data(stocks_cad)
#date_intersection = [i for i in stock_price.index if i in price_cad.index]
#prices = pd.concat([stock_price.loc[date_intersection], price_cad.loc[date_intersection]], axis=1)

#price_hedge = pdr.get_data_yahoo(hedge_name + hedge_name_cad, start=datetime(2016, 4, 1), end=datetime(2021, 6, 1))
#price_hedge = price_hedge['Adj Close'].ffill(axis=0).dropna()

if 'weights.pkl' in os.listdir(os.getcwd() + '/Data'):
    merged_weight = pd.read_pickle('Data/weights.pkl')
else:
    total_rtn, total_nv, total_weight = helper_fcn.get_total_rtn_nv_w(stock_price, equity_name, credit_name, alt_name)
    # in usd
    total_rtn_cad, total_nv_cad, total_weight_cad = helper_fcn.get_total_rtn_nv_w(price_cad,
                                                                                  equity_name_cad,
                                                                                  credit_name_cad,
                                                                                  alt_name_cad)
    # in cad
    mutual_date = [i for i in total_weight.index if i in total_weight_cad.index]

    merged_weight = pd.concat([total_weight.loc[mutual_date] / 2, total_weight_cad.loc[mutual_date] / 2], axis=1)
    merged_weight.to_pickle('weights.pkl')


if 'Signal.pkl' in os.listdir(os.getcwd() + '/Data'):
    signalSeries = pd.read_pickle('Data/Signal.pkl')
else:
    pass

# Rebalancing
should_allow = []
temp = []
x = 2016

for i in range(6):
    temp.append(date(x + i, 4, 1))
    temp.append(date(x + i, 10, 1))

rebalance_date = []

for i in list(temp):
    try:
        a = (merged_weight.loc[i])
        rebalance_date.append(i)
    except:
        try:
            a = (merged_weight.loc[i + timedelta(days=1)])
            rebalance_date.append(i + timedelta(days=1))
        except:
            try:
                a = (merged_weight.loc[i + timedelta(days=2)])
                rebalance_date.append(i + timedelta(days=2))
            except:
                try:
                    a = (merged_weight.loc[i + timedelta(days=3)])
                    rebalance_date.append(i + timedelta(days=3))
                except:
                    pass

for w in list(merged_weight.index):
    date = w.to_pydatetime().date()
    if date in rebalance_date:
        should_allow.append(True)
    else:
        should_allow.append(False)

erc_weight = merged_weight.loc[should_allow]
whole_port_data = pdr.get_data_yahoo('CAD=X', start=datetime(2016, 4, 1), end=datetime(2021, 6, 1))
adj_close_data = whole_port_data['Adj Close']
overnight_rate = pd.read_csv('Data/canadaOvernight.csv', index_col=0, parse_dates=True).sort_index()

counter = 90000
p_value = prices.loc[pd.to_datetime('2016-04-01'):pd.to_datetime('2021-06-01')].dropna()
p_value = (p_value[erc_weight.columns])

stock_price = prices[erc_weight.columns].dropna()
stock_price = stock_price.loc[pd.to_datetime('2016-04-01'):pd.to_datetime('2021-06-01')].dropna()
investment = []
cash = []

for i in range(len(erc_weight)):
    reb_date = erc_weight.index[i]

    try:
        end_date = erc_weight.index[i + 1] - timedelta(days=1)
    except:
        end_date = date(2021, 6, 1)

    reb_to_end_data = p_value[reb_date:end_date]
    reb_date = reb_to_end_data.index[0]
    end_date = reb_to_end_data.index[-1]
    allocated_money = counter * erc_weight.iloc[i]

    try:
        fx_rate = adj_close_data.loc[reb_date]
    except:
        fx_rate = adj_close_data.loc[reb_date.date() - timedelta(days=1)]

    us_name = [i for i in list(stock_price.columns) if (i[-2:] != 'TO')]
    price_in_cad = stock_price.copy().loc[reb_date]
    price_in_cad[us_name] *= fx_rate

    p_value[reb_date:end_date] = p_value[reb_date:end_date] * list(allocated_money.divide(price_in_cad))
    investment.extend([100000 + (i * 10000)] * len(p_value[reb_date:end_date]))
    cash.extend([10000 + (i * 1000)] * len(p_value[reb_date:end_date]))

    price_in_cad = p_value.copy().loc[end_date]

    try:
        fx_rate = adj_close_data.loc[end_date]
    except:
        fx_rate = adj_close_data.loc[end_date.date() - timedelta(days=1)]

    price_in_cad[us_name] *= fx_rate

    end_value = price_in_cad.sum()

    counter = 9000 + end_value

p_value['Cash'] = cash


# Last part
# Risk Adjustments

macro_data = pd.read_csv('MacroData.csv', index_col='DATE')
macro_data = macro_data.loc[macro_data.index >= '2000-03-01'].iloc[:-2, :]
macro_data = macro_data.applymap(lambda x: float(x))

inflation = macro_data['CPIAUCSL'].pct_change().dropna() * 100 * 12
Industrial_prod_growth = macro_data['INDPRO'].pct_change().dropna() * 100
risk_data = pd.DataFrame(inflation).join(Industrial_prod_growth).join(macro_data.iloc[:, 2:7])

risk_data['CreditPremium'] = (macro_data['BAMLC0A4CBBBEY'] - macro_data['BAMLC0A1CAAAEY']) - (
        macro_data['BAMLC0A4CBBBEY'] - macro_data['BAMLC0A1CAAAEY']).shift(1)

risk_data.columns = ['Inflation', 'IndustrialProdGrowth', 'T-Bill', 'Oil', 'Libor', 'House',
                     'Unemploy', 'CreditPremium']

risk_data['Unexpected Inflation'] = (risk_data['Inflation'] - risk_data['Inflation'].shift(1)) \
                                    - (risk_data['T-Bill'].shift(1) - risk_data['T-Bill'].shift(2))

risk_data = risk_data.dropna()

risk_data = risk_data[['IndustrialProdGrowth', 'Oil', 'Unemploy', 'House', 'CreditPremium', 'Unexpected Inflation']]

us_name_risk = alt_name + credit_name + equity_name + hedge_name
p_value[us_name_risk] = p_value[us_name_risk].multiply(p_value['Adj Close'], axis=0)

eq_name = equity_name + equity_name_cad
cr_name = credit_name + credit_name_cad
alternative_name = alt_name + alt_name_cad
hedging_name = hedge_name + hedge_name_cad

beta_portfolio = market_econ_regression(p_value.iloc[:, :-7], True)
beta_eq = market_econ_regression(p_value[eq_name])
beta_cr = market_econ_regression(p_value[cr_name])
beta_alt = market_econ_regression(p_value[alternative_name])

# down
downward_situation = senario_analysis(risk_data, scenario='down')
downward_situation.insert(0, 'constant', 1)
downward_equity = np.dot(np.array(downward_situation.iloc[:, :7]), np.array(beta_eq))
downward_situation['EQ Estimated Return'] = downward_equity
downward_credit = np.dot(np.array(downward_situation.iloc[:, :7]), np.array(beta_cr))
downward_situation['CR Estimated Return'] = downward_credit
downward_alternative = np.dot(np.array(downward_situation.iloc[:, :7]), np.array(beta_alt))
downward_situation['PE Estimated Return'] = downward_alternative
downward_port = np.dot(np.array(downward_situation.iloc[:, :7]), np.array(beta_portfolio))
downward_situation['Portfolio Estimated Return'] = downward_port

# up
upward_situation = senario_analysis(risk_data, scenario='up')
upward_situation.insert(0, 'constant', 1)
upward_equity = np.dot(np.array(upward_situation.iloc[:, :7]), np.array(beta_eq))
upward_situation['EQ Estimated Return'] = upward_equity
upward_credit = np.dot(np.array(upward_situation.iloc[:, :7]), np.array(beta_cr))
upward_situation['CR Estimated Return'] = upward_credit
upward_alternative = np.dot(np.array(upward_situation.iloc[:, :7]), np.array(beta_alt))
upward_situation['Alt Estimated Return'] = upward_alternative
upward_port = np.dot(np.array(upward_situation.iloc[:, :7]), np.array(beta_portfolio))

credit = p_value[credit_name].loc[:'2021-06-01'].sum(axis=1)
exposure_to_plot = calculate_exposure(p_value, '2021-06-01')
exposure_to_plot['Weight'].plot.pie(autopct='%.2f', fontsize=12, figsize=(8, 8))

macro_data = get_attribution(p_value, '2021-06-01')
macro_data['Returns Attribution'].sum()
macro_data['Returns Attribution'].plot.pie(autopct='%.2f', fontsize=12, figsize=(8, 8))

risk_attr = risk_attribution(p_value, '2021-06-01')
risk_attr['2021-06-01'].plot.pie(autopct='%.2f', fontsize=12, figsize=(8, 8))

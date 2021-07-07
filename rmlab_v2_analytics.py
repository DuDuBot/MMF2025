def goodPrint(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


########################################################################

from pandas_datareader import data as pdr
import quantstats as qs

from datetime import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np

np.random.seed(0)
import seaborn as sns

sns.set()
import utilityFuncs
import os

os.getcwd()
#######################################################################

"""## Data Pre Processing"""
stocks = ["SCO", "SPY", "GLD", "VWO", "IEF", "EMB", "lqd", "VNQ", "MNA", "CAD=X", "^IRX"]
# oil,sp500,gold,emerging_eq,us_7_10year_bonds,emerging bonds, hy_coroprate, reit etf, Hedge Fund, risk_free (13-week treasury bond), CAD
start = datetime(2010, 1, 1)
end = datetime(2020, 6, 1)

data = pdr.get_data_yahoo(stocks, start=start, end=end)
data = data["Adj Close"]

rf = data.iloc[1:, -1] / 252
cad = data.iloc[1:, -2] / 252
data = data.iloc[1:, :-2]
returns = data.pct_change().dropna()
clmns = 'Oil,SPX,Gold,EM EQ,US Bond,EMD,US HY,REIT,Hedge Fund'.split(',')
dataIdx = data.index.values
dataNamed = pd.DataFrame(data.values, index=dataIdx, columns=clmns).dropna()
rtnNamed = dataNamed.pct_change().dropna()

#######################################################################

# Portfolio Construction
tickerEquity = ['XLY', 'XLI', 'XLF', 'XLV', 'XLK', 'XLP']
tickerEqNamesUS = ["Consumer Discretionary", "Industrial", "Financial", "Health Care", "Technology", "Consumer Staples"]

tickerEquityCAD = ['XMD.TO', 'XFN.TO', 'ZUH.TO', 'XIT.TO', 'ZDJ.TO']
tickerEqNamesCAD = ["Mid_Small", "Financial", "Health Care", "Information Technology", "DJI"]

tickerCredit = ["EMB", "HYG", 'LQD', 'MBB']
tickerCreditNamesUSD = ["Emerging Markets", "High Yield", "Investment Grade", "Mortgage Backed Securities"]
tickerCreditCAD = ['ZEF.TO', 'XHY.TO', 'ZCS.TO', 'XQB.TO']
tickerCreditNamesCAD = ["Emerging Markets", "High Yield", "Corporate Bonds", "Investment Grade"]

# only used when regime changes
tickerHedge = ['IEF']
tickerHNamesUSD = ["US_Treasury"]
tickerHedgeCAD = ['CGL.TO']
tickerHNamesCAD = ["Gold_CAD"]

tickerAlts = ['PSP', 'IGF', 'VNQ', 'MNA']
tickerAltsNamesUSD = ["PE", "Infra", "REITs", "HF"]
tickerAltsCAD = ['CGR.TO', 'CIF.TO']
tickerAltsNamesCAD = ["REITs", "Infra"]

stocks = tickerEquity + tickerCredit + tickerAlts + tickerHedge + ["SPY", "CAD=X", "^IRX"]
stocksCAD = tickerEquityCAD + tickerCreditCAD + tickerAltsCAD + tickerHedgeCAD + ["SPY", "CAD=X", "^IRX"]

start = datetime(2010, 1, 1)
end = datetime(2020, 6, 1)

price, rtn = utilityFuncs.pull_data(stocks)
priceCAD, rtnCAD = utilityFuncs.pull_data(stocksCAD)
commonDate = [i for i in price.index if i in priceCAD.index]
priceMerged = pd.concat([price.loc[commonDate], priceCAD.loc[commonDate]], axis=1)

start = datetime(2015, 4, 1)
end = datetime(2020, 6, 1)
priceHedge = pdr.get_data_yahoo(tickerHedge + tickerHedgeCAD, start=start, end=end)["Adj Close"]
priceHedge = priceHedge.ffill(axis=0).dropna()

if 'weights.pkl' in os.listdir(os.getcwd() + '/Data'):
    weightMerged = pd.read_pickle('Data/weights.pkl')
else:
    rtnTotal, nvTotal, wTotal = utilityFuncs.make_port(price, tickerEquity, tickerCredit, tickerAlts)
    # Results for NYSE
    rtnTotalCAD, nvTotalCAD, wTotalCAD = utilityFuncs.make_port(priceCAD, tickerEquityCAD, tickerCreditCAD,
                                                                tickerAltsCAD)
    # Results for TSX

    mutualDate = [i for i in wTotal.index if i in wTotalCAD.index]

    weightMerged = pd.concat([wTotal.loc[mutualDate] / 2, wTotalCAD.loc[mutualDate] / 2], axis=1)
    weightMerged.to_pickle('weights.pkl')

######################################################################

# Regime Detection
if 'Signal.pkl' in os.listdir(os.getcwd() + '/Data'):
    signalSeries = pd.read_pickle('Data/Signal.pkl')
# else:
# HMM

#######################################################################

# Rebalancing and Portfolio Allocation
myMask = []
temp = []
x = 2015
weightsAll = weightMerged

for i in range(6):
    temp.append(date(x + i, 4, 1))
    temp.append(date(x + i, 10, 1))

trialList = list(temp)
rebalancing = []

# Getting all the rebalancing dates
for i in trialList:
    try:
        a = (weightsAll.loc[i])
        rebalancing.append(i)
    except:
        try:
            a = (weightsAll.loc[i + timedelta(days=1)])
            rebalancing.append(i + timedelta(days=1))
        except:
            try:
                a = (weightsAll.loc[i + timedelta(days=2)])
                rebalancing.append(i + timedelta(days=2))
            except:
                try:
                    a = (weightsAll.loc[i + timedelta(days=3)])
                    rebalancing.append(i + timedelta(days=3))
                except:
                    pass

for i in list(weightsAll.index):
    i = i.to_pydatetime().date()
    if i in (rebalancing):
        myMask.append(True)
    else:
        myMask.append(False)

# This dataframe contains all the portfolio weights
ERCWeight = weightsAll.loc[myMask]
start = datetime(2015, 4, 1)
end = datetime(2020, 6, 1)
fx = pdr.get_data_yahoo("CAD=X", start=start, end=end)
fxData = fx["Adj Close"]
oRates = pd.read_csv("Data/canadaOvernight.csv", index_col=0, parse_dates=True).sort_index()

# Performance Analysis for Main Portfolio
start = 90000
portfolioValue = priceMerged.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()
portfolioValue = (portfolioValue[ERCWeight.columns])

price = priceMerged[ERCWeight.columns].dropna()
price = price.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()
investment = []
cash = []

for i in range(len(ERCWeight)):
    rebalanceDate = ERCWeight.index[i]

    try:
        endDate = ERCWeight.index[i + 1] - timedelta(days=1)
    except:
        endDate = date(2020, 6, 1)

    relevantData = portfolioValue[rebalanceDate:endDate]
    rebalanceDate = relevantData.index[0]
    endDate = relevantData.index[-1]

    moneyAllocated = start * ERCWeight.iloc[i]

    try:
        fxConvert = fxData.loc[rebalanceDate]
    except:
        fxConvert = fxData.loc[rebalanceDate.date() - timedelta(days=1)]

    usTickers = [i for i in list(price.columns) if (i[-2:] != "TO")]
    priceinCAD = price.copy().loc[rebalanceDate]
    priceinCAD[usTickers] *= fxConvert

    noofUnits = moneyAllocated.divide(priceinCAD)

    portfolioValue[rebalanceDate:endDate] = portfolioValue[rebalanceDate:endDate] * list(noofUnits)
    investment.extend([100000 + (i * 10000)] * len(portfolioValue[rebalanceDate:endDate]))
    cash.extend([10000 + (i * 1000)] * len(portfolioValue[rebalanceDate:endDate]))

    priceinCAD = portfolioValue.copy().loc[endDate]

    try:
        fxConvert = fxData.loc[endDate]
    except:
        fxConvert = fxData.loc[endDate.date() - timedelta(days=1)]

    priceinCAD[usTickers] *= fxConvert

    endvalue = priceinCAD.sum()

    start = 9000 + endvalue

portfolioValue["Cash"] = cash

# Regime Strategy

trades = signalSeries.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()
moneyAccount = portfolioValue.Cash.copy()
openPos = 0
regimeDates = []

for i in range(len(moneyAccount)):
    try:
        currentIndex = moneyAccount.index[i]
        if trades[currentIndex] == 1 and openPos == 0:
            buyIndex = currentIndex
            buyPrice = priceHedge.loc[(currentIndex.date())]
            openPos = 1

        elif trades[moneyAccount.index[i]] == -1 and openPos == 1:
            sellPrice = priceHedge.loc[(currentIndex.date())]
            openPos = 0
            regimeDates.append([buyIndex, currentIndex])
    except:
        pass

priceHedge2 = priceHedge.copy()

for i in priceHedge.index:
    if i not in portfolioValue.index:
        priceHedge2.drop(i, inplace=True)

priceHedge = priceHedge2

tradeData = []
for i in range(len(regimeDates)):
    buyDate = regimeDates[i][0]
    sellDate = regimeDates[i][1]
    goldData = priceHedge.loc[buyDate:sellDate]["CGL.TO"] / priceHedge.loc[buyDate]["CGL.TO"]
    treaData = priceHedge.loc[buyDate:sellDate].IEF / priceHedge.loc[buyDate].IEF
    tradeData.append([goldData, treaData])

cashValue = [moneyAccount.iloc[0]]
treaValue = [0]
goldValue = [0]
j = 0
buyDates = [i[0] for i in regimeDates]
sellDates = [i[1] for i in regimeDates]
numberofDays = 0
openPos = False

for i in range(len(portfolioValue) - 1):

    currentIndex = portfolioValue.index[i]
    ORate = oRates.loc[currentIndex.date()] / 36500

    if currentIndex in rebalancing[1:]:

        if openPos == True:
            cashValue[i:i + numberofDays + 1] = np.add(cashValue[i:i + numberofDays + 1], 1000)
        else:
            cashValue[i] = cashValue[i] + 1000

    if openPos == True:

        if numberofDays > 0:
            numberofDays -= 1
            continue

        elif numberofDays == 0:
            cashValue[i] = goldValue[i] + treaValue[i] + cashValue[i]
            goldValue[i] = 0
            treaValue[i] = 0
            openPos = False

    if currentIndex in buyDates:

        numberofDays = len(tradeData[j][0]) - 2
        goldData = np.multiply(list(tradeData[j][0]), float(cashValue[i] / 2))
        treaData = np.multiply(list(tradeData[j][1]), float(cashValue[i] / 2))
        goldValue[i] = (cashValue[i] / 2)
        treaValue[i] = (cashValue[i] / 2)
        cashValue[i] = 0
        goldValue.extend(list(goldData[1:]))
        treaValue.extend(list(treaData[1:]))
        cashValue.extend(len(goldData[1:]) * [0])
        j += 1
        openPos = True



    else:

        cashValue.append((cashValue[i]) * (1 + float(ORate)))
        treaValue.append(0)
        goldValue.append(0)

portfolioValue["Cash"] = cashValue
portfolioValue["CGL.TO"] = goldValue
portfolioValue["IEF"] = treaValue

usTickers.append("IEF")
cadTickers = list(set(portfolioValue.columns) - set(usTickers))

portfolioValue["USDTickers"] = portfolioValue[usTickers].sum(axis=1)
portfolioValue["CADTickers"] = portfolioValue[cadTickers].sum(axis=1)

portfolioValue = portfolioValue.join(fxData)
portfolioValue.ffill(axis=0, inplace=True)
portfolioValue["USDTickers_CAD"] = portfolioValue["USDTickers"].multiply(portfolioValue["Adj Close"])
# portfolioValue.drop(["Adj Close"],inplace=True,axis=1)


portfolioValue["Principal"] = investment
portfolioValue["Value_CAD"] = portfolioValue["CADTickers"] + portfolioValue["USDTickers_CAD"]

rebalancing = portfolioValue[~portfolioValue['Principal'].diff().isin([0])].index
portfolioValue["Return"] = portfolioValue["Value_CAD"].pct_change()
portfolioValue.loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index), 'Return'] = (
                                                                                                               portfolioValue.loc[
                                                                                                                   list(
                                                                                                                       portfolioValue.loc[
                                                                                                                           portfolioValue.index.isin(
                                                                                                                               rebalancing)][
                                                                                                                       1:].index), 'Value_CAD']) / (
                                                                                                                   (
                                                                                                                       portfolioValue.shift(
                                                                                                                           1).loc[
                                                                                                                           list(
                                                                                                                               portfolioValue.loc[
                                                                                                                                   portfolioValue.index.isin(
                                                                                                                                       rebalancing)][
                                                                                                                               1:].index), 'Value_CAD']) + 10000) - 1

# Generate graphs for the portfolio
returnData = portfolioValue.Return.dropna()
qs.reports.full(returnData)

# Portfolio Exposures
plt.figure(figsize=(10, 5))
labels = list(ERCWeight.columns)
plt.stackplot(list(ERCWeight.index), ERCWeight.values.T, labels=labels)
plt.title("Weights Rebalancing Evolution")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
temp = portfolioValue[ERCWeight.columns].div(portfolioValue["Value_CAD"], axis=0)
plt.stackplot(list(portfolioValue.index), temp.T, labels=labels)
plt.title("Exposure by Asset Class")
plt.legend()

# Risk Models

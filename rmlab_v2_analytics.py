
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

# price = pdr.get_data_yahoo(stocks, start=start, end=end)
# price = price["Adj Close"]

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

#######################################################################

# Risk Models

df = pd.read_csv("MacroData.csv", index_col='DATE')
df = df.loc[df.index >= '2000-03-01'].iloc[:-2, :]
df = df.applymap(lambda x: float(x))
credit_risk_premium = (df['BAMLC0A4CBBBEY'] - df['BAMLC0A1CAAAEY']) - (
            df['BAMLC0A4CBBBEY'] - df['BAMLC0A1CAAAEY']).shift(1)

inflation = df['CPIAUCSL'].pct_change().dropna() * 100 * 12
Industrial_prod_growth = df['INDPRO'].pct_change().dropna() * 100
riskData = pd.DataFrame(inflation).join(Industrial_prod_growth).join(df.iloc[:, 2:7])
riskData['CreditPremium'] = credit_risk_premium
riskData.columns = ['Inflation', 'IndustrialProdGrowth', 'T-Bill', 'Oil', 'Libor', 'House', 'Unemploy', 'CreditPremium']
riskData['Unexpected Inflation'] = (riskData['Inflation'] - riskData['Inflation'].shift(1)) - (
            riskData['T-Bill'].shift(1) - riskData['T-Bill'].shift(2))
riskData = riskData.dropna()
riskData = riskData[['IndustrialProdGrowth', 'Oil', 'Unemploy', 'House', 'CreditPremium', 'Unexpected Inflation']]


# riskData.head()
# riskData.describe()
#
# riskData.corr()

# from google.colab import drive
# drive.mount('/content/drive')

# # riskReturns=portfolioValue.Return.dropna()
# portfolioValue['Return_temp']=(portfolioValue[Alt_ticker].sum(axis=1)).pct_change().dropna()
# # portfolioValue.loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Return_temp']=(portfolioValue.loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Value_CAD'])/((portfolioValue.shift(1).loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Value_CAD'])+10000)-1
# portfolioValue.loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Return_temp']=np.nan
# riskReturns = portfolioValue['Return_temp'].fillna(method='ffill')
# riskReturns.index = riskReturns.index.map(lambda x:pd.to_datetime(str(x)))
# # monthlyReturns=riskReturns.groupby([riskReturns.index.year,riskReturns.index.month]).sum()
# monthlyReturns = (riskReturns+1).groupby([riskReturns.index.year,riskReturns.index.month]).prod()-1
# monthlyReturns.index.names=["Year","Month"]
# monthlyReturns=monthlyReturns.reset_index(level=[0,1])
# indexList=[]
# for i in range(len(monthlyReturns)):
#   indexList.append(date(int(monthlyReturns.iloc[i].Year),int(monthlyReturns.iloc[i].Month),1))
# monthlyReturns.index=indexList
# monthlyReturns.drop(["Year","Month"],axis=1,inplace=True)
# monthlyReturns = monthlyReturns.set_index(pd.DatetimeIndex(monthlyReturns.index))
# monthlyReturns.columns=['Returns']
# X=riskData.loc["2015-04-01":"2020-03-01"][riskData.columns]
# Y=monthlyReturns.loc["2015-04-01":"2020-03-01"]
#
# X = sm.add_constant(X)
# model = sm.OLS(Y, X).fit()
# model.summary()
#
# # Basic correlogram
# sns.pairplot(X.join(Y))
# plt.show()


# model.params

def market_econ_regression(PortfolioValue, isPortfolio=False):
    # PortfolioValue = PortfolioValue.sum(axis=1)
    # riskReturns = PortfolioValue.pct_change().dropna()
    PortfolioValue = PortfolioValue.copy()
    PortfolioValue['Value'] = PortfolioValue.sum(axis=1)
    PortfolioValue['Return_temp'] = PortfolioValue['Value'].pct_change().dropna()
    if isPortfolio:
        PortfolioValue.loc[
            list(PortfolioValue.loc[PortfolioValue.index.isin(rebalancing)][1:].index), 'Return_temp'] = (
                                                                                                         PortfolioValue.loc[
                                                                                                             list(
                                                                                                                 PortfolioValue.loc[
                                                                                                                     PortfolioValue.index.isin(
                                                                                                                         rebalancing)][
                                                                                                                 1:].index), 'Value']) / (
                                                                                                                     (
                                                                                                                     PortfolioValue.shift(
                                                                                                                         1).loc[
                                                                                                                         list(
                                                                                                                             PortfolioValue.loc[
                                                                                                                                 PortfolioValue.index.isin(
                                                                                                                                     rebalancing)][
                                                                                                                             1:].index), 'Value']) + 10000) - 1
        riskReturns = PortfolioValue['Return_temp']
    else:
        PortfolioValue.loc[
            list(PortfolioValue.loc[PortfolioValue.index.isin(rebalancing)][1:].index), 'Return_temp'] = np.nan
        riskReturns = PortfolioValue['Return_temp'].fillna(method='ffill')
    riskReturns.index = riskReturns.index.map(lambda x: pd.to_datetime(str(x)))
    # riskReturns.plot()
    # plt.show()
    monthlyReturns = (riskReturns + 1).groupby([riskReturns.index.year, riskReturns.index.month]).prod() - 1
    monthlyReturns.index.names = ["Year", "Month"]
    monthlyReturns = monthlyReturns.reset_index(level=[0, 1])
    indexList = []
    for i in range(len(monthlyReturns)):
        indexList.append(date(int(monthlyReturns.iloc[i].Year), int(monthlyReturns.iloc[i].Month), 1))
    monthlyReturns.index = indexList
    monthlyReturns.drop(["Year", "Month"], axis=1, inplace=True)
    monthlyReturns = monthlyReturns.set_index(pd.DatetimeIndex(monthlyReturns.index))
    monthlyReturns.columns = ['Returns']
    X = riskData.loc["2015-04-01":"2020-03-01"][riskData.columns]
    Y = monthlyReturns.loc["2015-04-01":"2020-03-01"]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    # print(model.summary())
    return model.params


# Convert USD to CAD
US_ticker = tickerAlts + tickerCredit + tickerEquity + tickerHedge
portfolioValue[US_ticker] = portfolioValue[US_ticker].multiply(portfolioValue['Adj Close'], axis=0)

# sub class
EQ_ticker = tickerEquity + tickerEquityCAD
CR_ticker = tickerCredit + tickerCreditCAD
Alt_ticker = tickerAlts + tickerAlts
Hedge_ticker = tickerHedge + tickerHedgeCAD

beta_portfolio = market_econ_regression(portfolioValue.iloc[:, :-7], True)
beta_EQ = market_econ_regression(portfolioValue[EQ_ticker])
beta_CR = market_econ_regression(portfolioValue[CR_ticker])
beta_Alt = market_econ_regression(portfolioValue[Alt_ticker])


# scenario simulation (boostraping)
def boostraping(econ_data, scenario='down'):
    if scenario == 'down':
        samplePool = riskData.sort_values('Unemploy', ascending=0).iloc[:10, :]
        randomSample1 = list(np.random.choice(samplePool.iloc[:, 0], 3))
        randomSample2 = list(np.random.choice(samplePool.iloc[:, 1], 3))
        randomSample3 = list(np.random.choice(samplePool.iloc[:, 2], 3))
        randomSample4 = list(np.random.choice(samplePool.iloc[:, 3], 3))
        randomSample5 = list(np.random.choice(samplePool.iloc[:, 4], 3))
        randomSample6 = list(np.random.choice(samplePool.iloc[:, 5], 3))
        # randomSample7 = list(np.random.choice(samplePool.iloc[:,6],3))
    if scenario == 'up':
        samplePool = riskData.sort_values('Unemploy').iloc[:10, :]
        randomSample1 = list(np.random.choice(samplePool.iloc[:, 0], 3))
        randomSample2 = list(np.random.choice(samplePool.iloc[:, 1], 3))
        randomSample3 = list(np.random.choice(samplePool.iloc[:, 2], 3))
        randomSample4 = list(np.random.choice(samplePool.iloc[:, 3], 3))
        randomSample5 = list(np.random.choice(samplePool.iloc[:, 4], 3))
        randomSample6 = list(np.random.choice(samplePool.iloc[:, 5], 3))
        # randomSample7 = list(np.random.choice(samplePool.iloc[:,6],3))
    # simulatedScenario = pd.DataFrame([randomSample1,randomSample2,randomSample3,randomSample4,randomSample5,randomSample6,randomSample7]).T
    simulatedScenario = pd.DataFrame(
        [randomSample1, randomSample2, randomSample3, randomSample4, randomSample5, randomSample6]).T
    simulatedScenario.columns = econ_data.columns
    return simulatedScenario


downScenario = boostraping(riskData, scenario='down')
upScenario = boostraping(riskData, scenario='up')
downScenario.insert(0, 'constant', 1)
upScenario.insert(0, 'constant', 1)

upEquity = np.dot(np.array(upScenario.iloc[:, :7]), np.array(beta_EQ))
upScenario['EQ Estimated Return'] = upEquity
upCredit = np.dot(np.array(upScenario.iloc[:, :7]), np.array(beta_CR))
upScenario['CR Estimated Return'] = upCredit
upAlt = np.dot(np.array(upScenario.iloc[:, :7]), np.array(beta_Alt))
upScenario['Alt Estimated Return'] = upAlt
upPortfolio = np.dot(np.array(upScenario.iloc[:, :7]), np.array(beta_portfolio))

downEquity = np.dot(np.array(downScenario.iloc[:, :7]), np.array(beta_EQ))
downScenario['EQ Estimated Return'] = downEquity
downCredit = np.dot(np.array(downScenario.iloc[:, :7]), np.array(beta_CR))
downScenario['CR Estimated Return'] = downCredit
downAlt = np.dot(np.array(downScenario.iloc[:, :7]), np.array(beta_Alt))
downScenario['PE Estimated Return'] = downAlt
downPortfolio = np.dot(np.array(downScenario.iloc[:, :7]), np.array(beta_portfolio))
downScenario['Portfolio Estimated Return'] = downPortfolio

upScenario
downScenario

# Risk Exposure
# EQ = portfolioValue[tickerEquity].loc[:'2020-06-01'].sum(axis=1)
# portfolioValue.loc['2020-06-01'][:-7][tickerEquity]
CR = portfolioValue[tickerCredit].loc[:'2020-06-01'].sum(axis=1)


# Hedge = portfolioValue.loc[:'2020-06-01'].iloc[:,10:13].sum(axis=1)
# PE = portfolioValue.loc[:'2020-06-01'].iloc[:,13:15].sum(axis=1)
# Alternative = portfolioValue.loc[:'2020-06-01'].iloc[:,15].sum()
# pd.DataFrame({'EQ':EQ,'CR':CR,'Hedge':Hedge,'PE':PE,'Alternative':Alternative}).pct_change().dropna().cov()

def getExposure(portfolioValue, date='2020-06-01'):
    w = portfolioValue.loc[date][:-7] / (portfolioValue.loc[date][:-7].sum())
    EQw = w[tickerEquity].sum()
    CRw = w[tickerCredit].sum()
    Alt_w = w[tickerAlts].sum()
    Hedge_w = w[tickerHedge].sum()
    EQw_CAD = w[tickerEquityCAD].sum()
    CRw_CAD = w[tickerCreditCAD].sum()
    Alt_w_CAD = w[tickerAltsCAD].sum()
    Hedge_w_CAD = w[tickerHedgeCAD].sum()
    cash = w['Cash']
    # list of strings
    lst = [EQw, CRw, Alt_w, Hedge_w, EQw_CAD, CRw_CAD, Alt_w_CAD, Hedge_w_CAD, cash]
    df = pd.DataFrame(lst,
                      index=['EQ_USD', 'CR_USD', 'Alt_USD', 'Hedge_USD', 'EQ_CAD', 'CR_CAD', 'Alt_CAD', 'Hedge_CAD',
                             'Cash'], columns=['Weight'])
    return df


exposure = getExposure(portfolioValue, '2020-06-01')
exposure
exposure['Weight'].plot.pie(autopct='%.2f', fontsize=12, figsize=(8, 8))


# Return Attribution
def getReturn(PortfolioValue, date='2020-06-01'):
    PortfolioValue = PortfolioValue.copy()
    PortfolioValue['Value'] = PortfolioValue.sum(axis=1)
    PortfolioValue['Return_temp'] = PortfolioValue['Value'].pct_change().dropna()
    # if isPortfolio:
    #     PortfolioValue.loc[list(PortfolioValue.loc[PortfolioValue.index.isin(rebalancing)][1:].index),'Return_temp']=(PortfolioValue.loc[list(PortfolioValue.loc[PortfolioValue.index.isin(rebalancing)][1:].index),'Value'])/((PortfolioValue.shift(1).loc[list(PortfolioValue.loc[PortfolioValue.index.isin(rebalancing)][1:].index),'Value'])+10000)-1
    #     riskReturns = PortfolioValue['Return_temp']
    # else:
    PortfolioValue.loc[
        list(PortfolioValue.loc[PortfolioValue.index.isin(rebalancing)][1:].index), 'Return_temp'] = np.nan
    riskReturns = PortfolioValue['Return_temp'].fillna(method='ffill')
    return riskReturns.loc[:date]


# returnAttr = getReturn(portfolioValue,date='2020-06-01')

def getReturnAttribution(portfolioValue, date='2020-06-01'):
    EQ = getReturn(portfolioValue[tickerEquity], date)[-1]
    CR = getReturn(portfolioValue[tickerCredit], date)[-1]
    Alt = getReturn(portfolioValue[tickerAlts], date)[-1]
    # Hedge = getReturn(portfolioValue[tickerHedge],date)[-1]
    EQ_CAD = getReturn(portfolioValue[tickerEquityCAD], date)[-1]
    CR_CAD = getReturn(portfolioValue[tickerCreditCAD], date)[-1]
    Alt_CAD = getReturn(portfolioValue[tickerAltsCAD], date)[-1]
    # Hedge_CAD = getReturn(portfolioValue[tickerHedgeCAD],date)[-1]
    returns = [EQ, CR, Alt, EQ_CAD, CR_CAD, Alt_CAD]
    return_sum = np.array(returns).sum()
    EQw = EQ / return_sum
    CRw = CR / return_sum
    Alt_w = Alt / return_sum
    # Hedge_w = Hedge/return_sum
    EQw_CAD = EQ_CAD / return_sum
    CRw_CAD = CR_CAD / return_sum
    Alt_w_CAD = Alt_CAD / return_sum
    # Hedge_w_CAD = Hedge_CAD/return_sum
    returns_attr = [EQw, CRw, Alt_w, EQw_CAD, CRw_CAD, Alt_w_CAD]
    name = ['EQ_USD', 'CR_USD', 'Alt_USD', 'EQ_CAD', 'CR_CAD', 'Alt_CAD']
    df = pd.DataFrame({'Returns': returns, 'Returns Attribution': returns_attr, 'Name': name}).set_index('Name')
    return df


df = getReturnAttribution(portfolioValue, date='2020-06-01')
df['Returns Attribution'].sum()
df['Returns Attribution'].plot.pie(autopct='%.2f', fontsize=12, figsize=(8, 8))


# Risk Attribution
def getRiskAttribution(portfolioValue, date='2020-06-01'):
    # portfolioValue = portfolioValue.loc[:date]
    w = getExposure(portfolioValue, date)
    # print(w)
    EQ = getReturn(portfolioValue[tickerEquity], date)
    CR = getReturn(portfolioValue[tickerCredit], date)
    Alt = getReturn(portfolioValue[tickerAlts], date)
    # Hedge = getReturn(portfolioValue[tickerHedge],date).std()
    EQ_CAD = getReturn(portfolioValue[tickerEquityCAD], date)
    CR_CAD = getReturn(portfolioValue[tickerCreditCAD], date)
    Alt_CAD = getReturn(portfolioValue[tickerAltsCAD], date)
    # Hedge_CAD = getReturn(portfolioValue[tickerHedgeCAD],date).std()
    returns = [EQ, CR, Alt, EQ_CAD, CR_CAD, Alt_CAD]
    name = ['EQ_USD', 'CR_USD', 'Alt_USD', 'EQ_CAD', 'CR_CAD', 'Alt_CAD']
    w = w.loc[name] / w.loc[name].sum()
    df = pd.DataFrame(returns).T.dropna()
    df.columns = name
    # print(df)
    Q = df.cov()
    # print(Q)
    riskAttribution = np.dot(np.array(w.T), np.array(Q))
    risk = pd.DataFrame(riskAttribution, columns=name, index=[date])
    riskAttr = risk / risk.sum(axis=1)[0]
    return riskAttr.T


riskAttribution = getRiskAttribution(portfolioValue, '2020-06-01')

riskAttribution['2020-06-01'].plot.pie(autopct='%.2f', fontsize=12, figsize=(8, 8))
# riskAttribution.applymap(lambda x:x/(riskAttribution.sum(axis=1)).values).sum(axis=1)
#######################################################################

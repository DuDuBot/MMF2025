
from pandas_datareader import data as pdr
# ! pip install quantstats --upgrade --no-cache-dir
import quantstats as qs
import strategies
from tqdm import tqdm
from datetime import datetime
from datetime import date,timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
import time
import regimeDetection
import yfinance  as yf


from strategies import MVPort
from strategies import ERCRP


def Fit_RP(price,tickerEquity,N):
    ERCEquity=ERCRP()
    z=ERCEquity.get_allocations(price[tickerEquity],N)
    wEquity=pd.DataFrame(z,columns=tickerEquity,index=price.index)
    wEquity=wEquity.shift(1)
    wEquity.replace(np.nan,1/len(tickerEquity),inplace=True)
    rtnERCEquity=(wEquity*np.log(price[tickerEquity]).diff()).sum(axis=1)
    #rtnERCEquity=rtnERCEquity.loc[pd.to_datetime('2014-12-31'):]
    nvERCEquity=np.exp(rtnERCEquity.cumsum())
    shpERCEquity=rtnERCEquity.mean()/rtnERCEquity.std()*np.sqrt(252)
    #nvERCEquity.plot()
    #plt.show()
    print('Return:',round(rtnERCEquity.mean()*252,3))
    print('Std.:  ',round(rtnERCEquity.std()*16,3))
    print('Sharpe:',round(shpERCEquity,3))
    return [rtnERCEquity,nvERCEquity,wEquity.loc[rtnERCEquity.index]]




def Fit_MSR(rf,dfMix,N):
    MVMix=MVPort(rf.loc[dfMix.index])
    o=MVMix.get_allocations(dfMix.values,N)
    wMix=pd.DataFrame(o,columns=dfMix.columns,index=dfMix.index)
    wMix=wMix.shift(1)
    wMix.replace(np.nan,1/dfMix.shape[1],inplace=True)
    rtnMVOMix=(wMix*np.log(dfMix).diff()).sum(axis=1)
    #rtnMVOMix=rtnMVOMix.loc[pd.to_datetime('2014-12-31'):]
    nvMVOMix=np.exp(rtnMVOMix.cumsum())
    shpMVOMix=(rtnMVOMix-rf.loc[dfMix.index]).mean()/rtnMVOMix.std()*np.sqrt(252)

    print('Return:',round(rtnMVOMix.mean()*252,3))
    print('Std.:  ',round(rtnMVOMix.std()*16,3))
    print('Sharpe:',round(shpMVOMix,3))
    return [rtnMVOMix,nvMVOMix,wMix.loc[rtnMVOMix.index]]

def pull_data(stocks,start=datetime(2010,1,1),end = datetime(2020,6,1)):
    price = pdr.get_data_yahoo(stocks, start=start, end=end)
    price = price["Adj Close"]
    #cad=price.iloc[:,-2]

    price = price.drop(columns = ['CAD=X','^IRX'])
    price=price.dropna()
    rtn=np.log(price.dropna()).diff().dropna()
    return price,rtn



def make_port(price,tickerEquity,tickerCredit,tickerPE):
    rtnERCEquity,nvERCEquity,wEquity=Fit_RP(price,tickerEquity,1000)
    #rtnMVOEquity,nvMVOEquity,wEquity_MVO=Fit_MSR(rf,price[tickerEquity],1000)
    rtnERCCredit,nvERCCredit,wCredit=Fit_RP(price,tickerCredit,1000)
    #rtnMVOCredit,nvMVOCredit,wCredit_MVO=Fit_MSR(rf,price[tickerCredit],1000)
    rtnERCPE,nvERCPE,wPE=Fit_RP(price,tickerPE,1000)




    dfMix=pd.DataFrame(columns=['Equity','Credit'],index=nvERCCredit.index)
    dfMix.Equity=nvERCEquity.values
    dfMix.Credit=nvERCCredit.values



    rtnERCMix,nvERCMix,wMix=Fit_RP(dfMix,dfMix.columns,1000)
    wMix=wMix*0.9
    wMix['PE']=0.1
    weightsAll=pd.concat([(wEquity.T*wMix.Equity.values).T,(wCredit.T*wMix.Credit.values).T,(wPE.T*0.1).T],axis=1)
    weightsAll=weightsAll.dropna()
    #activeClm=tickerEquity+tickerCredit+tickerPE+tickerHedge+tickerAlternative
    activeClm=tickerEquity+tickerCredit+tickerPE

    rtn=np.log(price.dropna()).diff().dropna()
    rtnTotal=(weightsAll*rtn.loc[weightsAll.index[1:],activeClm]).sum(axis=1)

    nvTotal=np.exp(rtnTotal.cumsum()).plot()
    plt.show()
    shpTotal=rtnTotal.mean()/rtnTotal.std()*16
    print('Return:',round(rtnTotal.mean()*252,3))
    print('Std.:  ',round(rtnTotal.std()*16,3))
    print('Sharpe:',round(shpTotal,3))
    return rtnTotal,nvTotal,weightsAll

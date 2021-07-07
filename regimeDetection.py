"""# Regime Detection"""


from pandas_datareader import data as pdr
# ! pip install quantstats --upgrade --no-cache-dir
import quantstats as qs
import strategies
#import fix_yahoo_finance as yf
import yfinance as yf
from tqdm import tqdm
from datetime import datetime
from datetime import date,timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from hmmlearn import hmm
from sklearn.decomposition import PCA
import time

def percentile_data(X,period=3,extend=False):
    try:
        start=X.index[0].replace(year=X.index[0].year+period)
    except:
        start=X.index[0].replace(year=X.index[0].year+period,day=28)
    startDate=X.loc[:start].index[-1]
    dfIdx=X.loc[startDate:].index
    dfPercentile=pd.DataFrame(columns=X.columns,index=dfIdx)
    if extend:
        for i in dfIdx:
            dfPercentile.loc[i]=X.loc[:i].rank(pct=True).iloc[-1]
    else:
        for i in dfIdx:
            try:
                dfPercentile.loc[i]=X.loc[i.replace(year=i.year-period):i].rank(pct=True).iloc[-1]
            except:
                dfPercentile.loc[i]=X.loc[i.replace(year=i.year-period,day=28):i].rank(pct=True).iloc[-1]
    return dfPercentile

def fix_states(states, dataHMMTemp):
    tempList=[[0,dataHMMTemp[states==0].mean()]]
    tempList.append([1,dataHMMTemp[states==1].mean()])
    tempList.append([2,dataHMMTemp[states==2].mean()])
    ary=np.array(tempList)[np.argsort(np.array(tempList)[:, 1])]
    return pd.Series(states).replace(ary[:,0],[0,1,2]).values

def desc_by_state(mySeries,states):
    #rtnM=rtnMoney[mySeries.index]
    #descIndex=['Return','Std.','Sharpe','Skewness']
    descIndex=['Return','Std.','Skewness','Kurtosis','Count','T-Count']
    descColumns=['Seeking','Neutral','Aversion']
    df=pd.DataFrame(columns=descColumns,index=descIndex)
    for i in range(3):
        tempSeries=mySeries[states==i]
        df.iloc[0,i]=tempSeries.mean()*252
        df.iloc[1,i]=tempSeries.std()*np.sqrt(252)
        df.iloc[2,i]=tempSeries.skew()
        df.iloc[3,i]=tempSeries.kurtosis()
        df.iloc[4,i]=tempSeries.size
    df.iloc[5,[0,1]]=count_transaction(list(states))
    return df.astype(float).round(5)

def desc_by_threshold(mySeries,riskIndex,threshold=[0,0.25,0.55,1]):
    #rtnM=rtnMoney[mySeries.index]
    #descIndex=['Return','Std.','Sharpe','Skewness']
    descIndex=['Return','Std.','Skewness','Kurtosis','Count']
    descColumns=['Seeking','Neutral','Aversion']
    df=pd.DataFrame(columns=descColumns,index=descIndex)
    for i in range(3):
        tempSeries=mySeries[(riskIndex>threshold[i]) & (riskIndex<=threshold[i+1])]
        df.iloc[0,i]=tempSeries.mean()*252
        df.iloc[1,i]=tempSeries.std()*np.sqrt(252)
        df.iloc[2,i]=tempSeries.skew()
        df.iloc[3,i]=tempSeries.kurtosis()
        df.iloc[4,i]=tempSeries.size
    return df.astype(float).round(5)

def count_transaction(States):
    count=0
    count_half=0
    states=[i/2 for i in States]
    for i in range(len(states)-1):
        temp=states[i+1]-states[i]
        count=count+(temp>0)+(temp<0)
        count_half=abs(temp)+count_half
    return count,count_half


def trade_by_state(mySeries0,states):
    nv=[1]
    cash=1
    pos=0
    if states[0]==0.5:
        cash=0.5
        pos=0.5*(1-0.001)
    elif states[0]==1:
        cash=0
        pos=1-0.001
    nv.append(cash+pos*mySeries0[1]+pos)

    for i in range(1,states.size-1):
        if states[i]-states[i-1]==1:
            pos=cash*(1-0.001)
            cash=0
        elif states[i]-states[i-1]==0.5:
            if pos==0:
                pos=cash/2*(1-0.001)
                cash=cash/2
            else:
                pos=pos+cash*(1-0.001)
                cash=0
        elif states[i]-states[i-1]==-0.5:
            if cash==0:
                cash=pos/2*(1-0.001)
                pos=pos/2
            else:
                cash=cash+pos*(1-0.001)
                pos=0
        elif states[i]-states[i-1]==-1:
            cash=pos*(1-0.001)
            pos=0
        pos=pos*(1+mySeries0.iloc[i+1])
        nv.append(cash+pos)
    return pd.Series([i for i in nv],index=mySeries0.index).astype(float)


def rtn_by_year(rtnSeries):
    yearS=rtnSeries.index[0].year
    yearE=rtnSeries.index[-1].year
    descColumns=['Return','Std.','Sharpe','Skewness','MDD']
    df=pd.DataFrame(index=range(yearS,yearE+1),columns=descColumns)
    for i in df.index:
        tempData=rtnSeries.loc[rtnSeries.index.year==i]
        df.loc[i,'Return']=tempData.mean()*252
        df.loc[i,'Std.']=tempData.std()*np.sqrt(252)
        if df.loc[i,'Return']==0 or df.loc[i,'Std.']==0:
            df.loc[i,'Sharpe']=0
        else:
            df.loc[i,'Sharpe']=df.loc[i,'Return']/df.loc[i,'Std.']
        df.loc[i,'Skewness']=tempData.skew()
        tempCumValue=np.exp(tempData.cumsum())
        df.loc[i,'MDD']=min([(tempCumValue[j]-tempCumValue[:j].max())/tempCumValue[:j].max() for j in tempCumValue.index])
        #df.loc[i,'']
    df.loc['Overall']=np.nan
    df.iloc[-1,0]=rtnSeries.mean()*252
    df.iloc[-1,1]=rtnSeries.std()*np.sqrt(252)
    df.iloc[-1,2]=df.iloc[-1,0]/df.iloc[-1,1]
    cumValue=np.exp(rtnSeries.cumsum())
    df.iloc[-1,3]=rtnSeries.skew()
    df.iloc[-1,4]=min([(cumValue[j]-cumValue[:j].max())/cumValue[:j].max() for j in cumValue.index])
    return df
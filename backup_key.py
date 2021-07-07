# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:21:57 2020

@author: keyfe
"""


tickerEquity=['XLY','XLI','XLF','XLV','XLK','XLP']
# Consumer Discritionary, Industrial, Financial, Health Care,Technology,
# Consumer Staples
tickerCredit=["EMB","HYG",'LQD','MBB']
# EMD, HY, IG, MBS, Loan BKLN not enough history
tickerHedge=["GLD",'IEF']
tickerPE=['PSP','IGF','VNQ']
# PE. Infra, REITs
tickerAlternative=['MNA']
tickerBM=['SPY','HYG']
stocks = tickerEquity+tickerCredit+tickerPE+tickerHedge+tickerAlternative+["SPY","CAD=X","^IRX"]

start = datetime(2010,1,1)
end = datetime(2020,12,12)

price = pdr.get_data_yahoo(stocks, start=start, end=end)
price = price["Adj Close"]
price=price.dropna(how='any')
price = price.drop(columns = ['CAD=X','^IRX'])

rf = price.iloc[1:,-1]/252/100
cad=price.iloc[1:,-2]/252
cashValue=(1+rf).cumprod()
cashValue=cashValue.fillna(method='bfill')

rtn=np.log(price).diff().dropna()

clmns='EQ1,EQ2,EQ3,EQ4,EQ5,EQ6,CR1,CR2,CR3,CR4,Gold,Bond,PE,Inf,REITs,HF,SPY'.split(',')
dataIdx=price.index.values
priceNamed=pd.DataFrame(price.values,index=dataIdx,columns=clmns).dropna()
rtnNamed=priceNamed.pct_change().dropna()


rtnBM=(rtn[['SPY','LQD']]*np.array([0.6,0.4])).sum(axis=1)

rtnBM=rtnBM.loc[pd.to_datetime('2015-01-01'):]
nvBM=np.exp(rtnBM.cumsum())
shpBM=rtnBM.mean()/rtnBM.std()*16

rtnERCEquity,nvERCEquity,wEquity=utilityFuncs.Fit_RP(price,tickerEquity,1000)
rtnERCCredit,nvERCCredit,wCredit=utilityFuncs.Fit_RP(price,tickerCredit,1000)
rtnERCPE,nvERCPE,wPE=utilityFuncs.Fit_RP(price,tickerPE,1000)

dfMix=pd.DataFrame(columns=['Equity','Credit','PE'],index=nvERCCredit.index)
dfMix.Equity=nvERCEquity.values
dfMix.Credit=nvERCCredit.values
dfMix.PE=nvERCPE.values
rtnERCMix,nvERCMix,wMix=utilityFuncs.Fit_RP(dfMix,dfMix.columns,1000)

weightsAll=pd.concat([(wEquity.T*wMix.Equity.values).T,(wCredit.T*wMix.Credit.values).T,(wPE.T*wMix.PE.values).T],axis=1)
weightsAll=weightsAll*0.8

for i in tickerHedge:
    weightsAll[i]=0.075
    
weightsAll['MNA']=0.05

activeClm=tickerEquity+tickerCredit+tickerPE+tickerHedge+tickerAlternative
rtnTotal=(weightsAll*rtn.loc[weightsAll.index[1:],activeClm]).sum(axis=1)

shpTotal=rtnTotal.mean()/rtnTotal.std()*16
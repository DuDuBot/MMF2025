import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copulas.multivariate import VineCopula

np.random.seed(0)
sns.set()


def risk_model_outputs(benchmark_data, port_value, alts, credit, equity, alts_cad, credit_cad, equity_cad):
    macro_data = pd.read_csv('./data_2/MacroData.csv', index_col='DATE', parse_dates=True)
    macro_data = macro_data.loc[macro_data.index >= '2000-01-01'].iloc[:-2, :]
    macro_data = macro_data.iloc[:, :-1]
    macro_data = macro_data.applymap(lambda x: float(x)).dropna()
    credit_risk_premium = (macro_data['BAMLC0A4CBBBEY'] - macro_data['BAMLC0A1CAAAEY'])
    inflation = macro_data['CPIAUCSL'].pct_change().dropna() * 100 * 12
    industrial_growth = macro_data['INDPRO'].pct_change().dropna() * 100
    risk_output = pd.DataFrame(inflation).join(industrial_growth).join(macro_data.iloc[:, 2:7]).join(
        macro_data.iloc[:, 9:])
    risk_output['Credit Premium'] = credit_risk_premium
    risk_output.columns = ['Inflation', 'Industrial ProdGrowth', 'T-Bill', 'Oil', 'Libor', 'House', 'Unemploy',
                           '10 Yield curve', 'Term Premium', '5 Yield Curve', '2 Yield Curve', '1 Yield Curve',
                           'Credit Premium']
    risk_output['Unexpected Inflation'] = (risk_output['Inflation'] - risk_output['Inflation'].shift(1)) - (
                risk_output['T-Bill'].shift(1) - risk_output['T-Bill'].shift(2))
    risk_output['Yield spread'] = risk_output['10 Yield curve'] - risk_output['T-Bill']
    risk_output = risk_output.dropna()
    risk_output = risk_output[
        ['Industrial ProdGrowth', 'Credit Premium', '10 Yield curve', 'T-Bill', 'Yield spread', '5 Yield Curve',
         '2 Yield Curve', '1 Yield Curve']]

    input_1 = risk_output[['5 Yield Curve', '2 Yield Curve', '1 Yield Curve', 'T-Bill', '10 Yield curve']]
    input_1 = StandardScaler().fit_transform(input_1)

    pca = PCA(n_components=1)
    pc = pca.fit_transform(input_1)
    pca_df = pd.DataFrame(data=pc, columns=['principal component'])
    risk_output = risk_output[['Industrial ProdGrowth', 'Credit Premium', 'Yield spread']]
    risk_output['Yield Curve PCA'] = pca_df.values
    partial_risk_data = risk_output.loc['5-2016':'4-2021']
    partial_risk_data = partial_risk_data[
        ['Industrial ProdGrowth', 'Credit Premium', 'Yield spread', 'Yield Curve PCA']]

    SPY_data = pd.read_csv('./data/SPY.csv', index_col='Date', parse_dates=True)['Adj Close']
    monthly_spy = SPY_data.resample('m').last()
    monthly_spy = monthly_spy.pct_change().dropna()

    partial_risk_data['SP500 Return'] = list(monthly_spy.loc['5-2016':'4-2021'])
    partial_risk_data['Port_Returns'] = list(benchmark_data.loc['5-2016':'4-2021'].Port_Returns.dropna())

    X = partial_risk_data[['Industrial ProdGrowth', 'Credit Premium', 'Yield Curve PCA',
                           'Yield spread', 'SP500 Return']]
    Y = partial_risk_data['Port_Returns']
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()

    us_assets = alts + credit + equity
    port_value[us_assets] = port_value[us_assets].multiply(port_value['CAD=X'], axis=0)

    # Normal scenario
    normal_scenario = partial_risk_data.median()
    up_1 = pd.DataFrame(normal_scenario).T.copy()
    up_2 = pd.DataFrame(normal_scenario).T.copy()
    up_3 = pd.DataFrame(normal_scenario).T.copy()
    up_4 = pd.DataFrame(normal_scenario).T.copy()
    up_5 = pd.DataFrame(normal_scenario).T.copy()
    up_best = pd.DataFrame([2, 2, 1, -3, 0.15, 0]).T
    up_best.columns = up_5.columns

    up_1['SP500 Return'] = 0.15
    up_2['Industrial ProdGrowth'] = 2
    up_3['Credit Premium'] = 2
    up_4['Yield spread'] = 1
    up_5['Yield Curve PCA'] = -3
    up = pd.concat([up_1, up_2, up_3, up_4, up_5, up_best], axis=0)
    up = up.iloc[:, 0:5]

    down_1 = pd.DataFrame(normal_scenario).T.copy()
    down_2 = pd.DataFrame(normal_scenario).T.copy()
    down_3 = pd.DataFrame(normal_scenario).T.copy()
    down_4 = pd.DataFrame(normal_scenario).T.copy()
    down_5 = pd.DataFrame(normal_scenario).T.copy()

    down_1['SP500 Return'] = -0.25
    down_2['Industrial ProdGrowth'] = -10
    down_3['Credit Premium'] = -2
    down_4['Yield spread'] = -2
    down_5['Yield Curve PCA'] = 1
    down_worst = pd.DataFrame([-10, -2, -2, 1, -0.25, 0]).T  # worst scenario
    down_worst.columns = down_5.columns
    down = pd.concat([down_1, down_2, down_3, down_4, down_5, down_worst], axis=0)
    down = down.iloc[:, 0:5]

    down.insert(0, 'constant', 1)
    up.insert(0, 'constant', 1)
    port_up = np.dot(np.array(up), np.array(model.params))
    up['Portfolio Estimated Return'] = port_up
    port_down = np.dot(np.array(down), np.array(model.params))
    down['Portfolio Estimated Return'] = port_down

    data = partial_risk_data[partial_risk_data.columns[:-1]]
    copula = VineCopula('regular')
    copula.fit(data)
    smps = copula.sample(10000)
    smps.insert(0, 'constant', 1)
    smpl = np.dot(np.array(smps), np.array(model.params))
    smps['Portfolio Estimated Return'] = smpl
    up_1 = smps.sort_values('Portfolio Estimated Return', ascending=0).iloc[:3, :]
    down_1 = smps.sort_values('Portfolio Estimated Return', ascending=1).iloc[:3, :]

    return up, down, up_1, down_1

from scipy import stats
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import helper_functions
import quantstats as qs

np.random.seed(0)
sns.set()


def read_data_calculator(port_value, rebalance):
    port_rtn = port_value.Return.to_frame()
    port_rtn.index = pd.to_datetime(port_rtn.index)

    SPY_data = pd.read_csv('./data/SPY.csv', index_col=0)
    SPY_data['SPY'] = SPY_data['Adj Close'].pct_change()
    SPY_data = SPY_data['SPY'].to_frame()
    SPY_data.index = pd.to_datetime(SPY_data.index)

    rf = pd.read_csv('./data/^IRX.csv', index_col=0)
    rf['^IRX'] = rf['Adj Close'] / 36500
    rf = rf['^IRX'].to_frame()
    rf.index = pd.to_datetime(rf.index)

    data_process = SPY_data.merge(port_rtn, left_index=True, right_index=True).dropna()
    data_process = data_process.merge(rf, left_index=True, right_index=True).dropna()

    corr_SPY = round(np.corrcoef(data_process.SPY, data_process.Return)[0][1], 3)
    print('Correlation to SP500: ', corr_SPY)
    print('Kurtosis: ', round(stats.kurtosis(data_process.Return), 3))
    print('Skewness: ', round(stats.skew(data_process.Return), 3))
    vol = np.std(data_process.Return) * np.sqrt(252)
    print('Volatility: ', round(vol * 100, 3))
    excess_rtn = data_process.Return - data_process['^IRX']
    sharpe = np.mean(excess_rtn) / np.std(excess_rtn)
    print('Sharpe Ratio: ', round(sharpe * np.sqrt(252), 3))
    max_draw_down = qs.stats.max_drawdown(port_rtn).values[0]
    print('Max Drawdown in %: ', round(max_draw_down * 100, 3))
    compound_annual_growth_rate = qs.stats.cagr(port_rtn).values[0]
    print('CAGR in %: ', round(compound_annual_growth_rate * 100, 3))
    print('Sortino Ratio: ', round(qs.stats.sortino(data_process.Return), 3))

    print('[- VaR Calculations -]')

    alpha = 0.01
    miu = (np.mean(port_value.Return) + 1) ** 252 - 1
    sigma = np.std(port_value.Return) * np.sqrt(252)

    h = 1.  # horizon of 1 days
    miu_h = miu * (h / 252)
    sigma_h = sigma * np.sqrt(h / 252)

    value_at_risk = norm.ppf(1 - alpha) * sigma_h - miu_h
    value_pct = port_value.iloc[-1].Value_CAD * value_at_risk
    print('99% 1 day VaR: ', round(value_at_risk * 100, 2), "% or $", round(value_pct))

    cvar = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * sigma_h - miu_h
    value_pct = port_value.iloc[-1].Value_CAD * cvar
    print('99% 1 day CVaR/ES: ', round(cvar * 100, 2), "% or $", round(value_pct))

    h = 10.  # horizon of 10 days
    miu_h = miu * (h / 252)
    sigma_h = sigma * np.sqrt(h / 252)

    value_at_risk = norm.ppf(1 - alpha) * sigma_h - miu_h
    value_pct = port_value.iloc[-1].Value_CAD * value_at_risk
    print('99% 10 day VaR: ', round(value_at_risk * 100, 2), "% or $", round(value_pct))

    cvar = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * sigma_h - miu_h
    value_pct = port_value.iloc[-1].Value_CAD * cvar
    print('99% 10 day CVaR/ES: ', round(cvar * 100, 2), "% or $", round(value_pct))

    print('[- Returns - Period Wise -]')
    time_weighted_return = []
    for i in range(len(rebalance) - 1):
        i_1 = (port_value.index.get_loc(rebalance[i]))
        i_2 = (port_value.index.get_loc(rebalance[i + 1]))
        if i != 0:
            irr = round(100 * ((port_value.Value_CAD.iloc[i_2 - 1]
                                / port_value.Value_CAD.iloc[i_1 - 1]) - 1), 2)
            print("Period", i + 1, ": ", irr, "%")
        else:
            irr = round(100 * ((port_value.Value_CAD.iloc[i_2 - 1] / port_value.Value_CAD.iloc[i_1]) - 1), 2)
            print("Period", i + 1, ": ", irr, "%")
        time_weighted_return.append(irr / 100)
    print("Time Weighted Return:", (np.prod(np.add(time_weighted_return, 1)) - 1) * 100, "%")


def transaction_cost_calculator(port_value, rebalance):
    trans_cost = 0
    for i in range(len(rebalance)):
        rebalance_date = rebalance[i]
        if i == 0:
            trans_cost = port_value.loc[rebalance_date].Value_CAD * 0.001
        else:
            i_1 = (port_value.index.get_loc(rebalance_date))
            i_2 = i_1 - 1
            v_1 = port_value.iloc[i_1]
            v_2 = port_value.iloc[i_2]
            trans_cost += (sum((abs(v_1 - v_2)[port_value.columns[:25]])) * 0.001)
    print("Transaction Costs: $", round(trans_cost, 2))


def benchmark_comparison(port_value):
    hm_data = pd.read_csv("./data_2/Benchmark.csv", index_col=0, parse_dates=True).sort_index().loc["03/2016"
                                                                                                    :"05/2021"]
    hm_data.drop(['Name', 'Code', 'Return'], axis=1, inplace=True)
    hm_data.index = pd.to_datetime(hm_data.index)

    month_value = [100000]
    port_value.index = pd.to_datetime(port_value.index)
    for i in range(2016, 2022):
        for j in range(1, 13):
            try:
                monthly_data = port_value.loc[str(j) + "-" + str(i)].Value_CAD
                month_value.append(monthly_data.iloc[-1])
            except:
                pass

    hm_data['Portfolio'] = month_value[:-1]
    hm_data['Port_Returns'] = hm_data['Portfolio'].pct_change()
    hm_data['Bench_Returns'] = hm_data['Index'].pct_change()

    for i in range(2016, 2022):
        for j in [4, 10]:
            if (j == 4 and i == 2016) or (j == 10 and i == 2021):
                continue
            else:
                i_1 = hm_data.loc[str(j) + "-" + str(i)].index[0]
                i_2 = hm_data.index.get_loc(i_1)
                calculated_rtn = ((hm_data.iloc[i_2].Portfolio) / (hm_data.iloc[i_2 - 1].Portfolio + 10000)) - 1
                hm_data.loc[i_1, 'Port_Returns'] = calculated_rtn

    plt.plot(hm_data["Port_Returns"].iloc[2:].cumsum() * 100, label="Portfolio")
    plt.plot(hm_data["Bench_Returns"].iloc[2:].cumsum() * 100, label="Benchmark")
    plt.xlabel("Years")
    plt.ylabel("Returns")
    plt.legend()
    plt.title("Performance of Portfolio vs Benchmark")
    plt.show()

    tracking_error = np.std(hm_data["Port_Returns"] - hm_data["Bench_Returns"]) * np.sqrt(12)
    info_ratio = (qs.stats.cagr(hm_data["Port_Returns"]) - qs.stats.cagr(hm_data["Bench_Returns"])) \
                       / tracking_error
    print("Tracking Error: ", round(tracking_error, 3))
    print("Information Ratio: ", round(info_ratio, 3))

    SPY_data = pd.read_csv('./data/SPY.csv', index_col=0)
    SPY_data['SPY'] = SPY_data["Adj Close"]
    SPY_data = SPY_data['SPY'].to_frame()
    SPY_data.index = pd.to_datetime(SPY_data.index)
    SPY_data = SPY_data.loc['2016-03-01':'2021-05-31']
    SPY_monthly = SPY_data.resample('m').last()
    SPY_monthly = SPY_monthly.pct_change().dropna()
    (beta, alpha) = stats.linregress(list(SPY_monthly.SPY), list(hm_data.Port_Returns.dropna()))[0:2]
    print("Beta:", round(beta, 2))
    print("Alpha:", round(alpha * 12 * 100, 3))
    return hm_data


def get_portfolio_stats_and_graph(returns):
    qs.reports.full(returns)


def get_USD_CAD_exposure(port_value):
    ccy = ["CAD ", "USD"]
    exposure = np.vstack([port_value["CADTickers"] / port_value["Value_CAD"],
                   (1 - (port_value["CADTickers"] / port_value["Value_CAD"]))])
    plt.stackplot(port_value.index, exposure, labels=ccy)
    plt.title("USD/CAD Exposure")
    plt.legend()


def show_weights_changed(port_value, equity_name, equity_name_cad,
                         credit_name, credit_name_cad, alts_name, alts_names_cad):
    cate = ["Equity ", "Credit", "Alternatives", "Cash"]
    equity_value = ((port_value[equity_name_cad].sum(axis=1)) +
                    ((port_value[equity_name].multiply(port_value['CAD=X'],
                                                      axis=0).sum(axis=1)))).divide(port_value["Value_CAD"])
    credit_value = ((port_value[credit_name_cad].sum(axis=1)) +
                   ((port_value[credit_name].multiply(port_value['CAD=X'],
                                                      axis=0).sum(axis=1)))).divide(port_value["Value_CAD"])
    alts_value = ((port_value[alts_names_cad].sum(axis=1)) +
                 ((port_value[alts_name].multiply(port_value['CAD=X'],
                                                  axis=0).sum(axis=1)))).divide(port_value["Value_CAD"])
    cash_value = (port_value["Cash"]).divide(port_value["Value_CAD"])
    ax = plt.subplot(111)

    y = np.vstack([equity_value, credit_value, alts_value, cash_value])
    plt.stackplot(port_value.index, y, labels=cate)
    plt.title("Asset Class Exposure")
    ax.legend(loc='right', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, shadow=True)
    plt.show()


def get_nv_result(port_value, signals, equity_name, credit_name, alts_name, equity_ind_usd, equity_ind_cad,
                  credit_ind_usd, credit_ind_cad, alts_ind_usd, alts_ind_cad):
    print("---Exposures---")
    port_value.index = pd.to_datetime(port_value.index)
    print("Date: ", port_value.index[signals].date())
    print("Portfolio Value: ", round(port_value["Value_CAD"].iloc[signals]))

    nv_equity = port_value[equity_name].iloc[signals]
    nv_equity.index = equity_ind_usd + equity_ind_cad
    print("Equity Exposure: ", sum(round(nv_equity)))
    helper_functions.display_fcn(round(nv_equity).to_string())

    nv_credit = port_value[credit_name].iloc[signals]
    nv_credit.index = credit_ind_usd + credit_ind_cad
    print("Credit Exposure: ", sum(round(nv_credit)))
    helper_functions.display_fcn(round(nv_credit).to_string())

    nv_alts = port_value[alts_name].iloc[signals]
    nv_alts.index = alts_ind_usd + alts_ind_cad
    print("Alternatives Exposure: ", sum(round(nv_alts)))
    helper_functions.display_fcn(round(nv_alts).to_string())

    nv_cash = port_value["Cash"].iloc[signals]
    print("Cash: ", round(nv_cash))


def get_exposure_calculator(port_value, equity, equity_cad, credit, credit_cad, alts, alts_cad, date='2021-06-01'):
    exposure = port_value.loc[date][:-7] / (port_value.loc[date][:-7].sum())
    w_eq = exposure[equity].sum()
    w_cr = exposure[credit].sum()
    w_alt = exposure[alts].sum()
    w_eq_cad = exposure[equity_cad].sum()
    w_cr_cad = exposure[credit_cad].sum()
    w_alt_cad = exposure[alts_cad].sum()
    cash = exposure['Cash']

    lst = [w_eq, w_cr, w_alt, w_eq_cad, w_cr_cad, w_alt_cad, cash]
    df = pd.DataFrame(lst, index=['Equity_USD', 'Credit_USD', 'Alternative_USD',
                                  'Equity_CAD', 'Credit_CAD', 'Alternatives_CAD', 'Cash'], columns=['Weight'])
    return df


def get_attribution(port_value, rb, equity, equity_cad, credit, credit_cad, alts, alts_cad):
    v_p = port_value.copy()

    rb = list(rb)
    rb.append('2021-07-01')

    names = equity + credit + alts + equity_cad + credit_cad + alts_cad
    names.append('Cash')

    add_ons = [(v_p.loc[rb[i]:rb[i + 1], names].iloc[-2] -
                 v_p.loc[rb[i]:rb[i + 1], names].iloc[0]) for i in range(len(rb) - 1)]
    add_ons = pd.DataFrame(add_ons)
    attribution_rtn = pd.Series()
    attribution_rtn['US Equity'] = add_ons[equity].sum().sum()
    attribution_rtn['CAD Equity'] = add_ons[equity_cad].sum().sum()
    attribution_rtn['US Credit'] = add_ons[credit].sum().sum()
    attribution_rtn['CAD Credit'] = add_ons[credit_cad].sum().sum()
    attribution_rtn['US Alternative'] = add_ons[alts].sum().sum()
    attribution_rtn['CAD Alternative'] = add_ons[alts_cad].sum().sum()
    result = attribution_rtn * (v_p.Value_CAD[-1] - 200000) / attribution_rtn.sum()
    return result


def get_risk_attribution(break_down, break_down_cad, w):
    returns = [break_down[0], break_down[1], break_down[2], break_down_cad[0], break_down_cad[1], break_down_cad[2]]
    name = ['Equity_USD', 'Credit_USD', 'Alternative_USD', 'Equity_CAD', 'Credit_CAD', 'Alternatives_CAD']
    w = w.loc[name] / w.loc[name].sum()

    rtn = pd.DataFrame(returns).T.dropna()
    rtn.columns = name
    temp = rtn.iloc[-1000:].cov()
    risk_attribution = np.dot(np.array(w.T), np.array(temp))
    risk = pd.DataFrame(risk_attribution, columns=name, index=['Risk Attribution'])
    result = risk / risk.sum(axis=1)[0]
    result = result.T
    result = result * w.loc[result.index].values
    result = result / result.sum()
    result.index = ['US Equity', 'US Credit', 'US Alternative', 'CAD Equity', 'CAD Credit', 'CAD Alternative']
    return result

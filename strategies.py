# -*- coding: utf-8 -*-

import yfinance as yf
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.stats import norm
from datetime import datetime
from datetime import date,timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()


class MVPort:
    """
    Expected Shortfall (CVaR) portfolio
    """
    def __init__(self, rtnM):
        self.rtnM = rtnM

    def objection_error(self, weight, args):
        miu = args[0]
        cov = args[1]
        total_risk_of_portfolio = self.portfolio_std(weight, cov)
        total_return_of_portfolio = (weight*miu).sum()
        error=(self.rtnM.mean()-total_return_of_portfolio*252)/total_risk_of_portfolio/16
        # error=(np.array(rf*252)-total_return_of_portfolio*252)/total_risk_of_portfolio/16
        return error

    def get_signal(self, timeseries, initial_weights, tol = 1e-10):
        miu, cov = self.calculate_miu_cov(timeseries)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                        {'type': 'ineq', 'fun': lambda x: x})

        optimize_result = minimize(fun=self.objection_error,
                                    x0=initial_weights,
                                    args=[miu, cov],
                                    method='SLSQP',
                                    constraints=constraints,
                                    tol=tol,
                                    options={'disp': False})

        weight = optimize_result.x
        return weight

    def get_allocations(self, timeseries, rolling_window = 24):
        allocations = np.zeros(timeseries.shape)*np.nan
        initial_weights = [1 / timeseries.shape[1]] * timeseries.shape[1]
        for i in tqdm(range(rolling_window, timeseries.shape[0])):
            allocations[i,] = self.get_signal(timeseries.iloc[i-rolling_window:i+1,], initial_weights)
        return allocations

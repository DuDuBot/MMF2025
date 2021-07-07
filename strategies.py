# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:51:05 2020

@author: pulki
"""

#import fix_yahoo_finance as yf
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

    def price_to_log_return(self, timeseries):

        log_return = np.diff(np.log(timeseries), axis=0)
        return log_return

    def calculate_miu_cov(self, timeseries):
        log_return = self.price_to_log_return(timeseries)
        miu = np.mean(log_return, axis=0)
        cov = np.cov(log_return.T)
        return miu, cov

    def portfolio_std(self,  weight, cov):
        return np.sqrt(np.dot(np.dot(weight, cov), weight.T))

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
    
    
    
    
"""## Portfolio Construction"""

class ERCRP:
    """
    Simple Equal-Risk Contribution (ERC) portfolio
    """
    def __init__(self):
        pass

    def price_to_log_return(self, timeseries):
        """
        timeseries: M * N numpy ndarray
        return: M-1 * N numpy ndarray
        """
        log_return = np.diff(np.log(timeseries), axis=0)
        return log_return
    
    def calculate_cov_matrix(self, timeseries):
        log_return = self.price_to_log_return(timeseries)
        cov = np.cov(log_return.T)
        return cov

    def portfolio_risk(self, weight, cov):
        """
        weight: 1 * N numpy ndarray
        cov: N * N numpy ndarray
        return a float
        """
        total_risk_of_portfolio = np.sqrt(np.dot(np.dot(weight, cov), weight.T))
        return total_risk_of_portfolio
    
    def marginal_risk_contribution(self, weight, cov):
        """
        return a N * 1 numpy ndarray
        """
        ratio = np.dot(cov, weight.T) / self.portfolio_risk(weight, cov)
        risk_contribution = np.multiply(weight.T, ratio)
        return risk_contribution
    
    def objection_error(self, weight, args):
        cov = args[0]
        risk_target_percent = args[1]
        total_risk_of_portfolio = self.portfolio_risk(weight, cov)
        risk_contribution = self.marginal_risk_contribution(weight, cov)
        risk_target = np.multiply(risk_target_percent, total_risk_of_portfolio)
        error = np.sum(np.square(risk_contribution - risk_target))
        return error
    
    def get_signal(self, timeseries, initial_weights, risk_target_percent, tol = 1e-14):
        cov = self.calculate_cov_matrix(timeseries)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                        {'type': 'ineq', 'fun': lambda x: x})

        optimize_result = minimize(fun=self.objection_error,
                                    x0=initial_weights,
                                    args=[cov, risk_target_percent],
                                    constraints=constraints,
                                    tol=tol,
                                    method='SLSQP',
                                    options={'disp': False})

        weight = optimize_result.x
        #print(weight)
        return weight

    def get_allocations(self, timeseries, rolling_window = 24):
        allocations = np.zeros(timeseries.shape)*np.nan
        initial_weights = [1 / timeseries.shape[1]] * timeseries.shape[1]
        risk_target_percent = [1 / timeseries.shape[1]] * timeseries.shape[1]
        for i in tqdm(range(rolling_window, timeseries.shape[0])):
            allocations[i,] = self.get_signal(timeseries[i-rolling_window:i+1], initial_weights,
                                                risk_target_percent)
        return allocations
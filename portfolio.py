from scipy.optimize import minimize
from tqdm import tqdm
import numpy as np
import seaborn as sns

np.random.seed(0)
sns.set()

constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
               {'type': 'ineq', 'fun': lambda x: x}]


class CvarPortfolio:
    def __init__(self, return_matrix):
        self.return_matrix = return_matrix

    def mean_and_cov(self, time_series):
        log_return = np.diff(np.log(time_series), axis=0)
        return np.mean(log_return, axis=0), np.cov(log_return.T)

    def objection_error(self, weight, args):
        mean = args[0]
        cov = args[1]
        total_risk = np.sqrt(np.dot(np.dot(weight, cov), weight.T))
        total_return = (weight * mean).sum()
        error = (self.return_matrix.mean() - total_return * 252) / total_risk / 16
        return error

    def get_signal(self, time_series, initial_weights, tol=1e-10):
        mean, cov = self.mean_and_cov(time_series)

        optimize_result = minimize(fun=self.objection_error,
                                   x0=initial_weights,
                                   args=[mean, cov],
                                   method='SLSQP',
                                   constraints=constraints,
                                   tol=tol,
                                   options={'disp': False})

        return optimize_result.x

    def get_allocations(self, time_series, rolling_window=24):
        allocations = np.zeros(time_series.shape) * np.nan
        initial_weights = [1 / time_series.shape[1]] * time_series.shape[1]

        for i in tqdm(range(rolling_window, time_series.shape[0])):
            allocations[i, ] = self.get_signal(time_series.iloc[i - rolling_window:i + 1, ], initial_weights)

        return allocations


class ErcPortfolio:
    def __init__(self):
        pass

    def cov_matrix(self, time_series):
        log_return = np.diff(np.log(time_series), axis=0)
        return np.cov(log_return.T)

    def objection_error(self, weight, args):
        cov = args[0]
        risk_target_percent = args[1]

        total_risk = np.sqrt(np.dot(np.dot(weight, cov), weight.T))
        ratio = np.dot(cov, weight.T) / total_risk
        risk_contribution = np.multiply(weight.T, ratio)

        risk_target = np.multiply(risk_target_percent, total_risk)
        error = np.sum(np.square(risk_contribution - risk_target))
        return error

    def get_signal(self, time_series, initial_weights, risk_target_percent, tol=1e-14):
        cov = self.cov_matrix(time_series)

        optimize_result = minimize(fun=self.objection_error,
                                   x0=initial_weights,
                                   args=[cov, risk_target_percent],
                                   constraints=constraints,
                                   tol=tol,
                                   method='SLSQP',
                                   options={'disp': False})

        return optimize_result.x

    def get_allocations(self, time_series, rolling_window=24):
        allocations = np.zeros(time_series.shape) * np.nan
        initial_weights = [1 / time_series.shape[1]] * time_series.shape[1]
        risk_target_percent = [1 / time_series.shape[1]] * time_series.shape[1]

        for i in tqdm(range(rolling_window, time_series.shape[0])):
            allocations[i, ] = self.get_signal(time_series[i - rolling_window:i + 1], initial_weights,
                                              risk_target_percent)

        return allocations

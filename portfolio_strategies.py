from scipy.optimize import minimize
from tqdm import tqdm
import numpy as np
import seaborn as sns

np.random.seed(0)
sns.set()


class ERC:

    def __init__(self):
        pass

    def price_to_log_return(self, ts):
        log_return = np.diff(np.log(ts), axis=0)
        return log_return

    def calculate_cov_matrix(self, ts):
        log_return = self.price_to_log_return(ts)
        cov = np.cov(log_return.T)
        return cov

    def calculate_portfolio_risk(self, weight, cov):
        portfolio_risk = np.sqrt(np.dot(np.dot(weight, cov), weight.T))
        return portfolio_risk

    def marginal_risk_contribution(self, weight, cov):
        ratio = np.dot(cov, weight.T) / self.calculate_portfolio_risk(weight, cov)
        risk_contribution = np.multiply(weight.T, ratio)
        return risk_contribution

    def objection_error(self, weight, args):
        cov = args[0]
        risk_target_percent = args[1]
        portfolio_risk = self.calculate_portfolio_risk(weight, cov)
        marginal_risk = self.marginal_risk_contribution(weight, cov)
        risk_target = np.multiply(risk_target_percent, portfolio_risk)
        error = np.sum(np.square(marginal_risk - risk_target))
        return error

    def get_optimized_weight(self, ts, starting_weights, tgt_percent, tol=1e-14):
        const = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                       {'type': 'ineq', 'fun': lambda x: x})

        optimized_result = minimize(fun=self.objection_error,
                                    x0=starting_weights,
                                    args=[self.calculate_cov_matrix(ts), tgt_percent],
                                    constraints=const,
                                    tol=tol,
                                    method='SLSQP',
                                    options={'disp': False})
        return optimized_result.x

    def get_allocations(self, ts, rolling_wd=24):
        allocation_result = np.zeros(ts.shape) * np.nan
        starting_weights = [1 / ts.shape[1]] * ts.shape[1]
        risk_target_pert = [1 / ts.shape[1]] * ts.shape[1]

        for i in tqdm(range(rolling_wd, ts.shape[0])):
            allocation_result[i,] = self.get_optimized_weight(ts[i - rolling_wd: i + 1],
                                                              starting_weights, risk_target_pert)
        return allocation_result


class MomentumERC(ERC):

    def __init__(self):
        super().__init__()

    def calculate_cov_matrix(self, ts):
        log_return = self.price_to_log_return(ts)
        log_return_mean = log_return.mean(axis=0)
        log_return_quantile = np.quantile(log_return_mean, 0.49)
        for i in range(log_return.shape[1]):
            if log_return_mean[i] > log_return_quantile:
                log_return[:, i] = (log_return[:, i] - log_return_mean[i]) * 2 + log_return_mean[i]
            else:
                log_return[:, i] = (log_return[:, i] - log_return_mean[i]) * 0.5 + log_return_mean[i]
        cov = np.cov(log_return.T)
        return cov

from pandas import DataFrame
import numpy as np
from scipy.optimize import curve_fit
from scipy import log
import pwlf

# Base Regressor. Do not use. Use Derived Regressors.
class Regressor:
    def __init__(self):
        self.regression_func = None
        self.parameter_names: list[str] = None
        self.clustering_parameters: list[str] = None

    def set_clustering_parameters(self, clustering_parameters: list[str]):
        if clustering_parameters is None:
            self.clustering_parameters = self.parameter_names.copy()
            return
        for param in clustering_parameters:
            if param in self.parameter_names:
                self.clustering_parameters.append(param)

    def fit_row(self, x, y): # x,y are np.array
        result = {}
        params_opt, pcov = curve_fit(self.regression_func, x, y)
        result['parameters'] = params_opt

        # r-squared
        residual = y - self.regression_func(x, *params_opt)            
        y_mean = np.sum(y)/len(y)         
        ssreg = np.sum(residual**2)   
        sstot = np.sum((y - y_mean)**2)    
        result['r2'] = 1-(ssreg/sstot)

        return result
    
    def fit(self, data: DataFrame):
        fit_scores = []
        for pn in self.parameter_names:
            data[pn] = np.nan
        # for each row
        for i in range(len(data)):
            # Only select session scores
            row_y = data.iloc[i, 0:10].dropna()
            row_x = range(1, len(row_y) + 1)
            result = self.fit_row(row_x, row_y)
            fit_scores.append(result['r2'])
            for j in range(len(self.parameter_names)):
                data.iat[i, j - len(self.parameter_names)] = result['parameters'][j]
        return fit_scores

class LogRegressor(Regressor):
    def __init__(self, clustering_parameters: list[str] = None):
        Regressor.__init__(self)
        self.regression_func = lambda x, s, b: s * log(x) + b
        self.parameter_names = ['log_slope', 'log_bias']
        self.clustering_parameters = []
        self.set_clustering_parameters(clustering_parameters)

class LinearRegressor(Regressor):
    def __init__(self, clustering_parameters: list[str] = None):
        Regressor.__init__(self)
        self.regression_func = lambda x, s, b: s * x + b
        self.parameter_names = ['linear_slope', 'linear_bias']
        self.clustering_parameters = []
        self.set_clustering_parameters(clustering_parameters)

class PiecewiselinRegressor(Regressor):
    def __init__(self, clustering_parameters: list[str] = None):
        Regressor.__init__(self)
        self.parameter_names = ['knot', 'slope1', 'slope2']
        self.clustering_parameters = []
        self.set_clustering_parameters(clustering_parameters)
    
    # Use a library based on https://jekel.me/piecewise_linear_fit_py/pwlf.html
    # Before running, pip install pwlf
    def find_opt_knot(self, x, y) -> int:
        # Set different knot locations
        pwlf_each_row = pwlf.PiecewiseLinFit(x, y)
        r2_all_knots = {}
        for k in range(2, len(y)):
            pwlf_each_row.fit_with_breaks_opt([k])
            r2 = pwlf_each_row.r_squared()
            r2_all_knots[k] = r2
        # Find knot location with max r2
        max_r2 = max(r2_all_knots.values())
        opt_knot = [key for key in r2_all_knots if r2_all_knots[key] == max_r2]
        return opt_knot

    def is_unchanged_input(self, y) -> bool:
        for i in range(0, len(y) - 1):
            if y[i] != y[i+1]:
               return False
        return True

    def fit_row(self, x, y):
        result = {}
        if self.is_unchanged_input(y):
            result['parameters'] = [np.nan] * len(self.parameter_names)
            result['r2'] = np.nan
            return result
        opt_knot = self.find_opt_knot(x, y)
        pwlf_each_row = pwlf.PiecewiseLinFit(x, y)
        pwlf_each_row.fit_with_breaks_opt([opt_knot])
        slopes = pwlf_each_row.calc_slopes()
        result['parameters'] = np.insert(slopes, 0, opt_knot)
        result['r2'] = pwlf_each_row.r_squared()
        return result

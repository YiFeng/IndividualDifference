from pandas import DataFrame
import numpy as np
from scipy.optimize import curve_fit
from scipy import log

# Get intervention data
def get_intervention_data(no_missing_data: DataFrame) -> DataFrame:
    intervention_col_names = ['WmeanN_' + str(i) for i in range(1,11)]
    return no_missing_data[intervention_col_names]

# Preprocessing 10 sessions intervention data
def lin_func(x, s, b):
    y = s * x + b
    return y

# k means knot location, y0 means turning point, s1 means slope1, s2 means slope2
def piece_lin(x, k, y0, s1, s2):
    y = np.piecewise(x, [x < k], [lambda x:s1*x + y0-s1*k, lambda x:s2*x + y0-s2*x0])

# Base Regressor. Do not use. Use Derived Regressors.
class Regressor:
    def __init__(self):
        self.regression_func = None
        self.parameter_names: list[str] = None
        self.clustering_names: list[str] = None

    def fit_row(self, x, y):
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
    def __init__(self, option):
        Regressor.__init__(self)
        self.regression_func = lambda x, s, b: s * log(x) + b
        self.parameter_names = ['log_slope', 'log_bias']
        self.clustering_names = self.parameter_names[option]

'''
class LinearRegressor(Regressor):

class InterventionProcessor:
    def __init__(self, intervention_data):
        self.data: DataFrame = intervention_data
        self.regressors: list[Regressor] = []
        self.generated_col_names: set[str] = set()
        self.clustering_col_names: set[str] = set()
        self.clustering_model = None

    def register_regressor(self, reg: Regressor):
        self.regressors.append(reg)
    
    def fit(self):
        for r in self.regressors:
            result = r.fit(self.data)
            self.generated_col_names.add(r.parameter_names)
            self.clustering_col_names.add(r.clustering_names)

    def register_cluster_model(self, cluster):
        if self.clustering_model == None:
            self.clustering_model = cluster

    def get_max(self):
        self.generated_column_names.append('max')
        self.data['max'] = 111;

    def basic_analyze(self, option: list[str]):
        if 'max' in option:
            self.get_max()
        self.get_std_div()
        self.get_xxx()
    
    def mark_outlier(self):

    def cluster(self, clustering_col_names):
        # compare(clustering_col_names, self.generated_col_names)
        # data = select_non_outlier(self.data)
        results = clustering_model(self.data[self.generated_col_names])
        self.data[lable] = results;

    def get_clustered_data(self) -> DataFrame:
        # return non-outlier and clustered data
'''
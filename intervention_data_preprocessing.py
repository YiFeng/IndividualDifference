from pandas import DataFrame
from regressor import Regressor
from cluster import ClusterModel
import numpy as np
import re
from scipy import stats
import statsmodels
from statsmodels.stats.stattools import medcouple
import math
from math import e
from typing import List

class InterventionProcessor:
    @staticmethod
    def get_intervention_data(no_missing_data: DataFrame, intervention_col_names: List[str]) -> DataFrame:
        col_add_id = intervention_col_names.copy()
        col_add_id.insert(0, 'Unique')
        return no_missing_data[col_add_id]

    def __init__(self, no_missing_data):
        self.intervention_col_names = ['mean_' + str(i) for i in range(1,11)]
        self.data: DataFrame = self.get_intervention_data(no_missing_data, self.intervention_col_names)
        self.regressors: List[Regressor] = []
        self.generated_col_names: set[str] = set()
        self.clustering_col_names: set[str] = set()
        self.clustering_model = None

    def get_only_intervention_data(self) -> DataFrame:
        return self.data[self.intervention_col_names]

    def get_max(self):
        self.generated_col_names.add('max')
        self.data['max'] = self.get_only_intervention_data().max(axis=1, skipna=True, numeric_only=True)
        print('The average maximun span across sessions is {:.3f} with sd {:.3f}'.format(self.data['max'].mean(), self.data['max'].std())) 

    def get_mean(self):
        self.generated_col_names.add('mean')
        self.data['mean'] = self.get_only_intervention_data().mean(axis=1, skipna=True, numeric_only=True)
        print('The average span across sessions is {:.3f} with sd {:.3f}'.format(self.data['mean'].mean(), self.data['mean'].std())) 

    def get_std(self):
        self.generated_col_names.add('std')
        self.data['std'] = self.get_only_intervention_data().std(axis=1, skipna=True, numeric_only=True)
        print('The average standard deviation across sessions is {:.3f} with sd {:.3f}'.format(self.data['std'].mean(), self.data['std'].std()))
        
    def get_max_session(self):
        self.generated_col_names.add('max_session') 
        self.data['max_session'] = self.get_only_intervention_data().idxmax(axis=1)

    def basic_analyze(self):
        self.get_max()            
        self.get_std()
        self.get_mean()

    
    def mark_outlier_stewd(self, option: List[str]):
        self.data['outlier'] = False
        for col in option:
            if col not in self.data.columns:
                print('This column does not exsit: {}'.format(col))
                continue
            # med = stats.median_absolute_deviation(self.data[col])
            # median = self.data[col].median()
            print('The skew of {} is {:.3f}'.format(col, self.data[col].skew()))
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 -q1
            mc = medcouple(self.data[col])
            print('{}:MC={:.3f}'.format(col, mc))
            up = q3 + 1.5* (math.exp(4*mc)) *iqr
            down = q1 - 1.5* (math.exp(-3*mc)) *iqr
            print('interval of {} is {} to {}'.format(col, up, down))
            for i in range(len(self.data)):
                if self.data[col].iloc[i] >up or self.data[col].iloc[i] <down:
                    self.data['outlier'].iloc[i] = True
            print(self.data[self.data['outlier']])
    
    def mark_outlier_zscore(self, option: List[str]):
        self.data['outlier'] = False
        for col in option:
            if col not in self.data.columns:
                print('This column does not exsit: {}'.format(col))
                continue
            mean = self.data[col].mean()
            std = self.data[col].std()
            for i in range(len(self.data)):
                z_score = (self.data[col].iloc[i] - mean)/std
                if z_score >3 or z_score <-3:
                    self.data['outlier'].iloc[i] = True
        print(self.data[self.data['outlier']])

    def delete_outlier(self):
        self.data = self.data[self.data['outlier']==0]
        print('The sample size after delete outlier: {}'.format(len(self.data)))

    ####Regressor    
    def register_regressor(self, reg: Regressor):
        self.regressors.append(reg)
    
    def fit(self):
        for r in self.regressors:
            rsquare = r.fit(self.data)
            self.generated_col_names.update(r.parameter_names)
            print('The average r2 for {} is: {:.3f}'.format(type(r), np.nanmean(rsquare)))
            print('The num of objects can not fit by pwlf: {}'.format(self.data.isna().sum()))
            self.data = self.data.dropna(subset=['r2'])
            print('The sample size that can fit with pwlf: {}'.format(len(self.data)))
    
    ####Cluster 
    def register_cluster_model(self, cluster: ClusterModel):
        self.clustering_model = cluster

    def cluster(self):
        self.clustering_col_names.update(self.clustering_model.clustering_col_names)
        for col in self.clustering_col_names:
            if col not in self.generated_col_names:
                print('This column has not be generated:{}'.format(col))
                continue
        xx = self.clustering_model.clustering(self.data)
        return xx
        
    def get_clustered_data(self) -> DataFrame:
        return self.data


    ###Create labels
    def create_labels(self, dict_naming: dict, cluster_col: str, delete1cluster: bool, cluster_delete=None or str):
        if delete1cluster:
            self.data = self.data[self.data[cluster_col] != cluster_delete]
        self.data.replace({cluster_col: dict_naming}, inplace=True)

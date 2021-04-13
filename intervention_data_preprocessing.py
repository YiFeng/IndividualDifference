from pandas import DataFrame
from regressor import Regressor
from cluster import ClusterModel
import numpy as np
import re as re

class InterventionProcessor:
    @staticmethod
    def get_intervention_data(no_missing_data: DataFrame, intervention_col_names: list[str]) -> DataFrame:
        col_add_id = intervention_col_names.copy()
        col_add_id.insert(0, 'ID')
        return no_missing_data[col_add_id]

    def __init__(self, no_missing_data):
        self.intervention_col_names = ['WmeanN_' + str(i) for i in range(1,11)]
        self.data: DataFrame = self.get_intervention_data(no_missing_data, self.intervention_col_names)
        self.regressors: list[Regressor] = []
        self.generated_col_names: set[str] = set()
        self.clustering_col_names: set[str] = set()
        self.clustering_model = None

    def get_only_intervention_data(self) -> DataFrame:
        return self.data[self.intervention_col_names]

    def get_max(self):
        self.generated_col_names.add('max')
        self.data['max'] = self.get_only_intervention_data().max(axis=1, skipna=True, numeric_only=True) 
    
    def get_std(self):
        self.generated_col_names.add('std')
        self.data['std'] = self.get_only_intervention_data().std(axis=1, skipna=True, numeric_only=True)

    def get_max_session(self):
        self.generated_col_names.add('max_session') 
        self.data['max_session'] = self.get_only_intervention_data().idxmax(axis=1)

    def delete_crazy_sub(self):
        print('The sample size before delete crazy sub: {}'.format(len(self.data)))
        self.data = self.data[self.data['max'] < 15][self.data['std'] !=0]
        print('The sample size after delete crazy sub: {}'.format(len(self.data)))

    def basic_analyze(self):
        self.get_max()            
        self.get_std()
        self.delete_crazy_sub()
    
    def register_regressor(self, reg: Regressor):
        self.regressors.append(reg)
    
    def fit(self):
        for r in self.regressors:
            result = r.fit(self.data)
            self.generated_col_names.update(r.parameter_names)
            print('The mean r2 for {} is: {:.3f}'.format(type(r), np.nanmean(result)))

    def mark_outlier(self, option: list[str]):
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

    def register_cluster_model(self, cluster: ClusterModel):
        self.clustering_model = cluster

    def cluster(self):
        self.clustering_col_names.update(self.clustering_model.clustering_col_names)
        for col in self.clustering_col_names:
            if col not in self.generated_col_names:
                print('This column has not be generated:{}'.format(col))
                continue
        self.clustering_model.clustering(self.data)
        
    def get_clustered_data(self) -> DataFrame:
        return self.data
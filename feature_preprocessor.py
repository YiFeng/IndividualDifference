# Feature preprocessing
import raw_data_preprocessing as rdp
from scipy import stats
from pandas import DataFrame
from sklearn.experimental import enable_iterative_imputer 
from sklearn.preprocessing import OrdinalEncoder
import individual_differences_plot as plot
import pandas as pd
import numpy as np

class FeatureProcessor:
    @staticmethod
    def get_feature_data(no_missing_data: DataFrame, add_demographic: bool) -> DataFrame:
        if add_demographic:
            col_add_id = rdp.feature_col_names.copy() + rdp.demographic_columns
        else:
            col_add_id = rdp.feature_col_names.copy()
        col_add_id.insert(0, 'Unique')
        return no_missing_data[col_add_id]

    def __init__(self, no_missing_data: DataFrame, add_demographic: bool):
        self.data  = self.get_feature_data(no_missing_data, add_demographic)
        if add_demographic:
            self.feature_col_names = rdp.feature_col_names.copy() + rdp.demographic_columns
            self.feature_cate_names = rdp.feature_col_categ_names.copy() + rdp.demographic_columns
        else:
            self.feature_col_names = rdp.feature_col_names.copy()
            self.feature_cate_names = rdp.feature_col_categ_names.copy()
        self.feature_col_conti_names = rdp.feature_col_conti_names.copy()

#### Data clearning

    # Feature missing data
    def delete_missing_rows(self, crtieria: int):
        print('Original length of data is : {}'.format(len(self.data)))
        self.data = self.data[self.data[self.feature_col_names].isnull().sum(axis=1) < crtieria]
        print('Delete rows have >= crtieria missing values in features')
        print('The data length now is :{}'.format(len(self.data))) 

    def make_ordinary_var(self, target_col_name: str, categories):
        enc = OrdinalEncoder(categories=[categories], handle_unknown= 'use_encoded_value', unknown_value=np.nan)
        x = self.data[[target_col_name]]
        b = enc.fit_transform(x)
        self.data[target_col_name] = pd.Series(b[:,0])

    # Create dummy variables for categorical vars
    def make_dummy(self):
        self.data = pd.get_dummies(self.data)
        self.feature_col_names = self.data.columns.to_list()
        self.feature_col_names.remove('Unique')
        self.feature_cate_names = [x for x in self.feature_col_names if x not in self.feature_col_conti_names]
    
    # Feature correlation
    def corr_features(self):
        corr = self.data[self.feature_col_conti_names].corr()
        for i in range(len(self.feature_col_conti_names)):
            for j in range(len(self.feature_col_conti_names)):
                if corr.iloc[i,j] > 0.5 and i != j:
                    print('These two features correlated above 0.5:{}, {}'.format(self.feature_col_conti_names[i], self.feature_col_conti_names[j]))

    # Feature distribution
    def distri_features(self):
        plot.plot_distribution(self.data, self.feature_col_conti_names)
        

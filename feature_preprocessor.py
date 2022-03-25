# Feature preprocessing
import raw_data_preprocessing as rdp
from scipy import stats
from pandas import DataFrame
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import individual_differences_plot as plot
import pandas as pd

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

    # Feature missing data
    def delete_missing_rows(self):
        print('Original length of data is : {}'.format(len(self.data)))
        self.data = self.data[self.data[self.feature_col_names].isnull().sum(axis=1) < 6]
        print('Delete rows have >= 6 missing values in features')
        print('The data length now is :{}'.format(len(self.data))) 

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

    def features_impute_missing(self):
        # summarize the number of rows with missing values for each column
        for i in range(len(self.feature_col_names)):
	    # count number of rows with missing values
            feature = self.feature_col_names[i]
            n_miss = self.data[feature].isnull().sum()
            perc = n_miss / self.data.shape[0] * 100
            print('Feature {} missing counts: {}({:.1f} %)'.format(feature, n_miss, perc))
        # define imputer
        imputer = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending')
        self.data[self.feature_col_names] = imputer.fit_transform(self.data[self.feature_col_names])
        # print total missing
        print('============Finish impute continious variables=============')

        

# Discretize continuous features
    def discretize_features(self):
        # sleep, score > 11 means the level of sleepiness may impact a person’s activities 
        self.data['z_SES_category'] = pd.cut(x = self.data['z_SES'], bins = [-1.8,-0.5,0.8,2], labels = ['low_ses', 'middle_ses', 'high_ses'])
        # vgq, >20h means play more vg in the past year
        median = self.data['VGQ_pastyear'].median()
        self.data['VGQ_pastyear_category'] = pd.cut(x = self.data['VGQ_pastyear'], bins = [5,median,32], labels = ['less_play', 'more_play'])
        self.feature_cate_names = ['SES_category', 'VGQ_category']
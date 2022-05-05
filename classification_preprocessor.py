# Preprocessing data for classification
# Combine data, F test, chi test, feature selection, detect outlier, resample, normalize
from pandas import DataFrame
import pandas as pd
import numpy as np
import raw_data_preprocessing as rdp
import feature_preprocessor as fp
import scipy.stats as stats
import individual_differences_plot as plot
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import IsolationForest
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
import pickle
import os.path as path
import statsmodels.api as sm
from statsmodels.formula.api import ols


class ClassifyPreprocessor:
    @staticmethod
    def combine_cluster_feature(cluster_result, feature_preprocessed, label_name) -> DataFrame:   
        overlap_idlist = list(set(cluster_result['Unique']).intersection(set(feature_preprocessed['Unique'])))
        feature_preprocessed = feature_preprocessed.set_index('Unique')
        cluster_result = cluster_result.set_index('Unique')
        data = pd.concat([feature_preprocessed.loc[overlap_idlist,:], cluster_result.loc[overlap_idlist,label_name]], axis=1)
        return data

    def __init__(self, cluster_result: DataFrame, feature_preprocessed: DataFrame, label_name: str):
        self.data: DataFrame = self.combine_cluster_feature(cluster_result, feature_preprocessed, label_name)
        self.feature_numerical_names = rdp.feature_col_conti_names.copy()
        self.data_length = None
        self.label_name = label_name
        self.feature_names = self.data.columns.to_list()
        self.feature_names.remove(label_name)
        self.feature_cate_names = [x for x in self.feature_names if x not in self.feature_numerical_names]
    
    def transform_df_toarray(self, data, col_names):
        x = data[col_names].to_numpy()
        y = data[self.label_name].to_numpy()
        return x, y    
                    
    def features_impute_missing(self):
        # summarize the number of rows with missing values for each column
        for i in range(len(self.feature_names)):
	    # count number of rows with missing values
            feature = self.feature_names[i]
            n_miss = self.data[feature].isnull().sum()
            perc = n_miss / self.data.shape[0] * 100
            print('Feature {} missing counts: {}({:.1f} %)'.format(feature, n_miss, perc))
        # define imputer
        imputer = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, random_state=0, imputation_order='ascending')
        self.data[self.feature_names] = imputer.fit_transform(self.data[self.feature_names])
        # print total missing
        print('============Finish impute missing values in features=============')
    
    def split_train_test(self):
        # split a seperate testset
        y = self.data[self.label_name].to_numpy()
        df_train, df_test = train_test_split(self.data, test_size=0.20, random_state=0, stratify=y) 
        self.data_length = len(df_train)
        print('The length of training set before resample: {}'.format(len(df_train)))
        print('The length of test set: {}'.format(len(df_test)))
        print('The sample weight of train set before resample: {}'.format(Counter(df_train[self.label_name])))
        print('The sample weight of test set: {}'.format(Counter(df_test[self.label_name])))
        print('============Finish split train and test set=============')
        return df_train, df_test
    
    def get_mean_std(self, df_train):
        x_train_numerical = df_train[self.feature_numerical_names].to_numpy()
        scaler = StandardScaler()
        scaler.fit(x_train_numerical)
        mean_train = scaler.mean_
        std_train = scaler.var_
        return mean_train, std_train   

    def standardize_numerical(self, df_train_imputed, df_test, mean, std):
        # Standardize numerical features
        ### training set
        df_train_scaled = df_train_imputed.copy()
        df_train_scaled.iloc[:,0:len(mean)] = (df_train_scaled.iloc[:,0:len(mean)] - mean)/std
        ### test set
        df_test_scaled = df_test.copy()
        df_test_scaled.iloc[:,0:len(mean)] = (df_test_scaled.iloc[:,0:len(mean)] - mean)/std
        print('============Finish standardize train and test features=============')
        return df_train_scaled, df_test_scaled
    
    def feature_selection(self, df_train_scaled, k: int):
        x,y = self.transform_df_toarray(df_train_scaled, self.feature_numerical_names)
        ### feature selection using Mutual Information (https://towardsdatascience.com/learn-how-to-do-feature-selection-the-right-way-61bca8557bef)
        discrete_feature_indices = list(range(len(self.feature_numerical_names), len(self.feature_names)))
        ## mi_value = mutual_info_classif(x, y, discrete_features=discrete_feature_indices, n_neighbors=5, copy=True, random_state=0)        
        fs = SelectKBest(score_func=f_classif, k='all')
        fs.fit(x,y)
        importance_result = pd.DataFrame()
        importance_result['feature'] = self.feature_numerical_names
        importance_result['score'] = fs.scores_
        importance_result['pvalue'] = fs.pvalues_
        importance_result.sort_values(by=['score'], ascending=False, inplace=True)
        print('The features importance are: \n {}'.format(importance_result))
        self.feature_numerical_names = list(importance_result['feature'].head(k)) 

    def feature_selection_cate(self, df_train_scaled, k:int):
        x,y = self.transform_df_toarray(df_train_scaled, self.feature_cate_names)
        ### feature selection using Mutual Information (https://towardsdatascience.com/learn-how-to-do-feature-selection-the-right-way-61bca8557bef)
        ## mi_value = mutual_info_classif(x, y, discrete_features=discrete_feature_indices, n_neighbors=5, copy=True, random_state=0)        
        fs = SelectKBest(score_func=chi2,k='all')
        fs.fit(x,y)
        importance_result = pd.DataFrame()
        importance_result['feature'] = self.feature_cate_names
        importance_result['score'] = fs.scores_
        importance_result['pvalue'] = fs.pvalues_
        importance_result.sort_values(by=['score'], ascending=False, inplace=True)
        print('The top features are: \n{}'.format(importance_result))
        self.feature_cate_names = list(importance_result['feature'].head(k)) 
   
    def resample_train(self, df_train_scaled, target_sample_weight: dict[str,int], oversample:bool):
        self.feature_names = self.feature_cate_names + self.feature_numerical_names
        x_train, y_train = self.transform_df_toarray(df_train_scaled, self.feature_names)
        if oversample:
            oversample = SMOTE(sampling_strategy=target_sample_weight, random_state=42)
            features_resample, labels_resample = oversample.fit_resample(x_train, y_train)
            print('The sample weight after resample: {}'.format(Counter(labels_resample)))         
            print('============Finish resample train set=============')
        else:
            features_resample, labels_resample = x_train, y_train
        print('The input shape: {} and {}'.format(features_resample.shape, labels_resample.shape))
        return features_resample, labels_resample
    
    def save_data(self, x_train, y_train, df_test_standard, data_path: str):
        x_test, y_test = self.transform_df_toarray(df_test_standard, self.feature_names)
        with open(path.join(data_path, "train_x.array"), mode='wb') as f:
            pickle.dump(x_train, f)
            f.close()
        with open(path.join(data_path, "test_x.array"), mode='wb') as f:
            pickle.dump(x_test, f)
            f.close()

        with open(path.join(data_path, "train_y.array"), mode='wb') as f:
            pickle.dump(y_train, f)
            f.close()
        with open(path.join(data_path, "test_y.array"), mode='wb') as f:
            pickle.dump(y_test, f)
            f.close()

        with open(path.join(data_path, "original_len.int"), mode='wb') as f:
            pickle.dump(self.data_length, f)
            f.close()
        with open(path.join(data_path, "feature_names.list"), mode='wb') as f:
            pickle.dump(self.feature_names, f)
            f.close()
    
    '''      
    def delete_outlier(self):
        x, y = self.transform_df_toarray(self.feature_numerical_names)
        isf = IsolationForest(n_jobs=-1, random_state=1)
        outlier_result = isf.fit_predict(x,y)
        outlier_index = []
        for i in range(len(outlier_result)):
            if outlier_result[i] == -1:
                outlier_index.append(i)
        self.data.drop(index = self.data.iloc[outlier_index].index, inplace=True)
        print('The samplesize after delete outliers: {}'.format(len(self.data)))
        print('The sample weight after delete outliers: {}'.format(Counter(self.data[self.label_name])))
	'''
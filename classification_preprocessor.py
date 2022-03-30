# Preprocessing data for classification
# Combine data, F test, chi test, feature selection, detect outlier, resample, normalize
from pandas import DataFrame
import pandas as pd
import numpy as np
import raw_data_preprocessing as rdp
import feature_preprocessor as fp
import scipy.stats as stats
import individual_differences_plot as plot
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import IsolationForest
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler,OneHotEncoder
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
    
    def transform_df_toarray(self, col_names):
        x = self.data[col_names].to_numpy()
        y = self.data[self.label_name].to_numpy()
        return x, y    

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
   
    def feature_selection(self, k: int):
        x, y = self.transform_df_toarray(self.feature_names)
        fs = SelectKBest(score_func=f_classif, k=k)
        fs.fit(x,y)
        fs_result = {}
        for i in range(len(fs.scores_)):
            fs_result[self.feature_names[i]] = fs.scores_[i]
        fs_result = dict(sorted(fs_result.items(), key=lambda item: item[1], reverse=True))
        for key, value in fs_result.items() :
            print(key, value)
        print('The top {} features are: {}'.format(k, list(fs_result.keys())[:k]))
        self.feature_names = list(fs_result.keys())[:k] 
                
    def train_test_split(self):
        # split a seperate testset
        x, y = self.transform_df_toarray(self.feature_names)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
        print('The shape of training set before resample: {} and {}'.format(X_train.shape, y_train.shape))
        print('The sample weight before resample: {}'.format(Counter(y_train)))
        print('The shape of test set: {} and {}'.format(X_test.shape, y_test.shape))
        print('The sample weight of test set: {}'.format(Counter(y_test)))
        return X_train, X_test, y_train, y_test

    def resample_standardize(self, x_train, y_train, target_sample_weight: dict[str,int], oversample:bool):
        self.data_length = len(y_train)
        if oversample:
            oversample = SMOTE(sampling_strategy=target_sample_weight, random_state=42)
            features_resample, labels_resample = oversample.fit_resample(x_train, y_train)
            print('The sample weight after resample: {}'.format(Counter(labels_resample)))
        else:
            features_resample, labels_resample = x_train, y_train
        # Standardize numerical features
        scaler = StandardScaler()
        numerical_x = scaler.fit_transform(features_resample[:,0:len(self.feature_numerical_names)])
        category_x = features_resample[:,len(self.feature_numerical_names):]
        new_x = np.concatenate((numerical_x, category_x), axis=1)
        return new_x, labels_resample, self.data_length, self.feature_names
    
    def save_data(self, X_train, X_test, y_train, y_test, orig_len: int, feature_names, data_path: str):
        with open(path.join(data_path, "train_x.array"), mode='wb') as f:
            pickle.dump(X_train, f)
            f.close()
        with open(path.join(data_path, "test_x.array"), mode='wb') as f:
            pickle.dump(X_test, f)
            f.close()

        with open(path.join(data_path, "train_y.array"), mode='wb') as f:
            pickle.dump(y_train, f)
            f.close()
        with open(path.join(data_path, "test_y.array"), mode='wb') as f:
            pickle.dump(y_test, f)
            f.close()

        with open(path.join(data_path, "original_len.int"), mode='wb') as f:
            pickle.dump(orig_len, f)
            f.close()
        with open(path.join(data_path, "feature_names.list"), mode='wb') as f:
            pickle.dump(feature_names, f)
            f.close()
    
    '''
    def oneway_anova(self, data: DataFrame, factor: str, dependent_var: str):
        model = ols(dependent_var + ' ~ C('+ factor + ')', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table    
    
    def feature_test(self):
        # f test for numerical features
        for feature in self.feature_numerical_names:
            f, p = stats.f_oneway(self.data[feature][self.data['label'] == 0],
                                  self.data[feature][self.data['label'] == 1],
                                  self.data[feature][self.data['label'] == 2])
            result = self.oneway_anova(self.data, 'label', feature)
            f = result.iloc[0]['F']
            p = result.iloc[0]['PR(>F)']
            print('The ANOVA test result for {}: f:{:.3f}, p:{:.3f}'.format(feature, f, p))
        
        plot.bar_plot_cluster(self.data, self.feature_numerical_names)
	'''
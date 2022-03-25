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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
import pickle
import os.path as path
import statsmodels.api as sm
from statsmodels.formula.api import ols


class ClassifyPreprocessor:
    @staticmethod
    def combine_cluster_feature(cluster_result, feature_preprocessed, label_name) -> DataFrame:
        feature_preprocessed = feature_preprocessed.set_index('Unique')
        id_list = cluster_result['Unique'].tolist()
        cluster_result = cluster_result.set_index('Unique')
        data = pd.concat([feature_preprocessed.loc[id_list,:], cluster_result[label_name]], axis=1)
        return data


    def __init__(self, cluster_result: DataFrame, feature_preprocessed: DataFrame, label_name: str):
        self.data: DataFrame = self.combine_cluster_feature(cluster_result, feature_preprocessed, label_name)
        self.feature_numerical_names = rdp.feature_col_conti_names.copy()
        self.categorical_columns = rdp.feature_col_categ_names.copy()
        self.feature_names = rdp.feature_col_names.copy() 
        self.data_length = None
        self.label_name = label_name
    
    def transform_df_toarray(self, col_names):
        x = pd.get_dummies(self.data[col_names]).to_numpy()
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

    def oneway_anova(self, data: DataFrame, factor: str, dependent_var: str):
        model = ols(dependent_var + ' ~ C('+ factor + ')', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table    
    
    def feature_test(self):
        # f test for numerical features
        for feature in self.feature_numerical_names:
            '''
            f, p = stats.f_oneway(self.data[feature][self.data['label'] == 0],
                                  self.data[feature][self.data['label'] == 1],
                                  self.data[feature][self.data['label'] == 2])
            '''
            result = self.oneway_anova(self.data, 'label', feature)
            f = result.iloc[0]['F']
            p = result.iloc[0]['PR(>F)']
            print('The ANOVA test result for {}: f:{:.3f}, p:{:.3f}'.format(feature, f, p))
        
        plot.bar_plot_cluster(self.data, self.feature_numerical_names)
	    
    def feature_selection(self, k: int):
        x, y = self.transform_df_toarray(self.feature_numerical_names)
        fs = SelectKBest(score_func=f_classif, k=k)
        fs.fit(x,y)
        fs_result = {}
        for i in range(len(fs.scores_)):
            fs_result[self.feature_numerical_names[i]] = fs.scores_[i]
        fs_result = dict(sorted(fs_result.items(), key=lambda item: item[1], reverse=True))
        for key, value in fs_result.items() :
            print(key, value)
        print('The top {} features are: {}'.format(k, list(fs_result.keys())[:k]))
        self.feature_names = list(fs_result.keys())[:k] 
        # self.feature_names = self.feature_names + rdp.demographic_columns
    
    
    def resample_standardize(self, sample_weight: dict[int,int], oversample:bool):
        x, y = self.transform_df_toarray(self.feature_names)
        self.data_length = len(y)
        print('The sample weight before resample: {}'.format(Counter(y)))
        if oversample:
            oversample = SMOTE(sampling_strategy=sample_weight, random_state=42)
            features_resample, labels_resample = oversample.fit_resample(x, y)
            print('The sample weight after resample: {}'.format(Counter(labels_resample)))
        features_resample, labels_resample = x, y
        # Standardize
        scaler = StandardScaler()
        
        pipeline=make_column_transformer(
            (StandardScaler(),
            make_column_selector(dtype_include=np.number)),
            (OneHotEncoder(),
            make_column_selector(dtype_include=object)))

        new_x = pipeline.fit_transform(self.data[self.feature_names])
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
    
    
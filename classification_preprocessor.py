# Preprocessing data for classification
# Combine data, F test, chi test, feature selection, detect outlier, resample, normalize
from pandas import DataFrame
import pandas as pd
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
from sklearn.preprocessing import StandardScaler
import pickle
import os.path as path


class ClassifyPreprocessor:
    @staticmethod
    def combine_cluster_feature(cluster_result, feature_preprocessed) -> DataFrame:
        feature_preprocessed = feature_preprocessed.set_index('ID')
        id_list = cluster_result['ID'].tolist()
        cluster_result = cluster_result.set_index('ID')
        data = pd.concat([feature_preprocessed.loc[id_list,:], cluster_result['label']], axis=1)
        return data


    def __init__(self, cluster_result: DataFrame, feature_preprocessed: DataFrame):
        self.data: DataFrame = self.combine_cluster_feature(cluster_result, feature_preprocessed)
        self.feature_numerical_names = rdp.feature_col_names.copy()
        self.feature_names = rdp.feature_col_names.copy()
        self.data_length = None
    
    def transform_df_toarray(self, col_names):
        x = pd.get_dummies(self.data[col_names]).to_numpy()
        y = self.data['label'].to_numpy()
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
        print('The sample weight after delete outliers: {}'.format(Counter(self.data['label'])))
        
    def feature_test(self):
        # f test for numerical features
        for feature in self.feature_numerical_names:
            f, p = stats.f_oneway(self.data[feature][self.data['label'] == 0],
                                  self.data[feature][self.data['label'] == 1],
                                  self.data[feature][self.data['label'] == 2])
            print('The F test result for {}: f:{:.3f}, p:{:.3f}'.format(feature, f, p))
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
    
    def use_categorical_features(self):
        for i, feature in enumerate(self.feature_names):
            if feature == 'z_SES' or feature == 'VGQ_pastyear':
                self.feature_names[i] = feature + '_category'
        print('Use these features: {}'.format(self.feature_names))
    
    def resample_standardize(self, sample_weight: dict[int,int]):
        x, y = self.transform_df_toarray(self.feature_names)
        self.data_length = len(y)
        print('The sample weight before resample: {}'.format(Counter(y)))
        oversample = SMOTE(sampling_strategy=sample_weight, random_state=42)
        features_resample, labels_resample = oversample.fit_resample(x, y)
        print('The sample weight after resample: {}'.format(Counter(labels_resample)))
        # Standardize
        scaler = StandardScaler()
        return scaler.fit_transform(features_resample), labels_resample, self.data_length, self.feature_names

    def save_data(self, X, Y, orig_len: int, feature_names, data_path: str):
        with open(path.join(data_path, "input_x.array"), mode='wb') as f:
            pickle.dump(X, f)
            f.close()
        with open(path.join(data_path, "input_y.array"), mode='wb') as f:
            pickle.dump(Y, f)
            f.close()
        with open(path.join(data_path, "original_len.int"), mode='wb') as f:
            pickle.dump(orig_len, f)
            f.close()
        with open(path.join(data_path, "feature_names.list"), mode='wb') as f:
            pickle.dump(feature_names, f)
            f.close()
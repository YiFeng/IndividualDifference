# Feature preprocessing
import raw_data_preprocessing as rdp
from scipy import stats
from pandas import DataFrame
import individual_differences_plot as plot
import pandas as pd

class FeatureProcessor:
    @staticmethod
    def get_feature_data(no_missing_data: DataFrame) -> DataFrame:
        col_add_id = rdp.feature_col_names.copy()
        col_add_id.insert(0, 'ID')
        return no_missing_data[col_add_id]

    def __init__(self, no_missing_data: DataFrame):
        self.data  = self.get_feature_data(no_missing_data)
        self.feature_col_names = rdp.feature_col_names.copy()
        self.feature_cate_names: list[str] = []

    # Feature correlation
    def corr_features(self):
        corr = self.data[self.feature_col_names].corr()
        for i in range(len(self.feature_col_names)):
            for j in range(len(self.feature_col_names)):
                if corr.iloc[i,j] > 0.5 and i != j:
                    print('These two features correlated above 0.5:{}, {}'.format(self.feature_col_names[i], self.feature_col_names[j]))

    # Feature distribution
    def distri_features(self):
        plot.plot_distribution(self.data, self.feature_col_names)

    # Discretize continuous features
    def discretize_features(self):
        # sleep, score > 11 means the level of sleepiness may impact a person’s activities 
        self.data['SES_category'] = pd.cut(x = self.data['SES'], bins = [-1,10,13,22], labels = ['low_ses', 'middle_ses', 'high_ses'])
        # vgq, >20h means play more vg in the past year
        self.data['VGQ_category'] = pd.cut(x = self.data['VGQ'], bins = [-1,20,50], labels = ['less_play', 'more_play'])
        self.feature_cate_names = ['SES_category', 'VGQ_category']

# Feature F test (for numrical features)
# Feature chi test (for categorical features)


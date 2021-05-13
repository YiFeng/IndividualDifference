import pandas as pd
from pandas import DataFrame
import pingouin as pg
from scipy.stats import kendalltau, pearsonr, spearmanr

# Features that are used for training
feature_col_names: list[str] = [
    'z_Updating_pre', 'z_WM_pre', 'z_IC_pre','MRpre',\
    'TCA','CFQ','z_Grit_Ambition',\
    'Person_emot','Person_extraver', 'Person_opennes', 'Person_agreeable', 'Person_conscien',\
    'VGQ_pastyear','z_SES']

demographic_columns: list[str] = ['Study','Gender','Ethnicity','Hispanic','Algorithm']

def read_raw_data(filename: str) -> DataFrame:
    return pd.read_csv(filename, na_values=['', ' '])

# Select interested columns only. If find any missing in columns, delete that case(row).
def delete_missing_row(raw_data: DataFrame) -> DataFrame:
    print("The sample size of raw data is {}.".format(raw_data.shape[0]))
    no_missing_data = raw_data.dropna(axis='index', how='any', subset=feature_col_names)
    print("The sample size of no missing data is {}.".format(no_missing_data.shape[0]))
    return no_missing_data

def demographic_info(no_missing_data: DataFrame):
    for var in demographic_columns:
        print('Demographoc information:{}'.format(no_missing_data.groupby(var).size()))
    print('The mean age is {:.3f}, and std is {:.3f}'.format(no_missing_data['Age'].mean(), no_missing_data['Age'].std()))
    print('The age range is from {} to {}'.format(no_missing_data['Age'].min(), no_missing_data['Age'].max()))

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]

def descriptive_info(no_missing_data):
    description_result = pd.DataFrame(columns=['mean','sd'], index=feature_col_names)
    for feature in feature_col_names:
        description_result.loc[feature, 'mean'] = no_missing_data[feature].mean()
        description_result.loc[feature, 'sd'] = no_missing_data[feature].std()
    corr = no_missing_data[feature_col_names].corr()
    corr_p = no_missing_data[feature_col_names].corr(method=pearsonr_pval)


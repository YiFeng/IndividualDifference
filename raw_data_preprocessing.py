import pandas as pd
from pandas import DataFrame
import pingouin as pg
from scipy.stats import kendalltau, pearsonr, spearmanr
from typing import List

# Features that are used for training
feature_col_conti_names: List[str] = [
    'MR_accpre', 'WM_Zpre', 'IC_Zreverse_pre','NT_Zpre',\
    'TCA_Zscore','CFQMALscore','Avg_grit_amb',\
    'NEOFFIExtraversion','NEOFFIAgreeableness', 'NEOFFIConscientiousness', 'NEOFFIEmotionalstability', 'NEOFFIIntellectorOpenness',\
    'VGQHoursCategorySumPastYear_Z','ParentalEducationSum','SelfReportedSESRatingSum',\
    'Physicalhealth','Physicalfitness','Psychohealth']

feature_col_categ_names: List[str] = ['Gamified','Difficulty','Bilingual']
feature_col_names = feature_col_conti_names + feature_col_categ_names
demographic_columns: List[str] = ['Gender','Ethnicity','HispanicLatino']
intervention_col_names = ['mean_' + str(i) for i in range(1,11)]
valid_col_names = ['nback_training']

def read_raw_data(filename: str) -> DataFrame:
    return pd.read_csv(filename, na_values = ['',' '])

# Select interested columns only. If find any missing in columns, delete that case(row).
def delete_missing_row(raw_data: DataFrame) -> DataFrame:
    print("The sample size of raw data is {}.".format(raw_data.shape[0]))
    no_missing_data = raw_data.dropna(axis='index', how='any', subset=intervention_col_names + valid_col_names)
    print("The sample size of no missing data (listwise) is {}.".format(no_missing_data.shape[0]))
    return no_missing_data

def demographic_info(no_missing_data: DataFrame):
    for var in demographic_columns:
        print('Demographoc information:{}'.format(no_missing_data.groupby(var).size()))
    print('The mean age is {:.3f}, and std is {:.3f}'.format(no_missing_data['Age'].mean(), no_missing_data['Age'].std()))
    print('The age range is from {} to {}'.format(no_missing_data['Age'].min(), no_missing_data['Age'].max()))

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]

def descriptive_info(no_missing_data):
    description_result = pd.DataFrame(columns=['mean','sd','min','max'], index=feature_col_conti_names)
    for feature in feature_col_conti_names:
        description_result.loc[feature, 'mean'] = no_missing_data[feature].mean()
        description_result.loc[feature, 'sd'] = no_missing_data[feature].std()
        description_result.loc[feature, 'min'] = no_missing_data[feature].min()
        description_result.loc[feature, 'max'] = no_missing_data[feature].max()
    corr = no_missing_data[feature_col_conti_names].corr()
    corr_p = no_missing_data[feature_col_conti_names].corr(method=pearsonr_pval)
    for var in feature_col_categ_names:
        print('Categorical {} information:{}'.format(var,no_missing_data.groupby(var).size()))
    return description_result

'''
feature_col_names: List[str] = [
    'MR_accpre', 'WM_Zpre', 'IC_Zreverse_pre','NT_Zpre',\
    'TCA_Zscore','CFQMALscore','Avg_grit_amb',\
    'NEOFFIExtraversion','NEOFFIAgreeableness', 'NEOFFIConscientiousness', 'NEOFFIEmotionalstability', 'NEOFFIIntellectorOpenness',\
    'VGQHoursCategorySumPastYear_Z','ParentalEducationSum','SelfReportedSESRatingSum']
'''

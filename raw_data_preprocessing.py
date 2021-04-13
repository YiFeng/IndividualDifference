import pandas as pd
from pandas import DataFrame

# Features that are used for training
feature_col_names: list[str] = [
    'SES','VGQ','CFQ', 'Grit_Ambition',\
    'MRpre', 'DM_diff_pre', 'WM_pre', 'Updating_pre',\
    'Person_extraver', 'Person_agreeable', 'Person_conscien', 'Person_emot', 'Person_opennes']

def read_raw_data(filename: str) -> DataFrame:
    return pd.read_csv(filename, na_values=['', ' '])

# Select interested columns only. If find any missing in columns, delete that case(row).
def delete_missing_row(raw_data: DataFrame) -> DataFrame:
    print("The sample size of raw data is {}.".format(raw_data.shape[0]))
    no_missing_data = raw_data.dropna(axis='index', how='any', subset=feature_col_names)
    print("The sample size of no missing data is {}.".format(no_missing_data.shape[0]))
    return no_missing_data

# delete missing data
from pandas import DataFrame

def delete_missing_row(df_raw: DataFrame, feature_col: list[str]) -> DataFrame:
    print("The sample size of raw data is {}.".format(df_raw.shape[0]))
    df_used = df_raw.dropna(axis='index', how='any', subset=feature_col)
    print("The sample size of no missing data is {}.".format(df_used.shape[0]))
    return df_used
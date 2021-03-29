from pandas import DataFrame
from regressor import Regressor

# Get intervention data
def get_intervention_data(no_missing_data: DataFrame) -> DataFrame:
    intervention_col_names = ['WmeanN_' + str(i) for i in range(1,11)]
    return no_missing_data[intervention_col_names]

'''
class InterventionProcessor:
    def __init__(self, intervention_data):
        self.data: DataFrame = intervention_data
        self.regressors: list[Regressor] = []
        self.generated_col_names: set[str] = set()
        self.clustering_col_names: set[str] = set()
        self.clustering_model = None

    def register_regressor(self, reg: Regressor):
        self.regressors.append(reg)
    
    def fit(self):
        for r in self.regressors:
            result = r.fit(self.data)
            self.generated_col_names.add(r.parameter_names)
            self.clustering_col_names.add(r.clustering_names)

    def register_cluster_model(self, cluster):
        if self.clustering_model == None:
            self.clustering_model = cluster

    def get_max(self):
        self.generated_column_names.append('max')
        self.data['max'] = 111;

    def basic_analyze(self, option: list[str]):
        if 'max' in option:
            self.get_max()
        self.get_std_div()
        self.get_xxx()
    
    def mark_outlier(self):

    def cluster(self, clustering_col_names):
        # compare(clustering_col_names, self.generated_col_names)
        # data = select_non_outlier(self.data)
        results = clustering_model(self.data[self.generated_col_names])
        self.data[lable] = results;

    def get_clustered_data(self) -> DataFrame:
        # return non-outlier and clustered data
'''
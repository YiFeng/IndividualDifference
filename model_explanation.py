###### Using SHaP value to explain how features contribute to prediction model
###### See detail:https://shap.readthedocs.io/en/latest/index.html   
    
import shap    
    
    def shap_kernel(self, model_name: str, data_path):
        shap_all = []
        expected_all = []
        kfold_loo = KFold(n_splits=380, shuffle=False)
        X = self.X[:, self.used_feature_indx]
        for train, val in kfold_loo.split(X, self.Y):
            model = self.models[model_name]
            model.fit(X[train], self.Y[train])
            #shap
            explainer=shap.KernelExplainer(model.predict_proba, X[train])
            shap_values = explainer.shap_values(X[val])
            print(explainer.expected_value)
            shap_all.append(shap_values)
            expected_all.append(explainer.expected_value)
        # save shap
        save_data(data_path, shap_all, 'shap_values.data')
        save_data(data_path, expected_all,'shap_expected.data')

    def shap_tree(self, model_name: str, data_path):
        X = self.X[:, self.used_feature_indx]
        model = self.models[model_name]
        model.fit(X, self.Y)
        #shap
        explainer=shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        print(explainer.expected_value)
        shap.summary_plot(shap_values, X[:self.data_len_original], feature_names=self.feature_names)
        shap.summary_plot(shap_values[0][:self.data_len_original], X[:self.data_len_original], feature_names=self.feature_names)
        shap.summary_plot(shap_values[1][:self.data_len_original], X[:self.data_len_original], feature_names=self.feature_names)
        shap.summary_plot(shap_values[2][:self.data_len_original], X[:self.data_len_original], feature_names=self.feature_names)
        shap.dependence_plot('Video game background', shap_values[0][:self.data_len_original], X[:self.data_len_original], feature_names=self.feature_names)
        shap.dependence_plot('Video game background', shap_values[1][:self.data_len_original], X[:self.data_len_original], feature_names=self.feature_names)
        shap.dependence_plot('Video game background', shap_values[2][:self.data_len_original], X[:self.data_len_original], feature_names=self.feature_names)
        shap.dependence_plot('Working memory', shap_values[0][:self.data_len_original], X[:self.data_len_original], feature_names=self.feature_names)

    def read_shap(self, shap_all, class_index):
        shap_array = np.array(shap_all)
        shap_class = [e[class_index] for e in shap_array]
        shap_class = np.concatenate(shap_class, axis=0)
        return shap_class

    def return_correct_index(self, array):
        index_tuple = np.where(array[:self.data_len_original] == 1)
        list_index = [item for t in index_tuple for item in t]
        return list_index

    def shap_feature(self, shap_all_class, X_selected_feature, sub_index_list, class_index):
        for j in range(len(self.feature_names)):
            feature = X_selected_feature[sub_index_list][:,j]
            shap_feature = shap_all_class[class_index][sub_index_list][:,j]
            # corr(feature, shap_feature)
            plot.scatter_plot_shap(feature, shap_feature, class_index, self.feature_names[j])

    # visualize
    def shap_visualize(self, shap_all, expected_all, final_accuracies, final_pred_y, only_show_correct: bool):
        # each class shap
        if only_show_correct:
            sub_index_list = self.return_correct_index(final_accuracies)
        else:
                sub_index_list = list(range(self.data_len_original))
        print(len(sub_index_list))
        
        shap_all_class = []
        X_selected_feature = self.X[:, self.used_feature_indx]
        for class_index in range(3):
            shap_class = self.read_shap(shap_all, class_index)
            shap.summary_plot(shap_class[sub_index_list], X_selected_feature[sub_index_list], feature_names=self.feature_names)
            shap_all_class.append(shap_class)
        
        # shap summary
        shap.summary_plot(shap_all_class, X_selected_feature[sub_index_list], plot_type="bar", feature_names=self.feature_names, color=plot.cmap, class_inds=[0,1,2], class_names=['Low','High','Intermediate'])
        # each feature shap
        for i in range(3):
            self.shap_feature(shap_all_class, X_selected_feature, sub_index_list, i)
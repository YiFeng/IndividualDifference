# Classification procedure
# Model selection

import individual_differences_plot as plot
import multi_class_ovo_curve as roc_auc
import numpy as np
import classification_preprocessor as cp
import pandas as pd
from typing import List

# cross validation setting
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold
loo = LeaveOneOut()
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=0)


# models setting
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from ipywidgets import IntProgress

import traceback
import shap 

NAMES = ['Decision Tree','Random Forest', 'MLP', "Linear SVM", "RBF SVM", 'GDBT']

CLASSIFILES = [
    DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=0),
    RandomForestClassifier(n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="auto",
        class_weight="balanced",
        random_state=0),
    MLPClassifier(hidden_layer_sizes=(5,),solver='lbfgs', random_state=0),
    SVC(kernel="linear", C=0.01, tol=0.001, class_weight='balanced', random_state=0, probability=True),
    SVC(kernel="rbf", C=1, tol=0.001, gamma='scale', class_weight='balanced', random_state=0, probability=True),
    GradientBoostingClassifier(loss='deviance', n_estimators=100, max_depth=5,learning_rate=0.1, random_state=42)
    ]

dict_models = dict(zip(NAMES, CLASSIFILES))

# evaluate setting
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score,recall_score, confusion_matrix, roc_auc_score, cohen_kappa_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.feature_selection import SelectFromModel

# read or save data
import pickle
import os.path as path
def read_inputs(data_path, file_names):
    output = []
    for each_file in file_names:
        with open(path.join(data_path, each_file), mode='rb') as f:
            output.append(pickle.load(f))
            f.close()
    return output

def save_data(data_path, data, file_name):
    with open(path.join(data_path, file_name), mode='wb') as f:
        pickle.dump(data, f)
        f.close()

class ClassificationProcessor():
    def __init__(self, X, Y, original_length, feature_names, classes):
        self.X = X
        self.Y = Y
        self.data_len_original = original_length
        self.models = dict_models
        self.feature_names = feature_names
        self.classes = classes
        
    def train_evaluate_model(self, classifier_name: str, classifier, cross_validation, complete_x, complete_y, print_show_plot: bool, classes) -> int:
        accuracies = cross_val_score(classifier, complete_x, complete_y, scoring='f1_weighted', cv=cross_validation, n_jobs=-1)
        pred_y = cross_val_predict(classifier, complete_x, complete_y, cv=cross_validation, n_jobs=-1)
        prob_y = cross_val_predict(classifier, complete_x, complete_y, cv=cross_validation, n_jobs=-1, method='predict_proba')        
        # model performance (cross validation)
        #1) confusion matrix 2)Akaike information Criterion 3) ROC-AUC score
        cfm = confusion_matrix(complete_y[:self.data_len_original], pred_y[:self.data_len_original])
        # macro_roc_auc_ovo = roc_auc_score(complete_y[:self.data_len_original], pred_y[:self.data_len_original], multi_class = 'ovo', average = 'macro')
        # weighted_roc_auc_ovo = roc_auc_score(complete_y[:self.data_len_original], pred_y[:self.data_len_original],multi_class = 'ovo', average = 'weighted')
        mean_acc = np.mean(accuracies) 
        # plot confusion matrix
        if print_show_plot:
            print('The confusion matrix of {} is: {}'.format(classifier_name, cfm))
            print('{} Accuracy: {:.3f} ({:.3f})'.format(classifier_name, mean_acc, np.std(accuracies)))
            # print('f1 score: {:.3f}'.format(f1_score(complete_y[:self.data_len_original], pred_y[:self.data_len_original])))
            # print('precision: {:.3f}'.format(precision_score(complete_y[:self.data_len_original], pred_y[:self.data_len_original])))
            # print('recall: {:.3f}'.format(recall_score(complete_y[:self.data_len_original], pred_y[:self.data_len_original])))
            # print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} ""(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
            # print('The cogen kappa score is : {}'.format(kappa))
            plot.plot_confusion_matrix(classifier_name, cfm, classes=self.classes)
            # plot.plot_error_confusion(cfm)       
            # plot learning curve
            plot.plot_learning_curve(classifier, classifier_name, complete_x, complete_y, cross_validation)
        return accuracies, pred_y, prob_y # return acc for class 0,1,2
        
    def model_selection(self, feature_num: int):
        for name, clf in self.models.items():
            try:
              self.train_evaluate_model(name, clf, skf, self.X[:, 0:feature_num], self.Y, True, self.classes)
            except Exception as e:
              print('{}: got error: {}'.format(name, traceback.format_exc()))

    #### Feature selection    
    def feature_selection(self, x_test):
        estimator = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, 
                                        class_weight='balanced', random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, 
                                        warm_start=False, n_jobs=None, l1_ratio=None)
        sel_ = SelectFromModel(estimator)
        sel_.fit(self.X, self.Y)
        selected_feature_names = sel_.get_feature_names_out(self.feature_names)
        x_train_selected = sel_.transform(self.X)
        x_test_selected = sel_.transform(x_test)
        return x_train_selected, x_test_selected, selected_feature_names       

    ### Tune model
    # defining parameter range 
    def tune_model(self, model, param_grid, x_test, y_test):
        grid = GridSearchCV(model, param_grid, refit = True, verbose = 3,n_jobs=-1)         
        # fitting the model for grid search 
        grid.fit(self.X, self.Y)         
        # print best parameter after tuning 
        print(grid.best_params_) 
        grid_predictions = grid.predict(x_test)        
        # print classification report 
        print(classification_report(y_test, grid_predictions)) 

    ### final model
    def final_model(self, classifier, X_test, y_test):
        classifier.fit(self.X, self.Y)
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)
        classes = classifier.classes_
        # roc_auc.make_roc_auc(classes,X_test, y_test, y_proba)
        cfm = confusion_matrix(y_test, y_pred)
        print(cfm)
        plot.plot_confusion_matrix('confusion matrix', cfm, classes=classes)
    
    ###### Using SHaP value to explain how features contribute to prediction model
    ###### See detail:https://shap.readthedocs.io/en/latest/index.html      
    def shap_kernel(self, model_name: str, data_path):
        shap_all = []
        expected_all = []
        X = self.X
        for train, val in skf.split(X, self.Y):
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

    def shap_tree_cross(self, model_name: str, data_path):
        shap_all = []
        expected_all = []
        for train, val in skf.split(self.X, self.Y):
            print(train, val)
            model = self.models[model_name]
            model.fit(self.X[train], self.Y[train])
            #shap
            explainer=shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X[val])
            print(explainer.expected_value)
            shap_all.append(shap_values)
            expected_all.append(explainer.expected_value)
        # save shap
        save_data(data_path, shap_all, 'shap_values.data')
        save_data(data_path, expected_all,'shap_expected.data')

    
    def shap_tree(self, model_name: str, x_test, data_path):
        model = self.models[model_name]
        model.fit(self.X, self.Y)
        #shap
        explainer=shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        expected_value = explainer.expected_value
        print(explainer.expected_value)
        # save shap
        save_data(data_path, shap_values, 'shap_tree_values.data')
        save_data(data_path, expected_value,'shap_tree_expected.data')

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
            print(np.corrcoef(feature, shap_feature))
            plot.scatter_plot_shap(feature, shap_feature, class_index, self.feature_names[j])

    # visualize
    def shap_visualize(self, shap_all, expected_all, selected_feature_names):
        # each class shap
        sub_index_list = list(range(self.data_len_original))
        print(len(sub_index_list))
        sub_index_list
        shap_all_class = []
        X_selected_feature = self.X[sub_index_list]
        for class_index in range(2):
            shap_class = self.read_shap(shap_all, class_index)
            shap.summary_plot(shap_class[sub_index_list], X_selected_feature, feature_names=selected_feature_names)
            shap_all_class.append(shap_class)
        
        # shap summary
        shap.summary_plot(shap_all_class, X_selected_feature, plot_type="bar", 
                          feature_names=selected_feature_names, color=plot.cmap, class_inds=[0,1])
        # each feature shap
        for i in range(2):
            self.shap_feature(shap_all_class, X_selected_feature, sub_index_list, i)

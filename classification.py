# Classification procedure
# Model selection

import individual_differences_plot as plot
import multi_class_ovo_curve as roc_auc
import numpy as np
import classification_preprocessor as cp
import pandas as pd



# cross validation setting
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
loo = LeaveOneOut()
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=0)


# models setting
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from ipywidgets import IntProgress
import xgboost
from xgboost import XGBClassifier

import traceback

NAMES = ['Decision Tree','Random Forest', 'MLP', "Linear SVM", "RBF SVM", 'GDBT']

CLASSIFILES = [
    DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=0),
    RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=0),
    MLPClassifier(hidden_layer_sizes=(5,),solver='lbfgs', random_state=0),
    SVC(kernel="linear", C=0.025, class_weight='balanced',decision_function_shape='ovo', random_state=0, probability=True),
    SVC(kernel="rbf", C=1, class_weight='balanced',decision_function_shape='ovo', random_state=0, probability=True),
    GradientBoostingClassifier(loss='deviance', n_estimators=100, max_depth=5,learning_rate=0.1, random_state=42)
    ]

dict_models = dict(zip(NAMES, CLASSIFILES))

# evaluate setting
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import confusion_matrix
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score



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
    def __init__(self, X, Y, original_length, feature_names):
        self.X = X
        self.Y = Y
        self.data_len_original = original_length
        self.models = dict_models
        self.feature_names = feature_names
        self.used_feature_indx =  [0,1,2,3,4,5,6,7,8]
        
    def train_evaluate_model(self, classifier_name: str, classifier, cross_validation, complete_x, complete_y, print_show_plot: bool) -> int:
        accuracies = cross_val_score(classifier, complete_x, complete_y, scoring='f1_weighted', cv=cross_validation, n_jobs=-1)
        pred_y = cross_val_predict(classifier, complete_x, complete_y, cv=cross_validation, n_jobs=-1)
        prob_y = cross_val_predict(classifier, complete_x, complete_y, cv=cross_validation, n_jobs=-1, method='predict_proba')
        
        # model performance (cross validation)
        #1) confusion matrix 2)Akaike information Criterion 3) ROC-AUC score
        cfm = confusion_matrix(complete_y[:self.data_len_original], pred_y[:self.data_len_original])
        # macro_roc_auc_ovo = roc_auc_score(complete_y[:self.data_len_original], pred_y[:self.data_len_original], multi_class = 'ovo', average = 'macro')
        # weighted_roc_auc_ovo = roc_auc_score(complete_y[:self.data_len_original], pred_y[:self.data_len_original],multi_class = 'ovo', average = 'weighted')
        mean_acc = np.mean(accuracies[:self.data_len_original]) 

        # plot confusion matrix
        if print_show_plot:
            print('{} Accuracy: {:.3f} ({:.3f})'.format(classifier_name, mean_acc, np.std(accuracies[:self.data_len_original])))
            # print('f1 score: {:.3f}'.format(f1_score(complete_y[:self.data_len_original], pred_y[:self.data_len_original])))
            # print('precision: {:.3f}'.format(precision_score(complete_y[:self.data_len_original], pred_y[:self.data_len_original])))
            # print('recall: {:.3f}'.format(recall_score(complete_y[:self.data_len_original], pred_y[:self.data_len_original])))
            # print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} ""(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
            # print('The cogen kappa score is : {}'.format(kappa))
            plot.plot_confusion_matrix(classifier_name, cfm, classes=['short_benefit','long_benefit','much_benefit'])
            # plot.plot_error_confusion(cfm)
        
            # plot learning curve
            plot.plot_learning_curve(classifier, classifier_name, complete_x, complete_y, cross_validation)
        return cfm[0][0]/cfm[0].sum(), cfm[1][1]/cfm[1].sum(), cfm[2][2]/cfm[2].sum(), accuracies, pred_y, prob_y # return acc for class 0,1,2
    
    
    def model_selection(self, feature_num: int):
        for name, clf in self.models.items():
            try:
              self.train_evaluate_model(name, clf, skf, self.X[:, 0:feature_num], self.Y, True)
            except Exception as e:
              print('{}: got error: {}'.format(name, traceback.format_exc()))


    #### Feature selection
    def exhausive_feature_selection(self, model_name: str):
        # create object
        model = self.models[model_name]
        efs = EFS(model, min_features=1,
           max_features=10,scoring='accuracy',print_progress=True,cv=skf)
        efs = efs.fit(self.X, self.Y, custom_feature_names=self.feature_names)
        # print selected features
        best_idx = efs.best_idx_
        print('Best accuracy score: {:.3f}'.format(efs.best_score_))
        print('Best subset (indices): {}'.format(efs.best_idx_))
        print('Best subset (corresponding names):{}'.format(efs.best_feature_names_)) 
        self.train_evaluate_model(model_name, model, loo, self.X[:, best_idx], self.Y, True)
        self.used_feature_indx = best_idx
    
    def feature_selection(self, model_name: str):
        model = self.models[model_name]
        feature_selection_result = {}
        for i in range(1,15):
            X = self.X[:, 0:i]
            class0_acc, class1_acc, class2_acc, accuracies, pred_y, prob_y = self.train_evaluate_model(model_name +'with '+ str(i) + ' features', model, skf, X, self.Y, False)
            feature_selection_result[i] = [class0_acc, class1_acc, class2_acc]
        plot.plot_feature_selection_curve(feature_selection_result)

    ### final model
    def final_model(self, classifier, X_test, y_test):
        classifier.fit(self.X, self.Y)
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)
        classes = classifier.classes_
        # roc_auc.make_roc_auc(classes,X_test, y_test, y_proba)
        cfm = confusion_matrix(y_test, y_pred)
        print(cfm)


'''
        # get a stacking ensemble of models
    def get_stacking(self):
        # define the base models
        level0 = list()
        level0.append(('KNN', self.models['Nearest Neighbors']))
        level0.append(('RF', self.models['Random Forest']))
        # define meta learner model
        level1 = LogisticRegression()
        # define the stacking ensemble
        model = StackingClassifier(estimators=level0, final_estimator=level1, cv=loo)
        self.models['ensamble_knn_rf'] = model
        return model
    
    def final_model(self, model_name: str, feature_list: list[int]):
        class0_acc, class1_acc, class2_acc, accuracies, pred_y, prob_y, kappa = self.train_evaluate_model(model_name, self.models[model_name], loo, self.X[:, feature_list], self.Y, True)
        self.feature_names = list(self.feature_names[i] for i in feature_list)
        self.used_feature_indx = feature_list
        return accuracies, pred_y, prob_y, class0_acc, class1_acc, class2_acc, kappa

'''
# Classification procedure
# Model selection

import individual_differences_plot as plot
import numpy as np
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
from xgboost import XGBClassifier

import traceback

names = ["Logitic Reg", "Nearest Neighbors", 
          "Random Forest", 
         "Naive Bayes", 'Xgboost']

classifiers = [
    LogisticRegression(random_state=0, solver='liblinear'),
    KNeighborsClassifier(n_neighbors=5, weights='distance'),
    RandomForestClassifier(max_depth=5, n_estimators=50, max_features=1),
    GaussianNB(),
    XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)]
# evaluate setting
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import confusion_matrix

class ClassificationProcessor():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def train_evaluate_model(self, classifier_name, classifier, cross_validation, complete_x, complete_y):
        accuracies = cross_val_score(classifier, complete_x, complete_y, scoring='f1_weighted', cv=cross_validation, n_jobs=-1)
        pred_y = cross_val_predict(classifier, complete_x, complete_y, cv=cross_validation, n_jobs=-1)
        mean_acc = np.mean(accuracies)
        prob_y = cross_val_predict(classifier, complete_x, complete_y, cv=cross_validation, n_jobs=-1, method='predict_proba')
        print('{} Accuracy: {:.3f} ({:.3f})'.format(classifier_name, np.mean(accuracies), np.std(accuracies)))
        print('f1 score: {:.3f}'.format(f1_score(complete_y,pred_y, average='macro')))
        print('precision: {:.3f}'.format(precision_score(complete_y,pred_y, average='macro')))
        print('recall: {:.3f}'.format(recall_score(complete_y,pred_y, average='macro')))

        # plot confusion matrix
        cfm = confusion_matrix(complete_y[:344], pred_y[:344], labels=[0,1,2])
        print('{}: {}'.format(classifier_name, cfm))
        plot.plot_confusion_matrix(classifier_name, cfm, classes=[0, 1, 2])
        plot.plot_error_confusion(cfm)
        
        # plot learning curve
        plot.plot_learning_curve(classifier, cross_validation, complete_x, complete_y)
    
    def model_selection(self):
        for name, clf in zip(names, classifiers):
            try:
              self.train_evaluate_model(name, clf, skf, self.X, self.Y)
            except Exception as e:
              print('{}: got error: {}'.format(name, traceback.format_exc()))
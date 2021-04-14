import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import itertools
from yellowbrick.model_selection import LearningCurve

plot_folder_loc = '/datasets/googledrive/Yi_UCI_research/GSR other works/2020 Summer_predict individual training/plot'
colors = ['#4285F4','#DB4437','#F4B400','#0F9D58','m', 'y', 'k', 'w']

# plots about intervention data clustering
# piecewise lin plot
# predict for the determined points
def piecewise_lin_plot(x,y, pwlf, index):
    xHat = np.linspace(min(x), max(x), num=100)
    yHat = pwlf.predict(xHat)

    # plot the results
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.savefig(plot_folder_loc + '/piece_wise/fit_' + str(index) + '.png')

# plot cluster result
def plot_cluster_result(data: DataFrame, intervention_col_names: list[str]):
    df_cluster_type_list = []
    for i in range(len(data['label'].unique())):
        t = data[data['label'] == i][intervention_col_names]
        df_cluster_type_list.append(t)
        # each cluster plot
        fig, ax = plt.subplots(figsize=(9,4))
        x = range(1,11)
        ax.set_ylim(ymin=1, ymax=15)
        ax.set_xlim(xmin=1, xmax=10)
        ax.tick_params(labelsize=15)
        for index, row in t.iterrows():
            y = row
            ax.plot(x, y, linewidth=2, color=colors[i])
        plt.show()
    # plot clusters in one figure
    fig, ax = plt.subplots(figsize=(9,10))
    for i in range(len(df_cluster_type_list)):
        t_i = df_cluster_type_list[i]
        df_mean = t_i.mean()
        df_se = t_i.sem()
        t = pd.DataFrame({'mean': df_mean, 'se': df_se})
        x = range(1,11)
        y = t['mean']
        yerr = t['se']
        ci = 1.96 * np.std(y)/np.mean(y)
        label = 'cluster_' + str(i)
        ax.errorbar(x, y, yerr=yerr, color = colors[i],
                    elinewidth = 4, capsize = 4, label=label, linewidth=8)
        # ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
        ax.set_xlabel('Training Session', fontsize=24)
        ax.set_ylabel('N-back level', fontsize=24)
        # ax.set_title('cluster_using max, slope and sd', fontsize=28)
        ax.set_xlim(1,10)
        # ax.set_ylim(1,8)
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=20)
    fig.show()

# plot scatter plot for clustering
def plot_scatter_cluster(data: DataFrame, plot_col: list[str]):
    fig, ax = plt.subplots(figsize=(9,10))
    for i in range(len(data['label'].unique())):
        x = data[plot_col[0]][data['label']==i]
        y = data[plot_col[1]][data['label']==i]
        label = 'cluster_' + str(i)
        ax.scatter(x,y,c=colors[i], label=label)
        ax.set_xlabel(plot_col[0], fontsize=20)
        ax.set_ylabel(plot_col[1], fontsize=20)
        ax.legend(fontsize=20)
        # ax.set_title(cluster_result, fontsize=24)
        fig.show()

# bar plot   
def bar_plot_cluster(data: DataFrame, clustering_col_names: list[str]):
    for i in clustering_col_names:
        fig, ax = plt.subplots(figsize=(6,5))
        ax = sns.barplot(x="label", y=i, data=data, palette=colors)
        fig.show()

# plot features
# distribution of features
def plot_distribution(data: DataFrame, plot_col: list[str]):
    for feature in plot_col:
        fig, ax = plt.subplots()
        ax.hist(data[feature])
        ax.set_title(feature)

# category features & label
def cate_bar_plot(data: DataFrame, plot_col: list[str]):
    for feature in plot_col:
        df_plot = data.groupby(['label', feature]).size().reset_index().pivot(columns='label', index=feature, values=0)
        df_plot.plot(kind='bar', stacked=True)
        plt.show()

# confusion matrix
def plot_confusion_matrix(classifier_name, cm, classes, normalize=True):
    plt.figure(figsize=(10,6))
    cmap = plt.cm.Blues
    title = "Confusion Matrix"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j],
               horizontalalignment="center",
               color="black") 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(classifier_name)

def plot_error_confusion(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_conf_mx = cm / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()

# learning curve
def plot_learning_curve(classifier, cv, X, Y):
    sizes = np.linspace(0.3, 1.0, 10)
    visualizer = LearningCurve(classifier, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4)
    visualizer.fit(X, Y)        # Fit the data to the visualizer
    visualizer.show()           # Finalize and render the figure
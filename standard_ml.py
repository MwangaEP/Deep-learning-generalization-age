
#%%
# This program uses standard machine learning to predict the age structure of
# of Anopheles arabiensis mosquitoes reared in different insectaries (Ifakara and Glasgow)

# Principal component analysis is used to reduce the dimensionality of the data

# import all libraries

import this
import os
import io
import ast
import itertools
import collections
from time import time
from tqdm import tqdm

from itertools import cycle
import pickle
import random as rn
import datetime

import numpy as np
import pandas as pd

from random import randint
from collections import Counter

from sklearn.model_selection import ShuffleSplit, train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score, precision_recall_fscore_support

from imblearn.under_sampling import RandomUnderSampler

from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=None)
%matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

# This normalizes the confusion matrix and ensures neat plotting for all outputs.
# Function for plotting confusion matrcies

def plot_confusion_matrix(cm, classes,
                          normalize = True,
                          title = 'Confusion matrix',
                          xrotation=0,
                          yrotation=0,
                          cmap=plt.cm.Purples,
                          printout = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if printout:
            print("Normalized confusion matrix")
    else:
        if printout:
            print('Confusion matrix')

    if printout:
        print(cm)

    plt.figure(figsize=(6,4))

    plt.imshow(cm, interpolation='nearest', vmin = .2, vmax= 1.0,  cmap=cmap)
    plt.title(title)
    plt.colorbar()
    classes = classes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', weight = 'bold')
    plt.xlabel('Predicted label', weight = 'bold')
    plt.savefig(("./data/Fold\Standard_ml\_ml_tsne_0\Confusion_Matrix_" + figure_name + "_" + ".png"), dpi = 500, bbox_inches="tight")


#%%
# Visualizing outputs
# for visualizing confusion matrix once the model is trained

def visualize(figure_name, classes, predicted, true):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
#     print('visualize predicted classes', predicted)
#     print('visualize true classes', true)
    classes_pred = np.asarray(predicted)
    classes_true = np.asarray(true)
    print(classes_pred.shape)
    print(classes_true.shape)
    classes = ['1-9', '10-17']
    cnf_matrix = confusion_matrix(classes_true, classes_pred, labels = classes)
    plot_confusion_matrix(cnf_matrix, classes)


#%%
# importing dataframe
# read the full ifakara dataset

df = pd.read_csv("./data/Ifakara_data.dat", delimiter = '\t')
print(df.head())

# Checking class distribution in the data
print(Counter(df["Age"]))

# drops columns of no interest
df = df.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
df.head(10)

#%%

# reading 5% of the glasgow training data from the disk
# df_3 = pd.read_csv("./data/set_to_train_glasgow_05.csv")
# print(df_3.head())

# # Checking class distribution in the data
# print(Counter(df_3["Age"]))

# # drops columns of no interest
# df_3 = df_3.drop(['Unnamed: 0'], axis = 1)
# df_3.head(10)

# #%%

# # Concatinate 5% of  glasgow training data into full ifakara data before
# # training

# training_data = pd.concat([df, df_3], axis = 0, join = 'outer')

# # Checking the shape of the training data
# print('shape of training_data : {}'.format(training_data.shape))

# # print first 10 observations
# print('first ten observation of the training_data : {}'.format(training_data.head(10)))


# if we are not interested in intergrating glasgow data into ifakara data, we will just
# assign ifakara data to training data

training_data = df

# check last ten observations of the training data
training_data.tail(10)


#%%

# Renaming the age group into three classes

Age_group = []

# for row in training_data['Age']:
#     if row <= 5:
#         Age_group.append('1 - 5')

#     elif row > 5 and row <= 10:
#         Age_group.append('6 - 10')

#     else:
#         Age_group.append('11 - 17')

for row in training_data['Age']:
    if row <= 9:
        Age_group.append('1-9')

    else:
        Age_group.append('10-17')

print(Age_group)

training_data['Age_group'] = Age_group

# drop the column with Chronological Age and keep the age structure
training_data = training_data.drop(['Age'], axis = 1)
training_data.head(5)

#%%
# Dimension reduction with principal component analysis

# The idea here is to reduce the dimensianality of a dataset consisting of a large number
# of related variables while retaining as much variance in the data as possible. The algorthm
# finds a set of new varibles (principal componets) that the original variables are just
# linear combinations.

# define X (matrix of features) and y (list of labels)

X = training_data.iloc[:,:-1] # select all columns except the first one
y = training_data["Age_group"]

print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))

# creare a pipeline with standard scaler and PCA
seed = 42

pca_pipe = Pipeline([('scaler', StandardScaler()),
                      ('pca', decomposition.PCA(n_components = 8))])

# Use the pipeline to transform the training data
age_pca = pca_pipe.fit_transform(X)
print('First five observation : {}'.format(age_pca[:5]))

# Explore the explained variance
explained_var = pca_pipe.named_steps['pca'].explained_variance_ratio_
print('Explained variance : {}'.format(explained_var))

# transform X matrix with 8 number of components and y list of labels as arrays

X = np.asarray(age_pca)
y = np.asarray(y)
print(np.unique(y))


#%%

# define parameters
num_folds = 5 # split data into five folds
seed = 42 # seed value
scoring = 'accuracy' # metric for model evaluation

# specify cross-validation strategy
kf = KFold(n_splits = num_folds, shuffle = True, random_state = seed)

# make a list of models to test
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegressionCV(multi_class = 'auto', cv = kf, max_iter = 100, random_state = seed)))
models.append(('SVM', SVC(kernel = 'linear', gamma = 'auto', random_state = seed)))
# models.append(('RF', RandomForestClassifier(n_estimators = 500, random_state = seed)))
# models.append(('XGB', XGBClassifier(random_state = seed, n_estimators = 500)))

#%%

# Evaluate models to get the best perfoming model

results = []
names = []

skf = StratifiedKFold()

for name, model in models:
    cv_results = cross_val_score(
        model, X, y, cv = kf, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = 'Cross validation score for {0}: {1:.2%}'.format(
        name, cv_results.mean(), cv_results.std()
    )
    print(msg)

#%%

# Plotting the algorithm selection

sns.set(context = 'paper',
        style = 'whitegrid',
        palette = 'deep',
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Dejavu Sans'}))

plt.figure(figsize = (6, 4))
sns.boxplot(x = names, y = results, width = .4)
sns.despine(offset = 10, trim = True)
plt.xticks(rotation = 90)
plt.yticks(np.arange(0.2, 1.0 + .1, step = 0.1))
# plt.ylim(np.arange())
plt.ylabel('Accuracy', weight = 'bold')
plt.tight_layout()
# plt.savefig("./data/Fold\Standard_ml\selection_model_binary.png", dpi = 500, bbox_inches="tight")


# %%
# train XGB classifier and tune its hyper-parameters with randomized grid search

classifier = XGBClassifier()

# set hyparameter

estimators = [100, 500, 1000]
rate = [0.05, 0.10, 0.15, 0.20, 0.30]
depth = [2, 3, 4, 5, 6, 8, 10, 12, 15]
child_weight = [1, 3, 5, 7]
gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

param_grid = dict(n_estimators = estimators, learning_rate = rate, max_depth = depth,
                min_child_weight = child_weight, gamma = gamma, colsample_bytree = bytree)


# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores

save_predicted = [] # save predicted values for plotting averaged confusion matrix
save_true = [] # save true values for plotting averaged confusion matrix
num_rounds = 20


start = time()

for round in range(num_rounds):
    SEED = np.random.randint(0, 81470)

    for train_index, test_index in kf.split(X, y):

        # Split data into test and train

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # check the shape the splits
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        # generate models using all combinations of settings

        # RANDOMSED GRID SEARCH
        n_iter_search = 10
        rsCV = RandomizedSearchCV(verbose = 1,
                    estimator = classifier, param_distributions = param_grid, n_iter = n_iter_search,
                                scoring = scoring, cv = kf)

        rsCV_result = rsCV.fit(X_train, y_train)

        # print out results and give hyperparameter settings for best one
        means = rsCV_result.cv_results_['mean_test_score']
        stds = rsCV_result.cv_results_['std_test_score']
        params = rsCV_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))

        # print best parameter settings
        print("Best: %.2f using %s" % (rsCV_result.best_score_,
                                    rsCV_result.best_params_))

        # Insert the best parameters identified by randomized grid search into the base classifier
        classifier = XGBClassifier(**rsCV_result.best_params_)

        # Fitting the best classifier
        classifier.fit(X_train, y_train)

        # Predict X_test
        y_pred = classifier.predict(X_test)

        # Summarize outputs for plotting averaged confusion matrix

        for predicted, true in zip(y_pred, y_test):
            save_predicted.append(predicted)
            save_true.append(true)

        # summarize for plotting per class distribution

        classes = ['1-9', '10-17']
        local_cm = confusion_matrix(y_test, y_pred, labels = classes)
        local_report = classification_report(y_test, y_pred, labels = classes)

        local_kf_results = pd.DataFrame([("Accuracy", accuracy_score(y_test, y_pred)),
                                        ("params",str(rsCV_result.best_params_)),
                                        ("TRAIN",str(train_index)),
                                        ("TEST",str(test_index)),
                                        ("CM", local_cm),
                                        ("Classification report",
                                        local_report)]).T

        local_kf_results.columns = local_kf_results.iloc[0]
        local_kf_results = local_kf_results[1:]
        kf_results = kf_results.append(local_kf_results)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred, labels = classes)[3]
        local_acc = np.diag(local_cm)/local_support
        kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))


# %%

# plot confusion averaged for the validation set
figure_name = 'validation_0'
classes = np.unique(np.sort(y))
visualize(figure_name, classes, save_true, save_predicted)


# %%
# preparing dataframe for plotting per class accuracy

classes = ['1-9', '10-17']
rf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns = classes)
rf_per_class_acc_distrib.dropna().to_csv("./data/Fold\Standard_ml\_ml_pca_5\_rf_per_class_acc_distrib.csv")
rf_per_class_acc_distrib = pd.read_csv("./data/Fold\Standard_ml\_ml_pca_5\_rf_per_class_acc_distrib.csv", index_col=0)
rf_per_class_acc_distrib = np.round(rf_per_class_acc_distrib, 1)
rf_per_class_acc_distrib_describe = rf_per_class_acc_distrib.describe()
rf_per_class_acc_distrib_describe.to_csv("./data/Fold\Standard_ml\_ml_pca_5\_rf_per_class_acc_distrib.csv")

#%%
# plotting class distribution
sns.set(context = 'paper',
        style = 'whitegrid',
        palette = 'deep',
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Dejavu Sans'}))

plt.figure(figsize = (6, 4))

rf_per_class_acc_distrib = pd.melt(rf_per_class_acc_distrib, var_name="Label new")
# g = sns.pointplot(x="Label new", y="value", join = False, hue = "Label new",
#                 capsize = .1, scale= 4.5, errwidth = 4,
                # data = rf_per_class_acc_distrib)
g = sns.violinplot(x="Label new", y="value", hue = "Label new",
                data = rf_per_class_acc_distrib)

sns.despine(left=True)
plt.xticks(ha="right")
plt.yticks()
plt.ylim(ymin=0.5, ymax=1.0)
plt.xlabel(" ")
g.legend().set_visible(False)
# plt.legend(' ', frameon = False)
plt.ylabel("Prediction accuracy", weight = 'bold')
plt.grid(False)
plt.tight_layout()
plt.savefig("./data/Fold\Standard_ml\_ml_pca_5\_rf_per_class_acc_distrib.png", dpi = 500, bbox_inches="tight")

#%%

# save the trained model to disk for future use

with open('./data/Fold\Standard_ml\_ml_pca_5\classifier.pkl', 'wb') as fid:
     pickle.dump(classifier, fid)


# %%
# Loading new dataset for prediction (Glasgow dataset)
# start by loading the new test data

df_new = pd.read_csv("./data/set_to predict_glasgow_05.csv")
print(df_new.head())

# when predicting the whole glasgow dataset
# read full glasgow dataset

# df_2 = pd.read_csv("./data/glasgow_data.dat", delimiter = '\t')
# print(df_2.head())

# # Checking class distribution in glasgow data
# print(Counter(df_2["Age"]))

# # drops columns of no interest
# df_2 = df_2.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
# print(df_2.head(10))

# when predicting the whole glasgow and no glasgow data was intergrated in the trainig
# assign a full glasgow dataset to df_new

# df_new = df_2

# Checking class distribution in the data
print(Counter(df_new["Age"]))

# drops columns of no interest
df_new = df_new.drop(['Unnamed: 0'], axis=1)
df_new.head(10)



#%%
# Renaming the age group into three classes

Age_group_new = []

for row in df_new['Age']:
    if row <= 9:
        Age_group_new.append('1-9')

    else:
        Age_group_new.append('10-17')

print(Age_group_new)

df_new['Age_group_new'] = Age_group_new

# Drop age column which contain the chronological age of the mosquito, and
# keep age structure

df_new = df_new.drop(['Age'], axis = 1)
df_new.head(5)

#%%

# select X matrix of features and y list of labels

X_valid = df_new.iloc[:,:-1]
y_valid = df_new["Age_group_new"]

print('shape of X : {}'.format(X_valid.shape))
print('shape of y : {}'.format(y_valid.shape))

y_valid = np.asarray(y_valid)
print(np.unique(y_valid))

# tranform matrix of features with PCA

age_pca_valid = pca_pipe.fit_transform(X_valid)
print('First five observation : {}'.format(age_pca_valid[:5]))

# transform X and y matrices as arrays

age_pca_valid = np.asarray(age_pca_valid)
print(age_pca_valid)

#%%
# loading the classifier from the disk
with open('./data/Fold\Standard_ml\_ml_pca_5\classifier.pkl', 'rb') as fid:
     classifier_loaded = pickle.load(fid)

# generates output predictions based on the X_input passed

predictions = classifier_loaded.predict(age_pca_valid)

# Examine the accuracy of the model in predicting glasgow data

accuracy = accuracy_score(y_valid, predictions)
print("Accuracy:%.2f%%" %(accuracy * 100.0))

# compute precision, recall and f-score metrics

classes = ['1-9', '10-17']
cr_pca = classification_report(y_valid, predictions, labels = classes)
print(cr_pca)

#%%

# save classification report to disk as a csv

cr = pd.read_fwf(io.StringIO(cr_pca), header=0)
cr = cr.iloc[1:]
cr.to_csv('./data/Fold\Standard_ml\_ml_pca_5\classification_report_PCA_8_0.csv')

#%%

# plot the confusion matrix for the test data (glasgow data)
figure_name = 'test_0'
classes = np.unique(np.sort(y_valid))
visualize(figure_name, classes, predictions, y_valid)


##############################################
# Dimensionality reduction with t-SNE
##############################################

#%%
# Dimension reduction with t-Distributed Stochastic neigbour Embedding

# t-SNE a machine learning algorthims that converts similarities between
# data points to join probabilities, and tries to minimize the kullback-leibler
# divergence between the joint probabilities of the low-dimensional embedding and
# the high dimensional data.
#
# Drawback: It is possible to get different results with different initialization

# define X (matrix of features) and y (vector of labels)

start_time = time() # assess computational time the algorithm uses to transform data

X = training_data.iloc[:,:-1] # select all columns except the first one
y = training_data["Age_group"]
print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))

seed = 42

tsne_pipe = Pipeline([('scaler', StandardScaler()),
                      ('tsne', TSNE(n_components = 3,
                                    perplexity = 30,
                                    method='barnes_hut',
                                    random_state = seed))])

tsne_embedded = tsne_pipe.fit_transform(X)
print('First five tsne_embedded observation : {}'.format(tsne_embedded[:5]))
print('tsne_embedded shape : {}'.format(tsne_embedded.shape))

# transform X matrix with 10 number of components and y list of labels as arrays

X = np.asarray(tsne_embedded)
y = np.asarray(y)
print(np.unique(y))

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))

#%%
# train XGB classifier when dimensionality reduction is tsne and tune its hyper-parameters with randomized grid search

classifier = XGBClassifier()

# set hyparameter

estimators = [100, 500, 1000]
rate = [0.05, 0.10, 0.15, 0.20, 0.30]
depth = [2, 3, 4, 5, 6, 8, 10, 12, 15]
child_weight = [1, 3, 5, 7]
gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

param_grid = dict(n_estimators = estimators, learning_rate = rate, max_depth = depth,
                min_child_weight = child_weight, gamma = gamma, colsample_bytree = bytree)

# specify cross-validation strategy
num_folds = 5
scoring = 'accuracy' # metric for model evaluation
kf = KFold(n_splits = num_folds, shuffle = True, random_state = seed)

# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores

save_predicted = [] # save predicted values for plotting averaged confusion matrix
save_true = [] # save true values for plotting averaged confusion matrix

num_rounds = 20
start = time()


for round in range(num_rounds):
    SEED = np.random.randint(0, 81470)

    for train_index, test_index in kf.split(X, y):

        # Split data into test and train

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # check the shape the splits
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        # generate models using all combinations of settings

        # RANDOMSED GRID SEARCH
        n_iter_search = 10
        rsCV = RandomizedSearchCV(verbose = 1,
                    estimator = classifier, param_distributions = param_grid, n_iter = n_iter_search,
                                scoring = scoring, cv = kf)

        rsCV_result = rsCV.fit(X_train, y_train)

        # print out results and give hyperparameter settings for best one
        means = rsCV_result.cv_results_['mean_test_score']
        stds = rsCV_result.cv_results_['std_test_score']
        params = rsCV_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))

        # print best parameter settings
        print("Best: %.2f using %s" % (rsCV_result.best_score_,
                                    rsCV_result.best_params_))

        # Insert the best parameters identified by randomized grid search into the base classifier
        classifier = XGBClassifier(**rsCV_result.best_params_)

        # Fitting the best classifier
        classifier.fit(X_train, y_train)

        # Predict X_test
        y_pred = classifier.predict(X_test)

        # Summarize outputs for plotting averaged confusion matrix

        for predicted, true in zip(y_pred, y_test):
            save_predicted.append(predicted)
            save_true.append(true)

        # summarize for plotting per class distribution

        classes = ['1-9', '10-17']
        local_cm = confusion_matrix(y_test, y_pred, labels = classes)
        local_report = classification_report(y_test, y_pred, labels = classes)

        local_kf_results = pd.DataFrame([("Accuracy", accuracy_score(y_test, y_pred)),
                                        ("params",str(rsCV_result.best_params_)),
                                        ("TRAIN",str(train_index)),
                                        ("TEST",str(test_index)),
                                        ("CM", local_cm),
                                        ("Classification report",
                                        local_report)]).T

        local_kf_results.columns = local_kf_results.iloc[0]
        local_kf_results = local_kf_results[1:]
        kf_results = kf_results.append(local_kf_results)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred, labels = classes)[3]
        local_acc = np.diag(local_cm)/local_support
        kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# %%

# plot confusion averaged for the validation set
figure_name = 'validation_tsne_0'
classes = np.unique(np.sort(y))

sns.set(context = 'paper',
        style = 'whitegrid',
        palette = 'deep',
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Dejavu Sans'}))

visualize(figure_name, classes, save_true, save_predicted)


# %%
# preparing dataframe for plotting per class accuracy

classes = ['1-9', '10-17']
rf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns = classes)
rf_per_class_acc_distrib.dropna().to_csv("./data/Fold\Standard_ml\_ml_tsne_0\_rf_per_class_acc_distrib.csv")
rf_per_class_acc_distrib = pd.read_csv("./data/Fold\Standard_ml\_ml_tsne_0\_rf_per_class_acc_distrib.csv", index_col=0)
rf_per_class_acc_distrib = np.round(rf_per_class_acc_distrib, 1)
rf_per_class_acc_distrib_describe = rf_per_class_acc_distrib.describe()
rf_per_class_acc_distrib_describe.to_csv("./data/Fold\Standard_ml\_ml_tsne_0\_rf_per_class_acc_distrib.csv")

#%%
# plotting class distribution
sns.set(context = 'paper',
        style = 'whitegrid',
        palette = 'deep',
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Dejavu Sans'}))

plt.figure(figsize = (6, 4))

rf_per_class_acc_distrib = pd.melt(rf_per_class_acc_distrib, var_name="Label new")
# g = sns.pointplot(x="Label new", y="value", join = False, hue = "Label new",
#                 capsize = .1, scale= 4.5, errwidth = 4,
#                 data = rf_per_class_acc_distrib)


g = sns.violinplot(x="Label new", y="value", hue = "Label new",
                data = rf_per_class_acc_distrib)

sns.despine(left=True)
plt.xticks(ha="right")
plt.yticks()
plt.ylim(ymin=0.5, ymax=1.0)
plt.xlabel(" ")
g.legend().set_visible(False)
# plt.legend(' ', frameon = False)
plt.ylabel("Prediction accuracy", weight = 'bold')
plt.grid(False)
plt.tight_layout()
plt.savefig("./data/Fold\Standard_ml\_ml_tsne_0\_rf_per_class_acc_distrib.png", dpi = 500, bbox_inches="tight")

#%%

# save the trained model to disk for future use

with open('./data/Fold\Standard_ml\_ml_tsne_0\classifier.pkl', 'wb') as fid:
     pickle.dump(classifier, fid)


# %%
# Loading new dataset for prediction (Glasgow dataset)
# start by loading the new test data

# df_new = pd.read_csv("./data/set_to_predict_glasgow_05.csv")
# print(df_new.head())

# when predicting the whole glasgow dataset
# read full glasgow dataset

df_2 = pd.read_csv("./data/glasgow_data.dat", delimiter = '\t')
print(df_2.head())

# Checking class distribution in glasgow data
print(Counter(df_2["Age"]))

# drops columns of no interest
df_2 = df_2.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
print(df_2.head(10))

# when predicting the whole glasgow and no glasgow data was intergrated in the trainig
# assign a full glasgow dataset to df_new

df_new = df_2

# Checking class distribution in the data
# print(Counter(df_new["Age"]))

# # drops columns of no interest
# df_new = df_new.drop(['Unnamed: 0'], axis=1)
df_new.head(10)

#%%
# Renaming the age group into three classes

Age_group_new = []

for row in df_new['Age']:
    if row <= 9:
        Age_group_new.append('1-9')

    else:
        Age_group_new.append('10-17')

print(Age_group_new)

df_new['Age_group_new'] = Age_group_new

# Drop age column which contain the chronological age of the mosquito, and
# keep age structure

df_new = df_new.drop(['Age'], axis = 1)
df_new.head(5)

#%%

# select X matrix of features and y list of labels

X_valid = df_new.iloc[:,:-1]
y_valid = df_new["Age_group_new"]

print('shape of X : {}'.format(X_valid.shape))
print('shape of y : {}'.format(y_valid.shape))

y_valid = np.asarray(y_valid)
print(np.unique(y_valid))

# tranform matrix of features with tsne


age_tsne_valid = tsne_pipe.fit_transform(X_valid)
print('First five age_tsne_valid observation : {}'.format(age_tsne_valid[:5]))
print('age_tsne_valid shape : {}'.format(age_tsne_valid.shape))
# transform X and y matrices as arrays

age_tsne_valid = np.asarray(age_tsne_valid)
print(age_tsne_valid)

#%%
# loading the classifier from the disk
with open('./data/Fold\Standard_ml\_ml_tsne_0\classifier.pkl', 'rb') as fid:
     classifier_loaded = pickle.load(fid)

# generates output predictions based on the X_input passed

predictions = classifier_loaded.predict(age_tsne_valid)

# Examine the accuracy of the model in predicting glasgow data

accuracy = accuracy_score(y_valid, predictions)
print("Accuracy:%.2f%%" %(accuracy * 100.0))

# compute precision, recall and f-score metrics

classes = ['1-9', '10-17']
cr_pca = classification_report(y_valid, predictions, labels = classes)
print(cr_pca)

#%%

# save classification report to disk as a csv

cr = pd.read_fwf(io.StringIO(cr_pca), header=0)
cr = cr.iloc[1:]
cr.to_csv('./data/Fold\Standard_ml\_ml_tsne_0\classification_report_tsne_.csv')

#%%

# plot the confusion matrix for the test data (glasgow data)
figure_name = 'test_tsne_0'
classes = np.unique(np.sort(y_valid))
visualize(figure_name, classes, predictions, y_valid)

# %%

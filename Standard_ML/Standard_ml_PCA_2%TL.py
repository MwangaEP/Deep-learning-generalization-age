
#%%
# This program uses standard machine learning to predict the age structure of 
# of Anopheles arabiensis mosquitoes reared in different insectaries (Ifakara and Glasgow) ( using 2% transfer learning)

# Principal component analysis is used to reduce the dimensionality of the data

# import all libraries

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
    plt.savefig(("C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\_ml_pca_2\Confusion_Matrix_" + figure_name + "_" + ".png"), dpi = 500, bbox_inches="tight")
   

#%%
# Visualizing outputs
# for visualizing confusion matrix once the model is trained

def visualize(figure_name, classes, predicted, true):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
    # print('visualize predicted classes', predicted)
    # print('visualize true classes', true)
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

training_data = pd.read_csv("C:\Mannu\QMBCE\Thesis\Ifakara_data.dat", delimiter = '\t')
print(training_data.head())

# Checking class distribution in the data
print(Counter(training_data["Age"]))

# drops columns of no interest
training_data = training_data.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
training_data.head(10)

#%%

# reading 2% of the glasgow training data from the disk 
glasgow_train_df = pd.read_csv("C:\Mannu\QMBCE\Thesis\set_to_train_glasgow_02.csv")
print(glasgow_train_df.head())
print(glasgow_train_df.shape)
# Checking class distribution in the data
print(Counter(glasgow_train_df["Age"]))

# drops columns of no interest
glasgow_train_df = glasgow_train_df.drop(['Unnamed: 0'], axis = 1)
glasgow_train_df.head(10)

#%%

# Concatinate 2% of  glasgow training data into full ifakara data before 
# training 

training_data = pd.concat([training_data, glasgow_train_df], axis = 0, join = 'outer')

# Checking the shape of the training data
print('shape of training_data : {}'.format(training_data.shape))

# print first 10 observations
print('first ten observation of the training_data : {}'.format(training_data.head(10)))


#%%

# Renaming the age group into three classes

Age_group = []

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

X = np.asarray(training_data.iloc[:,:-1]) # select all columns except the first one 
y = np.asarray(training_data["Age_group"])

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
models.append(('RF', RandomForestClassifier(n_estimators = 500, random_state = seed)))
models.append(('XGB', XGBClassifier(random_state = seed, n_estimators = 500)))

#%%

# Evaluate models to get the best perfoming model

results = []
names = []

skf = StratifiedKFold()

for name, model in models:
    cv_results = cross_val_score(
        model, age_pca, y, cv = kf, scoring = scoring)
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
plt.yticks(np.arange(0.2, 1.0 + .1, step = 0.2))
# plt.ylim(np.arange())
plt.ylabel('Accuracy', weight = 'bold')
plt.tight_layout()
plt.savefig("C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\selection_model_binary.png", dpi = 500, bbox_inches="tight")


# %%
# train XGB classifier and tune its hyper-parameters with randomized grid search 

classifier = XGBClassifier()

classifier_pipeline = Pipeline([('scaler', StandardScaler()),
                                ('pca', decomposition.PCA(n_components = 8)),
                                ('XGB', XGBClassifier())])


# set hyparameter

estimators = [100, 500, 1000]
rate = [0.05, 0.10, 0.15, 0.20, 0.30]
depth = [2, 3, 4, 5, 6, 8, 10, 12, 15]
child_weight = [1, 3, 5, 7]
gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

param_grid = {'XGB__n_estimators': estimators, 
              'XGB__learning_rate': rate, 
              'XGB__max_depth': depth,
              'XGB__min_child_weight': child_weight, 
              'XGB__gamma': gamma, 
              'XGB__colsample_bytree': bytree}


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
                    estimator = classifier_pipeline, param_distributions = param_grid, n_iter = n_iter_search, 
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
        classifier = classifier_pipeline.set_params(**rsCV_result.best_params_)

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
figure_name = 'validation_1'
classes = np.unique(np.sort(y))

# plotting class distribution
sns.set(context = 'paper',
        style = 'whitegrid',
        palette = 'deep',
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Dejavu Sans'}))

plt.figure(figsize = (6, 4))
visualize(figure_name, classes, save_true, save_predicted)


# %%
# preparing dataframe for plotting per class accuracy

classes = ['1-9', '10-17']
rf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns = classes)
rf_per_class_acc_distrib.dropna().to_csv("C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\_ml_pca_2\_rf_per_class_acc_distrib.csv")
rf_per_class_acc_distrib = pd.read_csv("C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\_ml_pca_2\_rf_per_class_acc_distrib.csv", index_col=0)
rf_per_class_acc_distrib = np.round(rf_per_class_acc_distrib, 1)
rf_per_class_acc_distrib_describe = rf_per_class_acc_distrib.describe()
rf_per_class_acc_distrib_describe.to_csv("C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\_ml_pca_2\_rf_per_class_acc_distrib.csv")

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
plt.savefig("C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\_ml_pca_2\_rf_per_class_acc_distrib.png", dpi = 500, bbox_inches="tight")

#%%

# save the trained model to disk for future use

with open('C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\_ml_pca_2\classifier.pkl', 'wb') as fid:
     pickle.dump(classifier, fid)


# %%
# Loading new dataset for prediction (Glasgow dataset)
# start by loading the new test data 

glasgow_unseen_df = pd.read_csv("C:\Mannu\QMBCE\Thesis\set_to_predict_glasgow_02.csv")
print(glasgow_unseen_df.head())

# Checking class distribution in the data
print(Counter(glasgow_unseen_df["Age"]))

# drops columns of no interest
glasgow_unseen_df = glasgow_unseen_df.drop(['Unnamed: 0'], axis=1)
glasgow_unseen_df.head(10)

#%%
# Renaming the age group into three classes

Age_group_new = []

for row in glasgow_unseen_df['Age']:
    if row <= 9:
        Age_group_new.append('1-9')
    
    else:
        Age_group_new.append('10-17')

print(Age_group_new)

glasgow_unseen_df['Age_group_new'] = Age_group_new

# Drop age column which contain the chronological age of the mosquito, and 
# keep age structure

glasgow_unseen = glasgow_unseen_df.drop(['Age'], axis = 1)
glasgow_unseen.head(5)

#%%

# select X matrix of features and y list of labels

X_valid = np.asarray(glasgow_unseen_df.iloc[:,:-1])
y_valid = np.asarray(glasgow_unseen_df["Age_group_new"])

print('shape of X : {}'.format(X_valid.shape))
print('shape of y : {}'.format(y_valid.shape))
print(np.unique(y_valid))

#%%
# loading the classifier from the disk
with open('C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\_ml_pca_2\classifier.pkl', 'rb') as fid:
     classifier_loaded = pickle.load(fid)

# generates output predictions based on the X_input passed

predictions = classifier_loaded.predict(X_valid)

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
cr.to_csv('C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\_ml_pca_2\classification_report_PCA_8_0.csv')

#%%

# plot the confusion matrix for the test data (glasgow data)
figure_name = 'test_ug_2'
classes = np.unique(np.sort(y_valid))
visualize(figure_name, classes, predictions, y_valid)


#%%
# This program uses standard machine learning to predict the age structure of 
# of Anopheles arabiensis mosquitoes.

# Principal component analysis is used to reduce the dimensionality of the data

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

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    classes = classes[0]
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
    plt.savefig(("C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\Confusion_Matrix" + "_" +  figure_name + "_ " + ".png"), dpi = 500, bbox_inches="tight")
   

#%%
# Visualizing outputs
# for visualizing losses and metrics once the neural network fold is trained
def visualize(classes, figure_name, predicted, true):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
#     print('visualize predicted classes', predicted)
#     print('visualize true classes', true)
    classes_pred = np.argmax(predicted, axis=-1)
    classes_true = np.argmax(true, axis=-1)
    print(classes_pred.shape)
    print(classes_true.shape)
    cnf_matrix = confusion_matrix(classes_true, classes_pred)
    plot_confusion_matrix(cnf_matrix, classes)


#%%
# read the full ifakara dataset
df = pd.read_csv("C:\Mannu\QMBCE\Thesis\Ifakara_data.dat", delimiter = '\t')
# df = pd.read_csv("D:\QMBCE\Thesis\set_training.csv")
print(df.head())

# Checking class distribution in the data
print(Counter(df["Age"]))

# drops columns of no interest
df = df.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
# df = df.drop(['Unnamed: 0'], axis = 1)
df.head(10)

#%%

# reading the glasgow training data from the disk
df_3 = pd.read_csv("C:\Mannu\QMBCE\Thesis\set_to_train_glasgow_05.csv")
print(df_3.head())

# Checking class distribution in the data
print(Counter(df_3["Age"]))

# drops columns of no interest
df_3 = df_3.drop(['Unnamed: 0'], axis = 1)
df_3.head(10)

#%%

# intergrating the glasgow training data into full ifakara data before 
# training 

training_data = pd.concat([df, df_3], axis = 0, join = 'outer')

# Checking the shape of the training data
print('shape of training_data : {}'.format(training_data.shape))

# print first 10 observations
print('first ten observation of the training_data : {}'.format(training_data.head(10)))

# check last ten observations of the training data
training_data.tail(10)

#%%
# Dimension reduction with principle component analysis

# The idea here is to reduce the dimensianality of a dataset consisting of a large number 
# of related variables while retaining as much variance in the data as possible. The algorthm
# finds a set of new varibles (principle componets) that the original variables are just 
# linear combinations.

# define X (matrix of features) and y (list of labels)

X = training_data.iloc[:,1:] # select all columns except the first one 
y = training_data["Age"]

print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))
seed = 4

pca_pipe = Pipeline([('scaler', StandardScaler()),
                      ('pca', decomposition.PCA(n_components = 8, random_state = seed))])

age_pca = pca_pipe.fit_transform(X)
print('First five observation : {}'.format(age_pca[:5]))

explained_var = pca_pipe.named_steps['pca'].explained_variance_ratio_
print('Explained variance : {}'.format(explained_var))

# transform X matrix with 10 number of components and y list of labels as arrays

X = np.asarray(age_pca)
y = np.asarray(y)
print(np.unique(y))


# %%

# Renaming the age group into three classes
# Oganises the data into a format of lists of data, classes, labels.

y_age_group = np.where((y <= 5), 0, 0)
y_age_group = np.where((y >= 6) & (y <= 10), 1, y_age_group)
y_age_group = np.where((y >= 11), 2, y_age_group)

y_age_groups_list = [[ages] for ages in y_age_group]
age_group = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list))
age_group_classes = ["1-5", "6-10", "11-17"] 

# Labels default - all classification
labels, classes_default = [age_group], [age_group_classes]


#%%

# define parameters
num_folds = 5 # split data into five folds
seed = 4 # seed value
scoring = 'accuracy' # metric for model evaluation
# validation_size = 0.15

# specify cross-validation strategy
kf = KFold(n_splits = num_folds, shuffle = True, random_state = seed)

# make a list of models to test
models = []
models.append(('KNN', KNeighborsClassifier()))
# models.append(('LR', LogisticRegressionCV(multi_class = 'auto', cv = kf, max_iter = 2000, random_state = seed)))
# models.append(('SVM', SVC(kernel = 'rbf', gamma = 'auto', random_state = seed)))
models.append(('RF', RandomForestClassifier(n_estimators = 1000, random_state = seed)))
# models.append(('XGB', XGBClassifier(n_thread = 1, objective = 'multi:softmax', random_state = seed)))
# models.append(('MLP', MLPClassifier(max_iter = 2000, solver = 'sdg', random_state = seed)))

#%%

results = []
names = []

for name, model in models:
    cv_results = cross_val_score(
        model, X, age_group, cv = kf, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = 'Cross validation score for {0}: {1:.2%}'.format(
        name, cv_results.mean(), cv_results.std()
    )
    print(msg)

#%%
sns.set(context = 'paper',
        style = 'whitegrid',
        palette = 'deep',
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Dejavu Sans'}))

plt.figure(figsize = (6, 4))
sns.boxplot(x = names, y = results, width = .3)
sns.despine(offset = 10, trim = True)
plt.xticks(rotation = 90)
plt.yticks()
plt.ylabel('Accuracy', weight = 'bold')
plt.tight_layout()
plt.savefig("C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\selection_model.png", dpi = 500, bbox_inches="tight")

# %%
# train Random forest 

classifier = RandomForestClassifier(n_estimators=100, 
                                    criterion = "gini", 
                                    max_depth = None, 
                                    min_samples_split = 2, 
                                    min_samples_leaf = 1, 
                                    min_weight_fraction_leaf = 0., 
                                    max_features = "auto", 
                                    max_leaf_nodes = None, 
                                    min_impurity_decrease = 0., 
                                    min_impurity_split = None, 
                                    bootstrap = True, 
                                    oob_score = False, 
                                    n_jobs = None, 
                                    random_state = None, 
                                    verbose = 1, 
                                    warm_start = False, 
                                    class_weight = None, 
                                    ccp_alpha = 0.0, 
                                    max_samples = None)

# set hyparameter

estimators = [100, 500, 1000]
starts = [True, False]

param_grid = dict(n_estimators = estimators, warm_start = starts)


# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores

save_predicted = []
save_true = []

start = time()


for train_index, test_index in kf.split(X):

    # Split data into test and train

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = list(map(lambda y:y[train_index], labels)), list(map(lambda y:y[test_index], labels))

    
    y_train = np.asarray(y_train)
    y_train = np.squeeze(y_train)

    print(X_train.shape)
    print(y_train.shape)
    # generate models using all combinations of settings

    grid = GridSearchCV(estimator = classifier, param_grid = param_grid, scoring = scoring, cv = kf)
    grid_result = grid.fit(X_train, y_train)
    
    # print out results and give hyperparameter settings for best one
    
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%.2f (%.2f) with: %r" % (mean, stdev, param))

    ## print best parameter settings
    print("Best: %.2f using %s" % (grid_result.best_score_, grid_result.best_params_))

    classifier = RandomForestClassifier(**grid_result.best_params_)

    # Fitting the best classifier
    classifier.fit(X_train, y_train)

    # Predict X_test
    y_pred = classifier.predict(X_test)

    # change the dimension of y_test to array
    y_test = np.asarray(y_test)
    y_test = np.squeeze(y_test) # remove any single dimension entries from the arrays

    # Summarize outputs for plotting averaged confusion matrix

    for predicted, true in zip(y_pred, y_test):
        save_predicted.append(predicted)
        save_true.append(true)

    # summarize for plotting per class distribution

    local_cm = confusion_matrix(np.argmax(y_test, axis = -1), np.argmax(y_pred, axis = -1))
    local_report = classification_report(np.argmax(y_test, axis = -1), np.argmax(y_pred, axis = -1))

    local_kf_results = pd.DataFrame([("Accuracy", accuracy_score(np.argmax(y_test, axis = -1),
                                                                 np.argmax(y_pred, axis = -1))),
                                      ("params",str(grid_result.best_params_)),
                                      ("TRAIN",str(train_index)),
                                      ("TEST",str(test_index)),
                                      ("CM", local_cm),
                                      ("Classification report",
                                       local_report)]).T

    local_kf_results.columns = local_kf_results.iloc[0]
    local_kf_results = local_kf_results[1:]
    kf_results = kf_results.append(local_kf_results)

    # # per class accuracy
    # local_support = precision_recall_fscore_support(np.argmax(y_test, axis = -1), np.argmax(y_pred, axis = -1))[3]
    # local_acc = np.diag(local_cm)/local_support
    # kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))


# %%

# plot confusion averaged for the validation set
figure_name = 'validation'
visualize(classes_default, figure_name, save_true, save_predicted)

#%%

# save the trained model to disk for future use

with open('C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\classifier.pkl', 'wb') as fid:
     pickle.dump(classifier, fid)


# %%
# Loading new dataset for prediction 
# start by loading the new test data 

df_new = pd.read_csv("C:\Mannu\QMBCE\Thesis\set_to predict_glasgow_05.csv")
print(df_new.head())

# Checking class distribution in the data
print(Counter(df_new["Age"]))

# drops columns of no interest
df_new = df_new.drop(['Unnamed: 0'], axis=1)
df_new.head(10)

#%%

X_valid = df_new.iloc[:,1:]
y_valid = df_new["Age"]

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
# change labels

y_age_group_val = np.where((y_valid <= 5), 0, 0)
y_age_group_val = np.where((y_valid >= 6) & (y_valid <= 10), 1, y_age_group_val)
y_age_group_val = np.where((y_valid >= 11), 2, y_age_group_val)

y_age_groups_list_val = [[ages_val] for ages_val in y_age_group_val]
age_group_val = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list_val))
age_group_classes_val = ["1-5", "6-10", "11-17"]

labels_default_val, classes_default_val = [age_group_val], [age_group_classes_val]

#%%
# loading the classifier from the disk
with open('C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\classifier.pkl', 'rb') as fid:
     classifier_loaded = pickle.load(fid)


# change the dimension of y_test to array
y_validation = np.asarray(labels_default_val)
y_validation = np.squeeze(labels_default_val) # remove any single dimension entries from the arrays

# generates output predictions based on the X_input passed

predictions = classifier_loaded.predict(age_pca_valid)

accuracy = accuracy_score(np.argmax(y_validation, axis = -1), np.argmax(predictions, axis = -1))
print("Accuracy:%.2f%%" %(accuracy * 100.0))


cr_pca = classification_report(np.argmax(y_validation, axis=-1), np.argmax(predictions, axis=-1))
print(cr_pca)

#%%

# save classification report to disk as a csv

cr = pd.read_fwf(io.StringIO(cr_pca), header=0)
cr = cr.iloc[1:]
cr.to_csv('C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\classification_report_PCA_8_5%.csv')


#%%

# plot the confusion matrix for the test data (glasgow data)
figure_name = 'test'
visualize(classes_default_val, figure_name, predictions, y_validation)

# %%

# Using transfer learning coupled with dimensionality reduction to improve generalisability of 
# machine-learning predictions of mosquito ages from mid-infrared spectra

#%%
# Import libraries

import this
import os
import io
import ast
import json
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
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score

from imblearn.under_sampling import RandomUnderSampler

from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, metrics
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=None)
# %matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

# importing datasets
# read the full ifakara dataset

df = pd.read_csv("C:\Mannu\QMBCE\Thesis\Ifakara_data.dat", delimiter = '\t')
# df = pd.read_csv("D:\QMBCE\Thesis\set_training.csv")
print(df.head())

# Checking class distribution in Ifakara data
print(Counter(df["Age"]))

# drops columns of no interest
df = df.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
# df = df.drop(['Unnamed: 0'], axis = 1)
df.head(10)

#%%
# df_ifa_2 = pd.read_csv("C:\Mannu\QMBCE\Thesis\Arabiensis_lab_valid_matrix.dat", delimiter = '\t')
# print(df_ifa_2.head())

# # Checking class distribution in Ifakara data
# print(Counter(df_ifa_2["Cat3"]))

# # drops columns of no interest
# df_ifa_2 = df_ifa_2.drop(['Cat1', 'Cat2', 'Cat4', 'Cat5', 'Cat6', 'StoTime'], axis=1)
# # df = df.drop(['Unnamed: 0'], axis = 1)
# df_ifa_2.head(10)

# #%%
# Age = []

# for row in df_ifa_2['Cat3']:
#     if row == '01D':
#         Age.append(1)
    
#     elif row == '02D':
#         Age.append(2)
    
#     elif row == '03D':
#         Age.append(3)

#     elif row == '04D':
#         Age.append(4)

#     elif row == '05D':
#         Age.append(5)

#     elif row == '06D':
#         Age.append(6)

#     elif row == '07D':
#         Age.append(7)

#     elif row == '08D':
#         Age.append(8)

#     elif row == '09D':
#         Age.append(9)

#     elif row == '10D':
#         Age.append(10)

#     elif row == '11D':
#         Age.append(11)

#     elif row == '12D':
#         Age.append(12)

#     elif row == '13D':
#         Age.append(13)

#     elif row == '14D':
#         Age.append(14)

#     else:
#         Age.append(16)

# print(Age)

# df_ifa_2['Age'] = Age

# # drop the column with Chronological Age and keep the age structure
# df_ifa_2 = df_ifa_2.drop(['Cat3'], axis = 1) 
# df_ifa_2.head(5)

# #%%

# cols = df_ifa_2.columns.tolist()
# cols = cols[-1:] + cols[:-1]
# df_ifa_2 = df_ifa_2[cols]
# df_ifa_2

# #%%
# df = pd.concat([df, df_ifa_2], axis = 0, join = 'outer')

# # Checking the shape of the training data
# print('shape of training_data : {}'.format(df.shape))

# # print first 10 observations
# print('first ten observation of the training_data : {}'.format(df.head(10)))

# # check last ten observations of the training data
# df.tail(10)


#%%
# read full glasgow dataset

df_2 = pd.read_csv("C:\Mannu\QMBCE\Thesis\glasgow_data.dat", delimiter = '\t')
print(df_2.head())

# Checking class distribution in glasgow data 
print(Counter(df_2["Age"]))

# drops columns of no interest
df_2 = df_2.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
# df = df.drop(['Unnamed: 0'], axis = 1)
df_2.head(10)

#%%

# # spliting 5% and 2% set of the glasgow data and intergrate it to training data 
# # to allow CNN to learn for any differences and patterns from the spectra of mosquitoes 
# # reared in these two insectaries

# X_split = df_2.iloc[:,1:] # matrix of features
# y_split = df_2["Age"] # vector of labels
# print(X_split)

# seed = 42
# # size = 0.02 # split 2% of the glasgow data
# size = 0.05 # split 5% of the glasgow data

# rs = ShuffleSplit(n_splits = 10, test_size = size, random_state = seed)
# rs.get_n_splits(X_split)
# print(rs)

# for train_index_split, val_index_split in rs.split(X_split):
#     print("TRAIN:", train_index_split, "VALIDATION:", val_index_split)

# print(train_index_split.shape, val_index_split.shape)


# # saving glasgow set for prediction to disk
# set_to_predict = df.iloc[train_index_split,:]
# set_to_predict.to_csv("D:\QMBCE\Thesis\set_to_predict_glasgow_05.csv")

# # saving glasgow set to be concatinated in training to disk
# set_to_train = df.iloc[val_index_split,:]
# set_to_train.to_csv("D:\QMBCE\Thesis\set_to_train_glasgow_05.csv")

#%%

# since here the split regarded as validation here has only 5% of the glasgow dataset, 
# we will upload it here and concatinate it with the ifakara data for model training

# reading the 5% of the glasgow dataset from disk

df_3 = pd.read_csv("C:\Mannu\QMBCE\Thesis\set_to_train_glasgow_05.csv")
print(df_3.head())

print(df_3.shape)

# Checking class distribution in the data
print(Counter(df_3["Age"]))

# drops columns of no interest
df_3 = df_3.drop(['Unnamed: 0'], axis = 1)
df_3.head(10)

#%%

# Concat the glasgow training data into full ifakara data before 
# training 

training_data = pd.concat([df, df_3], axis = 0, join = 'outer')

# Checking the shape of the training data
print('shape of training_data : {}'.format(training_data.shape))

# print first 10 observations
print('first ten observation of the training_data : {}'.format(training_data.head(10)))

# check last ten observations of the training data
training_data.tail(10)

# if we are not interested in intergrating glasgow data into ifakara data, we will just 
# assign ifakara data to training data

# training_data = df

#%%

# create a new folder for the CNN outputs

def build_folder(Fold, to_build = False):
    if not os.path.isdir(Fold):
        if to_build == True:
            os.mkdir(Fold)
        else:
            print('Directory does not exists, not creating directory!')
    else:
        if to_build == True:
            raise NameError('Directory already exists, cannot be created!')

#%%

# This normalizes the confusion matrix and ensures neat plotting for all outputs.
# Function for plotting confusion matrcies

def plot_confusion_matrix(cm, classes, output, save_path, model_name, fold,
                          normalize=True,
                          title='Confusion matrix',
                          xrotation=0,
                          yrotation=0,
                          cmap=plt.cm.Purples,
                          printout=False):
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

    plt.imshow(cm, interpolation='nearest', vmin = 0.2, vmax = 1.0, cmap = cmap)
    # plt.title([title +' - '+ model_name])
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
    plt.savefig((save_path + "Confusion_Matrix_" + model_name + "_" + fold +"_"+ ".png"), dpi = 500, bbox_inches="tight")
    plt.savefig((save_path + "Confusion_Matrix_" + model_name + "_" + fold +"_"+ ".pdf"), dpi = 500, bbox_inches="tight")
    plt.close()

#%%
# Visualizing outputs

# for visualizing losses and metrics once the neural network fold is trained
def visualize(histories, save_path, model_name, fold, classes, outputs, predicted, true):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
#     print('visualize predicted classes', predicted)
#     print('visualize true classes', true)
    classes_pred = np.argmax(predicted, axis=-1)
    classes_true = np.argmax(true, axis=-1)
    print(classes_pred.shape)
    print(classes_true.shape)
    cnf_matrix = confusion_matrix(classes_true, classes_pred)
    plot_confusion_matrix(cnf_matrix, classes, outputs, save_path, model_name, fold)

#%%
# Data logging
# for logging data associated with the model

def log_data(log, name, fold, save_path):
    f = open((save_path+name+'_'+str(fold)+'_log.txt'), 'w')
    np.savetxt(f, log)
    f.close()

#%%

# Graphing the training data and validation
 
def graph_history(history, model_name, model_ver_num, fold, save_path):
    #not_validation = list(filter(lambda x: x[0:3] != "val", history.history.keys()))
    print('history.history.keys : {}'.format(history.history.keys()))
    filtered = filter(lambda x: x[0:3] != "val", history.history.keys())
    not_validation = list(filtered)
    for i in not_validation:
        plt.figure(figsize=(6, 4))
        # plt.title(i+"/ "+"val_"+i)
        plt.plot(history.history[i], label=i)
        plt.plot(history.history["val_"+i], label="val_"+i)
        plt.legend()
        plt.tight_layout()
        plt.grid(False)
        plt.xlabel("epoch", weight = 'bold')
        plt.ylabel(i)
        plt.savefig(save_path +model_name+"_"+str(model_ver_num)+"_"+str(fold)+"_"+i + ".png", dpi = 500, bbox_inches="tight")
        plt.savefig(save_path +model_name+"_"+str(model_ver_num)+"_"+str(fold)+"_"+i + ".pdf", dpi = 500, bbox_inches="tight")
        plt.close()

#%%
# Graphing the averaged training and validation histories 
 
# when plotting, smooth out the points by some factor (0.5 = rough, 0.99 = smooth)
# method taken from `Deep Learning with Python` by François Chollet

def smooth_curve(points, factor = 0.75):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def set_plot_history_data(ax, history, which_graph):

    if which_graph == 'accuracy':
        train = smooth_curve(history['accuracy'])
        valid = smooth_curve(history['val_accuracy'])

    epochs = range(1, len(train) + 1)
        
    trim = 0 # remove first 5 epochs
    # when graphing loss the first few epochs may skew the (loss) graph
    
    ax.plot(epochs[trim:], train[trim:], 'b', label = ('accuracy'))
    ax.plot(epochs[trim:], train[trim:], 'b', linewidth = 15, alpha = 0.1)
    
    ax.plot(epochs[trim:], valid[trim:], 'orange', label = ('val_accuracy'))
    ax.plot(epochs[trim:], valid[trim:], 'orange', linewidth = 15, alpha = 0.1)


def graph_history_averaged(combined_history):
    print('averaged_histories.keys : {}'.format(combined_history.keys()))
    fig, (ax1) = plt.subplots(nrows = 1,
                             ncols = 1,
                             figsize = (6, 4),
                             sharex = True)

    set_plot_history_data(ax1, combined_history, 'accuracy')
    
    # Accuracy graph
    ax1.set_ylabel('Accuracy', weight = 'bold')
    plt.xlabel('Epoch', weight = 'bold')
    # ax1.set_ylim(bottom = 0.3, top = 1.0)
    ax1.legend(loc = 'lower right')
    ax1.set_yticks(np.arange(0.3, 1.0, step = 0.1))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_visible(True)

    plt.tight_layout()
    plt.grid(False)
    plt.savefig("C:\Mannu\QMBCE\Thesis\Fold\_4comps_tsne_05_k_fold_publish_01\Averaged_graph.png", dpi = 500, bbox_inches="tight")
    plt.close()

#%%
# This takes a list of dictionaries, and combines them into a dictionary in which each key maps to a 
# list of all the appropriate values from the parameter dictionaries

def combine_dictionaries(list_of_dictionaries):
    
    combined_dictionaries = {}
    
    for individual_dictionary in list_of_dictionaries:
        
        for key_value in individual_dictionary:
            
            if key_value not in combined_dictionaries:
                
                combined_dictionaries[key_value] = []
            combined_dictionaries[key_value].append(individual_dictionary[key_value])

    return combined_dictionaries


#%%

# right now, no error checking, when all lists are either of same length or not the same length

def find_mean_from_combined_dicts(combined_dicts):
    
    dict_of_means = {}

    for key_value in combined_dicts:
        dict_of_means[key_value] = []

        # Length of longest list return the longest list within the list of a dictionary item
        length_of_longest_list = max([len(a) for a in combined_dicts[key_value]])
        temp_array = np.empty([len(combined_dicts[key_value]), length_of_longest_list])
        temp_array[:] = np.NaN

        for i, j in enumerate(combined_dicts[key_value]):
            temp_array[i][0:len(j)] = j
        mean_value = np.nanmean(temp_array, axis=0)

        dict_of_means[key_value] = mean_value.tolist()
    
    return dict_of_means


#%%
# Function to create deep CNN

# This function takes as an input a list of dictionaries. Each element in the list is a new hidden layer in the model. For each 
# layer the dictionary defines the layer to be used.

def create_models(model_shape, input_layer_dim):
    
    # parameter rate for l2 regularization
    regConst = 0.02
    
    # defining a stochastic gradient boosting optimizer
    sgd = tf.keras.optimizers.SGD(lr = 0.001, momentum = 0.9, 
                                    nesterov = True, clipnorm = 1.)
    
    # define categorical_crossentrophy as the loss function (multi-class problem i.e. 3 age classes)
    cce = 'categorical_crossentropy'
    # bce = 'binary_crossentropy'

    # input shape vector

    # change the input shape to avoid learning feautures independently. By changing the input shape to 
    # (input_layer_dim, ) it will learn some combination of feautures with the learnable weights of the 
    # network

    input_vec = tf.keras.Input(name = 'input', shape = (input_layer_dim, )) 

    for i, layerwidth in zip(range(len(model_shape)),model_shape):
        if i == 0:
            if model_shape[i]['type'] == 'c':

                # Convolution1D layer, which will learn filters from spectra 
                # signals with maxpooling1D and batch normalization:

                xd = tf.keras.layers.Conv1D(name=('Conv'+str(i+1)), filters=model_shape[i]['filter'], 
                 kernel_size = model_shape[i]['kernel'], strides = model_shape[i]['stride'],
                 activation = 'relu',
                 kernel_regularizer = regularizers.l2(regConst), 
                 kernel_initializer = 'he_normal')(input_vec)
                xd = tf.keras.layers.BatchNormalization(name=('batchnorm_'+str(i+1)))(xd)
                xd = tf.keras.layers.MaxPooling1D(pool_size=(model_shape[i]['pooling']))(xd)
                
                # A hidden layer

            elif model_shape[i]['type'] == 'd':
                xd = tf.keras.layers.Dense(name=('d'+str(i+1)), units=model_shape[i]['width'], activation='relu', 
                 kernel_regularizer = regularizers.l2(regConst), 
                 kernel_initializer='he_normal')(input_vec)
                xd = tf.keras.layers.BatchNormalization(name=('batchnorm_'+str(i+1)))(xd) 
                xd = tf.keras.layers.Dropout(name=('dout'+str(i+1)), rate=0.4)(xd)

        else:
            if model_shape[i]['type'] == 'c':
                
                # convulational1D layer

                xd = tf.keras.layers.Conv1D(name=('Conv'+str(i+1)), filters=model_shape[i]['filter'], 
                 kernel_size = model_shape[i]['kernel'], strides = model_shape[i]['stride'],
                 activation = 'relu',
                 kernel_regularizer = regularizers.l2(regConst), 
                 kernel_initializer='he_normal')(xd)
                xd = tf.keras.layers.BatchNormalization(name=('batchnorm_'+str(i+1)))(xd)
                xd = tf.keras.layers.MaxPooling1D(pool_size=(model_shape[i]['pooling']))(xd)
                
            elif model_shape[i]['type'] == 'd':
                if model_shape[i-1]['type'] == 'c':
                    xd = tf.keras.layers.Flatten()(xd)
                    
                xd = tf.keras.layers.Dropout(name=('dout'+str(i+1)), rate = 0.4)(xd)
                xd = tf.keras.layers.Dense(name=('d'+str(i+1)), units=model_shape[i]['width'], activation='relu', 
                 kernel_regularizer = regularizers.l2(regConst), 
                 kernel_initializer = 'he_normal')(xd)
                xd = tf.keras.layers.BatchNormalization(name=('batchnorm_'+str(i+1)))(xd) 
        
    # Project the vector onto a 3 unit output layer, and squash it with a 
    # softmax activation:

    x_age_group     = tf.keras.layers.Dense(name = 'age_group', units = 2, 
                     activation = 'softmax', 
                    #  activation = 'sigmoid',
                     kernel_regularizer = regularizers.l2(regConst), 
                     kernel_initializer = 'he_normal')(xd)

    outputs = []
    for i in ['x_age_group']:
        outputs.append(locals()[i])
    model = Model(inputs = input_vec, outputs = outputs)
    
    model.compile(loss = cce, metrics = ['accuracy'], 
                  optimizer=sgd)
    model.summary()
    return model


#############################################################################
######### Dimension reduction with principal component analysis (PCA) #######
#############################################################################


# #%%

# # Dimension reduction with principal component analysis

# # The idea here is to reduce the dimensianality of a dataset consisting of a large number 
# # of related variables while retaining as much variance in the data as possible. The algorthm
# # finds a set of new varibles (principal componets) that the original variables are just 
# # linear combinations.

# # define X (matrix of features) and y (vector of labels)

# X = training_data.iloc[:,1:] # select all columns except the first one 
# y = training_data["Age"]

# print('shape of X : {}'.format(X.shape))
# print('shape of y : {}'.format(y.shape))
# seed = 42

# # A pipeline containing standardization and PCA algorithm

# pca_pipe = Pipeline([('scaler', StandardScaler()),
#                       ('pca', decomposition.PCA(n_components = 8))])

# # Transform data into 8 principal componets 

# age_pca = pca_pipe.fit_transform(X)
# print('First five observation : {}'.format(age_pca[:5]))

# explained_var = pca_pipe.named_steps['pca'].explained_variance_ratio_
# print('Explained variance : {}'.format(explained_var))

# explained_variance = np.var(age_pca, axis=0)
# explained_variance_ratio = explained_variance/np.sum(explained_variance)

#%%
# sns.set(context = 'paper',
#         style = 'whitegrid',
#         palette = 'deep',
#         font_scale = 2.0,
#         color_codes = True,
#         rc = ({'font.family': 'Dejavu Sans'}))

# plt.figure(figsize = (6, 5))
# plt.style.use('seaborn')

# plt.plot(np.cumsum(explained_var))
# plt.grid(True)
# plt.xticks(np.arange(0, 14 + 2, step = 2), fontsize = 18)
# plt.yticks(fontsize = 18)
# plt.axhline(y = .99, color = 'r', linestyle= '-', linewidth = 0.5)
# plt.axvline(x = 8, color = 'r', linestyle= '-', linewidth = 0.5)
# plt.xlabel('Number of components', weight = 'bold', fontsize = 18)
# plt.ylabel('Commulative Explained variance', weight = 'bold', fontsize = 18);
# plt.savefig("C:\Mannu\QMBCE\Thesis\Fold\_no_comp_", dpi = 500, bbox_inches="tight")


#%%

# Plotting the PCA explained variance ratio

# sns.set(context = 'paper',
#         style = 'whitegrid',
#         palette = 'deep',
#         font_scale = 2.0,
#         color_codes = True,
#         rc = ({'font.family': 'Dejavu Sans'}))

# plt.figure(figsize = (6, 4))
# plt.style.use('seaborn')

# plt.plot(np.cumsum(explained_variance_ratio), linewidth = 3)
# plt.grid(True)
# plt.xticks(np.arange(0, 10 + 2, step = 2), fontsize = 18)
# plt.yticks(fontsize = 18)
# plt.axhline(y = .98, color = 'r', linestyle= '-', linewidth = 0.5)
# plt.axvline(x = 4, color = 'r', linestyle= '-', linewidth = 0.5)
# plt.xlabel('number of components', weight = 'bold', fontsize = 18)
# plt.ylabel('Explained variance ratio', weight = 'bold', fontsize = 18);
# plt.savefig("C:\Mannu\QMBCE\Thesis\Fold\_no_comp_", dpi = 500, bbox_inches="tight")

#%%
# # transform X matrix with 10 number of components and y list of labels as arrays

# X = np.asarray(age_pca)
# y = np.asarray(y)
# print(np.unique(y))


# # %%

# # Renaming the age group into three classes
# # Oganises the data into a format of lists of data, classes, labels.

# y_age_group = np.where((y <= 9), 0, 0)
# y_age_group = np.where((y >= 10), 1, y_age_group)


# # y_age_group = np.where((y <= 5), 0, 0)
# # y_age_group = np.where((y >= 6) & (y <= 10), 1, y_age_group)
# # y_age_group = np.where((y >= 11), 2, y_age_group)

# y_age_groups_list = [[ages] for ages in y_age_group]
# age_group = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list))
# age_group_classes = ["1-9", "10-17"] 
# # age_group_classes = ["1-5", "6-10", "11-17"] 

# # Labels default - all classification
# labels_default, classes_default, outputs_default = [age_group], [age_group_classes], ['x_age_group']


# #%%

# # Function to train the model

# # This function will split the data into training and validation, and call the create models function. 
# # This fucntion returns the model and training history.


# def train_models(model_to_test, save_path):

#     model_shape = model_to_test["model_shape"][0]
#     model_name = model_to_test["model_name"][0]
#     input_layer_dim = model_to_test["input_layer_dim"][0]
#     model_ver_num = model_to_test["model_ver_num"][0]
#     fold = model_to_test["fold"][0]
#     y_train = model_to_test["labels"][0]
#     X_train = model_to_test["features"][0]
#     classes = model_to_test["classes"][0]
#     outputs = model_to_test["outputs"][0]
#     compile_loss = model_to_test["compile_loss"][0]
#     compile_metrics = model_to_test["compile_metrics"][0]

#     model = create_models(model_shape, input_layer_dim)

# #   model.summary()
    
#     history = model.fit(x = X_train, 
#                         y = y_train,
#                         batch_size = 256, 
#                         verbose = 1, 
#                         epochs = 8000,
#                         validation_data = (X_val, y_val),
#                         callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
#                                     patience=400, verbose=1, mode='auto'), 
#                                     CSVLogger(save_path+model_name+"_"+str(model_ver_num)+'.csv', append=True, separator=';')])

#     model.save((save_path+model_name+"_"+str(model_ver_num)+"_"+str(fold)+"_"+'Model.h5'))
#     graph_history(history, model_name, model_ver_num, fold, save_path)
            
#     return model, history


# # Main training and prediction section for the PCA data

# # Functionality:
# # Define the CNN to be built.
# # Build a folder to output data into.
# # Call the model training.
# # Organize outputs and call visualization for plotting and graphing.



# outdir = "C:\Mannu\QMBCE\Thesis\Fold"
# build_folder(outdir, False)
  

# # set model parameters
# # model size when data dimension is reduced to 8 principle componets 

# # Options
# # Convolutional Layer:

# #     type = 'c'
# #     filter = optional number of filters
# #     kernel = optional size of the filters
# #     stride = optional size of stride to take between filters
# #     pooling = optional width of the max pooling

# # dense layer:

# #     type = 'd'
# #     width = option width of the layer

# model_size = [#{'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1}, 
#             #  {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
#             #  {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
#              {'type':'d', 'width':500},
#              {'type':'d', 'width':500},
#              {'type':'d', 'width':500},
#              {'type':'d', 'width':500},
#              {'type':'d', 'width':500},
#             #  {'type':'d', 'width':800},
#              {'type':'d', 'width':500}]


# # Name the model
# model_name = 'Baseline_CNN'
# label = labels_default
    
# # Split data into 10 folds for training/testing
# # Define cross-validation strategy 

# num_folds = 5
# kf = KFold(n_splits=num_folds, shuffle=True, random_state = seed)

# # Features
# features = X
    
# histories = []
# averaged_histories = []
# fold = 1
# train_model = True

# # Name a folder for the outputs to go into

# savedir = (outdir+"\_4comps_PCA_00_k_fold_publish_01")            
# build_folder(savedir, True)
# savedir = (outdir+"\_4comps_PCA_00_k_fold_publish_01\l")            
           

# # start model training on standardized data
   
# start_time = time()
# save_predicted = []
# save_true = []
# save_hist = []

# for train_index, test_index in kf.split(features):

#     # Split data into test and train

#     X_train, X_test = features[train_index], features[test_index]
#     y_train, y_test = list(map(lambda y:y[train_index], label)), list(map(lambda y:y[test_index], label))

#     # Further divide training dataset into train and validation dataset 
#     # with an 90:10 split
    
#     validation_size = 0.1
#     X_train, X_val, y_train, y_val = train_test_split(X_train,
#                                         *y_train, test_size = validation_size, random_state = seed)
    

#     # expanding to one dimension, because the conv layer expcte to, 1
#     X_train = X_train.reshape([X_train.shape[0], -1])
#     X_val = X_val.reshape([X_val.shape[0], -1])
#     X_test = X_test.reshape([X_test.shape[0], -1])



#     # Check the sizes of all newly created datasets
#     print("Shape of X_train:", X_train.shape)
#     print("Shape of X_val:", X_val.shape)
#     print("Shape of X_test:", X_test.shape)
#     print("Shape of y_train:", y_train.shape)
#     print("Shape of y_val:", y_val.shape)
#     # print("Shape of y_test:", y_test.shape)

#     input_layer_dim = len(X[0])

#     model_to_test = {
#         "model_shape" : [model_size], # defines the hidden layers of the model
#         "model_name"  : [model_name],
#         "input_layer_dim"  : [input_layer_dim], # size of input layer
#         "model_ver_num"  : [0],
#         "fold"  : [fold], # kf.split number on
#         "labels"   : [y_train],
#         "features" : [X_train],
#         "classes"  : [classes_default],
#         "outputs"   : [outputs_default],
#         "compile_loss": [{'age_group': 'categorical_crossentropy'}],
#         "compile_metrics" :[{'age_group': 'accuracy'}]
#         }

#     # Call function to train all the models from the dictionary
#     model, history = train_models(model_to_test, savedir)
#     histories.append(history)

#     print(X_test.shape)

#     # predict the unseen dataset/new dataset
#     y_predicted = model.predict(X_test)

#     # change the dimension of y_test to array
#     y_test = np.asarray(y_test)
#     y_test = np.squeeze(y_test) # remove any single dimension entries from the arrays

#     print('y predicted shape', y_predicted.shape)
#     print('y_test', y_test.shape)

#     # save predicted and true value in each iteration for plotting averaged confusion matrix

#     for pred, tru in zip(y_predicted, y_test):
#         save_predicted.append(pred)
#         save_true.append(tru)

#     hist = history.history
#     averaged_histories.append(hist)

#     # Plotting confusion matrix for each fold/iteration

#     visualize(histories, savedir, model_name, str(fold), classes_default, outputs_default, y_predicted, y_test)
#     # log_data(X_test, 'test_index', fold, savedir)

#     fold += 1

#     # Clear the Keras session, otherwise it will keep adding new
#     # models to the same TensorFlow graph each time we create
#     # a model with a different set of hyper-parameters.

#     K.clear_session()

#     # Delete the Keras model with these hyper-parameters from memory.
#     del model

# save_predicted = np.asarray(save_predicted)
# save_true = np.asarray(save_true)
# print('save predicted shape', save_predicted.shape)
# print('save.true shape', save_true.shape)

# # Plotting an averaged confusion matrix

# visualize(1, savedir, model_name, "Averaged", classes_default, outputs_default, save_predicted, save_true)

# end_time = time()
# print('Run time : {} s'.format(end_time-start_time))
# print('Run time : {} m'.format((end_time-start_time)/60))
# print('Run time : {} h'.format((end_time-start_time)/3600))

# #%%

# # combine all dictionaries together

# combn_dictionar = combine_dictionaries(averaged_histories)
# with open('C:\Mannu\QMBCE\Thesis\Fold\_4comps_PCA_00_k_fold_publish_01\combined_history_dictionaries.txt', 'w') as outfile:
#      json.dump(combn_dictionar, outfile)

# # find the average of all dictionaries 

# combn_dictionar_average = find_mean_from_combined_dicts(combn_dictionar)

# # Plot averaged histories
# graph_history_averaged(combn_dictionar_average)

# # %%
# # Loading new dataset for prediction 
# # start by loading the new test data 

# # df_new = pd.read_csv("C:\Mannu\QMBCE\Thesis\set_to_predict_glasgow_02.csv")

# # when predicting the whole glasgow and no glasgow data was intergrated in the trainig
# # assign a full glasgow dataset to df_new

# df_new = df_2

# print(df_new.head())

# # Checking class distribution in the data
# print(Counter(df_new["Age"]))

# # drops columns of no interest
# # df_new = df_new.drop(['Unnamed: 0'], axis=1)
# df_new.head(10)

# # %%

# # predicting new dataset with a model trained PCA transformed data 
# # define matrix of features and vector of labels

# X_valid = df_new.iloc[:,1:]
# y_valid = df_new["Age"]

# print('shape of X : {}'.format(X_valid.shape))
# print('shape of y : {}'.format(y_valid.shape))

# y_valid = np.asarray(y_valid)
# print(np.unique(y_valid))

# # tranform matrix of features with PCA 

# age_pca_valid = pca_pipe.fit_transform(X_valid)
# print('First five observation : {}'.format(age_pca_valid[:5]))

# # transform X and y matrices as arrays

# age_pca_valid = np.asarray(age_pca_valid)
# age_pca_valid = age_pca_valid.reshape([age_pca_valid.shape[0], -1])
# print(age_pca_valid)
# print(age_pca_valid.shape)

# #%%
# # change labels

# y_age_group_val = np.where((y_valid <= 9), 0, 0)
# y_age_group_val = np.where((y_valid >= 10), 1, y_age_group_val)

# # y_age_group_val = np.where((y_valid <= 5), 0, 0)
# # y_age_group_val = np.where((y_valid >= 6) & (y_valid <= 10), 1, y_age_group_val)
# # y_age_group_val = np.where((y_valid >= 11), 2, y_age_group_val)

# y_age_groups_list_val = [[ages_val] for ages_val in y_age_group_val]
# age_group_val = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list_val))
# age_group_classes_val = ["1-9", "10-17"]
# # age_group_classes_val = ["1-5", "6-10", "11-17"]

# labels_default_val, classes_default_val = [age_group_val], [age_group_classes_val]

# #%%

# # load model trained with PCA transformed data from the disk 

# reconstracted_model = tf.keras.models.load_model("C:\Mannu\QMBCE\Thesis\Fold\_4comps_PCA_00_k_fold_publish_01\lBaseline_CNN_0_2_Model.h5")

# # change the dimension of y_test to array
# y_validation = np.asarray(labels_default_val)
# y_validation = np.squeeze(labels_default_val) # remove any single dimension entries from the arrays

# # generates output predictions based on the X_input passed

# predictions = reconstracted_model.predict(age_pca_valid)

# # computes the loss based on the X_input you passed, along with any other metrics requested in the metrics param 
# # when model was compiled

# score = reconstracted_model.evaluate(age_pca_valid, y_validation, verbose = 1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# # Calculating precision, recall and f-1 scores metrics for the predicted samples 

# cr_pca = classification_report(np.argmax(y_validation, axis=-1), np.argmax(predictions, axis=-1))
# print(cr_pca)

# # save classification report to disk 
# cr = pd.read_fwf(io.StringIO(cr_pca), header=0)
# cr = cr.iloc[1:]
# cr.to_csv('C:\Mannu\QMBCE\Thesis\Fold\_4comps_PCA_00_k_fold_publish_01\classification_report_PCA_4_0%.csv')

# #%%

# # Plot the confusion matrix for predcited samples 
# visualize(2, savedir, model_name, "Test_set", classes_default_val, outputs_default, predictions, y_validation)


#############################################################################################
######### Dimension reduction with t-Distributed Stochastic neigbour Embedding (tsne) #######
#############################################################################################


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

X = training_data.iloc[:,1:] # select all columns except the first one 
y = training_data["Age"]

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

# transform X matrix with 3 number of components and y list of labels as arrays

X = np.asarray(tsne_embedded)
y = np.asarray(y)
print(np.unique(y))

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))

# %%

# Renaming the age group into three classes
# Oganises the data into a format of lists of data, classes, labels.

y_age_group = np.where((y <= 9), 0, 0)
y_age_group = np.where((y >= 10), 1, y_age_group)

y_age_groups_list = [[ages] for ages in y_age_group]
age_group = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list))
age_group_classes = ["1-9", "10-17"] 

# Labels default - all classification
labels_default, classes_default, outputs_default = [age_group], [age_group_classes], ['x_age_group']


#%%

# Function to train the model

# This function will split the data into training and validation, and call the create models function. 
# This fucntion returns the model and training history.

def train_models(model_to_test, save_path):

    model_shape = model_to_test["model_shape"][0]
    model_name = model_to_test["model_name"][0]
    input_layer_dim = model_to_test["input_layer_dim"][0]
    model_ver_num = model_to_test["model_ver_num"][0]
    fold = model_to_test["fold"][0]
    y_train = model_to_test["labels"][0]
    X_train = model_to_test["features"][0]
    classes = model_to_test["classes"][0]
    outputs = model_to_test["outputs"][0]
    compile_loss = model_to_test["compile_loss"][0]
    compile_metrics = model_to_test["compile_metrics"][0]

    model = create_models(model_shape, input_layer_dim)

#   model.summary()
    
    history = model.fit(x = X_train, 
                        y = y_train,
                        batch_size = 256, 
                        verbose = 1, 
                        epochs = 8000,
                        validation_data = (X_val, y_val),
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                    patience=400, verbose=1, mode='auto'), 
                                    CSVLogger(save_path+model_name+"_"+str(model_ver_num)+'.csv', append=True, separator=';')])

    model.save((save_path+model_name+"_"+str(model_ver_num)+"_"+str(fold)+"_"+'Model.h5'))
    graph_history(history, model_name, model_ver_num, fold, save_path)
            
    return model, history


# Main training and prediction section for the t-SNE transformed data

# Functionality:
# Define the CNN to be built.
# Build a folder to output data into.
# Call the model training.
# Organize outputs and call visualization for plotting and graphing.


outdir = "C:\Mannu\QMBCE\Thesis\Fold"
build_folder(outdir, False) 

# set model parameters
# model size when data dimension is reduced to 8 principle componets 

# Options
# Convolutional Layer:

#     type = 'c'
#     filter = optional number of filters
#     kernel = optional size of the filters
#     stride = optional size of stride to take between filters
#     pooling = optional width of the max pooling

# dense layer:

#     type = 'd'
#     width = option width of the layer

model_size = [#{'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1}, 
            #  {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
            #  {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500},
            #  {'type':'d', 'width':800},
             {'type':'d', 'width':500}]


# Name the model
model_name = 'Baseline_CNN'
label = labels_default
    
# Split data into 10 folds for training/testing
# Define cross-validation to be used for evaluation

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# Features
features = X
    
histories = []
averaged_histories = []
fold = 1
train_model = True

# Name a folder for the outputs to go into

savedir = (outdir+"\_4comps_tsne_05_k_fold_publish_01")            
build_folder(savedir, True)
savedir = (outdir+"\_4comps_tsne_05_k_fold_publish_01\l")            

# start model training on standardized data
   
start_time = time()
save_predicted = []
save_true = []
save_hist = []

for train_index, test_index in kf.split(features):

    # Split data into test and train

    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = list(map(lambda y:y[train_index], label)), list(map(lambda y:y[test_index], label))

    # Further divide training dataset into train and validation dataset 
    # with an 90:10 split
    
    validation_size = 0.1
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                        *y_train, test_size = validation_size, random_state = seed)
    

    # expanding to one dimension, because the conv layer expcte to, 1
    X_train = X_train.reshape([X_train.shape[0], -1])
    X_val = X_val.reshape([X_val.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])



    # Check the sizes of all newly created datasets
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_val:", y_val.shape)
    # print("Shape of y_test:", y_test.shape)

    input_layer_dim = len(X[0])

    model_to_test = {
        "model_shape" : [model_size], # defines the hidden layers of the model
        "model_name"  : [model_name],
        "input_layer_dim"  : [input_layer_dim], # size of input layer
        "model_ver_num"  : [0],
        "fold"  : [fold], # kf.split number on
        "labels"   : [y_train],
        "features" : [X_train],
        "classes"  : [classes_default],
        "outputs"   : [outputs_default],
        "compile_loss": [{'age_group': 'categorical_crossentropy'}],
        "compile_metrics" :[{'age_group': 'accuracy'}]
        }

    # Call function to train all the models from the dictionary
    model, history = train_models(model_to_test, savedir)
    histories.append(history)

    print(X_test.shape)

    # predict the unseen dataset/new dataset
    y_predicted = model.predict(X_test)

    # change the dimension of y_test to array
    y_test = np.asarray(y_test)
    y_test = np.squeeze(y_test) # remove any single dimension entries from the arrays

    print('y predicted shape', y_predicted.shape)
    print('y_test', y_test.shape)

    # save predicted and true value in each iteration for plotting averaged confusion matrix

    for pred, tru in zip(y_predicted, y_test):
        save_predicted.append(pred)
        save_true.append(tru)

    hist = history.history
    averaged_histories.append(hist)

    # Plotting confusion matrix for each fold/iteration

    visualize(histories, savedir, model_name, str(fold), classes_default, outputs_default, y_predicted, y_test)
    # log_data(X_test, 'test_index', fold, savedir)

    fold += 1

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.

    K.clear_session()

    # Delete the Keras model with these hyper-parameters from memory.
    del model

save_predicted = np.asarray(save_predicted)
save_true = np.asarray(save_true)
print('save predicted shape', save_predicted.shape)
print('save.true shape', save_true.shape)

# Plotting an averaged confusion matrix

visualize(1, savedir, model_name, "Averaged", classes_default, outputs_default, save_predicted, save_true)

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))

#%%

# combine all dictionaries together

combn_dictionar = combine_dictionaries(averaged_histories)
with open('C:\Mannu\QMBCE\Thesis\Fold\_4comps_tsne_05_k_fold_publish_01\combined_history_dictionaries.txt', 'w') as outfile:
     json.dump(combn_dictionar, outfile)

# find the average of all dictionaries 

combn_dictionar_average = find_mean_from_combined_dicts(combn_dictionar)

# Plot averaged histories
graph_history_averaged(combn_dictionar_average)

# %%

# start by loading the new test data 

df_new = pd.read_csv("C:\Mannu\QMBCE\Thesis\set_to predict_glasgow_05.csv")

# when predicting the whole glasgow and no glasgow data was intergrated in the trainig
# assign a full glasgow dataset to df_new

# df_new = df_2

print(df_new.head())

# Checking class distribution in the data
print(Counter(df_new["Age"]))

# drops columns of no interest
df_new = df_new.drop(['Unnamed: 0'], axis=1)
df_new.head(10)

#%%

# predicting new dataset with a model trained tsne transformed data 

# define matrix of features and vector of labels
X_valid = df_new.iloc[:,1:]
y_valid = df_new["Age"]

print('shape of X : {}'.format(X_valid.shape))
print('shape of y : {}'.format(y_valid.shape))

y_valid = np.asarray(y_valid)
print(np.unique(y_valid))

# tranform matrix of features with tsne 

tsne_embedded_valid = tsne_pipe.fit_transform(X_valid)
print('First five observation : {}'.format(tsne_embedded_valid[:5]))

# transform X and y matrices as arrays

tsne_embedded_valid = np.asarray(tsne_embedded_valid)
tsne_embedded_valid = tsne_embedded_valid.reshape([tsne_embedded_valid.shape[0], -1])
print(tsne_embedded_valid)


#%%
# change labels

y_age_group_val = np.where((y_valid <= 9), 0, 0)
y_age_group_val = np.where((y_valid >= 10), 1, y_age_group_val)

y_age_groups_list_val = [[ages_val] for ages_val in y_age_group_val]
age_group_val = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list_val))
age_group_classes_val = ["1-9", "10-17"]

labels_default_val, classes_default_val = [age_group_val], [age_group_classes_val]


#%%

# load model trained with tsne transformed data from the disk 

reconstracted_model = tf.keras.models.load_model("C:\Mannu\QMBCE\Thesis\Fold\_4comps_tsne_05_k_fold_publish_01\lBaseline_CNN_0_2_Model.h5")

# change the dimension of y_test to array
y_validation = np.asarray(labels_default_val)
y_validation = np.squeeze(labels_default_val) # remove any single dimension entries from the arrays

# generates output predictions based on the X_input passed

predictions = reconstracted_model.predict(tsne_embedded_valid)

# computes the loss based on the X_input you passed, along with any other metrics requested in the metrics param 
# when model was compiled

score = reconstracted_model.evaluate(tsne_embedded_valid, y_validation, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Calculating precision, recall and f-1 scores metrics for the predicted samples 

cr_tsne = classification_report(np.argmax(y_validation, axis=-1), np.argmax(predictions, axis=-1))
print(cr_tsne)

cr_tsne = pd.read_fwf(io.StringIO(cr_tsne), header=0)
cr_tsne = cr_tsne.iloc[1:]
cr_tsne.to_csv('C:\Mannu\QMBCE\Thesis\Fold\_4comps_tsne_05_k_fold_publish_01\classification_report_tsne_05.csv')

#%%
# Plotting the confusion matrix for the predicted samples 

visualize(2, savedir, model_name, "Test_set", classes_default_val, outputs_default, predictions, y_validation)

# %%

# Analysing Ifakara dataset to classfy the age of anopheles arabiensis

#%%
# import libraries

import this
import os
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
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix

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
%matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%
# read the full dataset
df = pd.read_csv("D:\QMBCE\Thesis\Ifakara_data.dat", delimiter = '\t')
print(df.head())

# Checking class distribution in the data
print(Counter(df["Age"]))

# drops columns of no interest
df = df.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
df.head(10)

#%%
# count the number of samples per age
class_counts = df.groupby('Age').size()
print('{}'.format(class_counts))
# X = df.iloc[:,:-1] # select everything except the last on column

# define X (matrix of features) and y (list of labels)

X = df.iloc[:,1:] # select all columns except the first one 
y = df["Age"]

print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))

# scale feautures with standardization
X = StandardScaler().fit_transform(X) # changed features to X and standardize it
print('X Standardized: {}'.format(X))

# Dimension reduction with principle component analysis

# The idea here is to reduce the dimensianality of a dataset consisting of a large number 
# of related variables while retaining as much variance in the data as possible. The algorthm
# finds a set of new varibles (principle componets) that the original variables are just 
# linear combinations.

seed = 4
# pca_pipe = Pipeline([('scaler', StandardScaler()),
#                       ('pca', decomposition.PCA(n_components = 10, random_state = seed))])

# age_pca = pca_pipe.fit_transform(X)
# print('First five observation : {}'.format(age_pca[:5]))

# explained_var = pca_pipe.named_steps['pca'].explained_variance_ratio_
# print('Explained variance : {}'.format(explained_var))

# Dimension reduction with t-Distributed Stochastic neigbour Embedding

# t-SNE a machine learning algorthims that converts similarities between
# data points to join probabilities, and tries to minimize the kullback-leibler 
# divergence between the joint probabilities of the low-dimensional embedding and 
# the high dimensional data.
# 
# Drawback: It is possible to get different results with different initialization

# tsne_pipe = Pipeline([('scaler', StandardScaler()),
#                           ('tsne', TSNE(n_components = 10,
#                                         perplexity = 30,
#                                         method='exact', 
#                                         random_state = seed))])

# tsne_embedded = tsne_pipe.fit_transform(X)
# print('First five tsne_embedded observation : {}'.format(tsne_embedded[:5]))
# print('tsne_embedded shape : {}'.format(tsne_embedded.shape))

# Dimension reduction with SpectraEmbedding

# Spectral embedding for non-linear dimensionality reduction. Implementing Lapcian 
# Eigenmaps algorthm to form an affinity matrix given by the specified fucntion and 
# applies spectral decomposition to the corresponding graph laplacians. 

# embedded_pipe = Pipeline([('scaler', StandardScaler()),
#                           ('s_embedding', SpectralEmbedding(n_components = 10, random_state = seed))]) 

# age_embedded = embedded_pipe.fit_transform(X)
# print('First five age_embedded observation : {}'.format(age_embedded[:5]))
# print('age embedded shape : {}'.format(age_embedded.shape))


# Dimension reduction with linear discreminant analysis

# Unlike PCA, LDA seeks to preserve as much disxriminatory power as possible for the dependent
# variable, while projecting the original data matrix onto a lower dimensional space. It utilizes 
# the classes in the dependent variable to devide the space of predictors into regions. Calculating 
# the distance between the mean and and the samples of each class (within-class variance) and 
# constructing the lower dimentinal-dimensional space with this criterion: maximizing the between 
# class variance and minimizing the within-class varience.

# lda_pipe = Pipeline([('scaler', StandardScaler()),
#                      ('lda', LinearDiscriminantAnalysis(n_components = 10))])

# age_lda = lda_pipe.fit(X, y).transform(X)
# print('First five age_lda observation : {}'.format(age_lda[:5]))
# print('age_lda shape : {}'.format(age_lda[:5]))   

# transform X matrix with 10 number of components and y list of labels as arrays

X = np.asarray(X)
y = np.asarray(y)
print(np.unique(y))

#%%
# visualize the majority of feautures with the most variance 

explained_variance_components = pca_pipe.named_steps['pca'].explained_variance_

plt.figure(figsize = (6, 4))
plt.bar(range(10), explained_variance_components, alpha =  0.5, align = 'center',
            label = 'individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal componets')
plt.savefig("D:\QMBCE\Thesis\Fold\componets_plot.png", dpi = 500, bbox_inches="tight")

#%%
# Renaming the age group 

y_age_group = np.where((y <= 5), 0, 0)
y_age_group = np.where((y >= 6) & (y <= 10), 1, y_age_group)
y_age_group = np.where((y >= 11), 2, y_age_group)

y_age_groups_list = [[ages] for ages in y_age_group]
age_group = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list))
age_group_classes = ["1-5", "6-10", "11-17"]
# print(y_age_groups_list)
# print(age_group)
# print(age_group_classes)

#%%
# Labels default - all classification
labels_default, classes_default, outputs_default = [age_group], [age_group_classes], ['x_age_group']

#%%
# Split into training / testing / validation

# split the final dataset into train and test with 80:20
testsize = 0.2
seed = 4
X_train, X_test, y_train, y_test = train_test_split(X,
                                        age_group, test_size = testsize, random_state = seed)

# Further divide training dataset into train and validation dataset 
# with an 90:10 split
validation_size = 0.1
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                        y_train, test_size = validation_size, random_state = seed)

# expanding to one dimension, because the conv layer expcte to, 1
X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Check the sizes of all newly created datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of y_test:", y_test.shape)


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
    # sns.set(context="paper",
    # style="whitegrid",
    # font_scale=2.0,
    # rc={"font.family": "Dejavu Sans"})

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title([title +' - '+ model_name])
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
        plt.title(i+"/ "+"val_"+i)
        plt.plot(history.history[i], label=i)
        plt.plot(history.history["val_"+i], label="val_"+i)
        plt.legend()
        plt.xlabel("epoch", weight = 'bold')
        plt.ylabel(i)
        plt.savefig(save_path +model_name+"_"+str(model_ver_num)+"_"+str(fold)+"_"+i + ".png", dpi = 500, bbox_inches="tight")
        plt.close()

#%%

def create_models(model_shape, input_layer_dim):
    
    # parameter rate for l2 regularization
    regConst = 0.02 
    
    # defining a stochastic gradient boosting optimizer
    sgd = tf.keras.optimizers.SGD(lr = 0.001, momentum = 0.5, 
                                    nesterov = True, clipnorm = 1.)
    
    # define categorical_crossentrophy as the loss function (multi-class problem)
    cce = 'categorical_crossentropy'

    # input shape vector
    input_vec = tf.keras.Input(name = 'input', shape = (input_layer_dim, 1))

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
                    
                xd = tf.keras.layers.Dropout(name=('dout'+str(i+1)), rate=0.4)(xd)
                xd = tf.keras.layers.Dense(name=('d'+str(i+1)), units=model_shape[i]['width'], activation='relu', 
                 kernel_regularizer = regularizers.l2(regConst), 
                 kernel_initializer = 'he_normal')(xd)
                xd = tf.keras.layers.BatchNormalization(name=('batchnorm_'+str(i+1)))(xd) 
        
    # Project the vector onto a 3 unit output layer, and squash it with a 
    # softmax activation:

    x_age_group     = tf.keras.layers.Dense(name = 'age_group', units = 3, 
                     activation = 'softmax', 
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

#%%

# Function to train the model
# This function will split the data into training and validation,
# and call the create models function. 
# This fucntion returns the model and training history.
# num_folds = 5
# validation_size = 0.1

def train_models(model_to_test, save_path):

    model_shape = model_to_test["model_shape"][0]
    model_name = model_to_test["model_name"][0]
    input_layer_dim = model_to_test["input_layer_dim"][0]
    model_ver_num = model_to_test["model_ver_num"][0]
    # fold = model_to_test["fold"][0]
    # y_train = model_to_test["labels"][0]
    # X_train = model_to_test["features"][0]
    classes = model_to_test["classes"][0]
    outputs = model_to_test["outputs"][0]
    compile_loss = model_to_test["compile_loss"][0]
    compile_metrics = model_to_test["compile_metrics"][0]

    model = create_models(model_shape, input_layer_dim)

#   model.summary()
    
    history = model.fit(x = X_train, 
                        y = y_train,
                        batch_size = 32, 
                        verbose = 1, 
                        epochs = 1500,
                        validation_data = (X_val, y_val),
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                    patience=200, verbose=1, mode='auto'), 
                                    CSVLogger(save_path+model_name+"_"+str(model_ver_num)+'.csv', append=True, separator=';')])

    model.save((save_path+model_name+"_"+str(model_ver_num)+"_"+str(fold)+"_"+'Model.h5'))
    graph_history(history, model_name, model_ver_num, fold, save_path)
            
    return model, history

#%%

# Main section
# Functionality:
# Oganises the data into a format of lists of data, classes, labels.
# Define the CNN to be built.
# Define the KFold validation to be used.
# Build a folder to output data into.
# Standardize and oragnise data into training/testing.
# Call the model training.
# Organize outputs and call visualization for plotting and graphing.

input_layer_dim = len(X[0])

outdir = "D:\QMBCE\Thesis\Fold"
build_folder(outdir, False)
  

# set model parameters

# change kernel to 2 when tsne used

model_size = [{'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1}, 
             {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
             {'type':'c', 'filter':8, 'kernel':1, 'stride':1, 'pooling':1}, 
             {'type':'c', 'filter':8, 'kernel':3, 'stride':1, 'pooling':1}, 
             {'type':'c', 'filter':8, 'kernel':1, 'stride':1, 'pooling':1}, 
             {'type':'d', 'width':400}]

# Name the model
model_name = 'Baseline_CNN'
label = labels_default
    
# Split data into 10 folds for training/testing
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# Features
# features = X
    
histories = []
fold = 1
train_model = True

# Name a folder for the outputs to go into
savedir = (outdir+"\Training_Folder_10comps_standardized")            
build_folder(savedir, True)
savedir = (outdir+"\Training_Folder_10comps_standardized\l")            

# strat model training
   
start_time = time()
save_predicted = []
save_true = []

model_to_test = {
    "model_shape" : [model_size], # defines the hidden layers of the model
    "model_name"  : [model_name],
    "input_layer_dim"  : [input_layer_dim], # size of input layer
    "model_ver_num"  : [0],
    # "fold"  : [fold], # kf.split number on
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

    # for pred, tru in zip(y_predicted, y_test):
    #     save_predicted.append(pred)
    #     save_true.append(tru)

    # Visualize the results
#     print(classes_default)
#     print(outputs_default)
#     print(predicted_labels)
#     print(true_labels)

visualize(histories, savedir, model_name, str(fold), classes_default, outputs_default, y_predicted, y_test)
# log_data(X_test, 'test_index', fold, savedir)

fold += 1

# Clear the Keras session, otherwise it will keep adding new
# models to the same TensorFlow graph each time we create
# a model with a different set of hyper-parameters.

K.clear_session()

# Delete the Keras model with these hyper-parameters from memory.
del model

# save_predicted = np.asarray(save_predicted)
# save_true = np.asarray(save_true)
# print('save predicted shape', save_predicted.shape)
# print('save.true shape', save_true.shape)

# visualize(1, savedir, model_name, "CM_matrix", classes_default, outputs_default, save_predicted, save_true)

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))


# %%

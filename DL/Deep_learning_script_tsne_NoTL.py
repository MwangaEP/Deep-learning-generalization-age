# In this script, we are predicting the age of Anopheles mosquitoes using deep learning, but here we do not include
# transfer learning, only the dimesnionality reduction technique is applied

#%%
# Import libraries

import os
import io
import ast
import json
import itertools
import collections
from time import time
from tqdm import tqdm

from itertools import cycle
import random as rn
import datetime

import numpy as np 
import pandas as pd

from random import randint
from collections import Counter 

from sklearn.model_selection import ShuffleSplit, train_test_split, KFold 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score

from imblearn.under_sampling import RandomUnderSampler

from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

from my_functions import build_folder
from my_functions import plot_confusion_matrix
from my_functions import visualize
from my_functions import log_data
from my_functions import graph_history, graph_history_averaged
from my_functions import combine_dictionaries, find_mean_from_combined_dicts
from my_functions import plot_cumulative_variance

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras import initializers
from keras.models import Sequential, Model
from keras import layers, metrics
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.models import model_from_json, load_model
from keras.regularizers import *
from keras.callbacks import CSVLogger
from keras import backend as K

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

ifakara_df = pd.read_csv("C:\Mannu\QMBCE\Thesis\Ifakara_data.dat", delimiter = '\t')
print(ifakara_df.head())

# Checking class distribution in Ifakara data
print(Counter(ifakara_df["Age"]))

# drops columns of no interest
ifakara_df = ifakara_df.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
ifakara_df.head(10)

#%%
# read full glasgow dataset

glasgow_df = pd.read_csv("C:\Mannu\QMBCE\Thesis\glasgow_data.dat", delimiter = '\t')
print(glasgow_df.head())

# Checking class distribution in glasgow data 
print(Counter(glasgow_df["Age"]))

# drops columns of no interest
glasgow_df = glasgow_df.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
glasgow_df.head(10)

#%%
# Function to create deep learning model

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
                xd = tf.keras.layers.Dropout(name=('dout'+str(i+1)), rate=0.5)(xd)

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
                    
                xd = tf.keras.layers.Dropout(name=('dout'+str(i+1)), rate = 0.5)(xd)
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

#%%

"""
    Dimension reduction with t-Distributed Stochastic neigbour Embedding

    t-SNE a machine learning algorthims that converts similarities between
    data points to join probabilities, and tries to minimize the kullback-leibler 
    divergence between the joint probabilities of the low-dimensional embedding and 
    the high dimensional data.

    Drawback: It is possible to get different results with different initialization

"""

# define X (matrix of features) and y (vector of labels)

start_time = time() # assess computational time the algorithm uses to transform data

# define X (matrix of features) and y (vector of labels)

X = ifakara_df.iloc[:,1:] # select all columns except the first one 
y = np.asarray(ifakara_df["Age"])
print(np.unique(y))

print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))
seed = 42

scaler = StandardScaler().fit(X = X) # fit the scaler
X_transformed = scaler.transform(X = X)


dim_reduction = TSNE(n_components = 3,
                    perplexity = 30,
                    method='barnes_hut', 
                    random_state = seed)

tsne_embedded = dim_reduction.fit_transform(X_transformed)

print('First five tsne_embedded observation : {}'.format(tsne_embedded[:5]))
print('tsne_embedded shape : {}'.format(tsne_embedded.shape))  
X = np.asarray(tsne_embedded)

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))


#%%
# Renaming the age group into three classes
# Oganises the data into a format of lists of data, classes, labels.

y_age_group = np.where((y <= 9), 0, 0)
y_age_group = np.where((y >= 10), 1, y_age_group)


# y_age_group = np.where((y <= 5), 0, 0)
# y_age_group = np.where((y >= 6) & (y <= 10), 1, y_age_group)
# y_age_group = np.where((y >= 11), 2, y_age_group)

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


# Main training and prediction section for the PCA data

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

"""
    Freezing the convulational layers, because we are passing the principal components, thus
    there is no need to extract features 

"""

model_size = [#{'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1}, 
            #  {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
            #  {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500},
             {'type':'d', 'width':500}]


# Name the model
model_name = 'Baseline_CNN'
label = labels_default
    
# Split data into 5 folds for training/testing
# Define cross-validation strategy 

num_folds = 5
random_seed = np.random.randint(0, 81470)
kf = KFold(n_splits=num_folds, shuffle=True, random_state = random_seed)

# Features
features = X
    
histories = []
averaged_histories = []
fold = 1
train_model = True

# Name a folder for the outputs to go into

savedir = (outdir+"\_tsne_00_k_fold_publish_01")            
build_folder(savedir, True)
savedir = (outdir+"\_tsne_00_k_fold_publish_01\l")            
           

# start model training on standardized data
   
start_time = time()
save_predicted = []
save_true = []
save_hist = []

for train_index, test_index in kf.split(features):

    # Split data into test and train

    X_trainset, X_test = features[train_index], features[test_index]
    y_trainset, y_test = list(map(lambda y:y[train_index], label)), list(map(lambda y:y[test_index], label))

    # Further divide training dataset into train and validation dataset 
    # with an 90:10 split
    
    validation_size = 0.1
    X_train, X_val, y_train, y_val = train_test_split(X_trainset,
                                        *y_trainset, test_size = validation_size, random_state = seed)
    

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
with open(savedir + '_combined_history_dictionaries_base_model.txt', 'w') as outfile:
     json.dump(combn_dictionar, outfile)

# find the average of all dictionaries 

combn_dictionar_average = find_mean_from_combined_dicts(combn_dictionar)

# Plot averaged histories
graph_history_averaged(combn_dictionar_average, savedir)

# %%
# predicting the unseen glasgow data 

# Checking class distribution in the data
print(Counter(glasgow_df["Age"]))

# drops columns of no interest
glasgow_df.head(10)

# %%

# predicting new dataset with a model trained PCA transformed data 
# define matrix of features and vector of labels

X_valid = glasgow_df.iloc[:,1:]
y_valid = np.asarray(glasgow_df["Age"])

print(np.unique(y_valid))
print('shape of X : {}'.format(X_valid.shape))
print('shape of y : {}'.format(y_valid.shape))

# tranform matrix of features with PCA 

X_valid_transformed = scaler.transform(X = X_valid)
age_tsne_valid = dim_reduction.fit_transform(X_valid_transformed)
print('First five observation : {}'.format(age_tsne_valid[:5]))

# transform X and y matrices as arrays

age_tsne_valid = np.asarray(age_tsne_valid)
age_pca_valid = age_tsne_valid.reshape([age_tsne_valid.shape[0], -1])
print(age_tsne_valid.shape)

#%%
# change labels

y_age_group_val = np.where((y_valid <= 9), 0, 0)
y_age_group_val = np.where((y_valid >= 10), 1, y_age_group_val)

y_age_groups_list_val = [[ages_val] for ages_val in y_age_group_val]
age_group_val = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list_val))
age_group_classes_val = ["1-9", "10-17"]

labels_default_val, classes_default_val = [age_group_val], [age_group_classes_val]

#%%

# load model trained with PCA transformed data from the disk 

model = tf.keras.models.load_model("C:\Mannu\QMBCE\Thesis\Fold\_tsne_00_k_fold_publish_01\lBaseline_CNN_0_2_Model.h5")

# change the dimension of y_test to array
y_validation = np.asarray(labels_default_val)
y_validation = np.squeeze(labels_default_val) # remove any single dimension entries from the arrays

# generates output predictions based on the X_input passed

predictions = model.predict(age_pca_valid)

# computes the loss based on the X_input you passed, along with any other metrics requested in the metrics param 
# when model was compiled

score = model.evaluate(age_pca_valid, y_validation, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Calculating precision, recall and f-1 scores metrics for the predicted samples 

cr_pca = classification_report(np.argmax(y_validation, axis=-1), np.argmax(predictions, axis=-1))
print(cr_pca)

# save classification report to disk 
cr = pd.read_fwf(io.StringIO(cr_pca), header = 0)
cr = cr.iloc[1:]
cr.to_csv('C:\Mannu\QMBCE\Thesis\Fold\_tsne_00_k_fold_publish_01\classification_report.csv')

#%%

# Plot the confusion matrix for predcited samples 
visualize(2, savedir, model_name, "Test_set", classes_default_val, outputs_default, predictions, y_validation)
 
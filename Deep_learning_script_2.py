# Using dimensionality reduction and deep learning to achive generalizability in predicting the age
# of anopheles arabiensis mosquitoes reared in different ecological conditions.

#%%
# Import libraries

import this
import os
import io
import ast
import itertools
import json
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
        font_scale = 2.0,
        color_codes=True,
        rc=None)
# %matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]


#%% FUNCTIONS

# Create a new folder for CNN

def build_folder(Fold, to_build = False):
    """Create a new folder for the CNN outputs"""
    if not os.path.isdir(Fold):
        if to_build == True:
            os.mkdir(Fold)
        else:
            print('Directory does not exists, not creating directory!')
    else:
        if to_build == True:
            raise NameError('Directory already exists, cannot be created!')

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

    plt.imshow(cm, interpolation='nearest', vmin = 0.2, vmax = 1.0, cmap=cmap)
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
    plt.close()

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

# Data logging
# for logging data associated with the model

def log_data(log, name, fold, save_path):
    f = open((save_path+name+'_'+str(fold)+'_log.txt'), 'w')
    np.savetxt(f, log)
    f.close()


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
        plt.tight_layout()
        plt.grid(False)
        plt.xlabel("epoch", weight = 'bold')
        plt.ylabel(i)
        plt.savefig(save_path +model_name+"_"+str(model_ver_num)+"_"+str(fold)+"_"+i + ".png", dpi = 500, bbox_inches="tight")
        plt.close()


# Function to create deep CNN

# This function takes as an input a list of dictionaries. Each element in the list is a new hidden layer in the model. For each
# layer the dictionary defines the layer to be used.

def create_models(model_shape, input_layer_dim):

    # parameter rate for l2 regularization
    regConst = 0.01

    # defining a stochastic gradient boosting optimizer
    sgd = tf.keras.optimizers.SGD(lr = 0.001, momentum = 0.9,
                                    nesterov = True, clipnorm = 1.)

    # define categorical_crossentrophy as the loss function (multi-class problem i.e. 3 age classes)
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

                xd = tf.keras.layers.Dropout(name=('dout'+str(i+1)), rate=0.5)(xd)
                xd = tf.keras.layers.Dense(name=('d'+str(i+1)), units=model_shape[i]['width'], activation='relu',
                 kernel_regularizer = regularizers.l2(regConst),
                 kernel_initializer = 'he_normal')(xd)
                xd = tf.keras.layers.BatchNormalization(name=('batchnorm_'+str(i+1)))(xd)

    # Project the vector onto a 3 unit output layer, and squash it with a
    # softmax activation:

    x_age_group     = tf.keras.layers.Dense(name = 'age_group', units = 2,
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

    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=32,
                        verbose=1,
                        epochs=200,
                        validation_data=(X_val, y_val),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=100, verbose=1, mode='auto'),
                                   CSVLogger(save_path+model_name+"_"+str(model_ver_num)+'.csv', append=True, separator=';')])

    model.save((save_path+model_name+"_"+str(model_ver_num) +
               "_"+str(fold)+"_"+'Model.h5'))
    graph_history(history, model_name, model_ver_num, fold, save_path)

#   model.summary()
    return model, history


# This takes a list of dictionaries, and combines them into a dictionary in which each key maps to a
# list of all the appropriate values from the parameter dictionaries

def combine_dictionaries(list_of_dictionaries):

    combined_dictionaries = {}

    for individual_dictionary in list_of_dictionaries:

        for key_value in individual_dictionary:

            if key_value not in combined_dictionaries:

                combined_dictionaries[key_value] = []
            combined_dictionaries[key_value].append(
                individual_dictionary[key_value])

    return combined_dictionaries

#%% IMPORTING DATASETS
# read the full ifakara dataset

df = pd.read_csv("./data/Ifakara_data.dat", delimiter = '\t')
print(df.head())

# Checking class distribution in Ifakara data
print(Counter(df["Age"]))

# drops columns of no interest
df = df.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
# df = df.drop(['Unnamed: 0'], axis = 1)
df.head(10)


#%%
# read full glasgow dataset

df_2 = pd.read_csv("./data/glasgow_data.dat", delimiter = '\t')
print(df_2.head())

# Checking class distribution in glasgow data
print(Counter(df_2["Age"]))

# drops columns of no interest
df_2 = df_2.drop(['Species', 'Status', 'Country', 'RearCnd', 'StoTime'], axis=1)
# df = df.drop(['Unnamed: 0'], axis = 1)
df_2.head(10)


#%%

# if we are not interested in intergrating glasgow data into ifakara data, we will just
# assign ifakara data to training data

training_data = df
training_data.head(10)

#%%

#

#####################################################################################
############ Training the whole spectra without dimensionality reduction ############
#                    (only standardization of feautures)
####################################################################################


#%%
# count the number of samples per age

class_counts = training_data.groupby('Age').size()
print('{}'.format(class_counts))
# X = df.iloc[:,:-1] # select everything except the last on column

# define X (matrix of features) and y (list of labels)

X = training_data.iloc[:,1:] # select all columns except the first one
y = training_data["Age"]

print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))

# scale feautures with standardization
X = StandardScaler().fit_transform(X) # changed features to X and standardize it
print('X Standardized: {}'.format(X))

# transform X matrix and y vector of labels into numpy arrays

X = np.asarray(X)
y = np.asarray(y)
print(np.unique(y))


#%%
# Renaming the age group into three classes
# Oganises the data into a format of lists of data, classes, labels.

y_age_group = np.where((y <= 9), 0, 0)
# y_age_group = np.where((y >= 6) & (y <= 10), 1, y_age_group)
y_age_group = np.where((y >= 10), 1, y_age_group)

y_age_groups_list = [[ages] for ages in y_age_group]
age_group = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list))
age_group_classes = ["1-9", "10-17"]

# Labels default - all classification
labels_default, classes_default, outputs_default = [age_group], [age_group_classes], ['x_age_group']


# %%


# Main training and prediction section for the standardized data

# Functionality:
# Define the CNN to be built.
# Build a folder to output data into.
# Call the model training.
# Organize outputs and call visualization for plotting and graphing.

input_layer_dim = len(X[0])

outdir = "./data/Fold"
build_folder(outdir, False)

# set model parameters
# model size when whole spectra is used

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

model_size = [{'type':'c', 'filter':16, 'kernel':8, 'stride':1, 'pooling':1},
             {'type':'c', 'filter':16, 'kernel':8, 'stride':1, 'pooling':1},
             {'type':'c', 'filter':16, 'kernel':4, 'stride':2, 'pooling':1},
             {'type':'c', 'filter':16, 'kernel':6, 'stride':1, 'pooling':1},
             {'type':'d', 'width':50}]

# Name the model
model_name = 'Baseline_CNN'
label = labels_default

# Split data into 10 folds for training/testing
# Define the KFold validation to be used.
seed = 42
num_folds = 5
kf = KFold(n_splits = num_folds, shuffle = True, random_state = seed)

# Features
features = X

histories = []
fold = 1
train_model = True

# Name a folder for the outputs to go into

savedir = (outdir+"\_allcomps_std_k_fold_publish_01")
build_folder(savedir, True)
savedir = (outdir+"\_allcomps_std_k_fold_publish_01\l")

# start model training on standardized data
averaged_histories = []
start_time = time()
save_predicted = []
save_true = []

for train_index, test_index in kf.split(features):

    # Split data into test and train

    X_trainset, X_test = features[train_index], features[test_index]
    y_trainset, y_test = list(map(lambda y:y[train_index], label)), list(map(lambda y:y[test_index], label))

    # Further divide training dataset into train and validation dataset
    # with an 90:10 split

    validation_size = 0.1
    X_train, X_val, y_train, y_val = train_test_split(X_trainset,
                                        *y_trainset, test_size = validation_size, random_state = 4)


    # expanding to one dimension, because the conv layer expcte to, 1
    X_train = np.expand_dims(X_train, axis = 2)
    X_val = np.expand_dims(X_val, axis = 2)
    X_test = np.expand_dims(X_test, axis = 2)

    # Check the sizes of all newly created datasets
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of X_test:", X_test.shape)
    # print("Shape of y_train:", y_train.shape)
    # print("Shape of y_val:", y_val.shape)
    # print("Shape of y_test:", y_test.shape)

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

    # save predicted values and true values for each fold to calculated averaged confusion matrix

    for pred, tru in zip(y_predicted, y_test):
        save_predicted.append(pred)
        save_true.append(tru)

    hist = history.history
    averaged_histories.append(hist)

    # plot confusion matrix after each iteration

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

# plot the everaged confusion matrix

visualize(1, savedir, model_name, "Averaged", classes_default, outputs_default, save_predicted, save_true)

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))


#%%

combn_dictionar = combine_dictionaries(averaged_histories)
with open('./data/Fold\_allcomps_std_k_fold_publish_01\combined_history_dictionaries.txt', 'w') as outfile:
     json.dump(combn_dictionar, outfile)


# %%
# Loading new dataset for prediction
# start by loading the new test data

# df_new = pd.read_csv("D:\QMBCE\Thesis\set_to_predict_glasgow_05.csv")

# when predicting the whole glasgow and no glasgow data was intergrated in the trainig
# assign a full glasgow dataset to df_new

df_new = df_2

print(df_new.head())

# Checking class distribution in the data
print(Counter(df_new["Age"]))

# drops columns of no interest
# df_new = df_new.drop(['Unnamed: 0'], axis=1)
df_new.head(10)

#%%

# define matrix of features and vector of labels

X_valid = df_new.iloc[:,1:]
y_valid = df_new["Age"]

print('shape of X : {}'.format(X_valid.shape))
print('shape of y : {}'.format(y_valid.shape))

y_valid = np.asarray(y_valid)
print(np.unique(y_valid))


# scale feautures with standardization (same treatment applied to the data used to train data)

X_valid_scaled = StandardScaler().fit_transform(X_valid) # changed features to X and standardize it
print('X Shape of standardized: {}'.format(X_valid_scaled.shape))

# transform X and y matrices as arrays
X_valid_scaled = np.asarray(X_valid_scaled)
X_valid_scaled = np.expand_dims(X_valid_scaled, axis = 2)
print(X_valid_scaled.shape)


#%%
# change labels

y_age_group_val = np.where((y_valid <= 9), 0, 0)
# y_age_group_val = np.where((y_valid >= 6) & (y_valid <= 10), 1, y_age_group_val)
y_age_group_val = np.where((y_valid >= 10), 1, y_age_group_val)

y_age_groups_list_val = [[ages_val] for ages_val in y_age_group_val]
age_group_val = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list_val))
age_group_classes_val = ["1-9", "10-17"]

labels_default_val, classes_default_val = [age_group_val], [age_group_classes_val]

#%%

# load model trained with standardized data from the disk

reconstructed_model = tf.keras.models.load_model("./data/Fold\_allcomps_std_k_fold_publish_01\lBaseline_CNN_0_3_Model.h5")

# change the dimension of y_test to array
y_validation = np.asarray(labels_default_val)
y_validation = np.squeeze(labels_default_val) # remove any single dimension entries from the arrays

# generates output predictions based on the X_input passed
print(X_valid_scaled.shape)

predictions = reconstructed_model.predict(X_valid_scaled)

# computes the loss based on the X_input you passed, along with any other metrics requested in the metrics param
# when model was compiled

score = reconstructed_model.evaluate(X_valid_scaled, y_validation, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Calculating precision, recall and f-1 scores metrics for the predicted samples

cr_standard = classification_report(np.argmax(y_validation, axis=-1), np.argmax(predictions, axis=-1))
print(cr_standard)

# save classification report to disk
cr = pd.read_fwf(io.StringIO(cr_standard), header=0)
cr = cr.iloc[1:]
cr.to_csv('./data/Fold\_allcomps_std_k_fold_publish_01\classification_report.csv')

#%%
# plot the confusion matrix for the predicted samples

visualize(2, savedir, model_name, "Test_set", classes_default_val, outputs_default, predictions, y_validation)

# %%

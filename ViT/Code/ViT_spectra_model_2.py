#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%%
# Print the current working directory
print("Current working directory:", os.getcwd())

#%%

# Set the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("New working directory:", os.getcwd())

#%%

import numpy as np
import pandas as pd
import io

from time import time

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report
from collections import Counter

from ViT.Code.helper_functions import (
    visualize,
    build_folder,
    log_data,
    graph_history,
    graph_history_averaged,
    combine_dictionaries,
    find_mean_from_combined_dicts
)

import json

import tensorflow as tf
from tensorflow import keras
from keras import (
    layers, 
    regularizers, 
    initializers, 
    optimizers,
    Model,
    callbacks
)
from keras import backend as K

import matplotlib.pyplot as plt # for making plots
import seaborn as sns

#%%
# results directory
save_dir = os.path.join("..", "Results")

#%%

# Check if TensorFlow is using GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is not using the GPU.")

#%%
# Function to prepare 1D spectral data into patches
def prepare_spectral_patches(x, patch_size):
    original_length = x.shape[0]
    total_patches = original_length // patch_size
    remainder = original_length % patch_size
    
    if remainder != 0:
        x = np.pad(x, (0, patch_size - remainder), 'constant')  # Pad with zeros if needed
        total_patches = (original_length + (patch_size - remainder)) // patch_size

    patches = x[:total_patches * patch_size].reshape(total_patches, patch_size)
    return patches


#%%

# Define the Vision Transformer model

def create_vit_model(input_shape, num_classes):

    '''
    Function to create a Vision Transformer model for 1D spectral data

    Parameters:
    input_shape (numpy array): shape of the input data
    num_classes (int): number of classes in the dataset
    patch_size (int): size of the patches to be extracted from the input data

    Returns:
    model: keras model, the compiled Vision Transformer model
    '''
    
    # Regularization constant
    regConst = 0.01

    # define categorical_crossentrophy as the loss function
    cce = 'categorical_crossentropy'
    # bce = 'binary_crossentropy'

    # Defining the Adam optimizer with custom parameters
    adam = optimizers.Adam(learning_rate=0.0001,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-07)


    # Step 0: Define the input layer
    inputs = layers.Input(shape=input_shape)
    # print(f"Input shape: {inputs.shape}")
    
    # Step 2: Patch Embedding
    # flat_patches = layers.Flatten()(inputs) # Flatten to (num_samples, num_patches * patch_size)
    # print(f"Shape after Flatten: {flat_patches.shape}")
    projection = layers.Dense(128,
                              kernel_initializer=initializers.he_normal(),
                              kernel_regularizer=regularizers.l2(regConst))(inputs) #(flat_patches)  # Linear projection to a higher dimension
    # print(f"Shape after Dense projection: {projection.shape}")

    # Step 3: Positional Encoding
    num_patches = input_shape[0]  # Number of patches
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=128)(tf.range(num_patches))
    position_embedding = tf.expand_dims(position_embedding, axis=0)  # Add batch dimension
    # print(f"Shape of position_embedding: {position_embedding.shape}")
    x = layers.Add()([projection, position_embedding])
    # print(f"Shape after Add: {x.shape}")

    # Step 4: Multi-Head Self-Attention
    for i in range(2):  # Number of transformer blocks
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(num_heads=16, key_dim=64)(x, x)
        # print(f"Shape after MultiHeadAttention block {i+1}: {attention_output.shape}")
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)  # Add & Norm
        # print(f"Shape after LayerNormalization block {i+1}: {x.shape}")

        # Feed Forward Network
        ffn_output = layers.Dense(128, 
                                  activation='relu',
                                  kernel_initializer=initializers.he_normal(),
                                  kernel_regularizer=regularizers.l2(regConst))(x)
        # print(f"Shape after first Dense in FFN block {i+1}: {ffn_output.shape}")
        ffn_output = layers.Dense(128,
                                  kernel_initializer=initializers.he_normal(),
                                  kernel_regularizer=regularizers.l2(regConst))(ffn_output)
        # print(f"Shape after second Dense in FFN block {i+1}: {ffn_output.shape}")
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)  # Add & Norm
        # print(f"Shape after second LayerNormalization block {i+1}: {x.shape}")

    # Step 5: Classification head
    x = layers.GlobalAveragePooling1D()(x)  # Global average pooling
    # print(f"Shape after GlobalAveragePooling1D: {x.shape}")
    x = layers.Dropout(0.5)(x)  # Dropout for regularization
    # print(f"Shape after Dropout: {x.shape}")

    # Debug print to check the shape before the final Dense layer
    # print(f"Shape before final Dense layer: {x.shape}")

    # Add new layers for transfer learning
    x = layers.Dense(128, 
                     activation='relu',
                     kernel_initializer=initializers.he_normal(), 
                     kernel_regularizer=regularizers.l2(regConst))(x)
    x = layers.Dropout(0.5)(x)  # Additional dropout layer

    # Add new layers for transfer learning
    x = layers.Dense(128, 
                     activation='relu',
                     kernel_initializer=initializers.he_normal(), 
                     kernel_regularizer=regularizers.l2(regConst))(x)
    x = layers.Dropout(0.5)(x)  # Additional dropout layer

    # Ensure num_classes is correctly defined
    # print(f"num_classes: {num_classes}")
    # assert num_classes == 2, f"Expected num_classes to be 2, but got {num_classes}"

    outputs = layers.Dense(num_classes, 
                           activation='softmax',
                           kernel_initializer=initializers.he_normal(),
                           kernel_regularizer=regularizers.l2(regConst))(x)  # Output layer

    # step 6: Define the complete model
    model = keras.Model(inputs, outputs)

    # step 7: Compile the model
    model.compile(
        optimizer=adam, 
        loss=cce, 
        metrics=['accuracy']
    )

    # Step 5: Print the model summary to see the architecture
    model.summary()

    return model

#%%

# Function to train the ViT model
def train_vit_model(X_train, y_train, X_val, y_val, input_shape, num_classes, batch_size, epochs):
    
    """
    This function trains a Vision Transformer (ViT) model on spectral data for age group classification.

    Parameters:
    X_train (numpy array): Training spectral data.
    y_train (numpy array): Training labels.
    X_val (numpy array): Validation spectral data.
    y_val (numpy array): Validation labels.
    input_layer_dim (int): The input dimension of your spectral data.
    num_classes (int): The number of output classes (e.g., age groups).
    batch_size (int): The batch size for training.
    epochs (int): The number of epochs for training.

    Returns:
    model: A trained Keras model.
    history: Training history of the model.
    """

    # Create the ViT model
    model = create_vit_model(input_shape, num_classes)
    
    # Train the model
    history = model.fit(
        x=X_train, 
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10,
                mode='auto') 
            # CSVLogger('training_log.csv')
            ]
    )

    # Save the model
    save_path = os.path.join("..", "Results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(
        save_path,
        f"{model_name}_{fold}_Model.keras"
    )
    model.save(file_path)

    #plot the training history
    graph_history(history, model_name, fold)
    
    return model, history

#%%

# Load your data here (use the path to your actual dataset)
ifakara_df = pd.read_csv(
    "C:\Mannu\QMBCE\Thesis\Ifakara_data.dat", 
    delimiter = '\t'
    )

# Checking class distribution in Ifakara data
print(Counter(ifakara_df["Age"]))

# drops columns of no interest
ifakara_df = ifakara_df.drop(
    [
        'Species', 
        'Status', 
        'Country', 
        'RearCnd', 
        'StoTime'
    ], axis=1
)

ifakara_df.head(10)

#%%
# load the dataset and use the model
start_time = time()

# Define cross-validation strategy (k-fold) 
num_folds = 5 # Number of folds
random_seed = np.random.randint(0, 81470) # Random seed for reproducibility

kf = KFold(
    n_splits=num_folds, 
    shuffle=True, 
    random_state = random_seed
)

# Name the model
model_name = 'ViT'
fold = 1

# main loop
if __name__ == "__main__":
    
    # Feature extraction
    X = np.asarray(ifakara_df.iloc[:, 1:])  # Spectral data (1D features)
    y = np.asarray(ifakara_df["Age"])       # Labels (Age groups)

    # Standardize the features
    scaler = StandardScaler().fit(X)
    X_transformed = scaler.transform(X)

    # Convert the target labels to binary classes
    y_age_group = np.where((y <= 9), 0, 0)
    y_age_group = np.where((y >= 10), 1, y_age_group)
    y_age_group_list = [[age] for age in y_age_group]
    y_binary = MultiLabelBinarizer().fit_transform(np.array(y_age_group_list))
    age_classes = ["1-9", "10-17"] 

    # Prepare the spectral data as patches (assuming patch size of 15)
    patch_size = 27
    X_patches = np.array([prepare_spectral_patches(x, patch_size) for x in X_transformed])
    print("Shape of X_patches:", X_patches.shape)

    # Split data into training and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_patches, 
    #     y_binary, 
    #     test_size=0.2, 
    #     random_state=42
    # )

    save_predicted = []
    save_true = []
    histories = []
    averaged_histories = []

    for train_index, test_index in kf.split(X_patches):

        # Split data into training and test sets

        X_trainset, X_test = X_patches[train_index], X_patches[test_index]
        y_trainset, y_test = y_binary[train_index], y_binary[test_index]

        # Print shapes to verify
        print(f"X_trainset shape: {X_trainset.shape}, X_test shape: {X_test.shape}")
        print(f"y_trainset shape: {y_trainset.shape}, y_test shape: {y_test.shape}")

        # Further divide training dataset into train and validation dataset 
        # with an 90:10 split

        validation_size = 0.1
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainset,
            y_trainset, 
            test_size = validation_size, 
            random_state = random_seed)
        
        # Print shapes to verify
        print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
        print(f"y_trainset shape: {y_train.shape}, y_val shape: {y_val.shape}")


        # Reshape for the model: (num_samples, num_patches, patch_size)
        input_shape = (X_train.shape[1], patch_size)

        # Train Vision Transformer model
        # Note: This will take some time to train
        # You can adjust the batch size and number of epochs in the function

        # Define the batch size and number of epochs
        n_epochs = 200
        size = 64
        numb_classes = 2

        # Train the model
        model, history = train_vit_model(
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            input_shape, 
            num_classes=numb_classes, 
            batch_size=size,
            epochs=n_epochs
        )

        histories.append(history)

        # Save the history of each fold
        hist = history.history 
        averaged_histories.append(hist)

        # predict the unseen dataset/new dataset
        test_prediction = model.predict(X_test)

        # change the dimension of y_test to array
        # y_test = np.asarray(y_test)
        # y_test = np.squeeze(y_test) # remove any single dimension entries from the arrays

        # print('y predicted shape', y_predicted.shape)
        # print('y_test', y_test.shape)

        # save predicted and true value in each iteration for plotting averaged confusion matrix

        for pred, tru in zip(test_prediction, y_test):
            save_predicted.append(pred)
            save_true.append(tru)

        # Plotting confusion matrix for each fold/iteration

        visualize(
            histories,  
            model_name, 
            str(fold), 
            age_classes, 
            test_prediction, 
            y_test
        )
        # # log_data(X_test, 'test_index', fold, savedir)

        fold += 1

        # Clear the Keras session, otherwise it will keep adding new
        # models to the same TensorFlow graph each time we create
        # a model with a different set of hyper-parameters.

        K.clear_session()

        # Delete the Keras model with these hyper-parameters from memory.
        del model

    # Plotting an averaged confusion matrix for all folds/iterations
    save_predicted = np.asarray(save_predicted)
    save_true = np.asarray(save_true)
    print('save predicted shape', save_predicted.shape)
    print('save.true shape', save_true.shape)

    visualize(
        1, 
        model_name, 
        "Averaged_training", 
        age_classes, 
        save_predicted, 
        save_true
    )

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))

#%%

# combine all dictionaries together for the base model training (using Ifakara data)
combn_dictionar = combine_dictionaries(averaged_histories)

# save the combined dictionary to a file

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Combine the directory path with the file name to create the full file path
file_path = os.path.join(save_dir, 'combined_history_dictionaries_ViT_train_model.txt')

# Write the combined dictionary to a file
with open(file_path, 'w') as outfile:
    json.dump(combn_dictionar, outfile)

# find the average of all dictionaries 

combn_dictionar_average = find_mean_from_combined_dicts(combn_dictionar)

# Plot averaged histories
graph_history_averaged(combn_dictionar_average)

#%%
# test_set classification report

cr1 = classification_report(
    np.argmax(save_true, axis=-1), 
    np.argmax(save_predicted, axis=-1)
)

# save classification report to disk
cr1 = pd.read_fwf(io.StringIO(cr1), header=0)
cr1 = cr1.iloc[0:]

# save classification report to disk
# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Combine the directory path with the file name to create the full file path
cr1_path = os.path.join(save_dir, 'cr_test_ifa.csv')

cr1.to_csv(cr1_path, index=False)

# %%
# load glasgow data
# Load your data here (use the path to your actual dataset)
glasgow_df = pd.read_csv(
    "C:\Mannu\QMBCE\Thesis\Glasgow_data.dat", 
    delimiter = '\t'
    )

# Checking class distribution in Ifakara data
print(Counter(glasgow_df["Age"]))

# drops columns of no interest
glasgow_df = glasgow_df.drop(
    [
        'Species', 
        'Status', 
        'Country', 
        'RearCnd', 
        'StoTime'
    ], axis=1
)

glasgow_df.head(10)

# %%

# Feature extraction for glasgow data
X_new = np.asarray(glasgow_df.iloc[:, 1:])  # Spectral data (1D features)
y_new = np.asarray(glasgow_df["Age"])       # Labels (Age groups)

# Standardize the features
X_new_scl = scaler.transform(X_new)

# Convert the target labels to binary classes
y_age_group_nw = np.where((y_new <= 9), 0, 0)
y_age_group_nw = np.where((y_new >= 10), 1, y_age_group_nw)
y_age_group_list_nw = [[age] for age in y_age_group_nw]
y_binary_nw = MultiLabelBinarizer().fit_transform(np.array(y_age_group_list_nw))

# Prepare the spectral data as patches (assuming patch size of 27 as used in training)
X_patches_new = np.array([prepare_spectral_patches(x, patch_size) for x in X_new_scl])
print("Shape of X_patches:", X_patches_new.shape)

#%%
# predict the unseen dataset/new dataset

# load the best model
predictor_model = keras.models.load_model(
    os.path.join(
        "..", 
        "Results", 
        "ViT_3_Model.keras"
    )
)

# predict the unseen dataset/new dataset

# computes the loss based on the X_input you passed, along with any other metrics requested in the metrics param 
# when model was compiled

score = predictor_model.evaluate(X_patches_new, y_binary_nw, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_prediction_new = predictor_model.predict(X_patches_new)

visualize(
        1, 
        model_name, 
        "Glasgow_data_no_TL", 
        age_classes, 
        test_prediction_new, 
        y_binary_nw
    )

# %%

# try transfer learning
# Function to create a transfer learning model based on a pre-trained model
def create_transfer_learning_model(base_model, learning_rate=0.001, reg_const=0.01):
    '''
    Function to create a transfer learning model based on a pre-trained model

    Parameters:
    base_model (keras Model): pre-trained model
    num_classes (int): number of classes in the new task
    learning_rate (float): learning rate for the optimizer
    reg_const (float): regularization constant

    Returns:
    model: keras model, the compiled transfer learning model
    '''
    
    # Freeze the layers of the base model
    # for layer in base_model.layers:
    #     layer.trainable = False

    # Freeze all layers except for the last two
    # for i, layer in enumerate(base_model.layers):
    #     if i < len(base_model.layers) - 2:  # Freeze all layers except the last two
    #         layer.trainable = False
    #     else:
    #         layer.trainable = True

    # Remove the final layer and use the last layer as the base
    inputs = base_model.inputs
    # x = base_model.output
    base_outputs = base_model.output

    # # Add custom layers on top of the base model
    # x = layers.Dense(128, 
    #                  activation='relu', 
    #                  kernel_initializer=initializers.he_normal(), 
    #                  kernel_regularizer=regularizers.l2(reg_const), 
    #                  name='tl_dense_1')(x)
    # x = layers.Dropout(0.5, name='tl_dropout_1')(x)  # Additional dropout layer

    # x = layers.Dense(128, 
    #                  activation='relu', 
    #                  kernel_initializer=initializers.he_normal(), 
    #                  kernel_regularizer=regularizers.l2(reg_const), 
    #                  name='tl_dense_2')(x)
    # x = layers.Dropout(0.5, name='tl_dropout_2')(x)  # Additional dropout layer

    # outputs = layers.Dense(2, # two classes classification
    #                        activation='softmax', 
    #                        kernel_initializer='he_normal', 
    #                        kernel_regularizer=regularizers.l2(reg_const),
    #                        name = 'output_tl_2')(base_outputs)

    # Define the complete model
    model = Model(inputs=inputs, outputs=base_outputs)

    # Define the Adam optimizer with custom parameters
    adm = optimizers.Adam(learning_rate=learning_rate,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-07)

    # Compile the model
    model.compile(
        optimizer=adm, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    return model

# Create the transfer learning model
transfer_learning_model = create_transfer_learning_model(
    predictor_model, 
    learning_rate=0.001, 
    reg_const=0.01
)

# Print the model summary
transfer_learning_model.summary()

#%%

# Split the data into training and validation sets
validation_size_tl = 0.95
X_train_tl, X_test_tl, y_train_tl, y_test_tl = train_test_split(
    X_patches_new,
    y_binary_nw, 
    test_size=validation_size_tl, 
    random_state=random_seed
)

# Print shapes to verify
print(f"X_train_tl shape: {X_train_tl.shape}, X_test_tl shape: {X_test_tl.shape}")
print(f"y_train_tl shape: {y_train_tl.shape}, y_test_tl shape: {y_test_tl.shape}")

# Transfer learning model training
start_time = time()

# Train the model
n_epochs_tl = 100  # Experiment with different numbers of epochs
batch_size_tl = 64  # Experiment with different batch sizes

history = transfer_learning_model.fit(
    x=X_train_tl, 
    y=y_train_tl,
    validation_data=(X_test_tl, y_test_tl),
    batch_size=batch_size_tl,
    epochs=n_epochs_tl,
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5,
            mode='auto'
        ) 
        # CSVLogger('training_log.csv')
    ]
)

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))

# Generate output predictions based on the X_input passed
predictions = transfer_learning_model.predict(X_test_tl)

# Evaluate model
score = transfer_learning_model.evaluate(X_test_tl, y_test_tl, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Visualize the results
visualize(
    1, 
    model_name, 
    "Glasgow_data_TL", 
    age_classes, 
    predictions, 
    y_test_tl
)

# %%

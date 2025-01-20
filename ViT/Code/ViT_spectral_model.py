
# This script implements Vision Transformer (ViT) for Spectra Data in Keras
# The model is designed to predict the age group of Anopheles mosquitoes based on spectral data.

#%% Imports

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report

from collections import Counter

import tensorflow as tf
from keras import layers, Model, regularizers
from vit_keras import vit
from keras.callbacks import CSVLogger

#%% Data Preprocessing Functions

# Function to prepare 1D spectral data into patches
# def prepare_spectral_patches(spectra, patch_size):
#     num_patches = len(spectra) // patch_size
#     patches = spectra[:num_patches * patch_size].reshape(-1, patch_size)
#     return patches

def prepare_spectral_patches(x, patch_size):
    original_length = x.shape[0]
    
    # Calculate the total number of complete patches
    total_patches = original_length // patch_size
    remainder = original_length % patch_size

    # Optionally handle remainder by trimming or padding
    if remainder != 0:
        # For example, you could pad with zeros
        # Or you could trim the input to fit
        x = np.pad(x, (0, patch_size - remainder), 'constant')  # Pad with zeros to make it a multiple
        total_patches = (original_length + (patch_size - remainder)) // patch_size

    # Create patches from the data
    patches = x[:total_patches * patch_size].reshape(total_patches, patch_size)

    return patches

#%% Model Creation

# Regularization constant
regConst = 0.01

# define categorical_crossentrophy as the loss function (multi-class problem i.e. 3 age classes)
cce = 'categorical_crossentropy'

# Defining the Adam optimizer with custom parameters
adam = tf.keras.optimizers.Adam(learning_rate=0.001,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-07)

# Create Vision Transformer model
def create_vit_model(input_layer_dim, num_classes):

    """
    This function creates a Vision Transformer (ViT) model for spectral data classification
    with He initialization and L2 regularization.
    
    Parameters:
    input_layer_dim (int): The input dimension of your spectral data.
    num_classes (int): The number of output classes (e.g., age groups).
    reg_const (float): Regularization constant for L2 regularization.
    
    Returns:
    model: A compiled Keras model with a Vision Transformer backbone and custom classification layers.
    """
    # Step 1: create a Vision Transformer model for spectra data

    vit_model = vit.vit_b32(
        image_size=input_layer_dim,  # Adjust the input dimension
        classes=num_classes,     # Number of output classes
        pretrained=False,            # We are building this from scratch, so no pre-trained weights
        include_top=False,           # Remove the top classification head
        pretrained_top=False,        # No pre-trained top
        weights='None'        # Use the weights from the ImageNet-21k pre-training

    )

    # Step 2: Add custom layers on top of the transformer backbone
    # Flatten the transformer output
    x = vit_model.output
    x = layers.Flatten()(x)

    # Dense layer with He initialization and L2 regularization
    x = layers.Dense(128, 
                     activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(regConst))(x)
    
    # Dropout layer to prevent overfitting
    x = layers.Dropout(0.5)(x)

    # Output layer (Final Dense layer) with softmax activation, He initialization, and L2 regularization
    outputs = layers.Dense(num_classes, 
                           activation='softmax',
                           kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(regConst))(x)

    # Step 3: Define the complete model
    model = Model(inputs=vit_model.input, outputs=outputs)

    # Step 4: Compile the model with categorical cross-entropy loss and accuracy metric
    model.compile(
        optimizer=adam, 
        loss=cce, 
        metrics=['accuracy']
        )
    
    # Step 5: Print the model summary to see the architecture
    model.summary()

    return model

#%% Model Training function

# Function to train the ViT model
def train_vit_model(X_train, y_train, X_val, y_val, input_layer_dim, num_classes, batch_size, epochs):
    
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
    model = create_vit_model(input_layer_dim, num_classes)
    
    # Train the model
    history = model.fit(
        x=X_train, 
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10), 
            CSVLogger('training_log.csv')
            ]
    )
    
    return model, history

#%% Main Script

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
# Example to show how to load the dataset and use the model
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
    
    # Prepare the spectral data as patches (assuming patch size of 32)
    patch_size = 15
    X_patches = np.array([prepare_spectral_patches(x, patch_size) for x in X_transformed])
    print("Shape of X_patches:", X_patches.shape)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_patches, 
        y_binary, 
        test_size=0.2, 
        random_state=42
    )
    
    # Reshape data to add a channel dimension for the ViT model
    # X_train = np.expand_dims(X_train, axis=-1)  # Expand dimensions for ViT
    # print("Shape of X_train:", X_train.shape)
    # X_val = np.expand_dims(X_val, axis=-1)      # Same for validation set
    # print("Shape of X_val:", X_val.shape)

    # Reshape your training and validation data appropriately
    X_train = X_train.reshape(X_train.shape[0], -1, patch_size)  # Flatten to shape (num_samples, num_patches * features_per_patch)
    print("Shape of X_train:", X_train.shape)
    X_val = X_val.reshape(X_val.shape[0], -1, patch_size)      # Same for validation set
    print("Shape of X_val:", X_val.shape)

    input_layer_dim = 111 * patch_size # Use the number of features in each patch
    
    # # Check if the flattened input size is a multiple of patch_size
    # if X_train.shape[1] % patch_size != 0:
    #     raise ValueError("The flattened input size must be a multiple of patch_size.")


    # Train Vision Transformer model
    # Note: This will take some time to train
    # You can adjust the batch size and number of epochs in the function

    # Define the batch size and number of epochs
    n_epochs = 100
    size = 16
    numb_classes = 2

    # Train the model
    model, history = train_vit_model(
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        input_layer_dim, 
        num_classes=numb_classes, 
        batch_size=size,
        epochs=n_epochs
    )
    
    # Evaluate the model on validation data
    val_predictions = model.predict(X_val)
    print(classification_report(np.argmax(y_val, axis=-1), np.argmax(val_predictions, axis=-1)))

# %%

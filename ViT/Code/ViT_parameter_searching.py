
#%%
import os
import numpy as np
import pandas as pd

from time import time

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

from ViT.Code.helper_functions import build_folder, visualize, log_data, graph_history, graph_history_averaged, combine_dictionaries, find_mean_from_combined_dicts

import json

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, initializers, optimizers, callbacks
from keras import backend as K

import matplotlib.pyplot as plt
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
def create_vit_model(input_shape, num_classes, learning_rate, reg_const, num_heads, key_dim):
    '''
    Function to create a Vision Transformer model for 1D spectral data

    Parameters:
    input_shape (numpy array): shape of the input data
    num_classes (int): number of classes in the dataset
    learning_rate (float): learning rate for the optimizer
    reg_const (float): regularization constant
    num_heads (int): number of attention heads
    key_dim (int): dimension of the key in the attention mechanism

    Returns:
    model: keras model, the compiled Vision Transformer model
    '''
    
    # Define the Adam optimizer with custom parameters
    adam = optimizers.Adam(learning_rate=learning_rate,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-07)

    # Step 0: Define the input layer
    inputs = layers.Input(shape=input_shape)
    
    # Step 2: Patch Embedding
    flat_patches = layers.Flatten()(inputs) # Flatten to (num_samples, num_patches * patch_size)
    projection = layers.Dense(128,
                              kernel_initializer=initializers.he_normal(),
                              kernel_regularizer=regularizers.l2(reg_const))(flat_patches)

    # Step 3: Positional Encoding
    num_patches = input_shape[0]  # Number of patches
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=128)(tf.range(num_patches))
    position_embedding = tf.expand_dims(position_embedding, axis=0)  # Add batch dimension
    x = layers.Add()([projection, position_embedding])

    # Step 4: Multi-Head Self-Attention
    for i in range(2):  # Number of transformer blocks
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)  # Add & Norm

        # Feed Forward Network
        ffn_output = layers.Dense(128, 
                                  activation='relu',
                                  kernel_initializer=initializers.he_normal(),
                                  kernel_regularizer=regularizers.l2(reg_const))(x)
        ffn_output = layers.Dense(128,
                                  kernel_initializer=initializers.he_normal(),
                                  kernel_regularizer=regularizers.l2(reg_const))(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)  # Add & Norm

    # Step 5: Classification head
    x = layers.GlobalAveragePooling1D()(x)  # Global average pooling
    x = layers.Dropout(0.5)(x)  # Dropout for regularization

    outputs = layers.Dense(num_classes, 
                           activation='softmax',
                           kernel_initializer=initializers.he_normal(),
                           kernel_regularizer=regularizers.l2(reg_const))(x)  # Output layer

    # Define the complete model
    model = keras.Model(inputs, outputs)

    # Compile the model
    model.compile(
        optimizer=adam, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    return model

#%%

# Function to train the ViT model
def train_vit_model(X_train, y_train, X_val, y_val, input_shape, num_classes, batch_size, epochs, learning_rate, reg_const, num_heads, key_dim):
    """
    This function trains a Vision Transformer (ViT) model on spectral data for age group classification.

    Parameters:
    X_train (numpy array): Training spectral data.
    y_train (numpy array): Training labels.
    X_val (numpy array): Validation spectral data.
    y_val (numpy array): Validation labels.
    input_shape (tuple): The input dimension of your spectral data.
    num_classes (int): The number of output classes (e.g., age groups).
    batch_size (int): The batch size for training.
    epochs (int): The number of epochs for training.
    learning_rate (float): Learning rate for the optimizer.
    reg_const (float): Regularization constant.
    num_heads (int): Number of attention heads.
    key_dim (int): Dimension of the key in the attention mechanism.

    Returns:
    model: A trained Keras model.
    history: Training history of the model.
    """

    # Create the ViT model
    model = create_vit_model(input_shape, num_classes, learning_rate, reg_const, num_heads, key_dim)
    
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
                patience=5,
                mode='auto') 
        ]
    )

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

# Define cross-validation strategy (k-fold) 
num_folds = 5 # Number of folds
random_seed = np.random.randint(0, 81470) # Random seed for reproducibility
kf = KFold(
    n_splits=num_folds, 
    shuffle=True, 
    random_state=random_seed
)

# Name the model
model_name = 'ViT'
fold = 1

# Define the parameter grid
param_grid = {
    'learning_rate': [0.0001, 0.001],
    'reg_const': [0.01, 0.001],
    'batch_size': [16, 32],
    'num_heads': [8, 16],
    'key_dim': [64, 128]
}

# Initialize variables to store the best model and best accuracy
best_accuracy = 0
best_params = None
best_model = None

start_time = time()

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

    # Perform grid search
    for learning_rate in param_grid['learning_rate']:
        for reg_const in param_grid['reg_const']:
            for batch_size in param_grid['batch_size']:
                for num_heads in param_grid['num_heads']:
                    for key_dim in param_grid['key_dim']:
                        fold_accuracies = []
                        for train_index, test_index in kf.split(X_patches):

                            # Split data into training and test sets
                            X_trainset, X_test = X_patches[train_index], X_patches[test_index]
                            y_trainset, y_test = y_binary[train_index], y_binary[test_index]

                            # Further divide training dataset into train and validation dataset 
                            validation_size = 0.1
                            X_train, X_val, y_train, y_val = train_test_split(
                                X_trainset,
                                y_trainset, 
                                test_size=validation_size, 
                                random_state=random_seed
                            )

                            # Reshape for the model: (num_samples, num_patches, patch_size)
                            input_shape = (X_train.shape[1], patch_size)

                            # Train the model
                            model, history = train_vit_model(
                                X_train, 
                                y_train, 
                                X_val, 
                                y_val, 
                                input_shape, 
                                num_classes=2, 
                                batch_size=batch_size,
                                epochs=50,
                                learning_rate=learning_rate,
                                reg_const=reg_const,
                                num_heads=num_heads,
                                key_dim=key_dim
                            )

                            # Evaluate the model
                            y_pred = model.predict(X_test)
                            y_pred_classes = np.argmax(y_pred, axis=1)
                            y_test_classes = np.argmax(y_test, axis=1)
                            accuracy = accuracy_score(y_test_classes, y_pred_classes)
                            fold_accuracies.append(accuracy)

                            # Clear the Keras session
                            K.clear_session()

                        # Calculate the mean accuracy for the current parameter combination
                        mean_accuracy = np.mean(fold_accuracies)
                        print(f"Params: lr={learning_rate}, reg={reg_const}, batch={batch_size}, heads={num_heads}, key_dim={key_dim} -> Accuracy: {mean_accuracy}")

                        # Update the best model if the current one is better
                        if mean_accuracy > best_accuracy:
                            best_accuracy = mean_accuracy
                            best_params = {
                                'learning_rate': learning_rate,
                                'reg_const': reg_const,
                                'batch_size': batch_size,
                                'num_heads': num_heads,
                                'key_dim': key_dim
                            }
                            best_model = model

    # Print the best parameters and best accuracy
    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best Parameters: {best_params}")

    # # Save the best model
    # save_path = os.path.join("..", "Results")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # file_path = os.path.join(save_path, f"{model_name}_best_model.keras")
    # best_model.save(file_path)

    # print(f"Best model saved to: {file_path}")

end_time = time()
print('Run time : {} s'.format(end_time-start_time))
print('Run time : {} m'.format((end_time-start_time)/60))
print('Run time : {} h'.format((end_time-start_time)/3600))


import os
import matplotlib.pyplot as plt # for making plots
import seaborn as sns

import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

#%%
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
    plt.xticks(tick_marks, classes, rotation = xrotation)
    plt.yticks(tick_marks, classes, rotation = yrotation)

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
    # plt.savefig((save_path + "Confusion_Matrix_" + model_name + "_" + fold +"_"+ ".pdf"), dpi = 500, bbox_inches="tight")
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


def graph_history_averaged(combined_history, save_path):
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
    ax1.set_yticks(np.arange(0.2, 1.0 + 0.1, step = 0.2))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_visible(True)

    plt.tight_layout()
    plt.grid(False)
    plt.savefig(save_path + "_Average" + ".png", dpi = 500, bbox_inches="tight")
    plt.close()

#%%
# This function takes a list of dictionaries, and combines them into a single dictionary in which each key maps to a 
# list of all the appropriate values from the parameters of the dictionaries

def combine_dictionaries(list_of_dictionaries):
    
    combined_dictionaries = {}
    
    for individual_dictionary in list_of_dictionaries:
        
        for key_value in individual_dictionary:
            
            if key_value not in combined_dictionaries:
                
                combined_dictionaries[key_value] = []
            combined_dictionaries[key_value].append(individual_dictionary[key_value])

    return combined_dictionaries


#%%

# This function calculates the average of the combined dictionaries either of same length or not the same length, 
# and return the mean

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

def plot_cumulative_variance(eigenvalues, n_components=0, figure_size=None, title=None, save_filename=None):
    """
    Plots the eigenvalues as bars and their cumulative sum to visualize
    the percent variance in the data explained by each principal component
    individually and by each principal component cumulatively.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_cumulative_variance
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Perform PCA and obtain eigenvalues:
        pca_X = PCA(X)
        eigenvalues = pca_X.L

        # Plot the cumulative variance from eigenvalues:
        plt = plot_cumulative_variance(eigenvalues,
                                       n_components=0,
                                       title='PCA on X',
                                       save_filename='PCA-X.pdf')
        plt.close()

    :param eigenvalues:
        a 0D vector of eigenvalues to analyze. It can be supplied as an attribute of the
        ``PCA`` class: ``PCA.L``.
    :param n_components: (optional)
        how many principal components you want to visualize (default is all).
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    bar_color = '#191b27'
    line_color = '#ff2f18'

    (n_eigenvalues, ) = np.shape(eigenvalues)

    if n_components == 0:
        n_retained = n_eigenvalues
    else:
        n_retained = n_components

    x_range = np.arange(1, n_retained+1)

    if figure_size is None:
        fig, ax1 = plt.subplots(figsize=(n_retained, 4))
    else:
        fig, ax1 = plt.subplots(figsize=figure_size)

    ax1.bar(x_range, eigenvalues[0:n_retained], color=bar_color, edgecolor=bar_color, align='center', zorder=2, label='Eigenvalue')
    ax1.set_ylabel('Eigenvalue', weight = 'bold')
    ax1.set_ylim(0, 1.05)
    # ax1.grid(zorder=0)
    ax1.set_xlabel('Principal component', weight = 'bold')

    ax2 = ax1.twinx()
    ax2.plot(x_range, np.cumsum(eigenvalues[0:n_retained])*100, 'o-', color=line_color, zorder=2, label='Cumulative')
    ax2.set_ylabel('Variance explained [%]', color=line_color, weight = 'bold')
    ax2.set_ylim(0,105)
    ax2.tick_params('y', colors=line_color)

    plt.xlim(0, n_retained+1)
    plt.xticks(x_range)

    if title != None:
        plt.title(title)

    if save_filename != None:
        plt.savefig(save_filename, dpi=500, bbox_inches='tight')

    return plt
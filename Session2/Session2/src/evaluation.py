import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(predictions, labels, experiment_name, print_matrix=False):
    """
        This function computes, prints and plots the confusion matrix.
        """
    classes = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city',
               'mountain', 'street', 'tallbuilding']
    label_size = 8
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(labels, predictions)
    if print_matrix:
        print(cnf_matrix)
    fig = plt.figure()
    plt.matshow(cnf_matrix)
    plt.colorbar()
    plt.xlabel('Predictions')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.savefig('confusion_matrix_' + experiment_name + '.jpg')
    return cnf_matrix

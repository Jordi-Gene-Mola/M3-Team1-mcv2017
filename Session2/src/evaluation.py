import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import metrics
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
    plt.savefig('confusion_matrix_' + experiment_name + '.png')
    return cnf_matrix

def rcurve(testDescriptors,test_labels,clf):

    classes = clf.classes_
    probas_ = clf.predict_proba(testDescriptors)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure()
    for i in range(len(classes)):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(test_labels, probas_[:,i], classes[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.title('Roc curve')
        plt.plot(fpr[i], tpr[i], label = classes[i] + 'AuC=' + str('%.2f'%roc_auc[i]))
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
    plt.show()

    # Compute micro-average ROC curve and ROC area
    test_labels_binary=label_binarize(test_labels,classes)
    fpr["micro"], tpr["micro"], thresholds = metrics.roc_curve(test_labels_binary.ravel(), probas_.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',lw=lw, label='ROC curve (AuC = %0.2f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve all classes')
    plt.legend(loc="lower right")
    plt.show()

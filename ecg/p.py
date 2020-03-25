import keras
import math
import numpy as np
import os
import sklearn
import sklearn.metrics as skm
import util

from keras.models import Model
from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.utils.multiclass import type_of_target

import matplotlib.pyplot as plt

#csv_file = open("/Users/gjy/Desktop/ecg-master/ecg/REFERENCE.csv", "r")
probs_file = open("/Users/gjy/Desktop/ecg-master/ecg/answersvk.txt", "r")

all_labels=[1, 1, 1, 0, 0, 1, 1, 2, 0, 1, 1, 1, 2, 1, 0, 1, 2, 1, 1, 2, 1, 3, 2, 2, 1, 1, 0, 1, 2, 2, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 2, 3, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 0, 1, 2, 2, 0, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 0, 2, 1, 0, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 2, 1, 1, 3, 0, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 3, 2, 1, 0, 1, 1, 2, 0, 2, 1, 1, 2, 0, 2, 3, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 2, 1, 2, 2, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 2, 1, 1, 3, 1, 2, 2, 3, 1, 1, 0, 2, 1, 2, 2, 2, 1, 2, 0, 0, 2, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
all_probs=[]
# for line in csv_file:
#     content1 = line.strip().split(',')
#     labels = content1[1]
#     if labels=='A':
#         all_labels.append(0)
#     if labels=='N':
#         all_labels.append(1)
#     if labels=='O':
#         all_labels.append(2)
#     if labels=='~':
#         all_labels.append(3)

for line in probs_file:
    content2 = line.strip().split(',')
    probs = content2[1]
    if probs == 'A':
        all_probs.append(0)
    if probs == 'N':
        all_probs.append(1)
    if probs == 'O':
        all_probs.append(2)
    if probs == '~':
        all_probs.append(3)

print("1111",all_labels)
print("2222",all_probs)

def plot_confusions(cm, xlabel, filename):
    cm = sklearn.preprocessing.normalize(cm, norm='l1', axis=1, copy=True)
    classes = ['A','N','O','~']
    plt.rcParams['figure.figsize'] = (8, 7)
    plt.pcolor(np.flipud(cm), cmap="Blues")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks + .5, classes, rotation=90, fontsize=16)
    plt.yticks(tick_marks + .5, reversed(classes), fontsize=16)
    plt.clim(0, 1)
    plt.ylabel("Real label", fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename,
                dpi=520,
                format='pdf',
                bbox_inches='tight')
cm = skm.confusion_matrix(all_labels,all_probs)

plot_confusions(cm, " predicted label", "/Users/gjy/Desktop/ecg-master/ecg/saved/vk_confusion_matrix.pdf")

report = skm.classification_report(
            all_labels,all_probs,
            target_names=['A','N','O','~'],
            digits=3)
acc=skm.accuracy_score(all_labels,all_probs)
scores = skm.precision_recall_fscore_support(
                    all_labels,all_probs,
                    average=None)
print("report",report)
print("acc",acc)

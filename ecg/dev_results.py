import keras
import math
import numpy as np
import os
import sklearn
import sklearn.metrics as skm
import sys
from selfattention import *
from keras.models import Model
from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.utils.multiclass import type_of_target

import matplotlib.pyplot as plt
import matplotlib
from keras import backend as K
from sklearn.metrics import precision_recall_curve

import load
import util
import json
#matplotlib inline
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


#model_path = "/home/yuanzihan/software/python_workspace/gjy/ecg-master/saved/selfatt0.60.001/1575027419-185/0.473-0.859-012-0.294-0.894.hdf5"
#model_path = "/home/yuanzihan/software/python_workspace/gjy/ecg-master/saved/nosum1111/1574826858-344/0.412-0.863-018-0.288-0.909.hdf5"
model_path = "/home/yuanzihan/software/python_workspace/gjy/ecg-master/saved/wuenda/1574574551-467/0.417-0.851-012-0.326-0.885.hdf5"
data_json = "/home/yuanzihan/software/python_workspace/gjy/ecg-master/examples/cinc17/dev.json"

preproc = util.load(os.path.dirname(model_path))
dataset = load.load_dataset(data_json)
ecgs, labels = preproc.process(*dataset)

model = keras.models.load_model(model_path)
#model = keras.models.load_model(model_path,custom_objects={'Self_Attention': Self_Attention})
probs = model.predict(ecgs, verbose=1)

#accuracy = accuracy_score(np.array(labels), probs)



#recall = recall_score(np.array(labels), probs)
#f1 = f1_score(np.array(labels), probs)



#model.summary()

####混淆矩阵
def plot_confusions(cm, xlabel, filename):
    cm = sklearn.preprocessing.normalize(cm, norm='l1', axis=1, copy=True)
    classes = preproc.classes
    matplotlib.rcParams['figure.figsize'] = (8, 7)
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
#cm = skm.confusion_matrix(np.argmax(labels, axis=2).ravel(),
                              #np.argmax(probs, axis=2).ravel())
#plot_confusions(cm, " predicted label", "/home/yuanzihan/software/python_workspace/gjy/ecg-master/saved/wuenda/1574574551-467/wuenda_confusion_matrix.pdf")
####

###可视化 CNN
#layer_CNN = Model(inputs=model.input, outputs=model.get_layer('self__attention_1').output)
#f1 = layer_CNN.predict(ecgs)
#weight
#layer_weight =  Model(inputs=model.input, outputs=model.get_layer('activation_51').output)
#f2 = layer_weight.predict(ecgs)
###input
#x = np.arange(0, 2047, 1)
#plt.plot(x, ecgs[0,512:2559,0])
#b=f1.reshape((852,18176,1))
#plt.plot(x, b[0,512:2559,0])
#plt.savefig("/home/yuanzihan/software/python_workspace/gjy/ecg-master/saved/selfatt0.60.001/1575027419-185/cnnselfattention.png",dpi=520)
###

def stats(ground_truth, preds):
    labels = range(ground_truth.shape[2])
    g = np.argmax(ground_truth, axis=2).ravel()
    p = np.argmax(preds, axis=2).ravel()
    stat_dict = {}
    for i in labels:
        # compute all the stats for each label
        tp = np.sum(g[g==i] == p[g==i])
        fp = np.sum(g[p==i] != p[p==i])
        fn = np.sum(g==i) - tp
        tn = np.sum(g!=i) - fp
        stat_dict[i] = (tp, fp, fn, tn)
    return stat_dict

def acc(ground_truth, preds):
    accuracy = accuracy_score(ground_truth, preds)
    return accuracy

def to_set(preds):
    idxs = np.argmax(preds, axis=2)
    return [list(set(r)) for r in idxs]

def set_stats(ground_truth, preds):
    labels = range(ground_truth.shape[2])
    ground_truth = to_set(ground_truth)
    preds = to_set(preds)
    stat_dict = {}
    for x in labels:
        tp = 0; fp = 0; fn = 0; tn = 0;
        for g, p in zip(ground_truth, preds):
            if x in g and x in p: # tp
                tp += 1
            if x not in g and x in p: # fp
                fp += 1
            if x in g and x not in p:
                fn += 1
            if x not in g and x not in p:
                tn += 1
        stat_dict[x] = (tp, fp, fn, tn)
    return stat_dict



def compute_f1(tp, fp, fn, tn):
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    specificity = tn / float(tn + fp)
    npv = tn / float(tn + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, tp + fn



def print_results(seq_sd, set_sd):
    print
    ("\t\t Seq F1    Set F1")
    seq_tf1 = 0;
    seq_tot = 0
    set_tf1 = 0;
    set_tot = 0
    for k, v in seq_sd.items():
        set_f1, n = compute_f1(*set_sd[k])
        set_tf1 += n * set_f1
        set_tot += n
        seq_f1, n = compute_f1(*v)
        seq_tf1 += n * seq_f1
        seq_tot += n
        print(
        "{:>10} {:10.3f} {:10.3f}".format(
            preproc.classes[k], seq_f1, set_f1))
    print(
    "{:>10} {:10.3f} {:10.3f}".format(
        "Average", seq_tf1 / float(seq_tot), set_tf1 / float(set_tot)))


def c_statistic_with_95p_confidence_interval(cstat, num_positives, num_negatives, z_alpha_2=1.96):
    """
    Calculates the confidence interval of an ROC curve (c-statistic), using the method described
    under "Confidence Interval for AUC" here:
      https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Area_Under_an_ROC_Curve.pdf
    Args:
        cstat: the c-statistic (equivalent to area under the ROC curve)
        num_positives: number of positive examples in the set.
        num_negatives: number of negative examples in the set.
        z_alpha_2 (optional): the critical value for an N% confidence interval, e.g., 1.96 for 95%,
            2.326 for 98%, 2.576 for 99%, etc.
    Returns:
        The 95% confidence interval half-width, e.g., the Y in X ± Y.
    """
    q1 = cstat / (2 - cstat)
    q2 = 2 * cstat ** 2 / (1 + cstat)
    numerator = cstat * (1 - cstat) \
                + (num_positives - 1) * (q1 - cstat ** 2) \
                + (num_negatives - 1) * (q2 - cstat ** 2)
    standard_error_auc = math.sqrt(numerator / (num_positives * num_negatives))
    return z_alpha_2 * standard_error_auc

def roc_auc(ground_truth, probs, index):
    gts = np.argmax(ground_truth, axis=2)
    n_gts = np.zeros_like(gts)
    n_gts[gts==index] = 1
    n_pos = np.sum(n_gts == 1)
    n_neg = n_gts.size - n_pos
    n_ps = probs[..., index].squeeze()
    n_gts, n_ps = n_gts.ravel(), n_ps.ravel()
    return n_pos, n_neg, skm.roc_auc_score(n_gts, n_ps)



def roc_auc_set(ground_truth, probs, index):
    gts = np.argmax(ground_truth, axis=2)
    max_ps = np.max(probs[...,index], axis=1)
    max_gts = np.any(gts==index, axis=1)
    pos = np.sum(max_gts)
    neg = max_gts.size - pos
    return pos, neg, skm.roc_auc_score(max_gts, max_ps)


def print_aucs(ground_truth, probs):
    seq_tauc = 0.0; seq_tot = 0.0
    set_tauc = 0.0; set_tot = 0.0
    print ("\t        AUC")
    for idx, cname in preproc.int_to_class.items():
        pos, neg, seq_auc = roc_auc(ground_truth, probs, idx)
        seq_tot += pos
        seq_tauc += pos * seq_auc
        seq_conf = c_statistic_with_95p_confidence_interval(seq_auc, pos, neg)
        pos, neg, set_auc = roc_auc_set(ground_truth, probs, idx)
        set_tot += pos
        set_tauc += pos * set_auc
        set_conf = c_statistic_with_95p_confidence_interval(set_auc, pos, neg)
        print ("{: <8}\t{:.3f} ({:.3f}-{:.3f})\t{:.3f} ({:.3f}-{:.3f})".format(cname, seq_auc, seq_auc-seq_conf,seq_auc+seq_conf,set_auc, set_auc-set_conf, set_auc+set_conf))
    print ("Average\t\t{:.3f}\t{:.3f}".format(seq_tauc/seq_tot, set_tauc/set_tot))

#print_results(stats(labels, probs), set_stats(labels, probs))
#print_aucs(labels, probs)
#print(acc(labels,probs))






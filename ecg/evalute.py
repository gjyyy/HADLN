import collections
import json
import keras
import numpy as np
import os
import sys
sys.path.append("../../../ecg")
import scipy.stats as sst
from keras.models import Model

import util
import load

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
#model_path = "/home/yuanzihan/software/python_workspace/gjy/ecg-master/saved/nosum1111/1574826858-344/0.412-0.863-018-0.288-0.909.hdf5"
model_path = "/home/yuanzihan/software/python_workspace/gjy/ecg-master/saved/wuenda/1574574551-467/0.417-0.851-012-0.326-0.885.hdf5"
data_path = "/home/yuanzihan/software/python_workspace/gjy/ecg-master/examples/cinc17/dev.json"

data = load.load_dataset(data_path)
preproc = util.load(os.path.dirname(model_path))
model = keras.models.load_model(model_path)

data_path = "/home/yuanzihan/software/python_workspace/gjy/ecg-master/examples/cinc17/train.json"
with open("/home/yuanzihan/software/python_workspace/gjy/ecg-master/examples/cinc17/train.json", 'r') as fid:
    train_labels = [json.loads(l)['labels'] for l in fid]
counts = collections.Counter(preproc.class_to_int[l[0]] for l in train_labels)
counts = sorted(counts.most_common(), key=lambda x: x[0])
counts = list(zip(*counts))[1]
smooth = 500
counts = np.array(counts)[None, None, :]
total = np.sum(counts) + counts.shape[1]
prior = (counts + smooth) / float(total)

probs = []
labels = []
for x, y  in zip(*data):
    x, y = preproc.process([x], [y])
    #probs.append(model.predict(x))
    layer_CNN = Model(inputs=model.input, outputs=model.get_layer('activation_34').output)
    f1 = layer_CNN.predict(x)
    probs.append(f1)
    labels.append(y)

print("111111",probs)

preds = []
ground_truth = []
for p, g in zip(probs, labels):
    preds.append(sst.mode(np.argmax(p / prior, axis=2).squeeze())[0][0])
    ground_truth.append(sst.mode(np.argmax(g, axis=2).squeeze())[0][0])

#preds=[1, 3, 2, 0, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 0, 2, 2, 1, 2, 1, 1, 1, 1, 2, 0, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 0, 3, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 0, 2, 1, 1, 2, 1, 1, 0, 1, 2, 1, 2, 2, 2, 1, 3, 1, 1, 2, 0, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 0, 0, 1, 1, 2, 2, 1, 2, 0, 1, 1, 3, 1, 1, 2, 1, 3, 2, 2, 0, 1, 2, 1, 2, 0, 0, 2, 0, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 0, 2, 1, 1, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 0, 2, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 0, 1, 2, 1, 0, 1, 2, 1, 1, 1, 2, 2, 0, 1, 1, 1, 2, 3, 1, 1, 1, 0, 2, 1, 2, 2, 1, 1, 1, 2, 1, 0, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 0, 1, 1, 1, 3, 1, 1, 1, 1, 0, 1, 2, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1, 2, 2, 1, 0, 1, 1, 1, 2, 1, 1, 2, 0, 2, 1, 1, 2, 2, 1, 1, 1, 0, 2, 1, 1, 2, 2, 2, 1, 3, 1, 0, 2, 2, 2, 1, 3, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 0, 1, 2, 2, 1, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 0, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 3, 1, 1, 0, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 0, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 0, 2, 3, 2, 2, 1, 3, 2, 1, 2, 2, 1, 1, 1, 1, 0, 3, 1, 2, 2, 2, 3, 1, 1, 0, 1, 2, 1, 2, 1, 1, 2, 1, 1, 0, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 0, 1, 0, 1, 2, 1, 1, 0, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 3, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 0, 1, 2, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 2, 0, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 0, 1, 0, 2, 0, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 0, 1, 2, 1, 2, 2, 0, 1, 1, 1, 1, 2, 1, 2, 1, 0, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 0, 0, 2, 2, 1, 1, 2, 1, 2, 1, 1, 0, 2, 1, 1, 1, 1, 2, 1, 0, 1, 0, 1, 3, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 3, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 0, 0, 1, 2, 1, 0, 3, 1, 1, 2, 3, 0, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 0, 2, 1, 1, 0, 2, 1, 2, 1, 2, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0, 0, 1, 2, 1, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 3, 1, 1, 0, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 0, 0]


# import sklearn.metrics as skm
# report = skm.classification_report(
#             ground_truth, preds,
#             target_names=preproc.classes,
#             digits=3)
# acc=skm.accuracy_score(ground_truth, preds)
# scores = skm.precision_recall_fscore_support(
#                     ground_truth,
#                     preds,
#                     average=None)
# print(report)
# print("111111",acc)
#print ("CINC Average {:3f}".format(np.mean(scores[2][:3])))
import os
import numpy as np
import random
import time
import glob
from Bio import SeqIO
from pybedtools import BedTool
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from sklearn import metrics
import h5py

INPUT_LENGTH = 1000
EPOCH = 200
BATCH_SIZE = 64
WORK_DIR = "./"

def run_model(data, model, save_dir):

    weights_file = os.path.join(save_dir, "model_weights.hdf5")
    model_file = os.path.join(save_dir, "single_model.hdf5")
    model.save(model_file)

    # Adadelta is recommended to be used with default values
    opt = Adadelta()

    # parallel_model = ModelMGPU(model, gpus=GPUS)
    parallel_model = model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train = data["train_data"]
    Y_train = data["train_labels"]
    X_validation = data["val_data"]
    Y_validation = data["val_labels"]
    X_test = data["test_data"]
    Y_test = data["test_labels"]

    from keras.utils.np_utils import to_categorical
    Y_train = to_categorical(Y_train, num_classes=None)
    Y_test = to_categorical(Y_test, num_classes=None)
    Y_validation = to_categorical(Y_validation, num_classes=None)

    _callbacks = []
    checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
    _callbacks.append(checkpointer)
    earlystopper = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    _callbacks.append(earlystopper)

    parallel_model.fit(X_train,
                       Y_train,
                       batch_size=BATCH_SIZE * GPUS,
                       epochs=EPOCH,
                       validation_data=(X_validation, Y_validation),
                       shuffle=True,
                       callbacks=_callbacks, verbose=1)

    Y_pred = parallel_model.predict(X_test)

    auc1 = metrics.roc_auc_score(Y_test[:,1], Y_pred[:,1])
    auc2 = metrics.roc_auc_score(Y_test[:,2], Y_pred[:,2])

    with open(os.path.join(save_dir, "auc.txt"), "w") as of:
        of.write("enhancer AUC: %f\n" % auc2)
        of.write("silencer AUC: %f\n" % auc1)

    [fprs, tprs, thrs] = metrics.roc_curve(Y_test[:,1], Y_pred[:, 1])
    sort_ix = np.argsort(np.abs(fprs - 0.1))
    fpr10_thr = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.05))
    fpr5_thr = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.03))
    fpr3_thr = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.01))
    fpr1_thr = thrs[sort_ix[0]]

    [fprs, tprs, thrs] = metrics.roc_curve(Y_test[:,2], Y_pred[:, 2])
    sort_ix = np.argsort(np.abs(fprs - 0.1))
    fpr10_thre = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.05))
    fpr5_thre = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.03))
    fpr3_thre = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.01))
    fpr1_thre = thrs[sort_ix[0]]

    with open(os.path.join(save_dir, "fpr_threshold_scores.txt"), "w") as of:
        of.write("silencer 10 \t %f\n" % fpr10_thr)
        of.write("5 \t %f\n" % fpr5_thr)
        of.write("3 \t %f\n" % fpr3_thr)
        of.write("1 \t %f\n\n" % fpr1_thr)
        of.write("enhancer 10 \t %f\n" % fpr10_thre)
        of.write("5 \t %f\n" % fpr5_thre)
        of.write("3 \t %f\n" % fpr3_thre)
        of.write("1 \t %f\n" % fpr1_thre)

def load_dataset(Dfile):

    print("reading enhancers...")
    data = {}
    with h5py.File(Dfile, "r") as inf:
        for _key in inf:
            data[_key] = inf[_key][()]
    return data

def train_model(Dfile,results_dir):

    model_file = WORK_DIR + "/source_files/model.hdf5"
    model = load_model(model_file)
   
    if not os.path.exists(Dfile):
        print("no data file"+Dfile)
        exit()
        
    data = load_dataset(Dfile)
    run_model(data, model, results_dir)

    
if __name__ == "__main__":

    import sys
    data = sys.argv[1]
    results_dir = sys.argv[2]
    if not os.path.exists(results_dir):
         os.mkdir(results_dir)
    train_model(data,results_dir)

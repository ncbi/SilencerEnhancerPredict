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
train_chromosomes = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr10", "chr11", "chr12", "chr13",
                     "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22"]
validation_chromosomes = ["chr7"]
test_chromosomes = ["chr8", "chr9"]

nucleotides = ['A', 'C', 'G', 'T']
INPUT_LENGTH = 1000
#FASTA_FILE = "/data/Dcode/common/hg38.fa"

def get_chrom2seq(fasta_file, capitalize=True):

    chrom2seq = {}
    for seq in SeqIO.parse(fasta_file, "fasta"):
        chrom2seq[seq.description.split()[0]] = seq.seq.upper() if capitalize else seq.seq

    return chrom2seq

def seq2one_hot(seq):

    d = np.array(['A', 'C', 'G', 'T'])

    return np.fromstring(str(seq.upper()), dtype='|S1')[:, np.newaxis] == d

def create_dataset(en_bed_file, sl_bed_file,neg_bed_file, data_file,fasta_file):

    chrom2seq = get_chrom2seq(fasta_file)

    print "Generating the positive dataset"

    en_beds = list(BedTool(en_bed_file))
    sl_beds = list(BedTool(sl_bed_file))
    neg_beds = list(BedTool(neg_bed_file))

    en_train_bed = [r for r in en_beds if r.chrom in train_chromosomes]
    en_val_bed = [r for r in en_beds if r.chrom in validation_chromosomes]
    en_test_bed = [r for r in en_beds if r.chrom in test_chromosomes]
    sl_train_bed = [r for r in sl_beds if r.chrom in train_chromosomes]
    sl_val_bed = [r for r in sl_beds if r.chrom in validation_chromosomes]
    sl_test_bed = [r for r in sl_beds if r.chrom in test_chromosomes]    

    pos_train_data = []
    pos_val_data = []
    pos_test_data = []

    pos_train_label = []
    pos_val_label = []
    pos_test_label = []
    for bed_list, data_list, label_list in zip([en_train_bed, en_val_bed, en_test_bed],
                                   [pos_train_data, pos_val_data, pos_test_data], 
				    [pos_train_label, pos_val_label,pos_test_label]):
    
        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == 1000:
                continue
            _vector = seq2one_hot(_seq)
            data_list.append(_vector)
            label_list.append(1)
    
    for bed_list, data_list, label_list in zip([sl_train_bed, sl_val_bed, sl_test_bed],
                                   [pos_train_data, pos_val_data, pos_test_data],
                                    [pos_train_label, pos_val_label,pos_test_label]):

        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == 1000:
                continue
            _vector = seq2one_hot(_seq)
            data_list.append(_vector)
            label_list.append(2)

    print "train enhancer/silencer samples: "+ str(len(pos_train_data))
    print "validation enhancer/silencer samples: "+ str(len(pos_val_data))
    print "test enhancer/silencer samples: "+ str(len(pos_test_data))

    print "Generating the negative dataset"

    neg_train_bed = [r for r in neg_beds if r.chrom in train_chromosomes]
    neg_val_bed = [r for r in neg_beds if r.chrom in validation_chromosomes]
    neg_test_bed = [r for r in neg_beds if r.chrom in test_chromosomes]

    neg_train_data = []
    neg_val_data = []
    neg_test_data = []

    for bed_list, data_list in zip([neg_train_bed, neg_val_bed, neg_test_bed],
                                   [neg_train_data, neg_val_data, neg_test_data]):
        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == 1000:
                continue
            _vector = seq2one_hot(_seq)
            data_list.append(_vector)

    print "train negative samples: " + str(len(neg_train_data))
    print "validation negative samples: " + str(len(neg_val_data))
    print "test negative samples: " + str(len(neg_test_data))

    print "Merging positive and negative to single matrices"

    pos_train_data_matrix = np.zeros((len(pos_train_data), INPUT_LENGTH, 4))
    for i in range(len(pos_train_data)):
        pos_train_data_matrix[i, :, :] = pos_train_data[i]
    pos_val_data_matrix = np.zeros((len(pos_val_data), INPUT_LENGTH, 4))
    for i in range(len(pos_val_data)):
        pos_val_data_matrix[i, :, :] = pos_val_data[i]
    pos_test_data_matrix = np.zeros((len(pos_test_data), INPUT_LENGTH, 4))
    for i in range(len(pos_test_data)):
        pos_test_data_matrix[i, :, :] = pos_test_data[i]

    neg_train_data_matrix = np.zeros((len(neg_train_data), INPUT_LENGTH, 4))
    for i in range(len(neg_train_data)):
        neg_train_data_matrix[i, :, :] = neg_train_data[i]
    neg_val_data_matrix = np.zeros((len(neg_val_data), INPUT_LENGTH, 4))
    for i in range(len(neg_val_data)):
        neg_val_data_matrix[i, :, :] = neg_val_data[i]
    neg_test_data_matrix = np.zeros((len(neg_test_data), INPUT_LENGTH, 4))
    for i in range(len(neg_test_data)):
        neg_test_data_matrix[i, :, :] = neg_test_data[i]

    test_data = np.vstack((pos_test_data_matrix, neg_test_data_matrix))
    train_data = np.vstack((pos_train_data_matrix, neg_train_data_matrix))
    val_data = np.vstack((pos_val_data_matrix, neg_val_data_matrix))
    
    i1 = np.zeros((len(pos_test_label),3))
    i1[np.array(pos_test_label)==1,0] = 1
    i1[np.array(pos_test_label)==2,1] = 1
    i = np.zeros((neg_test_data_matrix.shape[0],3))
    i[:,2] = 1
    test_label = np.vstack((i1,i))
    
    i1 = np.zeros((len(pos_val_label),3))
    i1[np.array(pos_val_label)==1,0] = 1
    i1[np.array(pos_val_label)==2,1] = 1
    i = np.zeros((neg_val_data_matrix.shape[0],3))
    i[:,2] = 1
    val_label = np.vstack((i1,i))

    i1 = np.zeros((len(pos_train_label),3))
    i1[np.array(pos_train_label)==1,0] = 1
    i1[np.array(pos_train_label)==2,1] = 1
    i = np.zeros((neg_train_data_matrix.shape[0],3))
    i[:,2] = 1
    train_label = np.vstack((i1,i))
    
    print(test_label.sum(axis=0))
    print(val_label.sum(axis=0))
    print(train_label.sum(axis=0))

    with h5py.File(data_file, "w") as of:
        of.create_dataset(name="test_data", data=test_data, compression="gzip")
        of.create_dataset(name="train_data", data=train_data, compression="gzip")
        of.create_dataset(name="val_data", data=val_data, compression="gzip")
        of.create_dataset(name="test_labels", data=test_label, compression="gzip")
        of.create_dataset(name="train_labels", data=train_label, compression="gzip")
        of.create_dataset(name="val_labels", data=val_label, compression="gzip")

if __name__ == "__main__":

    import sys
    enbed = sys.argv[1]
    slbed = sys.argv[2]
    negbed = sys.argv[3]
    data_file = sys.argv[4]
    FASTA_file = sys.argv[5]

    create_dataset(enbed,slbed,negbed,data_file,FASTA_file)

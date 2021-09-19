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

INPUT_LENGTH = 1000
nucleotides = ['A', 'C', 'G', 'T']

def get_chrom2seq(FASTA_FILE, capitalize=True):

    chrom2seq = {}
    for seq in SeqIO.parse(FASTA_FILE, "fasta"):
        chrom2seq[seq.description.split()[0]] = seq.seq.upper() if capitalize else seq.seq

    return chrom2seq


def seq2one_hot(seq):

    d = np.array(['A', 'C', 'G', 'T'])

    return np.fromstring(str(seq.upper()), dtype='|S1')[:, np.newaxis] == d

def create_dataset(bed_file, data_file, chrom2seq=None):

    if not chrom2seq:
        chrom2seq = get_chrom2seq()
        # return chrom2seq

    print "Generating the data"

    beds = list(BedTool(bed_file))

    train_bed = [r for r in beds if r.chrom in train_chromosomes]
    val_bed = [r for r in beds if r.chrom in validation_chromosomes]
    test_bed = [r for r in beds if r.chrom in test_chromosomes]

    train_data = []
    val_data = []
    test_data = []

    for bed_list, data_list in zip([train_bed, val_bed, test_bed],
                                   [train_data, val_data, test_data]):

        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == 1000:
                continue
            _vector = seq2one_hot(_seq)
            data_list.append(_vector)

    print len(train_data)
    print len(val_data)
    print len(test_data)

    train_data_matrix = np.zeros((len(train_data), INPUT_LENGTH, 4))
    for i in range(len(train_data)):
        train_data_matrix[i, :, :] = train_data[i]
    val_data_matrix = np.zeros((len(val_data), INPUT_LENGTH, 4))
    for i in range(len(val_data)):
        val_data_matrix[i, :, :] = val_data[i]
    test_data_matrix = np.zeros((len(test_data), INPUT_LENGTH, 4))
    for i in range(len(test_data)):
        test_data_matrix[i, :, :] = test_data[i]

    print "Saving to file:", dataset_save_file
    with h5py.File(dataset_save_file, "w") as of:
        of.create_dataset(name="test_data", data=test_data, compression="gzip")
        of.create_dataset(name="train_data", data=train_data, compression="gzip")
        of.create_dataset(name="val_data", data=val_data, compression="gzip")


def create_dataset(bedlist):

    chrom2seq = get_chrom2seq()
    dataset_save_file = posf+"fasta.hdf5"
    create_dataset_phase_two_unbinned(posf,dataset_save_file, chrom2seq=chrom2seq)


if __name__ == "__main__":

    # data_file = WORK_DIR + "/E118.H3K27ac.phase_two.hdf5"
    # results_dir = WORK_DIR + "/E118.H3K27ac.phase_two.results"
    import sys
    pos_file = sys.argv[1]
    create_datasets(pos_file)

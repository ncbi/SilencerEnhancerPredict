# CNN model for predicting enhancers and silencers
## How to use

### Generating data 

prerequisite: the whole_genome_sequence_file (downloaded from the USCS genome browse, it is too big to be included here).

        generate_data.py enhancers.bed silencers.bed controls.bed outputdata genome.fasta

example:

        generate_data.py ./examples/tempEN.bed ./examples/tempSL.bed ./examples/tempBK.bed temp.data.hdf5 hg19.fa

## Traing a model 
input: data file, and target directory for the results

        A data file includes training and validation sample sets. 
        Each sample set is represented by:
           a one-hot-encode DNA sequence matrix in a size of N * 1000 * 4, 
           a 3-class matrix in a size of N * 3. N is the number of samples.
 
example:
python train.py  ./examples/data_training.hdf5 ./examples/

output: in the directory ./examples/

        auc.txt

        fpr_threshold_scores.txt
        
        model_weights.hdf5
 
 
### Making predictions with a built model, 
input: data file, model file, 
 
example:
python train.py ./examples/data_prediction.hdf5 ./examples/model_weights.hdf5

output: ./examples/data_prediction.hdf5.predict.data

## Featuring

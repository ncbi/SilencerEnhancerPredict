# SilencerPred
## How to use
### Traing a model 
input: data file, and target directory for the results

A data file include a one-hot-encode DNA sequence matrix in a size of N * 1000 * 4, and a 3-class matrix in a size of N * 3. N is the number of samples.
 
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

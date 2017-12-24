#### RNN

C implementation of [WildML RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)

* Vanilla RNN w/ softmax output layer
    - For classification
* Vanilla RNN w/ unbounded output layer
    - For regression

#### Note
* Folder
    - Input: ./data/input
    - Output: ./data/output
* Input/Output sample format
    - Input file name: "exp_{dataset_identifier}_{#_of_sample}_{param_N_name}{param_N}_..._{iteration}.txt"
    - Output file name: "res_{dataset_identifier}_{#_of_sample}_{param_N_name}{param_N}_..._{iteration}.txt"
    1. \# of sample
    2. Dimension of input 0 (m/row, n/col)
    3. Row major serialized input matrix 0
    4. Dimension of output 0 (m/row, n/col)
    5. Row major serialized outputput matrix 0
    6. Repeat 2~5 until EOF
* Change `RNN_RAND_SEED` in RNN.c for difference initial model parameter
    - *This may be more important then model structure*


#### Parallel Programming Term Project
* Parallelize Matrix operation
    * Dimension of input/hidden/output layer should be big
* LSTM not required if the outcome does not matter




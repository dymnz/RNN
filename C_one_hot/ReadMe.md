#### RNN for one-hot I/O vector
* 1 hidden layer
    * Adjusable hidden cell count
* tanh() hidden node
* softmax() output node

#### Note
* Input and output should be one-hot vector

#### TODO
* ~~Dynamic learning rate~~
* ~~Gradient check~~
* Clean up the code
* Abstract matrix operation
* Abstract squash function derivative for gradeint descent calculation
* LSTM!

#### Reference
* http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/


#### Test

// RNN_Train_test()
```
loss at epoch:  1000 =   0.473759
loss at epoch:  2000 =   0.179101
loss at epoch:  3000 =   0.046927
loss at epoch:  4000 =   0.025632
loss at epoch:  5000 =   0.017457
loss at epoch:  6000 =   0.013084
loss at epoch:  7000 =   0.010352
loss at epoch:  8000 =   0.008499
loss at epoch:  9000 =   0.007173
Symbol: ['h', 'e', 'l', 'o']
-------------input_matrix
1.00000  0.00000  0.00000  0.00000  
0.00000  1.00000  0.00000  0.00000  
0.00000  0.00000  1.00000  0.00000  
0.00000  0.00000  1.00000  0.00000  
0.00000  0.00000  0.00000  1.00000  
-------------expected_output_matrix
0.00000  1.00000  0.00000  0.00000  
0.00000  0.00000  1.00000  0.00000  
0.00000  0.00000  1.00000  0.00000  
0.00000  0.00000  0.00000  1.00000  
1.00000  0.00000  0.00000  0.00000  
-------------predicted_output_matrix
0.00171  0.99546  0.00219  0.00064  
0.00003  0.00008  0.99972  0.00017  
0.00031  0.00084  0.99096  0.00790  
0.00586  0.00054  0.00555  0.98805  
0.99502  0.00204  0.00003  0.00291

```
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "common_math.h"


typedef struct {
	int i_dim;
	int o_dim;
	int h_dim;
	int bptt_truncate_len;

    Matrix_t *C;    // TxH
    Matrix_t *S;    // TxH
    Matrix_t *aS;    // TxH

    Matrix_t *Ig;    // TxH
    Matrix_t *Fg;    // TxH
    Matrix_t *Og;    // TxH

    Matrix_t *V;   // HxO
    Matrix_t *dV;   // HxO

    Matrix_t *Ui;   // IxH
    Matrix_t *Wi;   // HxH
    Matrix_t *dUi;  // IxH
    Matrix_t *dWi;  // HxH

    Matrix_t *Uf;   // IxH
    Matrix_t *Wf;   // HxH
    Matrix_t *dUf;  // IxH
    Matrix_t *dWf;  // HxH    

    Matrix_t *Uo;   // IxH
    Matrix_t *Wo;   // HxH
    Matrix_t *dUo;  // IxH
    Matrix_t *dWo;  // HxH    

    Matrix_t *Us;   // IxH
    Matrix_t *Ws;   // HxH      
    Matrix_t *dUs;  // IxH
    Matrix_t *dWs;  // HxH
} RNN_t;

typedef struct {
	Matrix_t **input_matrix_list;
	Matrix_t **output_matrix_list;
    int input_max_m;
    int input_n;      // This should be the same accoross all input matrix

    int output_max_m;
    int output_n;      // This should be the same accoross all output matrix

	int num_matrix;
} DataSet_t;

void TrainSet_init(DataSet_t *train_set, int num_matrix);
void DataSet_destroy(DataSet_t *train_set);

void RNN_init(
    RNN_t *RNN_storage, 
    int input_vector_len, 
    int output_vector_len,
    int hidden_layer_vector_len,
    int bptt_truncate_len
);

void RNN_destroy(RNN_t *RNN_storage);

void RNN_forward_propagation(
	RNN_t *RNN_storage,
	Matrix_t *input_matrix,
	Matrix_t *output_matrix
);

math_t RNN_loss_calculation(
    RNN_t *RNN_storage,
    Matrix_t *predicted_output_matrix,	// TxO
    Matrix_t *expected_output_matrix	// TxO
);

void RNN_BPTT(
	RNN_t *RNN_storage,
	Matrix_t *input_matrix,
	Matrix_t *predicted_output_matrix,
	Matrix_t *expected_output_matrix
);

void RNN_SGD(
    RNN_t *RNN_storage,
    Matrix_t *input_matrix,
    Matrix_t *expected_output_matrix,
    Matrix_t *predicted_output_matrix,
    math_t learning_rate    
);

void RNN_train(
    RNN_t *RNN_storage,
    DataSet_t *train_set,
    Matrix_t *predicted_output_matrix,
    math_t initial_learning_rate,
    int max_epoch,
    int print_loss_interval
);

void RNN_Predict(
    RNN_t *RNN_storage,
    Matrix_t *input_matrix,    
    Matrix_t *predicted_output_matrix
);

int RNN_Gradient_check(
    RNN_t *RNN_storage,
    DataSet_t *train_set,
    Matrix_t *predicted_output_matrix,
    math_t h,
    math_t error_threshold,
    int index_to_check
);

math_t gate_squash_func(math_t value);
math_t cell_state_squash_func(math_t value);
math_t cell_output_squash_func(math_t value);
math_t network_output_squash_func(math_t value);

math_t sigmoid(math_t value);
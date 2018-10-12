#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "common_math.h"


typedef struct {
    int i_dim;
    int o_dim;
    int h_dim;

    /* LSTM state */
    Matrix_t *Z_, *I_, *F_, *O_;        // TxH
    Matrix_t *Z, *I, *F, *O;            // TxH
    Matrix_t *C, *Y;                    // TxH

    /* LSTM model */    
    Matrix_t *Wz, *Wi, *Wf, *Wo;    // HxI
    Matrix_t *Rz, *Ri, *Rf, *Ro;    // HxH
    Matrix_t *Pi, *Pf, *Po;         // 1xH
    Matrix_t *Bz, *Bi, *Bf, *Bo;    // 1xH

    Matrix_t *dWz, *dWi, *dWf, *dWo;    // HxI
    Matrix_t *dRz, *dRi, *dRf, *dRo;    // HxH
    Matrix_t *dPi, *dPf, *dPo;          // 1xH
    Matrix_t *dBz, *dBi, *dBf, *dBo;    // 1xH  

    /* Output model */
    Matrix_t *V;                    // OxH
    Matrix_t *Bpo;                  // 1xO

    Matrix_t *dV;                    // OxH
    Matrix_t *dBpo;                  // 1xO

    Matrix_t *EdWz, *EdWi, *EdWf, *EdWo;    // HxI
    Matrix_t *EdRz, *EdRi, *EdRf, *EdRo;    // HxH
    Matrix_t *EdPi, *EdPf, *EdPo;          // 1xH
    Matrix_t *EdBz, *EdBi, *EdBf, *EdBo;    // 1xH  
    Matrix_t *EdV;                    // OxH
    Matrix_t *EdBpo;                  // 1xO

    Matrix_t *dEdWz, *dEdWi, *dEdWf, *dEdWo;    // HxI
    Matrix_t *dEdRz, *dEdRi, *dEdRf, *dEdRo;    // HxH
    Matrix_t *dEdPi, *dEdPf, *dEdPo;          // 1xH
    Matrix_t *dEdBz, *dEdBi, *dEdBf, *dEdBo;    // 1xH  
    Matrix_t *dEdV;                    // OxH
    Matrix_t *dEdBpo;                  // 1xO

    math_t gamma; // Adadelta update weight

} RNN_t;

typedef struct {
    int ending_epoch;    
    math_t last_training_loss;
    int best_epoch_cross;
    math_t best_cross_loss;
    char terminate_reason[FILENAME_LENGTH];
    RNN_t *RNN_best_model;
} RNN_result_t;

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
    int hidden_layer_vector_len
);

void RNN_destroy(RNN_t *RNN_storage);

void RNN_copy_model(RNN_t *source, RNN_t *dest);

void RNN_forward_propagation(
    RNN_t *RNN_storage,
    Matrix_t *input_matrix,
    Matrix_t *output_matrix
);

math_t RNN_loss_calculation(
    RNN_t *RNN_storage,
    Matrix_t *predicted_output_matrix,  // TxO
    Matrix_t *expected_output_matrix    // TxO
);

math_t RNN_find_set_loss(
    RNN_t * RNN_storage,
    DataSet_t *dataset,
    Matrix_t *predicted_output_matrix
);

math_t* RNN_RMSE(
    RNN_t *RNN_storage,
    Matrix_t *predicted_output_matrix,  // TxO
    Matrix_t *expected_output_matrix    // TxO
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
    Matrix_t *predicted_output_matrix
);

math_t RNN_train(
    RNN_t * RNN_storage,
    DataSet_t *train_set,
    Matrix_t *predicted_output_matrix,
    int max_epoch,
    int print_loss_interval,
    int gradient_check_interval
);


RNN_result_t* RNN_train_cross_valid(
    RNN_t * RNN_storage,
    DataSet_t *train_set,
    DataSet_t *cross_set,
    Matrix_t *predicted_output_matrix,
    int max_epoch,
    int cross_valid_patience,
    int print_loss_interval,
    int gradient_check_interval
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

void RNN_stepFP_init(
    RNN_t *RNN_storage,
    Matrix_t *input_matrix, // TxI
    Matrix_t *predicted_output_matrix   // TxO
);

void RNN_stepFP(
    RNN_t *RNN_storage,
    Matrix_t *input_matrix, // TxI
    Matrix_t *predicted_output_matrix   // TxO
);



void RNN_Gradient_Clipping(Matrix_t *matrix, math_t threshold);

math_t gate_squash_func(math_t value);
math_t gate_squash_derivative(math_t value);
math_t cell_state_squash_func(math_t value);
math_t cell_state_squash_derivative(math_t value);
math_t cell_output_squash_func(math_t value);
math_t cell_output_squash_derivative(math_t value);
math_t network_output_squash_func(math_t value);

math_t sigmoid(math_t value);
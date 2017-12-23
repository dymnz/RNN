#include "RNN.h"

#define RNN_RAND_SEED 1

void TrainSet_init(DataSet_t *train_set, int num_matrix) {
	train_set->num_matrix = num_matrix;
	train_set->input_matrix_list =
	    (Matrix_t **) malloc(num_matrix * sizeof(Matrix_t *));
	train_set->output_matrix_list =
	    (Matrix_t **) malloc(num_matrix * sizeof(Matrix_t *));
}

void DataSet_destroy(DataSet_t *train_set) {
	int i;
	for (i = 0; i < train_set->num_matrix; ++i) {
		matrix_free(train_set->input_matrix_list[i]);
		matrix_free(train_set->output_matrix_list[i]);
	}
	free(train_set->input_matrix_list);
	free(train_set->output_matrix_list);
	free(train_set);
}

void RNN_init(
    RNN_t *RNN_storage,
    int input_vector_len,
    int output_vector_len,
    int hidden_layer_vector_len,
    int bptt_truncate_len
) {
	unsigned int seed = RNN_RAND_SEED;

	int i_dim = input_vector_len;
	int o_dim = output_vector_len;
	int h_dim = hidden_layer_vector_len;

	RNN_storage->i_dim = i_dim;
	RNN_storage->o_dim = o_dim;
	RNN_storage->h_dim = h_dim;
	RNN_storage->bptt_truncate_len = bptt_truncate_len;

	/* LSTM state */
	// Size to be adjusted for different test sample size
	RNN_storage->Z_ = matrix_create(0, 0); RNN_storage->Z = matrix_create(0, 0);
	RNN_storage->I_ = matrix_create(0, 0); RNN_storage->I = matrix_create(0, 0);
	RNN_storage->F_ = matrix_create(0, 0); RNN_storage->F = matrix_create(0, 0);
	RNN_storage->O_ = matrix_create(0, 0); RNN_storage->O = matrix_create(0, 0);
	RNN_storage->C = matrix_create(0, 0);
	RNN_storage->Y = matrix_create(0, 0);

	/* LSTM model */
	RNN_storage->Wz = matrix_create(h_dim, i_dim); RNN_storage->dWz = matrix_create(h_dim, i_dim);
	RNN_storage->Wi = matrix_create(h_dim, i_dim); RNN_storage->dWi = matrix_create(h_dim, i_dim);
	RNN_storage->Wf = matrix_create(h_dim, i_dim); RNN_storage->dWf = matrix_create(h_dim, i_dim);
	RNN_storage->Wo = matrix_create(h_dim, i_dim); RNN_storage->dWo = matrix_create(h_dim, i_dim);
	matrix_random_with_seed(RNN_storage->Wz, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(RNN_storage->Wi, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(RNN_storage->Wf, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(RNN_storage->Wo, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);

	RNN_storage->Rz = matrix_create(h_dim, h_dim); RNN_storage->dRz = matrix_create(h_dim, h_dim);
	RNN_storage->Ri = matrix_create(h_dim, h_dim); RNN_storage->dRi = matrix_create(h_dim, h_dim);
	RNN_storage->Rf = matrix_create(h_dim, h_dim); RNN_storage->dRf = matrix_create(h_dim, h_dim);
	RNN_storage->Ro = matrix_create(h_dim, h_dim); RNN_storage->dRo = matrix_create(h_dim, h_dim);
	matrix_random_with_seed(RNN_storage->Rz, -sqrt(1 / i_dim), sqrt(1 / i_dim), &seed);
	matrix_random_with_seed(RNN_storage->Ri, -sqrt(1 / i_dim), sqrt(1 / i_dim), &seed);
	matrix_random_with_seed(RNN_storage->Rf, -sqrt(1 / i_dim), sqrt(1 / i_dim), &seed);
	matrix_random_with_seed(RNN_storage->Ro, -sqrt(1 / i_dim), sqrt(1 / i_dim), &seed);

	RNN_storage->Pi = matrix_create(1, h_dim); RNN_storage->dPi = matrix_create(1, h_dim);
	RNN_storage->Pf = matrix_create(1, h_dim); RNN_storage->dPf = matrix_create(1, h_dim);
	RNN_storage->Po = matrix_create(1, h_dim); RNN_storage->dPo = matrix_create(1, h_dim);
	matrix_random_with_seed(RNN_storage->Pi, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(RNN_storage->Pf, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(RNN_storage->Po, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);

	RNN_storage->Bz = matrix_create(1, h_dim); RNN_storage->dBz = matrix_create(1, h_dim);
	RNN_storage->Bi = matrix_create(1, h_dim); RNN_storage->dBi = matrix_create(1, h_dim);
	RNN_storage->Bf = matrix_create(1, h_dim); RNN_storage->dBf = matrix_create(1, h_dim);
	RNN_storage->Bo = matrix_create(1, h_dim); RNN_storage->dBo = matrix_create(1, h_dim);
	matrix_random_with_seed(RNN_storage->Bz, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(RNN_storage->Bi, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(RNN_storage->Bf, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(RNN_storage->Bo, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);

	/* Output model */
	RNN_storage->V = matrix_create(o_dim, h_dim); RNN_storage->dV = matrix_create(o_dim, h_dim);
	RNN_storage->Bpo = matrix_create(1, o_dim); RNN_storage->dBpo = matrix_create(1, o_dim);
	matrix_random_with_seed(RNN_storage->V, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(RNN_storage->Bpo, -sqrt(1 / h_dim), sqrt(1 / h_dim), &seed);
}

void RNN_destroy(RNN_t *RNN_storage) {
	matrix_free(RNN_storage->Z_); matrix_free(RNN_storage->Z);
	matrix_free(RNN_storage->I_); matrix_free(RNN_storage->I);
	matrix_free(RNN_storage->F_); matrix_free(RNN_storage->F);
	matrix_free(RNN_storage->O_); matrix_free(RNN_storage->O);
	matrix_free(RNN_storage->C);
	matrix_free(RNN_storage->Y);

	matrix_free(RNN_storage->Wz);
	matrix_free(RNN_storage->Wi);
	matrix_free(RNN_storage->Wf);
	matrix_free(RNN_storage->Wo);

	matrix_free(RNN_storage->Rz);
	matrix_free(RNN_storage->Ri);
	matrix_free(RNN_storage->Rf);
	matrix_free(RNN_storage->Ro);

	matrix_free(RNN_storage->Pi);
	matrix_free(RNN_storage->Pf);
	matrix_free(RNN_storage->Po);

	matrix_free(RNN_storage->Bz);
	matrix_free(RNN_storage->Bi);
	matrix_free(RNN_storage->Bf);
	matrix_free(RNN_storage->Bo);

	matrix_free(RNN_storage->V);
	matrix_free(RNN_storage->Bpo);

	free(RNN_storage);
}

void RNN_forward_propagation(
    RNN_t *RNN_storage,
    Matrix_t *input_matrix,	// TxI
    Matrix_t *predicted_output_matrix	// TxO
) {
	int i_dim = RNN_storage->i_dim;
	int o_dim = RNN_storage->o_dim;
	int t_dim = input_matrix->m;
	int h_dim = RNN_storage->h_dim;

	matrix_resize(RNN_storage->Z_, t_dim, h_dim); matrix_resize(RNN_storage->Z, t_dim, h_dim);
	matrix_resize(RNN_storage->I_, t_dim, h_dim); matrix_resize(RNN_storage->I, t_dim, h_dim);
	matrix_resize(RNN_storage->F_, t_dim, h_dim); matrix_resize(RNN_storage->F, t_dim, h_dim);
	matrix_resize(RNN_storage->O_, t_dim, h_dim); matrix_resize(RNN_storage->O, t_dim, h_dim);
	matrix_resize(RNN_storage->C, t_dim, h_dim);
	matrix_resize(RNN_storage->Y, t_dim, h_dim);

	math_t **X = input_matrix->data;
	math_t **P_O = predicted_output_matrix->data;

	math_t **Z_ = RNN_storage->Z_->data; 	math_t **Z = RNN_storage->Z->data;
	math_t **I_ = RNN_storage->I_->data; 	math_t **I = RNN_storage->I->data;
	math_t **F_ = RNN_storage->F_->data; 	math_t **F = RNN_storage->F->data;
	math_t **O_ = RNN_storage->O_->data; 	math_t **O = RNN_storage->O->data;
	math_t **C = RNN_storage->C->data;
	math_t **Y = RNN_storage->Y->data;

	math_t **Wz = RNN_storage->Wz->data; math_t **Rz = RNN_storage->Rz->data;
	math_t **Wi = RNN_storage->Wi->data; math_t **Ri = RNN_storage->Ri->data;
	math_t **Wf = RNN_storage->Wf->data; math_t **Rf = RNN_storage->Rf->data;
	math_t **Wo = RNN_storage->Wo->data; math_t **Ro = RNN_storage->Ro->data;

	math_t *Bz = RNN_storage->Bz->data[0];
	math_t *Bi = RNN_storage->Bi->data[0]; math_t *Pi = RNN_storage->Pi->data[0];
	math_t *Bf = RNN_storage->Bf->data[0]; math_t *Pf = RNN_storage->Pf->data[0];
	math_t *Bo = RNN_storage->Bo->data[0]; math_t *Po = RNN_storage->Po->data[0];

	math_t **V = RNN_storage->V->data;
	math_t *Bpo = RNN_storage->Bpo->data[0];

	clear_2d(Z_, t_dim, h_dim);
	clear_2d(I_, t_dim, h_dim);
	clear_2d(F_, t_dim, h_dim);
	clear_2d(O_, t_dim, h_dim);
	clear_2d(P_O, t_dim, o_dim);

	int h, i, o, r, t;

	// For t = 0
	for (h = 0; h < h_dim; ++h) {
		/* Block input / Input gate / Forget gate */
		for (i = 0; i < i_dim; ++i) {
			Z_[0][h] += Wz[h][i] * X[0][i];
			I_[0][h] += Wi[h][i] * X[0][i];
			F_[0][h] += Wf[h][i] * X[0][i];
		}

		Z_[0][h] += Bz[h];
		I_[0][h] += Bi[h];
		F_[0][h] += Bf[h];

		Z[0][h] = gate_squash_func(Z_[0][h]);
		I[0][h] = cell_state_squash_func(I_[0][h]);
		F[0][h] = cell_state_squash_func(F_[0][h]);

		/* Cell state */
		C[0][h] = Z[0][h] * I[0][h];

		/* Output gate */
		for (i = 0; i < i_dim; ++i) {
			O_[0][h] += Wo[h][i] * X[0][i];
		}
		O_[0][h] += Bo[h];
		O[0][h] = cell_state_squash_func(O_[0][h]);

		/* Block output */
		Y[0][h] = cell_output_squash_func(C[0][h]) * O[0][h];
	}

	// For t = 1 ... t_dim
	for (t = 1; t < t_dim; ++t) {
		for (h = 0; h < h_dim; ++h) {
			/* Block input / Input gate / Forget gate */
			for (i = 0; i < i_dim; ++i) {
				Z_[t][h] += Wz[h][i] * X[t][i];
				I_[t][h] += Wi[h][i] * X[t][i];
				F_[t][h] += Wf[h][i] * X[t][i];
			}
			for (r = 0; r < h_dim; ++r) {
				Z_[t][h] += Rz[h][r] * Y[t - 1][r];
				I_[t][h] += Ri[h][r] * Y[t - 1][r];
				F_[t][h] += Rf[h][r] * Y[t - 1][r];
			}

			Z_[t][h] += Bz[h];
			I_[t][h] += Pi[h] * C[t - 1][h] + Bi[h];
			F_[t][h] += Pf[h] * C[t - 1][h] + Bf[h];

			Z[t][h] = gate_squash_func(Z_[t][h]);
			I[t][h] = cell_state_squash_func(I_[t][h]);
			F[t][h] = cell_state_squash_func(F_[t][h]);

			/* Cell state */
			C[t][h] = Z[t][h] * I[t][h] + C[t - 1][h] * F[t][h];

			/* Output gate */
			for (i = 0; i < i_dim; ++i) {
				O_[t][h] += Wo[h][i] * X[t][i];
			}
			for (r = 0; r < h_dim; ++r) {
				O_[t][h] += Ro[h][r] * Y[t - 1][r];
			}
			O_[t][h] += Po[h] * C[t][h] + Bo[h];
			O[t][h] = cell_state_squash_func(O_[t][h]);

			/* Block output */
			Y[t][h] = cell_output_squash_func(C[t][h]) * O[t][h];
		}
	}

	/* Network output */
	for (t = 0; t < t_dim; ++t) {
		for (o = 0; o < o_dim; ++o) {
			for (r = 0; r < h_dim; ++r) {
				P_O[t][o] += V[o][r] * Y[t][r];
			}
			P_O[t][o] += Bpo[o];
		}
	}
}

void RNN_BPTT(
    RNN_t * RNN_storage,
    Matrix_t *input_matrix,				// TxI
    Matrix_t *predicted_output_matrix,	// TxO
    Matrix_t *expected_output_matrix	// TxO
) {
	int i_dim = RNN_storage->i_dim;
	int o_dim = RNN_storage->o_dim;
	int t_dim = input_matrix->m;
	int h_dim = RNN_storage->h_dim;

	math_t **X = input_matrix->data;
	math_t **P_O = predicted_output_matrix->data;
	math_t **E_O = expected_output_matrix->data;

	math_t **Z_ = RNN_storage->Z_->data; 	math_t **Z = RNN_storage->Z->data;
	math_t **I_ = RNN_storage->I_->data; 	math_t **I = RNN_storage->I->data;
	math_t **F_ = RNN_storage->F_->data; 	math_t **F = RNN_storage->F->data;
	math_t **O_ = RNN_storage->O_->data; 	math_t **O = RNN_storage->O->data;
	math_t **C = RNN_storage->C->data;
	math_t **Y = RNN_storage->Y->data;

	math_t **Wz = RNN_storage->Wz->data; math_t **Rz = RNN_storage->Rz->data;
	math_t **Wi = RNN_storage->Wi->data; math_t **Ri = RNN_storage->Ri->data;
	math_t **Wf = RNN_storage->Wf->data; math_t **Rf = RNN_storage->Rf->data;
	math_t **Wo = RNN_storage->Wo->data; math_t **Ro = RNN_storage->Ro->data;

	math_t **dWz = RNN_storage->dWz->data; math_t **dRz = RNN_storage->dRz->data;
	math_t **dWi = RNN_storage->dWi->data; math_t **dRi = RNN_storage->dRi->data;
	math_t **dWf = RNN_storage->dWf->data; math_t **dRf = RNN_storage->dRf->data;
	math_t **dWo = RNN_storage->dWo->data; math_t **dRo = RNN_storage->dRo->data;

	math_t *Bz = RNN_storage->Bz->data[0];
	math_t *Bi = RNN_storage->Bi->data[0]; math_t *Pi = RNN_storage->Pi->data[0];
	math_t *Bf = RNN_storage->Bf->data[0]; math_t *Pf = RNN_storage->Pf->data[0];
	math_t *Bo = RNN_storage->Bo->data[0]; math_t *Po = RNN_storage->Po->data[0];

	math_t *dBz = RNN_storage->dBz->data[0];
	math_t *dBi = RNN_storage->dBi->data[0]; math_t *dPi = RNN_storage->dPi->data[0];
	math_t *dBf = RNN_storage->dBf->data[0]; math_t *dPf = RNN_storage->dPf->data[0];
	math_t *dBo = RNN_storage->dBo->data[0]; math_t *dPo = RNN_storage->dPo->data[0];

	math_t **V = RNN_storage->V->data; 
	math_t **dV = RNN_storage->dV->data;
	math_t *Bpo = RNN_storage->Bpo->data[0]; 
	math_t *dBpo = RNN_storage->dBpo->data[0];

	int bptt_truncate_len = RNN_storage->bptt_truncate_len;

	clear_2d(dRz, h_dim, h_dim);
	clear_2d(dRi, h_dim, h_dim); clear_1d(dPi, h_dim);
	clear_2d(dRf, h_dim, h_dim); clear_1d(dPf, h_dim);
	clear_2d(dRo, h_dim, h_dim); clear_1d(dPo, h_dim);

	math_t *dP_O = (math_t *) malloc(o_dim * sizeof(math_t));
	math_t *dY = (math_t *) malloc(h_dim * sizeof(math_t));	clear_1d(dY, h_dim);
	math_t *dO = (math_t *) malloc(h_dim * sizeof(math_t)); clear_1d(dO, h_dim);
	math_t *dC = (math_t *) malloc(h_dim * sizeof(math_t)); clear_1d(dC, h_dim);
	math_t *dF = (math_t *) malloc(h_dim * sizeof(math_t)); clear_1d(dF, h_dim);
	math_t *dI = (math_t *) malloc(h_dim * sizeof(math_t)); clear_1d(dI, h_dim);
	math_t *dZ = (math_t *) malloc(h_dim * sizeof(math_t)); clear_1d(dZ, h_dim);


	int i, o, h;

	/* For t = t_dim - 1 */	
	for (o = 0; o < o_dim; ++o) {
		dP_O[o] = 2 * (P_O[t_dim - 1][o] - E_O[t_dim - 1][o]);
	}
	for (h = 0; h < h_dim; ++h) {
		for (o = 0; o < o_dim; ++o) {
			dY[h] += V[o][h] * dP_O[o];
		}

		dO[h] = 
			dY[h] * 
			cell_output_squash_func(C[t_dim - 1][h]) * 
			gate_squash_derivative(O_[t_dim - 1][h]);
		dC[h] = 
			dY[h] * 
			O[t_dim - 1][h] * 
			cell_output_squash_derivative(C[t_dim - 1][h])
			+
			Po[h] * dO[h];
		dF[h] = 
			dC[h] * 
			C[t_dim - 1 - 1][h] * 
			gate_squash_derivative(F_[t_dim - 1][h]);
		dI[h] = 
			dC[h] * 
			Z[t_dim - 1][h] *
			gate_squash_derivative(I_[t_dim - 1][h]);
		dZ[h] =
			dC[h] * 
			I[t_dim - 1][h] *
			gate_squash_derivative(Z_[t_dim - 1][h]);						
	}

	for (o = 0; o < o_dim; ++o) {
		for (h = 0; h < h_dim; ++h) {
			dV[o][h] = Y[t_dim - 1][h] * dP_O[o];
		}
		dBpo[o] = dP_O[o];
	}
	for (i = 0; i < i_dim; ++i) {
		for (h = 0; h < h_dim; ++h) {
			dWz[i][h] = X[t_dim - 1][i] * dZ[h];
			dWi[i][h] = X[t_dim - 1][i] * dI[h];
			dWf[i][h] = X[t_dim - 1][i] * dF[h];
			dWo[i][h] = X[t_dim - 1][i] * dO[h];
		}
	}
	for (h = 0; h < h_dim; ++h) {
		dBz[h] = dZ[h];
		dBi[h] = dI[h];
		dBf[h] = dF[h];
		dBo[h] = dO[h];
	}
	
	/* For t = t_dim - 2 ... 0 */
	for (t = t_dim - 2; t >= 0; --t) {

	}
}

void RNN_SGD(
    RNN_t * RNN_storage,
    Matrix_t *input_matrix,
    Matrix_t *expected_output_matrix,
    Matrix_t *predicted_output_matrix,
    math_t learning_rate
) {

	// math_t **Ui = RNN_storage->Ui->data;   // IxH
	// math_t **Wi = RNN_storage->Wi->data;   // HxH
	// math_t **dUi = RNN_storage->Ui->data;   // IxH
	// math_t **dWi = RNN_storage->Wi->data;   // HxH

	// math_t **Uf = RNN_storage->Uf->data;   // IxH
	// math_t **Wf = RNN_storage->Wf->data;   // HxH
	// math_t **dUf = RNN_storage->Uf->data;   // IxH
	// math_t **dWf = RNN_storage->Wf->data;   // HxH

	// math_t **Uo = RNN_storage->Uo->data;   // IxH
	// math_t **Wo = RNN_storage->Wo->data;   // HxH
	// math_t **dUo = RNN_storage->Uo->data;   // IxH
	// math_t **dWo = RNN_storage->Wo->data;   // HxH

	// math_t **Us = RNN_storage->Ug->data;   // IxH
	// math_t **Ws = RNN_storage->Wg->data;   // HxH
	// math_t **dUs = RNN_storage->Ug->data;   // IxH
	// math_t **dWs = RNN_storage->Wg->data;   // HxH

	// int i_dim = RNN_storage->i_dim;
	// int o_dim = RNN_storage->o_dim;
	// int h_dim = RNN_storage->h_dim;

	RNN_forward_propagation(
	    RNN_storage,
	    input_matrix,
	    predicted_output_matrix
	);

	RNN_BPTT(
	    RNN_storage,
	    input_matrix,				// TxI
	    predicted_output_matrix,	// TxO
	    expected_output_matrix		// TxO
	);

	// int m, n;

	// // Update U
	// for (m = 0; m < i_dim; ++m)
	// 	for (n = 0; n < h_dim; ++n)
	// 		Ui[m][n] -= learning_rate * dLdU[m][n];

	// // Update V
	// for (m = 0; m < h_dim; ++m)
	// 	for (n = 0; n < o_dim; ++n)
	// 		V[m][n] -= learning_rate * dLdV[m][n];

	// // Update W
	// for (m = 0; m < h_dim; ++m)
	// 	for (n = 0; n < h_dim; ++n)
	// 		W[m][n] -= learning_rate * dLdW[m][n];
}

void RNN_train(
    RNN_t * RNN_storage,
    DataSet_t *train_set,
    Matrix_t *predicted_output_matrix,
    math_t initial_learning_rate,
    int max_epoch,
    int print_loss_interval
) {
	int num_train = train_set->num_matrix;

	int i, e, t;
	math_t current_total_loss, last_total_loss = 987654321;

	Matrix_t *input_matrix, *expected_output_matrix;
	math_t learning_rate = initial_learning_rate;

	for (e = 0; e < max_epoch; ++e) {
		if (e > 0 && e % print_loss_interval == 0) {

			current_total_loss = 0.0;
			for (i = 0; i < num_train; ++i) {
				input_matrix = train_set->input_matrix_list[i];
				expected_output_matrix = train_set->output_matrix_list[i];
				RNN_Predict(
				    RNN_storage,
				    input_matrix,
				    predicted_output_matrix
				);

				current_total_loss += RNN_loss_calculation(
				                          RNN_storage,
				                          predicted_output_matrix,
				                          expected_output_matrix);
			}

			// Adjust learning rate if the loss increases
			if (last_total_loss < current_total_loss) {
				if (learning_rate / 2 > 1e-5)
					learning_rate /= 2;
			} else if (learning_rate * 1.1 < initial_learning_rate) {
				learning_rate *= 1.1;
			}

			last_total_loss = current_total_loss;

			int old_bptt_truncate_len = RNN_storage->bptt_truncate_len;
			RNN_storage->bptt_truncate_len = 10000;
			int gradient_check_result =
			    RNN_Gradient_check(
			        RNN_storage,
			        train_set,
			        predicted_output_matrix,
			        1e-3,
			        1e-2,
			        0
			    );
			RNN_storage->bptt_truncate_len = old_bptt_truncate_len;
			printf("average loss at epoch: %10d = %10.10lf LR: %lf\n",
			       e, current_total_loss / num_train, learning_rate);

			// Terminate the training process if the gradient check did not pass
			if (gradient_check_result != 0) {
				return;
			}
		}

		for (t = 0; t < num_train; ++t) {
			input_matrix = train_set->input_matrix_list[t];
			expected_output_matrix = train_set->output_matrix_list[t];

			RNN_SGD(
			    RNN_storage,
			    input_matrix,
			    expected_output_matrix,
			    predicted_output_matrix,
			    learning_rate
			);
		}
	}
}

// Cross entropy loss
math_t RNN_loss_calculation(
    RNN_t * RNN_storage,
    Matrix_t *predicted_output_matrix,	// TxO
    Matrix_t *expected_output_matrix	// TxO
) {
	math_t total_loss = 0.0, log_term = 0.0, delta;

	int t_dim = predicted_output_matrix->m;
	int o_dim = RNN_storage->o_dim;

	int t, o;
	for (t = 0; t < t_dim; ++t) {
		// expected_output_matrix is an one-hot vector
		log_term = 0;
		for (o = 0; o < o_dim; ++o) {
			delta =
			    expected_output_matrix->data[t][o] -
			    predicted_output_matrix->data[t][o];
			log_term += delta * delta;
		}
		total_loss += log_term;
	}

	return total_loss;
}

int RNN_Gradient_check(
    RNN_t * RNN_storage,
    DataSet_t *train_set,
    Matrix_t *predicted_output_matrix,
    math_t h,
    math_t error_threshold,
    int index_to_check
) {
	// Matrix_t *input_matrix, *expected_output_matrix;
	// input_matrix = train_set->input_matrix_list[index_to_check];
	// expected_output_matrix = train_set->output_matrix_list[index_to_check];

	// math_t **U = RNN_storage->input_weight_matrix->data;	// IxH
	// math_t **V = RNN_storage->output_weight_matrix->data;	// HxO
	// math_t **W = RNN_storage->internal_weight_matrix->data;	// HxH

	// math_t **dLdU = RNN_storage->input_weight_gradient->data;	// IxH
	// math_t **dLdV = RNN_storage->output_weight_gradient->data;	// HxO
	// math_t **dLdW = RNN_storage->internel_weight_gradient->data; // HxH

	// RNN_forward_propagation(
	//     RNN_storage,
	//     input_matrix,
	//     predicted_output_matrix
	// );

	// RNN_BPTT(
	//     RNN_storage,
	//     input_matrix,				// TxI
	//     predicted_output_matrix,	// TxO
	//     expected_output_matrix		// TxO
	// );

	// int i, m, n;
	// math_t old_model_param;
	// math_t total_loss_plus, total_loss_minus;
	// math_t estimated_gradient, calculated_gradient;
	// math_t relative_gradient_error;

	// Matrix_t *testing_model_list[] = {
	// 	RNN_storage->input_weight_gradient,		// U
	// 	RNN_storage->output_weight_gradient,		// V
	// 	RNN_storage->internel_weight_gradient	// W
	// };
	// math_t **UVW[] = {U, V, W};
	// math_t **dLdUVW[] = {dLdU, dLdV, dLdW};

	// Matrix_t *testing_model;
	// math_t **testing_matrix;
	// math_t **testing_gradient_matrix;

	// for (i = 0; i < 2; ++i) {
	// 	testing_model = testing_model_list[i];
	// 	testing_matrix = UVW[i];
	// 	testing_gradient_matrix = dLdUVW[i];

	// 	for (m = 0; m < testing_model->m; ++m) {
	// 		for (n = 0; n < testing_model->n; ++n) {
	// 			old_model_param = testing_matrix[m][n];
	// 			testing_matrix[m][n] = old_model_param + h;

	// 			RNN_forward_propagation(
	// 			    RNN_storage,
	// 			    input_matrix,
	// 			    predicted_output_matrix
	// 			);
	// 			total_loss_plus = RNN_loss_calculation(
	// 			                      RNN_storage,
	// 			                      predicted_output_matrix,
	// 			                      expected_output_matrix);

	// 			testing_matrix[m][n] = old_model_param - h;
	// 			RNN_forward_propagation(
	// 			    RNN_storage,
	// 			    input_matrix,
	// 			    predicted_output_matrix
	// 			);
	// 			total_loss_minus = RNN_loss_calculation(
	// 			                       RNN_storage,
	// 			                       predicted_output_matrix,
	// 			                       expected_output_matrix);
	// 			testing_matrix[m][n] = old_model_param;

	// 			estimated_gradient =
	// 			    (total_loss_plus - total_loss_minus) /
	// 			    (2.0 * h);
	// 			calculated_gradient = testing_gradient_matrix[m][n];
	// 			relative_gradient_error =
	// 			    fabs(estimated_gradient - calculated_gradient) /
	// 			    (fabs(estimated_gradient) + fabs(calculated_gradient));

	// 			if (relative_gradient_error > error_threshold) {
	// 				printf("-------------Gradient check error\n");
	// 				printf("For matrix %d [%d][%d]\n", i, m, n);
	// 				printf("+h loss: %lf\n", total_loss_plus);
	// 				printf("-h loss: %lf\n", total_loss_minus);
	// 				printf("estimated_gradient: %lf\n", estimated_gradient);
	// 				printf("calculated_gradient: %lf\n", calculated_gradient);
	// 				printf("relative_gradient_error: %lf\n", relative_gradient_error);
	// 				printf("---------------------------------------\n");
	// 				return 1;
	// 			}
	// 		}
	// 	}
	// }
	return 0;
}

void RNN_Predict(
    RNN_t * RNN_storage,
    Matrix_t *input_matrix,
    Matrix_t *predicted_output_matrix
) {
	RNN_forward_propagation(
	    RNN_storage,
	    input_matrix,
	    predicted_output_matrix
	);
}

math_t gate_squash_func(math_t value) {
	return sigmoid(value);
}
math_t gate_squash_derivative(math_t value) {
	math_t temp_v =  sigmoid(value);
	return temp_v * (1 - temp_v);
}

math_t cell_state_squash_func(math_t value) {
	return tanh(value);
}
math_t cell_state_squash_derivative(math_t value) {
	math_t temp_v =  tanh(value);
	return 1 - temp_v * temp_v;
}

math_t cell_output_squash_func(math_t value) {
	return tanh(value);
}
math_t cell_output_squash_derivative(math_t value) {
	math_t temp_v =  tanh(value);
	return 1 - temp_v * temp_v;
}


math_t sigmoid(math_t value) {
	return 2.0 / (1 + exp(-2 * value)) - 1;
}
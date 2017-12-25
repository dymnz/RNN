#include "RNN.h"

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
    int bptt_truncate_len,
    unsigned int seed
) {
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
	RNN_storage->Wz = matrix_create(h_dim, i_dim);
	RNN_storage->dWz = matrix_create(h_dim, i_dim);
	RNN_storage->Wi = matrix_create(h_dim, i_dim);
	RNN_storage->dWi = matrix_create(h_dim, i_dim);
	RNN_storage->Wf = matrix_create(h_dim, i_dim);
	RNN_storage->dWf = matrix_create(h_dim, i_dim);
	RNN_storage->Wo = matrix_create(h_dim, i_dim);
	RNN_storage->dWo = matrix_create(h_dim, i_dim);
	matrix_random_with_seed(
	    RNN_storage->Wz, -sqrt(2 / (h_dim + i_dim)),
	    sqrt(2 / (h_dim + i_dim)), &seed);
	matrix_random_with_seed(
	    RNN_storage->Wi, -sqrt(2 / (h_dim + i_dim)),
	    sqrt(2 / (h_dim + i_dim)), &seed);
	matrix_random_with_seed(
	    RNN_storage->Wf, -sqrt(2 / (h_dim + i_dim)),
	    sqrt(2 / (h_dim + i_dim)), &seed);
	matrix_random_with_seed(
	    RNN_storage->Wo, -sqrt(2 / (h_dim + i_dim)),
	    sqrt(2 / (h_dim + i_dim)), &seed);

	RNN_storage->Rz = matrix_create(h_dim, h_dim);
	RNN_storage->dRz = matrix_create(h_dim, h_dim);
	RNN_storage->Ri = matrix_create(h_dim, h_dim);
	RNN_storage->dRi = matrix_create(h_dim, h_dim);
	RNN_storage->Rf = matrix_create(h_dim, h_dim);
	RNN_storage->dRf = matrix_create(h_dim, h_dim);
	RNN_storage->Ro = matrix_create(h_dim, h_dim);
	RNN_storage->dRo = matrix_create(h_dim, h_dim);
	matrix_random_with_seed(
	    RNN_storage->Rz, -sqrt(1 / h_dim),
	    sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(
	    RNN_storage->Ri, -sqrt(1 / h_dim),
	    sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(
	    RNN_storage->Rf, -sqrt(1 / h_dim),
	    sqrt(1 / h_dim), &seed);
	matrix_random_with_seed(
	    RNN_storage->Ro, -sqrt(1 / h_dim),
	    sqrt(1 / h_dim), &seed);

	RNN_storage->Pi = matrix_create(1, h_dim);
	RNN_storage->dPi = matrix_create(1, h_dim);
	RNN_storage->Pf = matrix_create(1, h_dim);
	RNN_storage->dPf = matrix_create(1, h_dim);
	RNN_storage->Po = matrix_create(1, h_dim);
	RNN_storage->dPo = matrix_create(1, h_dim);
	matrix_random_with_seed(
	    RNN_storage->Pi, -sqrt(2 / (h_dim + 1)),
	    sqrt(2 / (h_dim + 1)), &seed);
	matrix_random_with_seed(
	    RNN_storage->Pf, -sqrt(2 / (h_dim + 1)),
	    sqrt(2 / (h_dim + 1)), &seed);
	matrix_random_with_seed(
	    RNN_storage->Po, -sqrt(2 / (h_dim + 1)),
	    sqrt(2 / (h_dim + 1)), &seed);

	RNN_storage->Bz = matrix_create(1, h_dim);
	RNN_storage->dBz = matrix_create(1, h_dim);
	RNN_storage->Bi = matrix_create(1, h_dim);
	RNN_storage->dBi = matrix_create(1, h_dim);
	RNN_storage->Bf = matrix_create(1, h_dim);
	RNN_storage->dBf = matrix_create(1, h_dim);
	RNN_storage->Bo = matrix_create(1, h_dim);
	RNN_storage->dBo = matrix_create(1, h_dim);
	matrix_random_with_seed(
	    RNN_storage->Bz, -sqrt(2 / (h_dim + 1)),
	    sqrt(2 / (h_dim + 1)), &seed);
	matrix_random_with_seed(
	    RNN_storage->Bi, -sqrt(2 / (h_dim + 1)),
	    sqrt(2 / (h_dim + 1)), &seed);
	matrix_random_with_seed(
	    RNN_storage->Bf, -sqrt(2 / (h_dim + 1)),
	    sqrt(2 / (h_dim + 1)), &seed);
	matrix_random_with_seed(
	    RNN_storage->Bo, -sqrt(2 / (h_dim + 1)),
	    sqrt(2 / (h_dim + 1)), &seed);

	/* Output model */
	RNN_storage->V = matrix_create(o_dim, h_dim);
	RNN_storage->dV = matrix_create(o_dim, h_dim);
	RNN_storage->Bpo = matrix_create(1, o_dim);
	RNN_storage->dBpo = matrix_create(1, o_dim);
	matrix_random_with_seed(
	    RNN_storage->V, -sqrt(2 / (h_dim + o_dim)),
	    sqrt(2 / (h_dim + o_dim)), &seed);
	matrix_random_with_seed(
	    RNN_storage->Bpo, -sqrt(1 / (h_dim + 1)),
	    sqrt(1 / (h_dim + 1)), &seed);
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

	matrix_resize(RNN_storage->Z_, t_dim, h_dim);
	matrix_resize(RNN_storage->Z, t_dim, h_dim);
	matrix_resize(RNN_storage->I_, t_dim, h_dim);
	matrix_resize(RNN_storage->I, t_dim, h_dim);
	matrix_resize(RNN_storage->F_, t_dim, h_dim);
	matrix_resize(RNN_storage->F, t_dim, h_dim);
	matrix_resize(RNN_storage->O_, t_dim, h_dim);
	matrix_resize(RNN_storage->O, t_dim, h_dim);
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
		O_[0][h] += Po[h] * C[0][h] + Bo[h];
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

	math_t **Rz = RNN_storage->Rz->data;
	math_t **Ri = RNN_storage->Ri->data;
	math_t **Rf = RNN_storage->Rf->data;
	math_t **Ro = RNN_storage->Ro->data;

	math_t **dWz = RNN_storage->dWz->data; math_t **dRz = RNN_storage->dRz->data;
	math_t **dWi = RNN_storage->dWi->data; math_t **dRi = RNN_storage->dRi->data;
	math_t **dWf = RNN_storage->dWf->data; math_t **dRf = RNN_storage->dRf->data;
	math_t **dWo = RNN_storage->dWo->data; math_t **dRo = RNN_storage->dRo->data;

	math_t *Pi = RNN_storage->Pi->data[0];
	math_t *Pf = RNN_storage->Pf->data[0];
	math_t *Po = RNN_storage->Po->data[0];

	math_t *dBz = RNN_storage->dBz->data[0];
	math_t *dBi = RNN_storage->dBi->data[0]; math_t *dPi = RNN_storage->dPi->data[0];
	math_t *dBf = RNN_storage->dBf->data[0]; math_t *dPf = RNN_storage->dPf->data[0];
	math_t *dBo = RNN_storage->dBo->data[0]; math_t *dPo = RNN_storage->dPo->data[0];

	math_t **V = RNN_storage->V->data;
	math_t **dV = RNN_storage->dV->data;
	math_t *dBpo = RNN_storage->dBpo->data[0];

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

	int i, o, h, t, r;

	// printf("V:\n");
	// matrix_print(RNN_storage->V);
	// printf("O:\n");
	// matrix_print(RNN_storage->O);
	// printf("Z:\n");
	// matrix_print(RNN_storage->Z);
	// printf("Z_:\n");
	// matrix_print(RNN_storage->Z_);
	// printf("F:\n");
	// matrix_print(RNN_storage->F);
	// printf("F_:\n");
	// matrix_print(RNN_storage->F_);
	// printf("I:\n");
	// matrix_print(RNN_storage->I);
	// printf("I_:\n");
	// matrix_print(RNN_storage->I_);
	// printf("Po:\n");
	// matrix_print(RNN_storage->Po);
	// printf("Pi:\n");
	// matrix_print(RNN_storage->Pi);
	// printf("Pf:\n");
	// matrix_print(RNN_storage->Pf);
	// printf("C:\n");
	// matrix_print(RNN_storage->C);
	// printf("_O:\n");
	// matrix_print(RNN_storage->O_);

	// printf("Rz:\n");
	// matrix_print(RNN_storage->Rz);
	// printf("Ri:\n");
	// matrix_print(RNN_storage->Ri);
	// printf("Rf:\n");
	// matrix_print(RNN_storage->Rf);
	// printf("Ro:\n");
	// matrix_print(RNN_storage->Ro);
	/* For t = t_dim - 1 */
	for (o = 0; o < o_dim; ++o) {
		dP_O[o] = 2 * (P_O[t_dim - 1][o] - E_O[t_dim - 1][o]);
	}
	for (h = 0; h < h_dim; ++h) {
		for (o = 0; o < o_dim; ++o) {
			dY[h] += V[o][h] * dP_O[o];
		}
	}
	for (h = 0; h < h_dim; ++h) {
		dO[h] =
		    dY[h] *
		    cell_output_squash_func(C[t_dim - 1][h]) *
		    cell_state_squash_derivative(O_[t_dim - 1][h]);
		dC[h] =
		    dY[h] *
		    O[t_dim - 1][h] *
		    cell_output_squash_derivative(C[t_dim - 1][h])
		    +
		    Po[h] * dO[h];
		dF[h] =
		    dC[h] *
		    C[t_dim - 1 - 1][h] *
		    cell_state_squash_derivative(F_[t_dim - 1][h]);
		dI[h] =
		    dC[h] *
		    Z[t_dim - 1][h] *
		    cell_state_squash_derivative(I_[t_dim - 1][h]);
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

	for (h = 0; h < h_dim; ++h) {
		for (i = 0; i < i_dim; ++i) {
			dWz[h][i] = dZ[h] * X[t_dim - 1][i];
			dWi[h][i] = dI[h] * X[t_dim - 1][i];
			dWf[h][i] = dF[h] * X[t_dim - 1][i];
			dWo[h][i] = dO[h] * X[t_dim - 1][i];
		}
	}
	for (h = 0; h < h_dim; ++h) {
		for (r = 0; r < h_dim; ++r) {
			dRz[h][r] = dZ[h] * Y[t_dim - 1 - 1][r];
			dRi[h][r] = dI[h] * Y[t_dim - 1 - 1][r];
			dRf[h][r] = dF[h] * Y[t_dim - 1 - 1][r];
			dRo[h][r] = dO[h] * Y[t_dim - 1 - 1][r];
		}
		dBz[h] = dZ[h];
		dBi[h] = dI[h];
		dBf[h] = dF[h];
		dBo[h] = dO[h];
		dPi[h] = C[t_dim - 1 - 1][h] * dI[h];
		dPf[h] = C[t_dim - 1 - 1][h] * dF[h];
		dPo[h] = C[t_dim - 1][h] * dO[h];
	}

	// printf("-------  t = %3d  -------\n", t_dim - 1);
	// printf("P_O:\n");
	// matrix_print(predicted_output_matrix);
	// printf("E_O:\n");
	// matrix_print(expected_output_matrix);
	// printf("dP_O:\n");
	// print_1d(dP_O, o_dim);
	// printf("dY:\n");
	// print_1d(dY, h_dim);
	// printf("dO:\n");
	// print_1d(dO, h_dim);
	// printf("dC:\n");
	// print_1d(dC, h_dim);
	// printf("dF:\n");
	// print_1d(dF, h_dim);
	// printf("dI:\n");
	// print_1d(dI, h_dim);
	// printf("dZ:\n");
	// print_1d(dZ, h_dim);
	// printf("------------\n");
	// sleep(1);
	// printf("----------\n");

	/* For t = t_dim - 2 ... 1 */
	for (t = t_dim - 2; t >= 1; --t) {
		clear_1d(dY, h_dim);
		for (o = 0; o < o_dim; ++o) {
			dP_O[o] = 2 * (P_O[t][o] - E_O[t][o]);
		}
		for (h = 0; h < h_dim; ++h) {
			for (o = 0; o < o_dim; ++o) {
				dY[h] += V[o][h] * dP_O[o];
			}

			for (r = 0; r < h_dim; ++r) {
				dY[h] +=
				    Rz[r][h] * dZ[r]
				    +
				    Ri[r][h] * dI[r]
				    +
				    Rf[r][h] * dF[r]
				    +
				    Ro[r][h] * dO[r];
			}
		}
		for (h = 0; h < h_dim; ++h) {
			dO[h] =
			    dY[h] *
			    cell_output_squash_func(C[t][h]) *
			    cell_state_squash_derivative(O_[t][h]);

			dC[h] =
			    dY[h] *
			    O[t][h] *
			    cell_output_squash_derivative(C[t][h])
			    +
			    Po[h] * dO[h]
			    +
			    Pi[h] * dI[h]
			    +
			    Pf[h] * dF[h]
			    +
			    dC[h] * F[t + 1][h];

			dF[h] =
			    dC[h] *
			    C[t - 1][h] *
			    cell_state_squash_derivative(F_[t][h]);
			dI[h] =
			    dC[h] *
			    Z[t][h] *
			    cell_state_squash_derivative(I_[t][h]);
			dZ[h] =
			    dC[h] *
			    I[t][h] *
			    gate_squash_derivative(Z_[t][h]);
		}
		for (o = 0; o < o_dim; ++o) {
			for (h = 0; h < h_dim; ++h) {
				dV[o][h] += Y[t][h] * dP_O[o];
			}
			dBpo[o] += dP_O[o];
		}

		for (h = 0; h < h_dim; ++h) {
			for (i = 0; i < i_dim; ++i) {
				dWz[h][i] += dZ[h] * X[t][i];
				dWi[h][i] += dI[h] * X[t][i];
				dWf[h][i] += dF[h] * X[t][i];
				dWo[h][i] += dO[h] * X[t][i];
			}
		}
		for (h = 0; h < h_dim; ++h) {
			for (r = 0; r < h_dim; ++r) {
				dRz[h][r] += dZ[h] * Y[t - 1][r];
				dRi[h][r] += dI[h] * Y[t - 1][r];
				dRf[h][r] += dF[h] * Y[t - 1][r];
				dRo[h][r] += dO[h] * Y[t - 1][r];
			}
			dBz[h] += dZ[h];
			dBi[h] += dI[h];
			dBf[h] += dF[h];
			dBo[h] += dO[h];
			dPi[h] += C[t - 1][h] * dI[h];
			dPf[h] += C[t - 1][h] * dF[h];
			dPo[h] += C[t][h] * dO[h];
		}

		// printf("-------  t = %3d  -------\n", t);
		// printf("P_O:\n");
		// matrix_print(predicted_output_matrix);
		// printf("E_O:\n");
		// matrix_print(expected_output_matrix);
		// printf("dP_O:\n");
		// print_1d(dP_O, o_dim);
		// printf("dY:\n");
		// print_1d(dY, h_dim);
		// printf("dO:\n");
		// print_1d(dO, h_dim);
		// printf("dC:\n");
		// print_1d(dC, h_dim);
		// printf("dF:\n");
		// print_1d(dF, h_dim);
		// printf("dI:\n");
		// print_1d(dI, h_dim);
		// printf("dZ:\n");
		// print_1d(dZ, h_dim);
		// printf("------------\n");
		// sleep(1);
		// printf("----------\n");
	}

	/* For t = 0 */
	clear_1d(dY, h_dim);
	for (o = 0; o < o_dim; ++o) {
		dP_O[o] = 2 * (P_O[0][o] - E_O[0][o]);
	}
	for (h = 0; h < h_dim; ++h) {
		for (o = 0; o < o_dim; ++o) {
			dY[h] += dP_O[o] * V[o][h];
		}
		for (r = 0; r < h_dim; ++r) {
			dY[h] +=
			    Rz[r][h] * dZ[r]
			    +
			    Ri[r][h] * dI[r]
			    +
			    Rf[r][h] * dF[r]
			    +
			    Ro[r][h] * dO[r];
		}
	}
	for (h = 0; h < h_dim; ++h) {
		dO[h] =
		    dY[h] *
		    cell_output_squash_func(C[0][h]) *
		    cell_state_squash_derivative(O_[0][h]);

		dC[h] =
		    dY[h] *
		    O[0][h] *
		    cell_output_squash_derivative(C[0][h])
		    +
		    Po[h] * dO[h]
		    +
		    Pi[h] * dI[h]
		    +
		    Pf[h] * dF[h]
		    +
		    dC[h] * F[1][h];
		dF[h] = 0.0;
		dI[h] =
		    dC[h] *
		    Z[0][h] *
		    cell_state_squash_derivative(I_[0][h]);

		dZ[h] =
		    dC[h] *
		    I[0][h] *
		    gate_squash_derivative(Z_[0][h]);
	}
	for (o = 0; o < o_dim; ++o) {
		for (h = 0; h < h_dim; ++h) {
			dV[o][h] += Y[0][h] * dP_O[o];
		}
		dBpo[o] += dP_O[o];
	}
	for (h = 0; h < h_dim; ++h) {
		for (i = 0; i < i_dim; ++i) {
			dWz[h][i] += dZ[h] * X[0][i];
			dWi[h][i] += dI[h] * X[0][i];
			dWf[h][i] += dF[h] * X[0][i];
			dWo[h][i] += dO[h] * X[0][i];
		}
	}
	for (h = 0; h < h_dim; ++h) {
		dBz[h] += dZ[h];
		dBi[h] += dI[h];
		dBf[h] += dF[h];
		dBo[h] += dO[h];
		dPo[h] += C[0][h] * dO[h];
	}
	// printf("-------  t = %3d  -------\n", 0);
	// printf("P_O:\n");
	// matrix_print(predicted_output_matrix);
	// printf("E_O:\n");
	// matrix_print(expected_output_matrix);
	// printf("dP_O:\n");
	// print_1d(dP_O, o_dim);
	// printf("dY:\n");
	// print_1d(dY, h_dim);
	// printf("dO:\n");
	// print_1d(dO, h_dim);
	// printf("dC:\n");
	// print_1d(dC, h_dim);
	// printf("dF:\n");
	// print_1d(dF, h_dim);
	// printf("dI:\n");
	// print_1d(dI, h_dim);
	// printf("dZ:\n");
	// print_1d(dZ, h_dim);
	// printf("------------\n");
	// sleep(1);
	// printf("----------\n");

	free(dP_O);
	free(dY);
	free(dO);
	free(dC);
	free(dF);
	free(dI);
	free(dZ);
}

void RNN_SGD(
    RNN_t * RNN_storage,
    Matrix_t *input_matrix,
    Matrix_t *expected_output_matrix,
    Matrix_t *predicted_output_matrix,
    math_t learning_rate
) {
	int i_dim = RNN_storage->i_dim;
	int o_dim = RNN_storage->o_dim;
	int h_dim = RNN_storage->h_dim;

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

	int h, i, r, o;


	for (h = 0; h < h_dim; ++h) {
		for (i = 0; i < i_dim; ++i) {
			Wz[h][i] -= learning_rate * dWz[h][i];
			Wi[h][i] -= learning_rate * dWi[h][i];
			Wf[h][i] -= learning_rate * dWf[h][i];
			Wo[h][i] -= learning_rate * dWo[h][i];
		}
		for (r = 0; r < h_dim; ++r) {
			Rz[h][r] -= learning_rate * dRz[h][r];
			Ri[h][r] -= learning_rate * dRi[h][r];
			Rf[h][r] -= learning_rate * dRf[h][r];
			Ro[h][r] -= learning_rate * dRo[h][r];
		}
		Pi[h] -= learning_rate * dPi[h];
		Pf[h] -= learning_rate * dPf[h];
		Po[h] -= learning_rate * dPo[h];

		Bz[h] -= learning_rate * dBz[h];
		Bi[h] -= learning_rate * dBi[h];
		Bf[h] -= learning_rate * dBf[h];
		Bo[h] -= learning_rate * dBo[h];
	}
	for (o = 0; o < o_dim; ++o) {
		for (h = 0; h < h_dim; ++h) {
			V[o][h] -= learning_rate * dV[o][h];
		}
		Bpo[o] -= learning_rate * dBpo[o];
	}
}

void RNN_train(
    RNN_t * RNN_storage,
    DataSet_t *train_set,
    Matrix_t *predicted_output_matrix,
    math_t initial_learning_rate,
    int max_epoch,
    int print_loss_interval,
    int gradient_check_interval
) {
	int num_train = train_set->num_matrix;

	int i, e, t;
	math_t current_total_loss, last_total_loss = 987654321;

	Matrix_t *input_matrix, *expected_output_matrix;
	math_t learning_rate = initial_learning_rate;

	for (e = 0; e < max_epoch; ++e) {
		if (e % print_loss_interval == 0)
			printf("average loss at epoch: %10d = %10.10lf LR: %lf\n",
			       e, current_total_loss / num_train, learning_rate);
		if (e > 0 && e % gradient_check_interval == 0) {

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
				if (learning_rate / 2 > 1e-6)
					learning_rate /= 2;
				else
					return;
			} else if (learning_rate * 1.1 < 1 * initial_learning_rate) {
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
			        1e-5,
			        1e-2,
			        0
			    );
			RNN_storage->bptt_truncate_len = old_bptt_truncate_len;

			// Terminate the training process if the gradient check did not pass
			if (gradient_check_result != 0) {
				printf("Gradient check error at epoch: %10d\n", e);
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
	Matrix_t *input_matrix, *expected_output_matrix;
	input_matrix = train_set->input_matrix_list[index_to_check];
	expected_output_matrix = train_set->output_matrix_list[index_to_check];

	math_t **Wz = RNN_storage->Wz->data; math_t **Rz = RNN_storage->Rz->data;
	math_t **Wi = RNN_storage->Wi->data; math_t **Ri = RNN_storage->Ri->data;
	math_t **Wf = RNN_storage->Wf->data; math_t **Rf = RNN_storage->Rf->data;
	math_t **Wo = RNN_storage->Wo->data; math_t **Ro = RNN_storage->Ro->data;

	math_t **dWz = RNN_storage->dWz->data; math_t **dRz = RNN_storage->dRz->data;
	math_t **dWi = RNN_storage->dWi->data; math_t **dRi = RNN_storage->dRi->data;
	math_t **dWf = RNN_storage->dWf->data; math_t **dRf = RNN_storage->dRf->data;
	math_t **dWo = RNN_storage->dWo->data; math_t **dRo = RNN_storage->dRo->data;

	math_t **Bz = RNN_storage->Bz->data;
	math_t **Bi = RNN_storage->Bi->data; math_t **Pi = RNN_storage->Pi->data;
	math_t **Bf = RNN_storage->Bf->data; math_t **Pf = RNN_storage->Pf->data;
	math_t **Bo = RNN_storage->Bo->data; math_t **Po = RNN_storage->Po->data;

	math_t **dBz = RNN_storage->dBz->data;
	math_t **dBi = RNN_storage->dBi->data; math_t **dPi = RNN_storage->dPi->data;
	math_t **dBf = RNN_storage->dBf->data; math_t **dPf = RNN_storage->dPf->data;
	math_t **dBo = RNN_storage->dBo->data; math_t **dPo = RNN_storage->dPo->data;

	math_t **V = RNN_storage->V->data;
	math_t **dV = RNN_storage->dV->data;
	math_t **Bpo = RNN_storage->Bpo->data;
	math_t **dBpo = RNN_storage->dBpo->data;

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

	int i, m, n;
	math_t old_model_param;
	math_t total_loss_plus, total_loss_minus;
	math_t estimated_gradient, calculated_gradient;
	math_t relative_gradient_error;

	Matrix_t *testing_model_list[] = {
		RNN_storage->Wz, RNN_storage->Wi, RNN_storage->Wf, RNN_storage->Wo,
		RNN_storage->Rz, RNN_storage->Ri, RNN_storage->Rf, RNN_storage->Ro,
		RNN_storage->Pi, RNN_storage->Pf, RNN_storage->Po,
		RNN_storage->Bz, RNN_storage->Bi, RNN_storage->Bf, RNN_storage->Bo,
		RNN_storage->V,  RNN_storage->Bpo
	};
	math_t **UVW[] = {
		Wz, Wi, Wf, Wo,
		Rz, Ri, Rf, Ro,
		Pi, Pf, Po,
		Bz, Bi, Bf, Bo,
		V,  Bpo
	};
	math_t **dLdUVW[] = {
		dWz, dWi, dWf, dWo,
		dRz, dRi, dRf, dRo,
		dPi, dPf, dPo,
		dBz, dBi, dBf, dBo,
		dV,  dBpo
	};

	Matrix_t *testing_model;
	math_t **testing_matrix;
	math_t **testing_gradient_matrix;

	for (i = 0; i < 17; ++i) {
		testing_model = testing_model_list[i];
		testing_matrix = UVW[i];
		testing_gradient_matrix = dLdUVW[i];

		for (m = 0; m < testing_model->m; ++m) {
			for (n = 0; n < testing_model->n; ++n) {
				old_model_param = testing_matrix[m][n];
				testing_matrix[m][n] = old_model_param + h;

				RNN_forward_propagation(
				    RNN_storage,
				    input_matrix,
				    predicted_output_matrix
				);
				total_loss_plus = RNN_loss_calculation(
				                      RNN_storage,
				                      predicted_output_matrix,
				                      expected_output_matrix);

				testing_matrix[m][n] = old_model_param - h;
				RNN_forward_propagation(
				    RNN_storage,
				    input_matrix,
				    predicted_output_matrix
				);
				total_loss_minus = RNN_loss_calculation(
				                       RNN_storage,
				                       predicted_output_matrix,
				                       expected_output_matrix);
				testing_matrix[m][n] = old_model_param;

				estimated_gradient =
				    (total_loss_plus - total_loss_minus) /
				    (2.0 * h);
				calculated_gradient = testing_gradient_matrix[m][n];
				relative_gradient_error =
				    fabs(estimated_gradient - calculated_gradient) /
				    (fabs(estimated_gradient) + fabs(calculated_gradient));

				if (relative_gradient_error > error_threshold) {
					printf("-------------Gradient check error\n");
					printf("For matrix %d [%d][%d]\n", i, m, n);
					printf("+h loss: %lf\n", total_loss_plus);
					printf("-h loss: %lf\n", total_loss_minus);
					printf("estimated_gradient: %lf\n", estimated_gradient);
					printf("calculated_gradient: %lf\n", calculated_gradient);
					printf("relative_gradient_error: %lf\n", relative_gradient_error);
					printf("---------------------------------------\n");
					return 1;
				}
			}
		}
	}
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

math_t cell_state_squash_func(math_t value) {
	return sigmoid(value);
}
math_t cell_state_squash_derivative(math_t value) {
	math_t temp_v =  sigmoid(value);
	return temp_v * (1.0 - temp_v);
}

math_t gate_squash_func(math_t value) {
	return tanh(value);
}
math_t gate_squash_derivative(math_t value) {
	math_t temp_v =  tanh(value);
	return 1.0 - temp_v * temp_v;
}

math_t cell_output_squash_func(math_t value) {
	return tanh(value);
}
math_t cell_output_squash_derivative(math_t value) {
	math_t temp_v =  tanh(value);
	return 1.0 - temp_v * temp_v;
}


math_t sigmoid(math_t value) {
	//return 2.0 / (1 + exp(-2 * value)) - 1;
	return 1.0 / (1 + exp(-value));
}
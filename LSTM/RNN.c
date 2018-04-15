#include "RNN.h"

void TrainSet_init(DataSet_t *train_set, int num_matrix) {
	train_set->num_matrix = num_matrix;
	train_set->input_matrix_list =
	    (Matrix_t **) malloc(num_matrix * sizeof(Matrix_t *));
	train_set->output_matrix_list =
	    (Matrix_t **) malloc(num_matrix * sizeof(Matrix_t *));

	train_set->input_max_m = 0;
	train_set->input_n = 0;
	train_set->output_max_m = 0;
	train_set->output_n = 0;
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
    unsigned int seed
) {
	int i_dim = input_vector_len;
	int o_dim = output_vector_len;
	int h_dim = hidden_layer_vector_len;

	RNN_storage->i_dim = i_dim;
	RNN_storage->o_dim = o_dim;
	RNN_storage->h_dim = h_dim;
	RNN_storage->gamma = 0.95;

	srand(seed);

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
	matrix_random(
	    RNN_storage->Wz, -sqrt(2.0f / (h_dim + i_dim)),
	    sqrt(2.0f / (h_dim + i_dim)));
	matrix_random(
	    RNN_storage->Wi, -sqrt(2.0f / (h_dim + i_dim)),
	    sqrt(2.0f / (h_dim + i_dim)));
	matrix_random(
	    RNN_storage->Wf, -sqrt(2.0f / (h_dim + i_dim)),
	    sqrt(2.0f / (h_dim + i_dim)));
	matrix_random(
	    RNN_storage->Wo, -sqrt(2.0f / (h_dim + i_dim)),
	    sqrt(2.0f / (h_dim + i_dim)));

	RNN_storage->Rz = matrix_create(h_dim, h_dim);
	RNN_storage->dRz = matrix_create(h_dim, h_dim);
	RNN_storage->Ri = matrix_create(h_dim, h_dim);
	RNN_storage->dRi = matrix_create(h_dim, h_dim);
	RNN_storage->Rf = matrix_create(h_dim, h_dim);
	RNN_storage->dRf = matrix_create(h_dim, h_dim);
	RNN_storage->Ro = matrix_create(h_dim, h_dim);
	RNN_storage->dRo = matrix_create(h_dim, h_dim);
	matrix_random(
	    RNN_storage->Rz, -sqrt(1.0f / h_dim),
	    sqrt(1.0f / h_dim));
	matrix_random(
	    RNN_storage->Ri, -sqrt(1.0f / h_dim),
	    sqrt(1.0f / h_dim));
	matrix_random(
	    RNN_storage->Rf, -sqrt(1.0f / h_dim),
	    sqrt(1.0f / h_dim));
	matrix_random(
	    RNN_storage->Ro, -sqrt(1.0f / h_dim),
	    sqrt(1.0f / h_dim));

	RNN_storage->Pi = matrix_create(1, h_dim);
	RNN_storage->dPi = matrix_create(1, h_dim);
	RNN_storage->Pf = matrix_create(1, h_dim);
	RNN_storage->dPf = matrix_create(1, h_dim);
	RNN_storage->Po = matrix_create(1, h_dim);
	RNN_storage->dPo = matrix_create(1, h_dim);
	matrix_random(
	    RNN_storage->Pi, -sqrt(2.0f / (h_dim + 1)),
	    sqrt(2.0f / (h_dim + 1)));
	matrix_random(
	    RNN_storage->Pf, -sqrt(2.0f / (h_dim + 1)),
	    sqrt(2.0f / (h_dim + 1)));
	matrix_random(
	    RNN_storage->Po, -sqrt(2.0f / (h_dim + 1)),
	    sqrt(2.0f / (h_dim + 1)));

	RNN_storage->Bz = matrix_create(1, h_dim);
	RNN_storage->dBz = matrix_create(1, h_dim);
	RNN_storage->Bi = matrix_create(1, h_dim);
	RNN_storage->dBi = matrix_create(1, h_dim);
	RNN_storage->Bf = matrix_create(1, h_dim);
	RNN_storage->dBf = matrix_create(1, h_dim);
	RNN_storage->Bo = matrix_create(1, h_dim);
	RNN_storage->dBo = matrix_create(1, h_dim);
	matrix_random(
	    RNN_storage->Bz, -sqrt(2.0f / (h_dim + 1)),
	    sqrt(2.0f / (h_dim + 1)));
	matrix_random(
	    RNN_storage->Bi, -sqrt(2.0f / (h_dim + 1)),
	    sqrt(2.0f / (h_dim + 1)));
	matrix_random(
	    RNN_storage->Bf, -sqrt(2.0f / (h_dim + 1)),
	    sqrt(2.0f / (h_dim + 1)));

	// Large bias on forget gate to learn to remember
	for (int i = 0; i < h_dim; ++i)
		RNN_storage->Bf->data[0][i] = 1.0f;

	matrix_random(
	    RNN_storage->Bo, -sqrt(2.0f / (h_dim + 1)),
	    sqrt(2.0f / (h_dim + 1)));

	/* Output model */
	RNN_storage->V = matrix_create(o_dim, h_dim);
	RNN_storage->dV = matrix_create(o_dim, h_dim);
	RNN_storage->Bpo = matrix_create(1, o_dim);
	RNN_storage->dBpo = matrix_create(1, o_dim);
	matrix_random(
	    RNN_storage->V, -sqrt(2.0f / (h_dim + o_dim)),
	    sqrt(2.0f / (h_dim + o_dim)));
	matrix_random(
	    RNN_storage->Bpo, -sqrt(1.0f / (h_dim + 1)),
	    sqrt(1.0f / (h_dim + 1)));

	/* Adadelta */
	RNN_storage->EdWz = matrix_create(h_dim, i_dim);
	RNN_storage->EdWi = matrix_create(h_dim, i_dim);
	RNN_storage->EdWf = matrix_create(h_dim, i_dim);
	RNN_storage->EdWo = matrix_create(h_dim, i_dim);
	RNN_storage->EdRz = matrix_create(h_dim, h_dim);
	RNN_storage->EdRi = matrix_create(h_dim, h_dim);
	RNN_storage->EdRf = matrix_create(h_dim, h_dim);
	RNN_storage->EdRo = matrix_create(h_dim, h_dim);
	RNN_storage->EdPi = matrix_create(1, h_dim);
	RNN_storage->EdPf = matrix_create(1, h_dim);
	RNN_storage->EdPo = matrix_create(1, h_dim);
	RNN_storage->EdBz = matrix_create(1, h_dim);
	RNN_storage->EdBi = matrix_create(1, h_dim);
	RNN_storage->EdBf = matrix_create(1, h_dim);
	RNN_storage->EdBo = matrix_create(1, h_dim);
	RNN_storage->EdV = matrix_create(o_dim, h_dim);
	RNN_storage->EdBpo = matrix_create(1, o_dim);

	RNN_storage->dEdWz = matrix_create(h_dim, i_dim);
	RNN_storage->dEdWi = matrix_create(h_dim, i_dim);
	RNN_storage->dEdWf = matrix_create(h_dim, i_dim);
	RNN_storage->dEdWo = matrix_create(h_dim, i_dim);
	RNN_storage->dEdRz = matrix_create(h_dim, h_dim);
	RNN_storage->dEdRi = matrix_create(h_dim, h_dim);
	RNN_storage->dEdRf = matrix_create(h_dim, h_dim);
	RNN_storage->dEdRo = matrix_create(h_dim, h_dim);
	RNN_storage->dEdPi = matrix_create(1, h_dim);
	RNN_storage->dEdPf = matrix_create(1, h_dim);
	RNN_storage->dEdPo = matrix_create(1, h_dim);
	RNN_storage->dEdBz = matrix_create(1, h_dim);
	RNN_storage->dEdBi = matrix_create(1, h_dim);
	RNN_storage->dEdBf = matrix_create(1, h_dim);
	RNN_storage->dEdBo = matrix_create(1, h_dim);
	RNN_storage->dEdV = matrix_create(o_dim, h_dim);
	RNN_storage->dEdBpo = matrix_create(1, o_dim);
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

	// /* Network output */
	// for (t = 0; t < t_dim; ++t) {
	// 	for (o = 0; o < o_dim; ++o) {
	// 		for (r = 0; r < h_dim; ++r) {
	// 			P_O[t][o] += V[o][r] * Y[t][r];
	// 		}
	// 		P_O[t][o] += Bpo[o];
	// 	}
	// }

	/* Network output */
	for (t = 0; t < t_dim; ++t) {
		for (o = 0; o < o_dim; ++o) {
			for (r = 0; r < h_dim; ++r) {
				P_O[t][o] += V[o][r] * Y[t][r];
			}
			P_O[t][o] += Bpo[o];
			//P_O[t][o] = network_output_squash_func(P_O[t][o]);
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


	/* For t = t_dim - 1 */
	for (o = 0; o < o_dim; ++o) {
		dP_O[o] = 2.0f * (P_O[t_dim - 1][o] - E_O[t_dim - 1][o]);
		//* P_O[t_dim - 1][o] * (1 - P_O[t_dim - 1][o]);
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

	/* For t = t_dim - 2 ... 1 */
	for (t = t_dim - 2; t >= 1; --t) {
		clear_1d(dY, h_dim);
		for (o = 0; o < o_dim; ++o) {
			dP_O[o] = 2 * (P_O[t][o] - E_O[t][o]); //* P_O[t][o] * (1 - P_O[t][o]);
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
	}

	/* For t = 0 */
	clear_1d(dY, h_dim);
	for (o = 0; o < o_dim; ++o) {
		dP_O[o] = 2 * (P_O[0][o] - E_O[0][o]); // * P_O[0][o] * (1 - P_O[0][o]);
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

	free(dP_O);
	free(dY);
	free(dO);
	free(dC);
	free(dF);
	free(dI);
	free(dZ);
}

void RNN_Gradient_Clipping(Matrix_t *matrix, math_t threshold) {
	int m, n;
	for (m = 0; m < matrix->m; ++m) {
		for (n = 0; n < matrix->n; ++n) {
			if (fabs(matrix->data[m][n]) > threshold)
				matrix->data[m][n] *= threshold / fabs(matrix->data[m][n]);
		}
	}
}

void RNN_SGD(
    RNN_t * RNN_storage,
    Matrix_t *input_matrix,
    Matrix_t *expected_output_matrix,
    Matrix_t *predicted_output_matrix
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

	math_t **EdWz = RNN_storage->EdWz->data; math_t **EdRz = RNN_storage->EdRz->data;
	math_t **EdWi = RNN_storage->EdWi->data; math_t **EdRi = RNN_storage->EdRi->data;
	math_t **EdWf = RNN_storage->EdWf->data; math_t **EdRf = RNN_storage->EdRf->data;
	math_t **EdWo = RNN_storage->EdWo->data; math_t **EdRo = RNN_storage->EdRo->data;
	math_t *EdBz = RNN_storage->EdBz->data[0];
	math_t *EdBi = RNN_storage->EdBi->data[0]; math_t *EdPi = RNN_storage->EdPi->data[0];
	math_t *EdBf = RNN_storage->EdBf->data[0]; math_t *EdPf = RNN_storage->EdPf->data[0];
	math_t *EdBo = RNN_storage->EdBo->data[0]; math_t *EdPo = RNN_storage->EdPo->data[0];
	math_t **EdV = RNN_storage->EdV->data;
	math_t *EdBpo = RNN_storage->EdBpo->data[0];

	math_t **dEdWz = RNN_storage->dEdWz->data; math_t **dEdRz = RNN_storage->dEdRz->data;
	math_t **dEdWi = RNN_storage->dEdWi->data; math_t **dEdRi = RNN_storage->dEdRi->data;
	math_t **dEdWf = RNN_storage->dEdWf->data; math_t **dEdRf = RNN_storage->dEdRf->data;
	math_t **dEdWo = RNN_storage->dEdWo->data; math_t **dEdRo = RNN_storage->dEdRo->data;
	math_t *dEdBz = RNN_storage->dEdBz->data[0];
	math_t *dEdBi = RNN_storage->dEdBi->data[0]; math_t *dEdPi = RNN_storage->dEdPi->data[0];
	math_t *dEdBf = RNN_storage->dEdBf->data[0]; math_t *dEdPf = RNN_storage->dEdPf->data[0];
	math_t *dEdBo = RNN_storage->dEdBo->data[0]; math_t *dEdPo = RNN_storage->dEdPo->data[0];
	math_t **dEdV = RNN_storage->dEdV->data;
	math_t *dEdBpo = RNN_storage->dEdBpo->data[0];


	math_t gamma = RNN_storage->gamma;
	math_t ep = 1e-6;

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

	// Adadelta update

	for (h = 0; h < h_dim; ++h) {
		for (i = 0; i < i_dim; ++i) {
			EdWz[h][i] = gamma * EdWz[h][i] + (dWz[h][i] * dWz[h][i]) * (1 - gamma);
			EdWi[h][i] = gamma * EdWi[h][i] + (dWi[h][i] * dWi[h][i]) * (1 - gamma);
			EdWf[h][i] = gamma * EdWf[h][i] + (dWf[h][i] * dWf[h][i]) * (1 - gamma);
			EdWo[h][i] = gamma * EdWo[h][i] + (dWo[h][i] * dWo[h][i]) * (1 - gamma);
		}
	}

	for (h = 0; h < h_dim; ++h) {
		for (r = 0; r < h_dim; ++r) {
			EdRz[h][r] = gamma * EdRz[h][r] + (dRz[h][r] * dRz[h][r]) * (1 - gamma);
			EdRi[h][r] = gamma * EdRi[h][r] + (dRi[h][r] * dRi[h][r]) * (1 - gamma);
			EdRf[h][r] = gamma * EdRf[h][r] + (dRf[h][r] * dRf[h][r]) * (1 - gamma);
			EdRo[h][r] = gamma * EdRo[h][r] + (dRo[h][r] * dRo[h][r]) * (1 - gamma);
		}
	}

	for (h = 0; h < h_dim; ++h) {
		EdPi[h] = gamma * EdPi[h] + (dPi[h] * dPi[h]) * (1 - gamma);
		EdPf[h] = gamma * EdPf[h] + (dPf[h] * dPf[h]) * (1 - gamma);
		EdPo[h] = gamma * EdPo[h] + (dPo[h] * dPo[h]) * (1 - gamma);

		EdBz[h] = gamma * EdBz[h] + (dBz[h] * dBz[h]) * (1 - gamma);
		EdBi[h] = gamma * EdBi[h] + (dBi[h] * dBi[h]) * (1 - gamma);
		EdBf[h] = gamma * EdBf[h] + (dBf[h] * dBf[h]) * (1 - gamma);
		EdBo[h] = gamma * EdBo[h] + (dBo[h] * dBo[h]) * (1 - gamma);
	}

	for (o = 0; o < o_dim; ++o) {
		for (h = 0; h < h_dim; ++h) {
			EdV[o][h] = gamma * EdV[o][h] + (dV[o][h] * dV[o][h]) * (1 - gamma);
		}
	}
	for (o = 0; o < o_dim; ++o) {
		EdBpo[o] = gamma * EdBpo[o] + (dBpo[o]  * dBpo[o]) * (1 - gamma);
	}

	for (h = 0; h < h_dim; ++h) {
		for (i = 0; i < i_dim; ++i) {
			math_t d_z = sqrt(dEdWz[h][i] + ep) / sqrt(EdWz[h][i] + ep) * dWz[h][i];
			math_t d_i = sqrt(dEdWi[h][i] + ep) / sqrt(EdWi[h][i] + ep) * dWi[h][i];
			math_t d_f = sqrt(dEdWf[h][i] + ep) / sqrt(EdWf[h][i] + ep) * dWf[h][i];
			math_t d_o = sqrt(dEdWo[h][i] + ep) / sqrt(EdWo[h][i] + ep) * dWo[h][i];

			Wz[h][i] -= d_z;
			Wi[h][i] -= d_i;
			Wf[h][i] -= d_f;
			Wo[h][i] -= d_o;

			dEdWz[h][i] = gamma * dEdWz[h][i] + (d_z * d_z) * (1 - gamma);
			dEdWi[h][i] = gamma * dEdWi[h][i] + (d_i * d_i) * (1 - gamma);
			dEdWf[h][i] = gamma * dEdWf[h][i] + (d_f * d_f) * (1 - gamma);
			dEdWo[h][i] = gamma * dEdWo[h][i] + (d_o * d_o) * (1 - gamma);
		}
		for (r = 0; r < h_dim; ++r) {
			math_t d_z = sqrt(dEdRz[h][r] + ep) / sqrt(EdRz[h][r] + ep) * dRz[h][r];
			math_t d_i = sqrt(dEdRi[h][r] + ep) / sqrt(EdRi[h][r] + ep) * dRi[h][r];
			math_t d_f = sqrt(dEdRf[h][r] + ep) / sqrt(EdRf[h][r] + ep) * dRf[h][r];
			math_t d_o = sqrt(dEdRo[h][r] + ep) / sqrt(EdRo[h][r] + ep) * dRo[h][r];

			Rz[h][r] -= d_z;
			Ri[h][r] -= d_i;
			Rf[h][r] -= d_f;
			Ro[h][r] -= d_o;

			dEdRz[h][r] = gamma * dEdRz[h][r] + (d_z * d_z) * (1 - gamma);
			dEdRi[h][r] = gamma * dEdRi[h][r] + (d_i * d_i) * (1 - gamma);
			dEdRf[h][r] = gamma * dEdRf[h][r] + (d_f * d_f) * (1 - gamma);
			dEdRo[h][r] = gamma * dEdRo[h][r] + (d_o * d_o) * (1 - gamma);
		}
		math_t d_z;
		math_t d_i = sqrt(dEdPi[h] + ep) / sqrt(EdPi[h] + ep) * dPi[h];
		math_t d_f = sqrt(dEdPf[h] + ep) / sqrt(EdPf[h] + ep) * dPf[h];
		math_t d_o = sqrt(dEdPo[h] + ep) / sqrt(EdPo[h] + ep) * dPo[h];

		Pi[h] -= d_i;
		Pf[h] -= d_f;
		Po[h] -= d_o;

		dEdPi[h] = gamma * dEdPi[h] + (d_i * d_i) * (1 - gamma);
		dEdPf[h] = gamma * dEdPf[h] + (d_f * d_f) * (1 - gamma);
		dEdPo[h] = gamma * dEdPo[h] + (d_o * d_o) * (1 - gamma);

		d_z = sqrt(dEdBz[h] + ep) / sqrt(EdBz[h] + ep) * dBz[h];
		d_i = sqrt(dEdBi[h] + ep) / sqrt(EdBi[h] + ep) * dBi[h];
		d_f = sqrt(dEdBf[h] + ep) / sqrt(EdBf[h] + ep) * dBf[h];
		d_o = sqrt(dEdBo[h] + ep) / sqrt(EdBo[h] + ep) * dBo[h];

		Bz[h] -= d_z;
		Bi[h] -= d_i;
		Bf[h] -= d_f;
		Bo[h] -= d_o;

		dEdBz[h] = gamma * dEdBz[h] + (d_z * d_z) * (1 - gamma);
		dEdBi[h] = gamma * dEdBi[h] + (d_i * d_i) * (1 - gamma);
		dEdBf[h] = gamma * dEdBf[h] + (d_f * d_f) * (1 - gamma);
		dEdBo[h] = gamma * dEdBo[h] + (d_o * d_o) * (1 - gamma);
	}
	for (o = 0; o < o_dim; ++o) {
		for (h = 0; h < h_dim; ++h) {
			math_t d_o = sqrt(dEdV[o][h] + ep) / sqrt(EdV[o][h] + ep) * dV[o][h];
			V[o][h] -= d_o;
			dEdV[o][h] = gamma * dEdV[o][h] + (d_o * d_o) * (1 - gamma);
		}
		math_t d_o = sqrt(dEdBpo[o] + ep) / sqrt(EdBpo[o] + ep) * dBpo[o];
		Bpo[o] -= d_o;
		dEdBpo[o] = gamma * dEdBpo[o] + (d_o * d_o) * (1 - gamma);
	}
}

math_t RNN_train(
    RNN_t * RNN_storage,
    DataSet_t *train_set,
    Matrix_t *predicted_output_matrix,
    int max_epoch,
    int print_loss_interval,
    int gradient_check_interval
)  {
	int num_train = train_set->num_matrix;

	int i, e, t;
	math_t current_total_loss;

	Matrix_t *input_matrix, *expected_output_matrix;

	for (e = 0; e < max_epoch; ++e) {
		if ((e > 0 && e % print_loss_interval == 0) || (e == max_epoch - 1)) {
			current_total_loss = 0.0;
			for (i = 0; i < num_train; ++i) {
				input_matrix = train_set->input_matrix_list[i];
				expected_output_matrix = train_set->output_matrix_list[i];
				RNN_Predict(
				    RNN_storage,
				    input_matrix,
				    predicted_output_matrix
				);

				current_total_loss +=
				    RNN_loss_calculation(
				        RNN_storage,
				        predicted_output_matrix,
				        expected_output_matrix) / expected_output_matrix->m;
			}
			printf("average loss at epoch: %10d = %10.10lf\n",
			       e, current_total_loss / num_train);
		}

		if (e > 0 && e % gradient_check_interval == 0) {
			int gradient_check_result =
			    RNN_Gradient_check(
			        RNN_storage,
			        train_set,
			        predicted_output_matrix,
			        1e-4,
			        1e-2,
			        0
			    );

			// Terminate the training process if the gradient check did not pass
			if (gradient_check_result != 0) {
				printf("Gradient check error at epoch: %10d\n", e);
				return -1;
			}
		}

		for (t = 0; t < num_train; ++t) {
			input_matrix = train_set->input_matrix_list[t];
			expected_output_matrix = train_set->output_matrix_list[t];

			RNN_SGD(
			    RNN_storage,
			    input_matrix,
			    expected_output_matrix,
			    predicted_output_matrix
			);
		}
	}

	return current_total_loss / num_train;
}

// Square loss
math_t RNN_loss_calculation(
    RNN_t * RNN_storage,
    Matrix_t *predicted_output_matrix,	// TxO
    Matrix_t *expected_output_matrix	// TxO
) {
	math_t total_loss = 0.0, log_term = 0.0, delta;

	int t_dim = expected_output_matrix->m;
	int o_dim = RNN_storage->o_dim;

	int t, o;
	for (t = 0; t < t_dim; ++t) {
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


// RMSE
math_t* RNN_RMSE(
    RNN_t * RNN_storage,
    Matrix_t *predicted_output_matrix,	// TxO
    Matrix_t *expected_output_matrix	// TxO
) {
	int t_dim = expected_output_matrix->m;
	int o_dim = RNN_storage->o_dim;

	math_t *loss_list = (math_t *) malloc(o_dim * sizeof(math_t));
	math_t delta;

	int t, o;
	for (t = 0; t < t_dim; ++t) {
		for (o = 0; o < o_dim; ++o) {
			delta =
			    expected_output_matrix->data[t][o] -
			    predicted_output_matrix->data[t][o];
			loss_list[o] += delta * delta;
		}				
	}

	for (o = 0; o < o_dim; ++o) {
		loss_list[o] = sqrt(loss_list[o] / t_dim);
	}

	return loss_list;
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

	int num_parameter = sizeof(testing_model_list) / sizeof(Matrix_t *);
	for (i = 0; i < num_parameter; ++i) {
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

				if (estimated_gradient > 1e-6 &&
				        relative_gradient_error > error_threshold) {
					printf("-------------Gradient check error\n");
					printf("For matrix %d [%d][%d]\n", i, m, n);
					printf("+h loss: %.13lf\n", total_loss_plus);
					printf("-h loss: %.13lf\n", total_loss_minus);
					printf("estimated_gradient: %.13lf\n", estimated_gradient);
					printf("calculated_gradient: %.13lf\n", calculated_gradient);
					printf("relative_gradient_error: %.13lf\n", relative_gradient_error);
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

math_t network_output_squash_func(math_t value) {
	return sigmoid(value);
}

math_t sigmoid(math_t value) {
	//return 2.0 / (1 + exp(-2 * value)) - 1;
	return 1.0 / (1 + exp(-value));
}
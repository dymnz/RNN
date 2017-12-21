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
	RNN_storage->i_dim = input_vector_len;
	RNN_storage->o_dim = output_vector_len;
	RNN_storage->h_dim = hidden_layer_vector_len;
	RNN_storage->bptt_truncate_len = bptt_truncate_len;

	RNN_storage->C = matrix_create(0, 0);
	RNN_storage->S = matrix_create(0, 0);
	RNN_storage->aS = matrix_create(0, 0);
	RNN_storage->Ig = matrix_create(0, 0);
	RNN_storage->Fg = matrix_create(0, 0);
	RNN_storage->Og = matrix_create(0, 0);
	RNN_storage->V
	    = matrix_create(hidden_layer_vector_len, output_vector_len);
	RNN_storage->dV
	    = matrix_create(hidden_layer_vector_len, output_vector_len);

	RNN_storage->Ui
	    = matrix_create(input_vector_len, hidden_layer_vector_len);
	RNN_storage->Wi
	    = matrix_create(hidden_layer_vector_len, hidden_layer_vector_len);
	RNN_storage->Uf
	    = matrix_create(input_vector_len, hidden_layer_vector_len);
	RNN_storage->Wf
	    = matrix_create(hidden_layer_vector_len, hidden_layer_vector_len);
	RNN_storage->Uo
	    = matrix_create(input_vector_len, hidden_layer_vector_len);
	RNN_storage->Wo
	    = matrix_create(hidden_layer_vector_len, hidden_layer_vector_len);
	RNN_storage->Us
	    = matrix_create(input_vector_len, hidden_layer_vector_len);
	RNN_storage->Ws
	    = matrix_create(hidden_layer_vector_len, hidden_layer_vector_len);


	unsigned int seed = RNN_RAND_SEED;

	matrix_random_with_seed(
	    RNN_storage->V,
	    -sqrt(1 / hidden_layer_vector_len),
	    sqrt(1 / hidden_layer_vector_len),
	    &seed);
	matrix_random_with_seed(
	    RNN_storage->Ui,
	    -sqrt(1 / input_vector_len),
	    sqrt(1 / input_vector_len),
	    &seed);
	matrix_random_with_seed(
	    RNN_storage->Uf,
	    -sqrt(1 / input_vector_len),
	    sqrt(1 / input_vector_len),
	    &seed);
	matrix_random_with_seed(
	    RNN_storage->Uo,
	    -sqrt(1 / input_vector_len),
	    sqrt(1 / input_vector_len),
	    &seed);
	matrix_random_with_seed(
	    RNN_storage->Ug,
	    -sqrt(1 / input_vector_len),
	    sqrt(1 / input_vector_len),
	    &seed);

	matrix_random_with_seed(
	    RNN_storage->Wi,
	    -sqrt(1 / hidden_layer_vector_len),
	    sqrt(1 / hidden_layer_vector_len),
	    &seed);
	matrix_random_with_seed(
	    RNN_storage->Wf,
	    -sqrt(1 / hidden_layer_vector_len),
	    sqrt(1 / hidden_layer_vector_len),
	    &seed);
	matrix_random_with_seed(
	    RNN_storage->Wo,
	    -sqrt(1 / hidden_layer_vector_len),
	    sqrt(1 / hidden_layer_vector_len),
	    &seed);
	matrix_random_with_seed(
	    RNN_storage->Wg,
	    -sqrt(1 / hidden_layer_vector_len),
	    sqrt(1 / hidden_layer_vector_len),
	    &seed);
}

void RNN_destroy(RNN_t *RNN_storage) {
	matrix_free(RNN_storage->C);
	matrix_free(RNN_storage->S);
	matrix_free(RNN_storage->aS);
	matrix_free(RNN_storage->V);
	matrix_free(RNN_storage->Ig);
	matrix_free(RNN_storage->Fg);
	matrix_free(RNN_storage->Og);

	matrix_free(RNN_storage->Ui);
	matrix_free(RNN_storage->Wi);
	matrix_free(RNN_storage->dUi);
	matrix_free(RNN_storage->dWi);

	matrix_free(RNN_storage->Uf);
	matrix_free(RNN_storage->Wf);
	matrix_free(RNN_storage->dUf);
	matrix_free(RNN_storage->dWf);

	matrix_free(RNN_storage->Uo);
	matrix_free(RNN_storage->Wo);
	matrix_free(RNN_storage->dUo);
	matrix_free(RNN_storage->dWo);

	matrix_free(RNN_storage->Ug);
	matrix_free(RNN_storage->Wg);
	matrix_free(RNN_storage->dUg);
	matrix_free(RNN_storage->dWg);

	matrix_free(RNN_storage->dWg);
	matrix_free(RNN_storage->dWg);
	matrix_free(RNN_storage->dWg);
	matrix_free(RNN_storage->dWg);
	free(RNN_storage);
}

void RNN_forward_propagation(
    RNN_t *RNN_storage,
    Matrix_t *input_matrix,	// TxI
    Matrix_t *output_matrix	// TxO
) {
	int i_dim = RNN_storage->i_dim;
	int o_dim = RNN_storage->o_dim;
	int t_dim = input_matrix->m;
	int h_dim = RNN_storage->h_dim;

	matrix_resize(RNN_storage->C, t_dim, h_dim);
	matrix_resize(RNN_storage->S, t_dim, h_dim);
	matrix_resize(RNN_storage->aS, t_dim, h_dim);
	matrix_resize(RNN_storage->Ig, t_dim, h_dim);
	matrix_resize(RNN_storage->Fg, t_dim, h_dim);
	matrix_resize(RNN_storage->Og, t_dim, h_dim);

	math_t **X = input_matrix->data;
	math_t **O = output_matrix->data;

	math_t **C = RNN_storage->C->data;    // TxH
	math_t **S = RNN_storage->S->data;    // TxH
	math_t **aS = RNN_storage->aS->data;    // TxH

	math_t **Ig = RNN_storage->Ig->data;	// TxH
	math_t **Fg = RNN_storage->Fg->data;	// TxH
	math_t **Og = RNN_storage->Og->data;	// TxH

	math_t **Ui = RNN_storage->Ui->data;   // IxH
	math_t **Wi = RNN_storage->Wi->data;   // HxH

	math_t **Uf = RNN_storage->Uf->data;   // IxH
	math_t **Wf = RNN_storage->Wf->data;   // HxH

	math_t **Uo = RNN_storage->Uo->data;   // IxH
	math_t **Wo = RNN_storage->Wo->data;   // HxH

	math_t **Us = RNN_storage->Ug->data;   // IxH
	math_t **Ws = RNN_storage->Wg->data;   // HxH

	int m, n, r, t;

	clear_2d(C, t_dim, h_dim);
	clear_2d(S, t_dim, h_dim);
	clear_2d(aS, t_dim, h_dim);
	clear_2d(O, t_dim, o_dim);

	clear_2d(Ig, t_dim, h_dim);
	clear_2d(Fg, t_dim, h_dim);
	clear_2d(Og, t_dim, h_dim);

	// For t = 0
	for (n = 0; n < h_dim; ++n) {
		// Input gate / Forget gate / Cell state
		for (r = 0; r < i_dim; ++r) {
			Ig[0][n] += X[0][r] * Ui[r][n];
			Fg[0][n] += X[0][r] * Uf[r][n];
			aS[0][n] += X[0][r] * Us[r][n];
		}

		Ig[0][n] = gate_squash_func(Ig[0][n]);
		Fg[0][n] = gate_squash_func(Fg[0][n]);
		aS[0][n] = cell_state_squash_func(aS[0][n]);
		S[0][n] = Ig[0][n] * aS[0][n];

		// Output gate
		for (r = 0; r < i_dim; ++r)
			Og[0][n] += X[0][r] * Uo[r][n];
		for (r = 0; r < h_dim; ++r)
			Og[0][n] += S[0][r] * Wo[r][n];

		Og[0][n] = gate_squash_func(Og[0][n]);

		// Cell output
		C[0][n] = Og[0][n] * cell_output_squash_func(S[0][n]);
	}

	// For t = 1 ... t_dim
	for (t = 1; t < t_dim; ++t) {
		// Input gate / Forget gate / Cell state
		for (n = 0; n < h_dim; ++n) {
			for (r = 0; r < i_dim; ++r) {
				Ig[t][n] += X[t][r] * Ui[r][n];
				Fg[t][n] += X[t][r] * Uf[r][n];
				aS[t][n] += X[t][r] * Us[r][n];
			}
			for (r = 0; r < h_dim; ++r) {
				Ig[t][n] += S[t - 1][r] * Ui[r][n];
				Fg[t][n] += S[t - 1][r] * Uf[r][n];
			}

			Ig[t][n] = gate_squash_func(Ig[t][n]);
			Fg[t][n] = gate_squash_func(Fg[t][n]);
			aS[t][n] = cell_state_squash_func(aS[t][n]);
			S[t][n] = Fg[t][n] * S[t - 1][n] +
			          Ig[t][n] * aS[t][n];

			// Output gate
			for (r = 0; r < i_dim; ++r)
				Og[t][n] += X[t][r] * Uo[r][n];
			for (r = 0; r < h_dim; ++r)
				Og[t][n] += S[t][r] * Wo[r][n];

			Og[t][n] = gate_squash_func(Og[t][n]);

			// Cell output
			C[t][n] = Og[t][n] * cell_output_squash_func(S[t][n]);
		}
	}

	// Network output
	for (t = 0; t < t_dim; ++t) {
		for (n = 0; n < o_dim; ++n) {
			for (r = 0; r < h_dim; ++r) {
				O[t][n] += C[t][r] * V[r][n];
			}
		}
		O[t][n] = network_output_squash_func(O[t][n]);
	}
}

// Cross entropy loss
math_t RNN_loss_calculation(
    RNN_t *RNN_storage,
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

void RNN_BPTT(
    RNN_t *RNN_storage,
    Matrix_t *input_matrix,				// TxI
    Matrix_t *predicted_output_matrix,	// TxO
    Matrix_t *expected_output_matrix	// TxO
) {
	int t_dim = input_matrix->m;
	int i_dim = RNN_storage->i_dim;
	int o_dim = RNN_storage->o_dim;
	int h_dim = RNN_storage->h_dim;

	int bptt_truncate_len = RNN_storage->bptt_truncate_len;

	math_t **X = input_matrix->data;
	math_t **O = output_matrix->data;

	math_t **C = RNN_storage->C->data;    // TxH
	math_t **S = RNN_storage->S->data;    // TxH
	math_t **aS = RNN_storage->aS->data;    // TxH

	math_t **Ig = RNN_storage->Ig->data;	// TxH
	math_t **Fg = RNN_storage->Fg->data;	// TxH
	math_t **Og = RNN_storage->Og->data;	// TxH

	math_t **V = RNN_storage->V->data;   // HxO
	math_t **dV = RNN_storage->dV->data;   // HxO

	math_t **Ui = RNN_storage->Ui->data;   // IxH
	math_t **Wi = RNN_storage->Wi->data;   // HxH
	math_t **dUi = RNN_storage->Ui->data;   // IxH
	math_t **dWi = RNN_storage->Wi->data;   // HxH

	math_t **Uf = RNN_storage->Uf->data;   // IxH
	math_t **Wf = RNN_storage->Wf->data;   // HxH
	math_t **dUf = RNN_storage->Uf->data;   // IxH
	math_t **dWf = RNN_storage->Wf->data;   // HxH

	math_t **Uo = RNN_storage->Uo->data;   // IxH
	math_t **Wo = RNN_storage->Wo->data;   // HxH
	math_t **dUo = RNN_storage->Uo->data;   // IxH
	math_t **dWo = RNN_storage->Wo->data;   // HxH

	math_t **Us = RNN_storage->Ug->data;   // IxH
	math_t **Ws = RNN_storage->Wg->data;   // HxH
	math_t **dUs = RNN_storage->Ug->data;   // IxH
	math_t **dWs = RNN_storage->Wg->data;   // HxH

	clear_2d(dUi, i_dim, h_dim);
	clear_2d(dUf, i_dim, h_dim);
	clear_2d(dUo, i_dim, h_dim);
	clear_2d(dUs, i_dim, h_dim);

	clear_2d(dWi, h_dim, h_dim);
	clear_2d(dWf, h_dim, h_dim);
	clear_2d(dWo, h_dim, h_dim);
	clear_2d(dWs, h_dim, h_dim);


	int c, k, o, s, f;

	math_t **delta_Og = create_2d(t_dim, h_dim);
	math_t **delta_Ig = create_2d(t_dim, h_dim);
	math_t **delta_Fg = create_2d(t_dim, h_dim);

	math_t **e_C = create_2d(t_dim, h_dim);
	math_t **e_S = create_2d(t_dim, h_dim);

	clear_2d(delta_Og, t_dim, h_dim);
	clear_2d(delta_Ig, t_dim, h_dim);
	clear_2d(delta_Fg, t_dim, h_dim);
	clear_2d(e_C, t_dim, h_dim);
	clear_2d(e_S, t_dim, h_dim);

	// For t = T - 1
	for (m = 0; m < h_dim; ++m) {
		for (n = 0; n < o_dim; ++n) {
			// Outer product
			dV[m][n] += C[t_dim - 1][m] * delta_o[t_dim - 1][n];
		}
	}
	// Cell output
	for (c = 0; c < h_dim; ++c) {
		for (k = 0; k < o_dim ++k) {
			e_C[t_dim - 1][c] +=
			    2 * (O[t_dim - 1][c] - Y[t_dim - 1][c]) *
			    V[c][k];
		}
	}
	// Output gate
	for (o = 0; o < h_dim; ++o) {
		for (c = 0; c < h_dim; ++c) {
			delta_Og[t_dim - 1][o] +=
			    cell_output_squash_func(S[t_dim - 1][c]) *
			    e_C[t_dim - 1][c];
		}
		delta_Og[t_dim - 1][o] *= gate_squash_derivative(Og[t_dim - 1][h]);
	}
	// States
	for (s = 0; s < h_dim; ++s) {
		e_S[t_dim - 1][s] =
		    Og[t_dim - 1][s] *
		    cell_output_squash_derivative(S[t_dim - 1][s]) *
		    e_C[t_dim - 1][s];
		for (c = 0; c < h_dim; ++c) {
			e_S[t_dim - 1][s] +=
			    Wo[s][c] * delta_Og[t_dim - 1][c];
		}
	}
	// Cell
	for (c = 0; c < h_dim; ++c) {
		delta_C[t_dim - 1][c] =
		    Ig[t_dim - 1][c] *
		    cell_state_squash_derivative(aS[t_dim - 1][c]) *
		    e_S[t_dim - 1][c];
	}
	// Forget gate
	for (o = 0; o < h_dim; ++o) {
		for (s = 0; s < h_dim; ++s) {
			delta_Fg[t_dim - 1][o] +=
			    S[t_dim - 1 - 1][s] *
			    e_S[t_dim - 1][s];
		}
		delta_Fg[t_dim - 1][o] *= gate_squash_derivative(Fg[t_dim - 1][h]);
	}
	// Input gate
	for (o = 0; o < h_dim; ++o) {
		for (c = 0; c < h_dim; ++c) {
			delta_Ig[t_dim - 1][o] +=
			    cell_state_squash_derivative(aS[t_dim - 1][s]) *
			    e_S[t_dim - 1][s];
		}
		delta_Ig[t_dim - 1][o] *= gate_squash_derivative(Ig[t_dim - 1][h]);
	}

	free_2d(delta_Og, t_dim);
	free_2d(delta_Ig, t_dim);
	free_2d(delta_Fg, t_dim);
	free_2d(e_C, t_dim);
	free_2d(e_S, t_dim);
}

void RNN_SGD(
    RNN_t * RNN_storage,
    Matrix_t *input_matrix,
    Matrix_t *expected_output_matrix,
    Matrix_t *predicted_output_matrix,
    math_t learning_rate
) {

	math_t **Ui = RNN_storage->Ui->data;   // IxH
	math_t **Wi = RNN_storage->Wi->data;   // HxH
	math_t **dUi = RNN_storage->Ui->data;   // IxH
	math_t **dWi = RNN_storage->Wi->data;   // HxH

	math_t **Uf = RNN_storage->Uf->data;   // IxH
	math_t **Wf = RNN_storage->Wf->data;   // HxH
	math_t **dUf = RNN_storage->Uf->data;   // IxH
	math_t **dWf = RNN_storage->Wf->data;   // HxH

	math_t **Uo = RNN_storage->Uo->data;   // IxH
	math_t **Wo = RNN_storage->Wo->data;   // HxH
	math_t **dUo = RNN_storage->Uo->data;   // IxH
	math_t **dWo = RNN_storage->Wo->data;   // HxH

	math_t **Us = RNN_storage->Ug->data;   // IxH
	math_t **Ws = RNN_storage->Wg->data;   // HxH
	math_t **dUs = RNN_storage->Ug->data;   // IxH
	math_t **dWs = RNN_storage->Wg->data;   // HxH

	int i_dim = RNN_storage->i_dim;
	int o_dim = RNN_storage->o_dim;
	int h_dim = RNN_storage->h_dim;

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

	int m, n;

	// Update U
	for (m = 0; m < i_dim; ++m)
		for (n = 0; n < h_dim; ++n)
			Ui[m][n] -= learning_rate * dLdU[m][n];

	// Update V
	for (m = 0; m < h_dim; ++m)
		for (n = 0; n < o_dim; ++n)
			V[m][n] -= learning_rate * dLdV[m][n];

	// Update W
	for (m = 0; m < h_dim; ++m)
		for (n = 0; n < h_dim; ++n)
			W[m][n] -= learning_rate * dLdW[m][n];
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

	math_t **U = RNN_storage->input_weight_matrix->data;	// IxH
	math_t **V = RNN_storage->output_weight_matrix->data;	// HxO
	math_t **W = RNN_storage->internal_weight_matrix->data;	// HxH

	math_t **dLdU = RNN_storage->input_weight_gradient->data;	// IxH
	math_t **dLdV = RNN_storage->output_weight_gradient->data;	// HxO
	math_t **dLdW = RNN_storage->internel_weight_gradient->data; // HxH

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
		RNN_storage->input_weight_gradient,		// U
		RNN_storage->output_weight_gradient,		// V
		RNN_storage->internel_weight_gradient	// W
	};
	math_t **UVW[] = {U, V, W};
	math_t **dLdUVW[] = {dLdU, dLdV, dLdW};

	Matrix_t *testing_model;
	math_t **testing_matrix;
	math_t **testing_gradient_matrix;

	for (i = 0; i < 2; ++i) {
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

math_t gate_squash_func(math_t value) {
	return sigmoid(value);
}
math_t gate_squash_derivative(math_t value) {
	return value * (1 - value);
}


math_t cell_state_squash_func(math_t value) {
	return tanh(value);
}
math_t cell_state_squash_derivative(math_t value) {
	return 1 - value * value;
}
math_t cell_output_squash_func(math_t value) {
	return tanh(value);
}
math_t cell_output_squash_derivative(math_t value) {
	return 1 - value * value;
}

math_t network_output_squash_func(math_t value) {
	return value;
}


math_t sigmoid(math_t value) {
	return 2.0 / (1 + exp(-2 * value)) - 1;
}
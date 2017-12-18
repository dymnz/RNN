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
	RNN_storage->V
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
	RNN_storage->Ug
	    = matrix_create(input_vector_len, hidden_layer_vector_len);
	RNN_storage->Wg
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
	matrix_free(RNN_storage->V);

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

	math_t **X = input_matrix->data;
	math_t **O = output_matrix->data;

	math_t **C = RNN_storage->C->data;    // TxH
	math_t **S = RNN_storage->S->data;    // TxH

	math_t **Ui = RNN_storage->Ui->data;   // IxH
	math_t **Wi = RNN_storage->Wi->data;   // HxH

	math_t **Uf = RNN_storage->Uf->data;   // IxH
	math_t **Wf = RNN_storage->Wf->data;   // HxH

	math_t **Uo = RNN_storage->Uo->data;   // IxH
	math_t **Wo = RNN_storage->Wo->data;   // HxH

	math_t **Ug = RNN_storage->Ug->data;   // IxH
	math_t **Wg = RNN_storage->Wg->data;   // HxH

	math_t *Ig = (math_t *) malloc(h_dim * sizeof(math_t));    // 1xH
	math_t *Fg = (math_t *) malloc(h_dim * sizeof(math_t));    // 1xH
	math_t *Og = (math_t *) malloc(h_dim * sizeof(math_t));    // 1xH
	math_t *G = (math_t *) malloc(h_dim * sizeof(math_t));    // 1xH

	int m, n, r, t;

	clear_2d(G, t_dim, h_dim);
	clear_2d(C, t_dim, h_dim);
	clear_2d(S, t_dim, h_dim);
	clear_2d(O, t_dim, o_dim);

	clear_1d(Ig, h_dim);
	clear_1d(Fg, h_dim);
	clear_1d(Og, h_dim);
	clear_1d(G, h_dim);

	// For t = 0
	for (n = 0; n < h_dim; ++n) {
		for (r = 0; r < i_dim; ++r) {
			Ig[n] += X[0][r] * Ui[r][n];
			Fg[n] += X[0][r] * Uf[r][n];
			Og[n] += X[0][r] * Uo[r][n];
			G[n] += X[0][r] * Ug[r][n];
		}
	}
	for (n = 0; n < h_dim; ++n) {
		Ig[n] = gate_squash_func(Ig[n]);
		Fg[n] = gate_squash_func(Fg[n]);
		Og[n] = gate_squash_func(Og[n]);
		G[n] = internal_squash_func(G[n]);

		C[0][n] = G[n] * Ig[n];
		S[0][n] = internal_squash_func(C[0][n]) * Og[n];
	}

	for (t = 1; t < t_dim; ++t) {
		clear_1d(Ig, h_dim);
		clear_1d(Fg, h_dim);
		clear_1d(Og, h_dim);
		clear_1d(G, h_dim);
		for (n = 0; n < h_dim; ++n) {
			for (r = 0; r < i_dim; ++r) {
				// S[t] = X[t]*U
				Ig[n] += X[t][r] * Ui[r][n];
				Fg[n] += X[t][r] * Uf[r][n];
				Og[n] += X[t][r] * Uo[r][n];
				G[n] += X[t][r] * Ug[r][n];
			}
			for (r = 0; r < h_dim; ++r) {
				// S[t] = X[t]*U + S[t-1]*W
				Ig[n] += S[t - 1][r] * Wi[r][n];
				Fg[n] += S[t - 1][r] * Wf[r][n];
				Og[n] += S[t - 1][r] * Wo[r][n];
				G[n] += S[t - 1][r] * Wg[r][n];
			}
			Ig[n] = gate_squash_func(Ig[n]);
			Fg[n] = gate_squash_func(Fg[n]);
			Og[n] = gate_squash_func(Og[n]);
			G[n] = internal_squash_func(G[n]);

			C[t][n] = C[t - 1][n] * Fg[n] + G[n] * Ig[n];
			S[t][n] = internal_squash_func(C[0][n]) * Og[n];
		}
	}

	for (t = 0; t < t_dim; ++t) {
		for (n = 0; n < o_dim; ++n) {
			for (r = 0; r < h_dim; ++r) {
				O[t][n] += S[t][r] * V[r][n];
			}
		}
		O[t][n] = output_squash_func(O[t][n]);
	}

	free(Ig);
	free(Fg);
	free(Og);
	free(G);
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

	math_t **Ui = RNN_storage->Ui->data;   // IxH
	math_t **Wi = RNN_storage->Wi->data;   // HxH

	math_t **Uf = RNN_storage->Uf->data;   // IxH
	math_t **Wf = RNN_storage->Wf->data;   // HxH

	math_t **Uo = RNN_storage->Uo->data;   // IxH
	math_t **Wo = RNN_storage->Wo->data;   // HxH

	math_t **Ug = RNN_storage->Ug->data;   // IxH
	math_t **Wg = RNN_storage->Wg->data;   // HxH

	math_t **delta_o
	    = create_2d(t_dim, o_dim);
	math_t *delta_t
	    = (math_t *) malloc(h_dim * sizeof(math_t));
	math_t *temp_delta_t
	    = (math_t *) malloc(h_dim * sizeof(math_t));


	math_t **X = input_matrix->data;
	math_t **O = predicted_output_matrix->data;
	math_t **Y = expected_output_matrix->data;

	clear_2d(dLdU, i_dim, h_dim);
	clear_2d(dLdV, h_dim, o_dim);
	clear_2d(dLdW, h_dim, h_dim);

	int t, o, bptt_t, m, n;

	/*
		Derivative of loss w.r.t. output layer
	 */

	// Delta o
	for (t = 0; t < t_dim; ++t) {
		for (o = 0; o < o_dim; ++o) {
			delta_o[t][o] = output_squash_derivative(O[t][o], Y[t][o]);
		}
	}

	/*
		BPTT
	 */
	// math_t **dLdU = input_weight_gradient->data;		// IxH
	// math_t **dLdV = output_weight_gradient->data;	// HxO
	// math_t **dLdW = internel_weight_gradient->data;	// HxH
	for (t = t_dim - 1; t >= 0; --t) {
		// Update dLdV += outer(delta_o[t], S[t]')
		for (m = 0; m < h_dim; ++m) {
			for (n = 0; n < o_dim; ++n) {
				// Outer product
				dLdV[m][n] += S[t][m] * delta_o[t][n];
			}
		}

		// Update delta_t = V' dot delta_o[t] * (1 - S[t]^2)
		// HxO * Ox1 .* Hx1
		for (m = 0; m < h_dim; ++m) {
			delta_t[m] = 0;
			for (n = 0; n < o_dim; ++n) {
				delta_t[m] += V[m][n] * delta_o[t][n];
			}
			delta_t[m] *= (1 - S[t][m] * S[t][m]);
		}

		// BPTT: From t to 0, S[-1] = [0]
		int bptt_min = t - bptt_truncate_len < 0 ? 0 : t - bptt_truncate_len;
		for (bptt_t = t; bptt_t >= bptt_min; bptt_t--)
		{
			// Update dLdW += outer(delta_t, S[t-1])
			if (bptt_t - 1 >= 0) {
				for (m = 0; m < h_dim; ++m) {
					for (n = 0; n < h_dim; ++n) {
						dLdW[m][n] += S[bptt_t - 1][m] * delta_t[n];
					}
				}
			}

			// Update dLdU[x[bptt_step]] += delta_t
			for (m = 0; m < i_dim; ++m) {
				for (n = 0; n < h_dim; ++n) {
					dLdU[m][n] += X[bptt_t][m] * delta_t[n];
				}
			}

			// Update delta_t = W' dot delta_o[t] * (1 - S[t-1]^2)
			// HxO * Ox1 .* Hx1
			for (m = 0; m < h_dim; ++m)
				temp_delta_t[m] = delta_t[m];

			for (m = 0; m < h_dim; ++m) {
				delta_t[m] = 0;
				for (n = 0; n < h_dim; ++n) {
					delta_t[m] += temp_delta_t[n] * W[m][n];
				}
				if (bptt_t - 1 >= 0)
					delta_t[m] *= (1 - S[bptt_t - 1][m] * S[bptt_t - 1][m]);
			}
		}
	}

	free_2d(delta_o, t_dim);
	free(delta_t);
	free(temp_delta_t);
}

void RNN_SGD(
    RNN_t *RNN_storage,
    Matrix_t *input_matrix,
    Matrix_t *expected_output_matrix,
    Matrix_t *predicted_output_matrix,
    math_t learning_rate
) {

	math_t **U = RNN_storage->input_weight_matrix->data;	// IxH
	math_t **V = RNN_storage->output_weight_matrix->data;	// HxO
	math_t **W = RNN_storage->internal_weight_matrix->data;	// HxH

	math_t **dLdU = RNN_storage->input_weight_gradient->data;	// IxH
	math_t **dLdV = RNN_storage->output_weight_gradient->data;	// HxO
	math_t **dLdW = RNN_storage->internel_weight_gradient->data; // HxH

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
			U[m][n] -= learning_rate * dLdU[m][n];

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
    RNN_t *RNN_storage,
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
    RNN_t *RNN_storage,
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

math_t internal_squash_func(math_t value) {
	return tanh(value);
}

math_t output_squash_derivative(
    math_t predicted_output,
    math_t expected_output
) {
	// The derivative of squared error (O - Y)
	return 2 * (predicted_output - expected_output);
}

math_t gate_squash_func(math_t value) {
	return sigmoid(value);
}

math_t output_squash_func(math_t value) {
	return value;
}

math_t sigmoid(math_t value) {
	return 2.0 / (1 + exp(-2 * value)) - 1;
}
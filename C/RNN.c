#include "RNN.h"

void TrainSet_init(TrainSet_t *train_set, int num_matrix) {
	train_set->num_matrix = num_matrix;
	train_set->input_matrix_list =
	    (Matrix_t **) malloc(num_matrix * sizeof(Matrix_t *));
	train_set->output_matrix_list =
	    (Matrix_t **) malloc(num_matrix * sizeof(Matrix_t *));
}

void TrainSet_destroy(TrainSet_t *train_set) {
	int i;
	for (i = 0; i < train_set->num_matrix; ++i) {
		matrix_free(train_set->input_matrix_list[i]);
		matrix_free(train_set->output_matrix_list[i]);
	}
	free(train_set->input_matrix_list);
	free(train_set->output_matrix_list);
}

void RNN_init(RNN_t *RNN_storage) {
	int input_vector_len = RNN_storage->input_vector_len;
	int output_vector_len = RNN_storage->output_vector_len;
	int hidden_layer_vector_len = RNN_storage->hidden_layer_vector_len;

	RNN_storage->input_weight_matrix
	    = matrix_create(input_vector_len, hidden_layer_vector_len);
	if (!RNN_storage->input_weight_matrix) {
		printf("input_weight_matrix init error\n");
		exit(69);
	}

	RNN_storage->output_weight_matrix
	    = matrix_create(hidden_layer_vector_len, output_vector_len);
	if (!RNN_storage->output_weight_matrix) {
		printf("output_weight_matrix init error\n");
		exit(69);
	}

	RNN_storage->internal_weight_matrix
	    = matrix_create(hidden_layer_vector_len, hidden_layer_vector_len);
	if (!RNN_storage->internal_weight_matrix) {
		printf("internal_weight_matrix init error\n");
		exit(69);
	}

	unsigned int seed = 1;
	matrix_random_with_seed(
	    RNN_storage->input_weight_matrix,
	    -sqrt(1 / input_vector_len),
	    sqrt(1 / input_vector_len),
	    &seed);
	matrix_random_with_seed(
	    RNN_storage->output_weight_matrix,
	    -sqrt(1 / hidden_layer_vector_len),
	    sqrt(1 / hidden_layer_vector_len),
	    &seed);
	matrix_random_with_seed(
	    RNN_storage->internal_weight_matrix,
	    -sqrt(1 / hidden_layer_vector_len),
	    sqrt(1 / hidden_layer_vector_len),
	    &seed);

	RNN_storage->internal_state_matrix = matrix_create(0, 0);
}

void RNN_destroy(RNN_t *RNN_storage) {
	matrix_free(RNN_storage->input_weight_matrix);
	matrix_free(RNN_storage->output_weight_matrix);
	matrix_free(RNN_storage->internal_weight_matrix);
	matrix_free(RNN_storage->internal_state_matrix);
	free(RNN_storage);
}

void RNN_forward_propagation(
    RNN_t *RNN_storage,
    Matrix_t *input_matrix,	// TxI
    Matrix_t *output_matrix	// TxO
) {
	int i_dim = RNN_storage->input_vector_len;
	int o_dim = RNN_storage->output_vector_len;
	int t_dim = input_matrix->m;
	int h_dim = RNN_storage->hidden_layer_vector_len;

	matrix_resize(
	    RNN_storage->internal_state_matrix,
	    t_dim,
	    RNN_storage->hidden_layer_vector_len);

	math_t **X = input_matrix->data;
	math_t **O = output_matrix->data;

	math_t **S = RNN_storage->internal_state_matrix->data;	// TxH
	math_t **U = RNN_storage->input_weight_matrix->data;	// IxH
	math_t **V = RNN_storage->output_weight_matrix->data;	// HxO
	math_t **W = RNN_storage->internal_weight_matrix->data;	// HxH

	int m, n, r, t;

	for (m = 0; m < t_dim; ++m) {
		for (n = 0; n < h_dim; ++n) {
			S[m][n] = 0.0;
		}
	}

	// S[0] = X[0]*U as S[-1] = [0]
	// 1xH =  1xI*IxH
	for (n = 0; n < h_dim; ++n) {
		for (r = 0; r < i_dim; ++r) {
			S[0][n] += X[0][r] * U[r][n];
		}
	}
	for (n = 0; n < h_dim; ++n) {
		S[0][n] = internal_squash_func(S[0][n]);
	}

	// S[t] = X[t]*U + S[t-1]*W
	// 1xH =  1xI*IxH + 1xH*HxH
	for (t = 1; t < t_dim; ++t) {
		for (n = 0; n < h_dim; ++n) {
			for (r = 0; r < i_dim; ++r) {
				// S[t] = X[t]*U
				S[t][n] += X[t][r] * U[r][n];
			}
			for (r = 0; r < h_dim; ++r) {
				// S[t] = X[t]*U + S[t-1]*W
				S[t][n] += S[t - 1][r] * W[r][n];
			}
			S[t][n] = internal_squash_func(S[t][n]);
		}
	}


	// O[t] = S[t] * V
	// 1xO = 1xH * HxO
	math_t *temp_vector = (math_t *) malloc(o_dim * sizeof(math_t));
	for (t = 0; t < t_dim; ++t) {
		for (n = 0; n < o_dim; ++n) {
			temp_vector[n] = 0.0;
			for (r = 0; r < h_dim; ++r) {
				// O[t] = S[t] * V
				temp_vector[n] += S[t][r] * V[r][n];
			}
		}
		output_squash_func(temp_vector, O[t], o_dim);
	}

	free(temp_vector);
}

// Cross entropy loss
math_t RNN_loss_calculation(
    RNN_t *RNN_storage,
    Matrix_t *predicted_output_matrix,	// TxO
    Matrix_t *expected_output_matrix	// TxO
) {
	math_t total_loss = 0.0, log_term = 0.0, delta;

	int t_dim = predicted_output_matrix->m;
	int o_dim = RNN_storage->output_vector_len;

	// printf("----------------\n");
	// matrix_print(predicted_output_matrix);

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
    Matrix_t *expected_output_matrix,	// TxO
    Matrix_t *input_weight_gradient,	// dLdU IxH
    Matrix_t *output_weight_gradient,	// dLdV HxO
    Matrix_t *internel_weight_gradient	// dLdW HxH
) {
	int t_dim = input_matrix->m;
	int i_dim = RNN_storage->input_vector_len;
	int o_dim = RNN_storage->output_vector_len;
	int h_dim = RNN_storage->hidden_layer_vector_len;

	int bptt_truncate_len = RNN_storage->bptt_truncate_len;

	math_t **delta_o
	    = create_2d(t_dim, o_dim);
	math_t *delta_t
	    = (math_t *) malloc(h_dim * sizeof(math_t));
	math_t *temp_delta_t
	    = (math_t *) malloc(h_dim * sizeof(math_t));

	math_t **S = RNN_storage->internal_state_matrix->data;	// TxH
	math_t **V = RNN_storage->output_weight_matrix->data;	// HxO
	math_t **W = RNN_storage->internal_weight_matrix->data;	// HxH

	math_t **dLdU = input_weight_gradient->data;	// IxH
	math_t **dLdV = output_weight_gradient->data;	// HxO
	math_t **dLdW = internel_weight_gradient->data; // HxH

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
				// if (X[bptt_t][m] == 1) {
				// 	for (n = 0; n < h_dim; ++n) {
				// 		dLdU[m][n] += delta_t[n];
				// 	}
				// 	break;
				// }
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
    Matrix_t *input_weight_gradient,
    Matrix_t *output_weight_gradient,
    Matrix_t *internel_weight_gradient,
    math_t learning_rate
) {

	math_t **U = RNN_storage->input_weight_matrix->data;	// IxH
	math_t **V = RNN_storage->output_weight_matrix->data;	// HxO
	math_t **W = RNN_storage->internal_weight_matrix->data;	// HxH

	math_t **dLdU = input_weight_gradient->data;	// IxH
	math_t **dLdV = output_weight_gradient->data;	// HxO
	math_t **dLdW = internel_weight_gradient->data; // HxH

	int i_dim = RNN_storage->input_vector_len;
	int o_dim = RNN_storage->output_vector_len;
	int h_dim = RNN_storage->hidden_layer_vector_len;

	RNN_forward_propagation(
	    RNN_storage,
	    input_matrix,
	    predicted_output_matrix
	);

	RNN_BPTT(
	    RNN_storage,
	    input_matrix,				// TxI
	    predicted_output_matrix,	// TxO
	    expected_output_matrix,		// TxO
	    input_weight_gradient,		// dLdU IxH
	    output_weight_gradient,		// dLdV HxO
	    internel_weight_gradient	// dLdW HxH
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
    TrainSet_t *train_set,
    Matrix_t *predicted_output_matrix,
    Matrix_t *input_weight_gradient,
    Matrix_t *output_weight_gradient,
    Matrix_t *internel_weight_gradient,
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
			RNN_Gradient_check(
			    RNN_storage,
			    train_set,
			    predicted_output_matrix,
			    input_weight_gradient,
			    output_weight_gradient,
			    internel_weight_gradient,
			    1e-3,
			    1e-2,
			    0
			);
			RNN_storage->bptt_truncate_len = old_bptt_truncate_len;
			printf("average loss at epoch: %10d = %10.10lf LR: %lf\n",
			       e, current_total_loss / num_train, learning_rate);
		}

		for (t = 0; t < num_train; ++t) {
			input_matrix = train_set->input_matrix_list[t];
			expected_output_matrix = train_set->output_matrix_list[t];

			RNN_SGD(
			    RNN_storage,
			    input_matrix,
			    expected_output_matrix,
			    predicted_output_matrix,
			    input_weight_gradient,
			    output_weight_gradient,
			    internel_weight_gradient,
			    learning_rate
			);

		}

	}
}

void RNN_Gradient_check(
    RNN_t *RNN_storage,
    TrainSet_t *train_set,
    Matrix_t *predicted_output_matrix,
    Matrix_t *input_weight_gradient,
    Matrix_t *output_weight_gradient,
    Matrix_t *internel_weight_gradient,
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

	math_t **dLdU = input_weight_gradient->data;	// IxH
	math_t **dLdV = output_weight_gradient->data;	// HxO
	math_t **dLdW = internel_weight_gradient->data; // HxH

	RNN_forward_propagation(
	    RNN_storage,
	    input_matrix,
	    predicted_output_matrix
	);

	RNN_BPTT(
	    RNN_storage,
	    input_matrix,				// TxI
	    predicted_output_matrix,	// TxO
	    expected_output_matrix,		// TxO
	    input_weight_gradient,		// dLdU IxH
	    output_weight_gradient,		// dLdV HxO
	    internel_weight_gradient	// dLdW HxH
	);

	int i, m, n;
	math_t old_model_param;
	math_t total_loss_plus, total_loss_minus;
	math_t estimated_gradient, calculated_gradient;
	math_t relative_gradient_error;

	Matrix_t *testing_model_list[] = {
		input_weight_gradient,		// U
		output_weight_gradient,		// V
		internel_weight_gradient	// W
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
					exit(1);
				}
			}
		}
	}
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
	//matrix_print(predicted_output_matrix);
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

void output_squash_func(math_t *vector, math_t *result, int dim) {
	int i;
	for (i = 0; i < dim; ++i) {
		result[i] = vector[i];
	}
}

math_t sigmoid(math_t value) {
	return 2.0 / (1 + exp(-2 * value)) - 1;
}
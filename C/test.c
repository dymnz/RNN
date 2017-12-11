#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common_math.h"
#include "RNN.h"
#include "file_process.h"

#define FILE_NAME_LENGTH 255

void file_prepare(
    char train_file[],
    char test_file[],
    char loss_file[],
    char result_file[],
    char train_file_name[],
    char test_file_name[],
    char loss_file_name[],
    char result_file_name[]
) {
	char file_postfix[] = ".txt";
	char output_file_prefix[] = "./test_data/output/";
	char input_file_prefix[] = "./test_data/input/";

	char train_file_prefix[] = "exp_";
	char test_file_prefix[] = "exp_";
	char loss_file_prefix[] = "loss_";
	char result_file_prefix[] = "res_";

	strcat(train_file, input_file_prefix);
	strcat(test_file, input_file_prefix);
	strcat(loss_file, output_file_prefix);
	strcat(result_file, output_file_prefix);

	strcat(train_file, train_file_prefix);
	strcat(test_file, test_file_prefix);
	strcat(loss_file, loss_file_prefix);
	strcat(result_file, result_file_prefix);

	strcat(train_file, train_file_name);
	strcat(test_file, test_file_name);
	strcat(loss_file, loss_file_name);
	strcat(result_file, result_file_name);

	strcat(train_file, file_postfix);
	strcat(test_file, file_postfix);
	strcat(loss_file, file_postfix);
	strcat(result_file, file_postfix);
}

void uniform_random_with_seed_test() {
	int round = 1000000;

	unsigned int seed = 10;
	math_t lower_bound = 101;
	math_t upper_bound = 1100;

	int array_size = (int)(upper_bound - lower_bound) + 1;
	int *array = (int *) malloc(array_size * sizeof(math_t));

	int i;
	for (i = 0; i < round; ++i) {
		math_t rand_num =
		    uniform_random_with_seed(lower_bound, upper_bound, &seed);
		int index = (int)rand_num - (int)lower_bound;
		++array[index];
	}

	for (i = 0; i < array_size; ++i) {
		printf("%5d: %5d\n", i + (int)lower_bound, array[i]);
	}
}
void matrix_prepare(Matrix_t **m_ptr, int m, int n, math_t *data) {

	*m_ptr = matrix_create(m, n);
	Matrix_t *matrix = *m_ptr;
	if (!matrix)
		exit(69);

	for (m = 0; m < matrix->m; ++m)
		for (n = 0; n < matrix->n; ++n)
			matrix->data[m][n] = data[m * matrix->n + n];
}

void RNN_FP_test() {
	int T = 4, I = 4;
	int H = 3;
	math_t data_in[] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};

	Matrix_t *input_matrix, *output_matrix;
	matrix_prepare(&input_matrix, T, I, data_in);
	output_matrix = matrix_create(T, I);

	printf("-------------input_matrix\n");
	matrix_print(input_matrix);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));

	RNN_storage->input_vector_len = I;

	RNN_storage->output_vector_len = I;
	RNN_storage->hidden_layer_vector_len = H;
	RNN_storage->bptt_truncate_len = 4;

	RNN_init(RNN_storage);

	printf("-------------input_weight_matrix\n");
	matrix_print(RNN_storage->input_weight_matrix);
	printf("-------------output_weight_matrix\n");
	matrix_print(RNN_storage->output_weight_matrix);
	printf("-------------internal_weight_matrix\n");
	matrix_print(RNN_storage->internal_weight_matrix);

	RNN_forward_propagation(RNN_storage, input_matrix, output_matrix);
	printf("-------------internal_state_matrix\n");
	matrix_print(RNN_storage->internal_state_matrix);
	printf("-------------output_matrix\n");
	matrix_print(output_matrix);
}

void RNN_Loss_test() {
	int T = 4, I = 4;
	int H = 3, O = 4;

	// hell
	math_t data_in[] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0
	};

	// ello
	math_t data_out[] = {
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};

	Matrix_t *input_matrix, *expected_output_matrix, *predicted_output_matrix;
	matrix_prepare(&input_matrix, T, I, data_in);
	matrix_prepare(&expected_output_matrix, T, O, data_out);
	predicted_output_matrix = matrix_create(T, O);

	printf("-------------input_matrix\n");
	matrix_print(input_matrix);
	printf("-------------expected_output_matrix\n");
	matrix_print(expected_output_matrix);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));

	RNN_storage->input_vector_len = I;

	RNN_storage->output_vector_len = O;
	RNN_storage->hidden_layer_vector_len = H;
	RNN_storage->bptt_truncate_len = 4;

	RNN_init(RNN_storage);
	RNN_forward_propagation(RNN_storage, input_matrix, predicted_output_matrix);

	printf("-------------predicted_output_matrix\n");
	matrix_print(predicted_output_matrix);

	math_t total_loss = RNN_loss_calculation(
	                        RNN_storage,
	                        predicted_output_matrix,
	                        expected_output_matrix);
	printf("total_loss: %lf\n", total_loss);
	printf("expected_loss: %lf\n", log(I));

	RNN_destroy(RNN_storage);
	matrix_free(input_matrix);
	matrix_free(expected_output_matrix);
	matrix_free(predicted_output_matrix);
}

void RNN_BPTT_test() {
	int T = 5, I = 4;
	int H = 4, O = 4;

	// hell
	math_t data_in[] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};

	// ello
	math_t data_out[] = {
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 0, 0, 0
	};

	Matrix_t *input_matrix, *expected_output_matrix, *predicted_output_matrix;
	matrix_prepare(&input_matrix, T, I, data_in);
	matrix_prepare(&expected_output_matrix, T, I, data_out);
	predicted_output_matrix = matrix_create(T, I);

	printf("-------------input_matrix\n");
	matrix_print(input_matrix);
	printf("-------------expected_output_matrix\n");
	matrix_print(expected_output_matrix);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));

	RNN_storage->input_vector_len = I;

	RNN_storage->output_vector_len = I;
	RNN_storage->hidden_layer_vector_len = H;
	RNN_storage->bptt_truncate_len = 4;

	RNN_init(RNN_storage);
	RNN_forward_propagation(RNN_storage, input_matrix, predicted_output_matrix);

	printf("-------------predicted_output_matrix\n");
	matrix_print(predicted_output_matrix);

	math_t total_loss = RNN_loss_calculation(
	                        RNN_storage,
	                        predicted_output_matrix,
	                        expected_output_matrix);
	printf("total_loss: %lf\n", total_loss);
	printf("expected_loss: %lf\n", log(I));

	Matrix_t *input_weight_gradient;
	Matrix_t *output_weight_gradient;
	Matrix_t *internel_weight_gradient;
	input_weight_gradient = matrix_create(I, H);
	output_weight_gradient = matrix_create(H, O);
	internel_weight_gradient = matrix_create(H, H);

	printf("RNN_BPTT\n");

	RNN_BPTT(
	    RNN_storage,
	    input_matrix,
	    predicted_output_matrix,
	    expected_output_matrix,
	    input_weight_gradient,
	    output_weight_gradient,
	    internel_weight_gradient
	);
	printf("-------------input_weight_gradient\n");
	matrix_print(input_weight_gradient);
	printf("-------------output_weight_gradient\n");
	matrix_print(output_weight_gradient);
	printf("-------------internel_weight_gradient\n");
	matrix_print(internel_weight_gradient);

	RNN_destroy(RNN_storage);
	matrix_free(input_matrix);
	matrix_free(expected_output_matrix);
	matrix_free(predicted_output_matrix);
	matrix_free(input_weight_gradient);
	matrix_free(output_weight_gradient);
	matrix_free(internel_weight_gradient);
}

void RNN_Train_test() {
	int T = 5, I = 4;
	int H = 4, O = 4;

	math_t initial_learning_rate = 0.005;
	int max_epoch = 10000;
	int print_loss_interval = 1000;

	// hell
	math_t data_in[] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};
	// ello
	math_t data_out[] = {
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 0, 0, 0
	};
	Matrix_t *input_matrix, *expected_output_matrix;
	matrix_prepare(&input_matrix, T, I, data_in);
	matrix_prepare(&expected_output_matrix, T, I, data_out);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));
	RNN_storage->input_vector_len = I;
	RNN_storage->output_vector_len = I;
	RNN_storage->hidden_layer_vector_len = H;
	RNN_storage->bptt_truncate_len = 4;
	RNN_init(RNN_storage);

	// Storage for RNN_train()
	Matrix_t *input_weight_gradient;
	Matrix_t *output_weight_gradient;
	Matrix_t *internel_weight_gradient;
	Matrix_t *predicted_output_matrix;
	input_weight_gradient = matrix_create(I, H);
	output_weight_gradient = matrix_create(H, O);
	internel_weight_gradient = matrix_create(H, H);

	// Prepare test
	TrainSet_t *train_set = (TrainSet_t *) malloc(sizeof(TrainSet_t));
	TrainSet_init(train_set, 1);
	train_set->input_matrix_list[0] = input_matrix;
	train_set->output_matrix_list[0] = expected_output_matrix;

	train_set->output_max_m = T;
	predicted_output_matrix = matrix_create(train_set->output_max_m, O);

	RNN_train(
	    RNN_storage,
	    train_set,
	    predicted_output_matrix,
	    input_weight_gradient,
	    output_weight_gradient,
	    internel_weight_gradient,
	    initial_learning_rate,
	    max_epoch,
	    print_loss_interval
	);

	printf("Symbol: ['h', 'e', 'l', 'o']\n");
	printf("-------------input_matrix\n");
	matrix_print(input_matrix);
	printf("-------------expected_output_matrix\n");
	matrix_print(expected_output_matrix);
	printf("-------------predicted_output_matrix\n");
	RNN_Predict(
	    RNN_storage,
	    input_matrix,
	    predicted_output_matrix
	);


	TrainSet_destroy(train_set);
	RNN_destroy(RNN_storage);
	matrix_free(predicted_output_matrix);
	matrix_free(input_weight_gradient);
	matrix_free(output_weight_gradient);
	matrix_free(internel_weight_gradient);
}

void read_set_from_file_test() {
	char train_file[] = "./test_data/exp.txt";
	char loss_file[] = "./test_data/loss.txt";
	char result_file[] = "./test_data/res.txt";

	int H = 2;
	int bptt_truncate_len = 10;

	math_t initial_learning_rate = 0.005;
	int max_epoch = 100000;
	int print_loss_interval = 100;

	TrainSet_t *train_set = read_set_from_file(train_file);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));
	RNN_storage->input_vector_len = train_set->input_n;
	RNN_storage->output_vector_len = train_set->output_n;
	RNN_storage->hidden_layer_vector_len = H;
	RNN_storage->bptt_truncate_len = bptt_truncate_len;
	RNN_init(RNN_storage);
	printf("%d %d %d %d\n",
	       train_set->input_n, train_set->output_n,
	       train_set->input_max_m,
	       train_set->output_max_m);

	// Storage for RNN_train()
	Matrix_t *input_weight_gradient;
	Matrix_t *output_weight_gradient;
	Matrix_t *internel_weight_gradient;
	Matrix_t *predicted_output_matrix;
	input_weight_gradient = matrix_create(train_set->input_n, H);
	output_weight_gradient = matrix_create(H, train_set->output_n);
	internel_weight_gradient = matrix_create(H, H);

	predicted_output_matrix = matrix_create(
	                              train_set->output_max_m,
	                              train_set->output_n);

	RNN_train(
	    RNN_storage,
	    train_set,
	    predicted_output_matrix,
	    input_weight_gradient,
	    output_weight_gradient,
	    internel_weight_gradient,
	    initial_learning_rate,
	    max_epoch,
	    print_loss_interval
	);


	FILE *pRes = fopen(result_file, "a");
	fprintf(pRes, "%d\n", train_set->num_matrix);
	fclose(pRes);

	FILE *pLoss = fopen(loss_file, "a");



	int i;
	math_t loss, total_loss = 0.0f;
	for (i = 0; i < train_set->num_matrix; ++i) {
		RNN_Predict(
		    RNN_storage,
		    train_set->input_matrix_list[i],
		    predicted_output_matrix
		);

		loss = RNN_loss_calculation(
		           RNN_storage,
		           predicted_output_matrix,
		           train_set->output_matrix_list[i]);
		fprintf(pLoss, "%lf\n", loss);
		total_loss += loss;
		write_matrix_to_file(result_file, train_set->input_matrix_list[i]);
		write_matrix_to_file(result_file, predicted_output_matrix);
	}
	printf("average loss: %lf\n", total_loss / train_set->num_matrix);


	TrainSet_destroy(train_set);
	RNN_destroy(RNN_storage);
	matrix_free(predicted_output_matrix);
	matrix_free(input_weight_gradient);
	matrix_free(output_weight_gradient);
	matrix_free(internel_weight_gradient);
}



void RNN_cross_valid() {
	char train_file[FILE_NAME_LENGTH];
	char test_file[FILE_NAME_LENGTH];
	char loss_file[FILE_NAME_LENGTH];
	char result_file[FILE_NAME_LENGTH];

	char train_file_name[] = "2_1_CT5";
	char test_file_name[] = "10_2";
	char loss_file_name[] = "10_2";
	char result_file_name[] = "10_2";

	file_prepare(
	    train_file,
	    test_file,
	    loss_file,
	    result_file,
	    train_file_name,
	    test_file_name,
	    loss_file_name,
	    result_file_name
	);

	int H = 3;
	int bptt_truncate_len = 4;

	math_t initial_learning_rate = 0.001;
	int max_epoch = 500000;
	int print_loss_interval = 1000;

	TrainSet_t *train_set = read_set_from_file(train_file);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));
	RNN_storage->input_vector_len = train_set->input_n;
	RNN_storage->output_vector_len = train_set->output_n;
	RNN_storage->hidden_layer_vector_len = H;
	RNN_storage->bptt_truncate_len = bptt_truncate_len;
	RNN_init(RNN_storage);
	printf("%d %d %d %d\n",
	       train_set->input_n, train_set->output_n,
	       train_set->input_max_m,
	       train_set->output_max_m);

	// Storage for RNN_train()
	Matrix_t *input_weight_gradient;
	Matrix_t *output_weight_gradient;
	Matrix_t *internel_weight_gradient;
	Matrix_t *predicted_output_matrix;
	input_weight_gradient = matrix_create(train_set->input_n, H);
	output_weight_gradient = matrix_create(H, train_set->output_n);
	internel_weight_gradient = matrix_create(H, H);

	predicted_output_matrix = matrix_create(
	                              train_set->output_max_m,
	                              train_set->output_n);

	RNN_train(
	    RNN_storage,
	    train_set,
	    predicted_output_matrix,
	    input_weight_gradient,
	    output_weight_gradient,
	    internel_weight_gradient,
	    initial_learning_rate,
	    max_epoch,
	    print_loss_interval
	);

	TrainSet_destroy(train_set);
	train_set = read_set_from_file(test_file);

	FILE *pRes = fopen(result_file, "w");
	fprintf(pRes, "%d\n", train_set->num_matrix);
	fclose(pRes);

	FILE *pLoss = fopen(loss_file, "w");

	int i;
	math_t loss, total_loss = 0.0f;
	for (i = 0; i < train_set->num_matrix; ++i) {
		RNN_Predict(
		    RNN_storage,
		    train_set->input_matrix_list[i],
		    predicted_output_matrix
		);

		loss = RNN_loss_calculation(
		           RNN_storage,
		           predicted_output_matrix,
		           train_set->output_matrix_list[i]);
		fprintf(pLoss, "%lf\n", loss);
		total_loss += loss;
		write_matrix_to_file(result_file, train_set->input_matrix_list[i]);
		write_matrix_to_file(result_file, predicted_output_matrix);
	}
	printf("average loss: %lf\n", total_loss / train_set->num_matrix);


	TrainSet_destroy(train_set);
	RNN_destroy(RNN_storage);
	matrix_free(predicted_output_matrix);
	matrix_free(input_weight_gradient);
	matrix_free(output_weight_gradient);
	matrix_free(internel_weight_gradient);
}

int main()
{

	//RNN_FP_test();
	//RNN_Loss_test();
	//RNN_BPTT_test();
	//RNN_Train_test();
	//read_set_from_file_test();
	RNN_cross_valid();
	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common_math.h"
#include "RNN.h"
#include "file_process.h"
#include "util.h"

#define DEFAULT_RAND_SEED 5

#define DEFAULT_TRAIN_FILE_NAME "debug"
#define DEFAULT_TEST_FILE_NAME "debug"

unsigned int rand_seed = DEFAULT_RAND_SEED;
char train_file_name_arg[FILE_NAME_LENGTH] =
    DEFAULT_TRAIN_FILE_NAME;
char test_file_name_arg[FILE_NAME_LENGTH] =
    DEFAULT_TEST_FILE_NAME;

// HHHH / SEED
//   11 /    8 : 0.208137

int hidden_cell_num = 4;
math_t initial_learning_rate = 0.001;
int max_epoch = 100;
int print_loss_interval = 10;
int gradient_check_interval = 10;

int RNN_model_training_example() {
	printf("RNN_model_training_example\n");

	/*
	File I/O param
	*/
	char train_file_name[FILE_NAME_LENGTH]  = {0};
	char test_file_name[FILE_NAME_LENGTH] = {0};
	char loss_file_name[FILE_NAME_LENGTH] = {0};
	char result_file_name[FILE_NAME_LENGTH] = {0};

	strcat(train_file_name, "exp_");
	strcat(train_file_name, train_file_name_arg);

	strcat(test_file_name, "exp_");
	strcat(test_file_name, test_file_name_arg);

	strcat(loss_file_name, "loss_");
	strcat(loss_file_name, test_file_name_arg);

	strcat(result_file_name, "res_");
	strcat(result_file_name, test_file_name_arg);

	char train_file[FILE_NAME_LENGTH] = {0};
	char test_file[FILE_NAME_LENGTH] = {0};
	char loss_file[FILE_NAME_LENGTH] = {0};
	char result_file[FILE_NAME_LENGTH] = {0};

	IO_file_prepare(
	    train_file,
	    test_file,
	    loss_file,
	    result_file,
	    train_file_name,
	    test_file_name,
	    loss_file_name,
	    result_file_name
	);

	/*
	Storage prepare
	*/
	printf("Working on training file...\n");
	DataSet_t *train_set = read_set_from_file(train_file);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));
	RNN_init(
	    RNN_storage,
	    train_set->input_n,
	    train_set->output_n,
	    hidden_cell_num,
	    rand_seed
	);
	printf(" - RNN paramerter - \n");
	printf("Input vector length: %d\n", train_set->input_n);
	printf("Output vector length: %d\n", train_set->output_n);
	printf("Hidden cell num: %d\n", hidden_cell_num);
	printf("Rand seed : %u\n", rand_seed);
	printf("----------------\n");

	// Storage for RNN_train()
	Matrix_t *predicted_output_matrix;
	predicted_output_matrix = matrix_create(
	                              train_set->output_max_m,
	                              train_set->output_n);

	/*
	Start training with training file
	*/
	printf("Start training. Max epoch: %d Initital learning rate: % lf\n",
	       max_epoch, initial_learning_rate);
	int epoch;
	epoch = RNN_train(
	    RNN_storage,
	    train_set,
	    predicted_output_matrix,
	    initial_learning_rate,
	    max_epoch,
	    print_loss_interval,
	    gradient_check_interval
	);

	/*
	Testing file forward propagation
	*/
	DataSet_destroy(train_set);

	printf("Working on testing file...\n");
	train_set = read_set_from_file(test_file);
	
	matrix_free(predicted_output_matrix);
	predicted_output_matrix = matrix_create(
	                              train_set->output_max_m,
	                              train_set->output_n);

	FILE *pRes = fopen(result_file, "w");
	fprintf(pRes, " %d\n", train_set->num_matrix);
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
		fprintf(pLoss, " % lf\n", loss);
		total_loss += loss;
		write_matrix_to_file(result_file, train_set->input_matrix_list[i], "a");
		write_matrix_to_file(result_file, predicted_output_matrix, "a");
	}
	printf("average loss: % lf\n", total_loss / train_set->num_matrix);
	fclose(pLoss);

	/*
	Dump model
	*/
	char model_file_prefix[] = ". / data / model / ";
	printf("Model dump...\n");
	// Matrix_dump(
	//     "InputWeight_SEMG_2_CT5_0_BPTT4",
	//     model_file_prefix,
	//     RNN_storage->input_weight_matrix
	// );
	// Matrix_dump(
	//     "InternalWeight_SEMG_2_CT5_0_BPTT4",
	//     model_file_prefix,
	//     RNN_storage->internal_weight_matrix
	// );
	// Matrix_dump(
	//     "OutputWeight_SEMG_2_CT5_0_BPTT4",
	//     model_file_prefix,
	//     RNN_storage->output_weight_matrix
	// );

	/*
	Clean up
	*/
	DataSet_destroy(train_set);
	RNN_destroy(RNN_storage);
	matrix_free(predicted_output_matrix);

	return epoch;
}

int RNN_model_train_timed() {
	printf("RNN_model_training_example\n");

	/*
	File I/O param
	*/
	char train_file_name[FILE_NAME_LENGTH]  = {0};
	char test_file_name[FILE_NAME_LENGTH] = {0};
	char loss_file_name[FILE_NAME_LENGTH] = {0};
	char result_file_name[FILE_NAME_LENGTH] = {0};

	strcat(train_file_name, "exp_");
	strcat(train_file_name, train_file_name_arg);

	strcat(test_file_name, "exp_");
	strcat(test_file_name, test_file_name_arg);

	strcat(loss_file_name, "loss_");
	strcat(loss_file_name, test_file_name_arg);

	strcat(result_file_name, "res_");
	strcat(result_file_name, test_file_name_arg);

	char train_file[FILE_NAME_LENGTH] = {0};
	char test_file[FILE_NAME_LENGTH] = {0};
	char loss_file[FILE_NAME_LENGTH] = {0};
	char result_file[FILE_NAME_LENGTH] = {0};

	IO_file_prepare(
	    train_file,
	    test_file,
	    loss_file,
	    result_file,
	    train_file_name,
	    test_file_name,
	    loss_file_name,
	    result_file_name
	);

	/*
	Storage prepare
	*/
	printf("Working on training file...\n");
	DataSet_t *train_set = read_set_from_file(train_file);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));
	RNN_init(
	    RNN_storage,
	    train_set->input_n,
	    train_set->output_n,
	    hidden_cell_num,
	    rand_seed
	);
	printf(" - RNN paramerter - \n");
	printf("Input vector length: %d\n", train_set->input_n);
	printf("Output vector length: %d\n", train_set->output_n);
	printf("Hidden cell num: %d\n", hidden_cell_num);
	printf("Rand seed : %u\n", rand_seed);
	printf("----------------\n");

	// Storage for RNN_train()
	Matrix_t *predicted_output_matrix;
	predicted_output_matrix = matrix_create(
	                              train_set->output_max_m,
	                              train_set->output_n);

	/*
	Start training with training file
	*/
	printf("Start training. Max epoch: %d Initital learning rate: % lf\n",
	       max_epoch, initial_learning_rate);

	mytspec start_time, end_time;
	int epoch;
	get_time(start_time);
	epoch = RNN_train(
	    RNN_storage,
	    train_set,
	    predicted_output_matrix,
	    initial_learning_rate,
	    max_epoch,
	    print_loss_interval,
	    gradient_check_interval
	);
	get_time(end_time);
	printf("done!\nElapsed_time: %3.8lf\nepoch_per_second: %3.8lf\n",
	       elapsed_time(end_time, start_time),
	       (double) epoch / elapsed_time(end_time, start_time));

	DataSet_destroy(train_set);
	RNN_destroy(RNN_storage);
	matrix_free(predicted_output_matrix);

	return epoch;
}

int RNN_model_import_example() {
	printf("RNN_model_import_example\n");

	/*
	File I/O param
	*/
	char train_file_name[FILE_NAME_LENGTH]  = {0};
	char test_file_name[FILE_NAME_LENGTH] = {0};
	char loss_file_name[FILE_NAME_LENGTH] = {0};
	char result_file_name[FILE_NAME_LENGTH] = {0};

	strcat(train_file_name, "exp_");
	strcat(train_file_name, train_file_name_arg);

	strcat(test_file_name, "exp_");
	strcat(test_file_name, test_file_name_arg);

	strcat(loss_file_name, "loss_");
	strcat(loss_file_name, test_file_name_arg);

	strcat(result_file_name, "res_");
	strcat(result_file_name, test_file_name_arg);

	char train_file[FILE_NAME_LENGTH] = {0};
	char test_file[FILE_NAME_LENGTH] = {0};
	char loss_file[FILE_NAME_LENGTH] = {0};
	char result_file[FILE_NAME_LENGTH] = {0};

	IO_file_prepare(
	    train_file,
	    test_file,
	    loss_file,
	    result_file,
	    train_file_name,
	    test_file_name,
	    loss_file_name,
	    result_file_name
	);

	/*
	Storage prepare
	*/
	printf("Working on testing file...\n");
	DataSet_t *train_set = read_set_from_file(test_file);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));
	RNN_init(
	    RNN_storage,
	    train_set->input_n,
	    train_set->output_n,
	    hidden_cell_num,
	    rand_seed
	);
	printf(" - RNN paramerter - \n");
	printf("Input vector length: %d\n", train_set->input_n);
	printf("Output vector length: %d\n", train_set->output_n);
	printf("Hidden cell num: %d\n", hidden_cell_num);
	printf("----------------\n");

	// Storage for RNN_train()
	Matrix_t *predicted_output_matrix;
	predicted_output_matrix = matrix_create(
	                              train_set->output_max_m,
	                              train_set->output_n);

	/*
	Import trained model
	*/
	// printf("Import model...\n");
	// read_matrix_from_file(
	// 	". / data / model / InputWeight_SEMG_2_CT5_0_BPTT4.txt",
	// 	RNN_storage->input_weight_matrix
	// );
	// read_matrix_from_file(
	// 	". / data / model / InternalWeight_SEMG_2_CT5_0_BPTT4.txt",
	// 	RNN_storage->internal_weight_matrix
	// );
	// read_matrix_from_file(
	// 	". / data / model / OutputWeight_SEMG_2_CT5_0_BPTT4.txt",
	// 	RNN_storage->output_weight_matrix
	// );

	/*
	Testing file forward propagation
	*/
	FILE *pRes = fopen(result_file, "w");
	fprintf(pRes, " %d\n", train_set->num_matrix);
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
		fprintf(pLoss, " % lf\n", loss);
		total_loss += loss;
		write_matrix_to_file(result_file, train_set->input_matrix_list[i], "a");
		write_matrix_to_file(result_file, predicted_output_matrix, "a");
	}
	printf("average loss: % lf\n", total_loss / train_set->num_matrix);
	fclose(pLoss);

	/*
	Clean up
	*/
	DataSet_destroy(train_set);
	RNN_destroy(RNN_storage);
	matrix_free(predicted_output_matrix);

	return 0;
}

int main(int argc, char *argv[]) {

	if (argc < 9) {
		printf(
		    "Usage: ./rnn "
		    "train_file_name/"
		    "test_file_name/"
		    "hidden_cell_num/"
		    "max_epoch/"
		    "initial_learning_rate/"
		    "print_loss_interval/"
		    "gradient_check_interval/"
		    "rand_seed"
		    "\n");
		printf(
		    "Example: ./rnn "
		    "SEMG_10_CT5_0 "
		    "SEMG_10_CT5_0 "
		    "4 "
		    "10000 "
		    "0.001 "
		    "1000 "
		    "100 "
		    "4 "
		    "\n");
		exit(60);
	}

	strncpy(train_file_name_arg, argv[1], FILE_NAME_LENGTH);
	strncpy(test_file_name_arg, argv[2], FILE_NAME_LENGTH);

	hidden_cell_num = atoi(argv[3]);
	max_epoch = atoi(argv[4]);
	initial_learning_rate = atof(argv[5]);
	print_loss_interval = atoi(argv[6]);
	gradient_check_interval = atoi(argv[7]);
	rand_seed = atoi(argv[8]);


	//return RNN_model_train_timed();
	return RNN_model_training_example();
	//return RNN_model_import_example();
}


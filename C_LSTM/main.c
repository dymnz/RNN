#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common_math.h"
#include "RNN.h"
#include "file_process.h"

#define RNN_RAND_SEED 5

unsigned int seed = RNN_RAND_SEED;

int RNN_model_training_example() {
	printf("RNN_model_training_example\n");
	/*
		RNN model param
	*/
	int H = 3;
	int bptt_truncate_len = 4;

	math_t initial_learning_rate = 0.001;
	int max_epoch = 100000;
	int print_loss_interval = 20000;
    int gradient_check_interval = 100;

	/*
		File I/O param
	 */
	char train_file_name[] = "exp_SEMG_10_CT5_0";
	char test_file_name[] = "exp_SEMG_10_CT5_0";
	char loss_file_name[] = "loss_SEMG_10_CT5_0";
	char result_file_name[] = "res_SEMG_10_CT5_0";

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
	    H,
	    bptt_truncate_len,
	    seed
	);
	printf("-RNN paramerter-\n");
	printf("Input vector length: %d\n", train_set->input_n);
	printf("Output vector length: %d\n", train_set->output_n);
	printf("Hidden vector length: %d\n", H);
	printf("BPTT truncate length: %d\n", bptt_truncate_len);
	printf("----------------\n");

	// Storage for RNN_train()
	Matrix_t *predicted_output_matrix;
	predicted_output_matrix = matrix_create(
	                              train_set->output_max_m,
	                              train_set->output_n);

	/*
		Start training with training file
	*/
	printf("Start training. Max epoch: %d Initital learning rate: %lf\n",
	       max_epoch, initial_learning_rate);

	RNN_train(
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
		write_matrix_to_file(result_file, train_set->input_matrix_list[i], "a");
		write_matrix_to_file(result_file, predicted_output_matrix, "a");
	}
	printf("average loss: %lf\n", total_loss / train_set->num_matrix);
	fclose(pLoss);

	/*
		Dump model
	 */
	// char model_file_prefix[] = "./data/model/";
	// printf("Model dump...\n");
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

	return 0;
}


int RNN_model_import_example() {
	printf("RNN_model_import_example\n");
	/*
		RNN model param
	*/
	int H = 3;
	int bptt_truncate_len = 4;

	/*
		File I/O param
	 */
	char train_file_name[] = "exp_SEMG_2_CT5_0";
	char test_file_name[] = "exp_SEMG_2_CT5_0";
	char loss_file_name[] = "loss_SEMG_2_CT5_0";
	char result_file_name[] = "res_SEMG_2_CT5_0";

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
	    H,
	    bptt_truncate_len,
	    seed
	);
	printf("-RNN paramerter-\n");
	printf("Input vector length: %d\n", train_set->input_n);
	printf("Output vector length: %d\n", train_set->output_n);
	printf("Hidden vector length: %d\n", H);
	printf("BPTT truncate length: %d\n", bptt_truncate_len);
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
	// 	"./data/model/InputWeight_SEMG_2_CT5_0_BPTT4.txt", 
	// 	RNN_storage->input_weight_matrix
	// );
	// read_matrix_from_file(
	// 	"./data/model/InternalWeight_SEMG_2_CT5_0_BPTT4.txt", 
	// 	RNN_storage->internal_weight_matrix
	// );
	// read_matrix_from_file(
	// 	"./data/model/OutputWeight_SEMG_2_CT5_0_BPTT4.txt", 
	// 	RNN_storage->output_weight_matrix
	// );

	/*
		Testing file forward propagation
	 */
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
		write_matrix_to_file(result_file, train_set->input_matrix_list[i], "a");
		write_matrix_to_file(result_file, predicted_output_matrix, "a");
	}
	printf("average loss: %lf\n", total_loss / train_set->num_matrix);
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
	if (argc >= 2)
		seed = atoi(argv[1]);
	
	return RNN_model_training_example();
	//return RNN_model_import_example();
}


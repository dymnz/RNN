#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common_math.h"
#include "RNN.h"
#include "file_process.h"
#include "serial.h"

#define DEFAULT_RAND_SEED 5

#define DEFAULT_TRAIN_filename "debug"
#define DEFAULT_TEST_filename "debug"
#define DEFAULT_CROSS_filename "debug"

unsigned int rand_seed = DEFAULT_RAND_SEED;
char train_filename_arg[FILENAME_LENGTH] =
    DEFAULT_TRAIN_filename;
char test_filename_arg[FILENAME_LENGTH] =
    DEFAULT_TEST_filename;
char cross_filename_arg[FILENAME_LENGTH] =
    DEFAULT_CROSS_filename;
char serial_portname_arg[FILENAME_LENGTH] =
    DEFAULT_CROSS_filename;

int hidden_cell_num = 4;
math_t initial_learning_rate = 0.001;
int max_epoch = 100;
int cross_valid_patience = 10;
int print_loss_interval = 10;
int gradient_check_interval = 10;

int RNN_model_training_example() {
	printf("RNN_model_training_example\n");

	/*
	File I/O param
	*/
	char train_filename[FILENAME_LENGTH]  = {0};
	char test_filename[FILENAME_LENGTH] = {0};
	char cross_filename[FILENAME_LENGTH] = {0};
	char loss_filename[FILENAME_LENGTH] = {0};
	char result_filename[FILENAME_LENGTH] = {0};

	strcat(train_filename, "exp_");
	strcat(train_filename, train_filename_arg);

	strcat(test_filename, "exp_");
	strcat(test_filename, test_filename_arg);

	strcat(cross_filename, "exp_");
	strcat(cross_filename, cross_filename_arg);

	strcat(loss_filename, "loss_");
	strcat(loss_filename, test_filename_arg);

	strcat(result_filename, "res_");
	strcat(result_filename, test_filename_arg);

	char train_file[FILENAME_LENGTH] = {0};
	char test_file[FILENAME_LENGTH] = {0};
	char cross_file[FILENAME_LENGTH] = {0};
	char loss_file[FILENAME_LENGTH] = {0};
	char result_file[FILENAME_LENGTH] = {0};

	IO_file_prepare(
	    train_file,
	    test_file,
	    cross_file,
	    loss_file,
	    result_file,
	    train_filename,
	    test_filename,
	    cross_filename,
	    loss_filename,
	    result_filename
	);


	srand(rand_seed);

	/*
	Storage prepare
	*/
	printf("Working on training file...\n");
	DataSet_t *train_set = read_set_from_file(train_file);
	DataSet_t *cross_set = read_set_from_file(cross_file);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));
	RNN_init(RNN_storage,
	         train_set->input_n,
	         train_set->output_n,
	         hidden_cell_num
	        );

	printf(" - RNN paramerter - \n");
	printf("Input vector length: %d\n", train_set->input_n);
	printf("Output vector length: %d\n", train_set->output_n);
	printf("Hidden cell num: %d\n", hidden_cell_num);
	printf("Rand seed : %u\n", rand_seed);
	printf("----------------\n");

	// Storage for RNN_train()
	Matrix_t *predicted_output_matrix;
	printf("%d %d\n",  train_set->output_max_m, train_set->output_n);
	predicted_output_matrix = matrix_create(
	                              max(train_set->output_max_m, cross_set->output_max_m),
	                              train_set->output_n);

	/*
	Start training with training file
	*/
	printf("Start training. Max epoch: %d Initital learning rate: % lf\n",
	       max_epoch, initial_learning_rate);
	RNN_result_t *RNN_train_result;

	RNN_train_result = RNN_train_cross_valid(
	                       RNN_storage,
	                       train_set,
	                       cross_set,
	                       predicted_output_matrix,
	                       max_epoch,
	                       cross_valid_patience,
	                       print_loss_interval,
	                       gradient_check_interval
	                   );

	printf("Hidden: %5d\t Error: %10.10lf\n",
	       hidden_cell_num, RNN_train_result->last_training_loss);

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

	printf("RMSE outputs are multiplied by 90 for test\n");
	printf(
	    "* ./rnn %s %s %s %d %d %d %d %d\n",
	    train_filename_arg,
	    test_filename_arg,
	    cross_filename_arg,
	    hidden_cell_num,
	    max_epoch,
	    print_loss_interval,
	    gradient_check_interval,
	    rand_seed
	);
	printf("  * train loss at epoch: %10d = %10.10lf \n"
	       "  * best cross loss at epoch: %10d = %10.10lf \n",
	       RNN_train_result->ending_epoch,
	       RNN_train_result->last_training_loss,
	       RNN_train_result->best_epoch_cross,
	       RNN_train_result->best_cross_loss
	      );

	int i, r;
	math_t *loss_list;
	for (i = 0; i < train_set->num_matrix; ++i) {
		matrix_resize(
		    predicted_output_matrix,
		    train_set->input_matrix_list[i]->m,
		    train_set->output_n
		);
		RNN_Predict(
		    RNN_train_result->RNN_best_model,
		    train_set->input_matrix_list[i],
		    predicted_output_matrix
		);

		loss_list = RNN_RMSE(
		                RNN_storage,
		                predicted_output_matrix,
		                train_set->output_matrix_list[i]);

		printf("  * RMSE: ");
		for (r = 0; r < RNN_storage->o_dim; ++r) {
			printf("%8.5lf\t", loss_list[r] * 90);
			fprintf(pLoss, "%8.5lf\n", loss_list[r]);
		}
		printf("\n");
		write_matrix_to_file(result_file, train_set->input_matrix_list[i], "a");
		write_matrix_to_file(result_file, predicted_output_matrix, "a");

		free(loss_list);
	}


	fclose(pLoss);

	/*
	Dump model
	*/
	char model_file_prefix[] = "./data/model/TestModel_";
	printf("Model dump...\n");
	/* LSTM model */

	Matrix_dump("Wz", model_file_prefix, RNN_train_result->RNN_best_model->Wz);
	Matrix_dump("Wi", model_file_prefix, RNN_train_result->RNN_best_model->Wi);
	Matrix_dump("Wf", model_file_prefix, RNN_train_result->RNN_best_model->Wf);
	Matrix_dump("Wo", model_file_prefix, RNN_train_result->RNN_best_model->Wo);

	Matrix_dump("Rz", model_file_prefix, RNN_train_result->RNN_best_model->Rz);
	Matrix_dump("Ri", model_file_prefix, RNN_train_result->RNN_best_model->Ri);
	Matrix_dump("Rf", model_file_prefix, RNN_train_result->RNN_best_model->Rf);
	Matrix_dump("Ro", model_file_prefix, RNN_train_result->RNN_best_model->Ro);

	Matrix_dump("Pi", model_file_prefix, RNN_train_result->RNN_best_model->Pi);
	Matrix_dump("Pf", model_file_prefix, RNN_train_result->RNN_best_model->Pf);
	Matrix_dump("Po", model_file_prefix, RNN_train_result->RNN_best_model->Po);

	Matrix_dump("Bz", model_file_prefix, RNN_train_result->RNN_best_model->Bz);
	Matrix_dump("Bi", model_file_prefix, RNN_train_result->RNN_best_model->Bi);
	Matrix_dump("Bf", model_file_prefix, RNN_train_result->RNN_best_model->Bf);
	Matrix_dump("Bo", model_file_prefix, RNN_train_result->RNN_best_model->Bo);

	/* Output model */
	Matrix_dump("V", model_file_prefix, RNN_train_result->RNN_best_model->V);
	Matrix_dump("Bpo", model_file_prefix, RNN_train_result->RNN_best_model->Bpo);


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
	File I/O param
	*/

	char train_filename[FILENAME_LENGTH]  = {0};
	char test_filename[FILENAME_LENGTH] = {0};
	char cross_filename[FILENAME_LENGTH] = {0};
	char loss_filename[FILENAME_LENGTH] = {0};
	char result_filename[FILENAME_LENGTH] = {0};

	strcat(train_filename, "exp_");
	strcat(train_filename, train_filename_arg);

	strcat(test_filename, "exp_");
	strcat(test_filename, test_filename_arg);

	strcat(cross_filename, "exp_");
	strcat(cross_filename, cross_filename_arg);

	strcat(loss_filename, "loss_");
	strcat(loss_filename, test_filename_arg);

	strcat(result_filename, "res_");
	strcat(result_filename, test_filename_arg);

	char train_file[FILENAME_LENGTH] = {0};
	char test_file[FILENAME_LENGTH] = {0};
	char cross_file[FILENAME_LENGTH] = {0};
	char loss_file[FILENAME_LENGTH] = {0};
	char result_file[FILENAME_LENGTH] = {0};

	IO_file_prepare(
	    train_file,
	    test_file,
	    cross_file,
	    loss_file,
	    result_file,
	    train_filename,
	    test_filename,
	    cross_filename,
	    loss_filename,
	    result_filename
	);

	srand(rand_seed);

	/*
	Storage prepare
	*/
	printf("Working on testing file...\n");
	printf("reading %s\n", test_file);
	DataSet_t *train_set = read_set_from_file(test_file);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));
	RNN_init(
	    RNN_storage,
	    train_set->input_n,
	    train_set->output_n,
	    hidden_cell_num
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
	char model_file_prefix[] = "./data/model/TestModel_";
	printf("Import model...\n");
	/* LSTM model */
	Matrix_load("Wz", model_file_prefix, RNN_storage->Wz);
	Matrix_load("Wi", model_file_prefix, RNN_storage->Wi);
	Matrix_load("Wf", model_file_prefix, RNN_storage->Wf);
	Matrix_load("Wo", model_file_prefix, RNN_storage->Wo);

	Matrix_load("Rz", model_file_prefix, RNN_storage->Rz);
	Matrix_load("Ri", model_file_prefix, RNN_storage->Ri);
	Matrix_load("Rf", model_file_prefix, RNN_storage->Rf);
	Matrix_load("Ro", model_file_prefix, RNN_storage->Ro);

	Matrix_load("Pi", model_file_prefix, RNN_storage->Pi);
	Matrix_load("Pf", model_file_prefix, RNN_storage->Pf);
	Matrix_load("Po", model_file_prefix, RNN_storage->Po);

	Matrix_load("Bz", model_file_prefix, RNN_storage->Bz);
	Matrix_load("Bi", model_file_prefix, RNN_storage->Bi);
	Matrix_load("Bf", model_file_prefix, RNN_storage->Bf);
	Matrix_load("Bo", model_file_prefix, RNN_storage->Bo);

	/* Output model */
	Matrix_load("V", model_file_prefix, RNN_storage->V);
	Matrix_load("Bpo", model_file_prefix, RNN_storage->Bpo);

	/*
	Testing file forward propagation
	*/
	FILE *pRes = fopen(result_file, "w");
	fprintf(pRes, " %d\n", train_set->num_matrix);
	fclose(pRes);

	FILE *pLoss = fopen(loss_file, "w");

	int i, r;
	math_t *loss_list;
	for (i = 0; i < train_set->num_matrix; ++i) {
		matrix_resize(
		    predicted_output_matrix,
		    train_set->input_matrix_list[i]->m,
		    train_set->output_n
		);
		RNN_Predict(
		    RNN_storage,
		    train_set->input_matrix_list[i],
		    predicted_output_matrix
		);

		loss_list = RNN_RMSE(
		                RNN_storage,
		                predicted_output_matrix,
		                train_set->output_matrix_list[i]);

		printf("  * RMSE: ");
		for (r = 0; r < RNN_storage->o_dim; ++r) {
			printf("%8.5lf\t", loss_list[r] * 140);
			fprintf(pLoss, "%8.5lf\n", loss_list[r]);
		}
		printf("\n");
		write_matrix_to_file(result_file, train_set->input_matrix_list[i], "a");
		write_matrix_to_file(result_file, predicted_output_matrix, "a");

		free(loss_list);
	}


	fclose(pLoss);

	/*
	Clean up
	*/
	DataSet_destroy(train_set);
	RNN_destroy(RNN_storage);
	matrix_free(predicted_output_matrix);

	return 0;
}

int RNN_model_serial_stepFP_example() {
	printf("RNN_model_import_example\n");


	int serial_id = serial_init(serial_portname_arg);

	if (serial_id < 0) {
		printf("%s port open Error\n", serial_portname_arg);
		return -1;
	}

	/*
	File I/O param
	*/
	char train_filename[FILENAME_LENGTH]  = {0};
	char test_filename[FILENAME_LENGTH] = {0};
	char cross_filename[FILENAME_LENGTH] = {0};
	char loss_filename[FILENAME_LENGTH] = {0};
	char result_filename[FILENAME_LENGTH] = {0};

	strcat(train_filename, "exp_");
	strcat(train_filename, train_filename_arg);

	strcat(test_filename, "exp_");
	strcat(test_filename, test_filename_arg);

	strcat(cross_filename, "exp_");
	strcat(cross_filename, cross_filename_arg);

	strcat(loss_filename, "loss_");
	strcat(loss_filename, test_filename_arg);

	strcat(result_filename, "res_");
	strcat(result_filename, test_filename_arg);

	char train_file[FILENAME_LENGTH] = {0};
	char test_file[FILENAME_LENGTH] = {0};
	char cross_file[FILENAME_LENGTH] = {0};
	char loss_file[FILENAME_LENGTH] = {0};
	char result_file[FILENAME_LENGTH] = {0};

	IO_file_prepare(
	    train_file,
	    test_file,
	    cross_file,
	    loss_file,
	    result_file,
	    train_filename,
	    test_filename,
	    cross_filename,
	    loss_filename,
	    result_filename
	);

	srand(rand_seed);

	/*
	Storage prepare
	*/
	printf("Working on testing file...\n");
	printf("reading %s\n", test_file);
	DataSet_t *train_set = read_set_from_file(test_file);

	RNN_t *RNN_storage
	    = (RNN_t *) malloc(sizeof(RNN_t));
	RNN_init(
	    RNN_storage,
	    train_set->input_n,
	    train_set->output_n,
	    hidden_cell_num
	);
	printf(" - RNN paramerter - \n");
	printf("Input vector length: %d\n", train_set->input_n);
	printf("Output vector length: %d\n", train_set->output_n);
	printf("Hidden cell num: %d\n", hidden_cell_num);
	printf("----------------\n");

	// Storage for RNN_train()
	Matrix_t *predicted_output_matrix;
	predicted_output_matrix = matrix_create(
	                              2,
	                              train_set->output_n);

	Matrix_t * step_input_matrix;
	step_input_matrix = matrix_create(
	                        2,
	                        train_set->input_n);

	/*
	Import trained model
	*/
	char model_file_prefix[] = "./data/model/TestModel_";
	printf("Import model...\n");
	/* LSTM model */
	Matrix_load("Wz", model_file_prefix, RNN_storage->Wz);
	Matrix_load("Wi", model_file_prefix, RNN_storage->Wi);
	Matrix_load("Wf", model_file_prefix, RNN_storage->Wf);
	Matrix_load("Wo", model_file_prefix, RNN_storage->Wo);

	Matrix_load("Rz", model_file_prefix, RNN_storage->Rz);
	Matrix_load("Ri", model_file_prefix, RNN_storage->Ri);
	Matrix_load("Rf", model_file_prefix, RNN_storage->Rf);
	Matrix_load("Ro", model_file_prefix, RNN_storage->Ro);

	Matrix_load("Pi", model_file_prefix, RNN_storage->Pi);
	Matrix_load("Pf", model_file_prefix, RNN_storage->Pf);
	Matrix_load("Po", model_file_prefix, RNN_storage->Po);

	Matrix_load("Bz", model_file_prefix, RNN_storage->Bz);
	Matrix_load("Bi", model_file_prefix, RNN_storage->Bi);
	Matrix_load("Bf", model_file_prefix, RNN_storage->Bf);
	Matrix_load("Bo", model_file_prefix, RNN_storage->Bo);

	/* Output model */
	Matrix_load("V", model_file_prefix, RNN_storage->V);
	Matrix_load("Bpo", model_file_prefix, RNN_storage->Bpo);

	/*
	Testing file forward propagation
	*/
	union float_packet temp_packet;
	union float_packet packet_array[in_packet_num];

	int rdlen, wlen, ch;
	int init_complete = 0;
	do {
		rdlen = read(serial_id, &packet_array, in_packet_num * packet_size);	// Read one byte
		if (rdlen != in_packet_num * packet_size) {
			printf("rdlen != packet_num * packet_size\n");
			return -1;
		}

		for (ch = 0; ch < step_input_matrix->n; ++ch)
			step_input_matrix->data[0][ch] = packet_array[ch].float_value;


		if (!init_complete) {
			RNN_stepFP_init(
			    RNN_storage,
			    step_input_matrix,
			    predicted_output_matrix
			);
			init_complete = 1;
		}
		else {
			RNN_stepFP(
			    RNN_storage,
			    step_input_matrix,
			    predicted_output_matrix
			);
		}


		temp_packet.float_value = predicted_output_matrix->data[0][0];

		//printf("%f\n", temp_packet.float_value);
		///*
		wlen = write(serial_id,
		             temp_packet.byte_array,
		             out_packet_num * packet_size);
		//*/

		if (wlen != out_packet_num * packet_size) {
			printf("Error from write: %d, %d\n", wlen, errno);
			return -1;
		}
	} while (1);


	/*
	Clean up
	*/
	DataSet_destroy(train_set);
	RNN_destroy(RNN_storage);
	matrix_free(predicted_output_matrix);
	matrix_free(step_input_matrix);

	return 0;
}

int main(int argc, char *argv[]) {
	if (argc < 10) {
		printf(
		    "Usage: ./rnn "
		    "train_filename/"
		    "test_filename/"
		    "cross_filename/"
		    "hidden_cell_num/"
		    "max_epoch/"
		    "cross_valid_patience/"
		    "print_loss_interval/"
		    "gradient_check_interval/"
		    "rand_seed"
		    "\n");
		exit(60);
	}

	strncpy(train_filename_arg, argv[1], FILENAME_LENGTH);
	strncpy(test_filename_arg, argv[2], FILENAME_LENGTH);
	strncpy(cross_filename_arg, argv[3], FILENAME_LENGTH);

	hidden_cell_num = atoi(argv[4]);
	max_epoch = atoi(argv[5]);
	cross_valid_patience = atoi(argv[6]);
	print_loss_interval = atoi(argv[7]);
	gradient_check_interval = atoi(argv[8]);
	rand_seed = atoi(argv[9]);




	//return RNN_model_training_example();
	//return RNN_model_import_example();

	strncpy(serial_portname_arg, argv[10], FILENAME_LENGTH);
	return RNN_model_serial_stepFP_example();
}
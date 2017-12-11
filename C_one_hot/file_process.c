#include "file_process.h"


TrainSet_t *read_set_from_file(char *file_name) {
	FILE *pFile = fopen (file_name, "r");
	if (!pFile) {
		printf("%s read error\n", file_name);
		exit(69);
	}

	int num_matrix;
	fscanf(pFile, "%d", &num_matrix);
	printf("Reading %d matrix from %s\n", num_matrix, file_name);

	TrainSet_t *train_set = (TrainSet_t *) malloc(sizeof(TrainSet_t));
	TrainSet_init(train_set, num_matrix);

	int i, r, j, i_m, i_n, o_m, o_n;
	for (i = 0; i < num_matrix; ++i) {
		// Read input vector
		fscanf(pFile, "%d %d", &i_m, &i_n);

		if (train_set->input_max_m < i_m) {
			train_set->input_max_m = i_m;
			train_set->output_max_m = i_m;
		}

		Matrix_t *input_matrix = matrix_create(i_m, i_n);

		for (r = 0; r < i_m; ++r) {
			for (j = 0; j < i_n; ++j) {
				fscanf(pFile, "%lf", &(input_matrix->data[r][j]));
			}
		}

		// Read output vector
		fscanf(pFile, "%d %d", &o_m, &o_n);
		if (o_m != i_m) {
			printf("input/output time step mis-match\n");
			exit(47);
		}

		Matrix_t *output_matrix = matrix_create(o_m, o_n);

		for (r = 0; r < o_m; ++r) {
			for (j = 0; j < o_n; ++j) {
				fscanf(pFile, "%lf", &(output_matrix->data[r][j]));
			}
		}

		train_set->input_matrix_list[i] = input_matrix;
		train_set->output_matrix_list[i] = output_matrix;
	}
	train_set->input_n = i_n;
	train_set->output_n = o_n;

	fclose(pFile);

	return train_set;
}

void write_matrix_to_file(char *file_name, Matrix_t *matrix) {
	FILE *pFile = fopen (file_name, "w+");
	if (!pFile) {
		printf("%s write error\n", file_name);
		exit(69);
	}

	fprintf(pFile, "%d %d\n", matrix->m, matrix->n);

	int i, r;
	for (i = 0; i < matrix->m; ++i) {
		for (r = 0; r < matrix->n; ++r) {
			fprintf(pFile, "%lf\t", matrix->data[i][r]);
		}
	}

	fclose(pFile);
}


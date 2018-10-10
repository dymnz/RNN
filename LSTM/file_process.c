#include "file_process.h"

void IO_file_prepare(
    char train_file[],
    char test_file[],
    char cross_file[],
    char loss_file[],
    char result_file[],
    char train_file_name[],
    char test_file_name[],
    char cross_file_name[],
    char loss_file_name[],
    char result_file_name[]
) {
	const char file_postfix[] = ".txt";
	const char output_file_prefix[] = "./data/output/";
	const char input_file_prefix[] = "./data/input/";

	const char train_file_prefix[] = "";
	const char test_file_prefix[] = "";
	const char cross_file_prefix[] = "";
	const char loss_file_prefix[] = "";
	const char result_file_prefix[] = "";

	strcat(train_file, input_file_prefix);
	strcat(test_file, input_file_prefix);
	strcat(cross_file, input_file_prefix);
	strcat(loss_file, output_file_prefix);
	strcat(result_file, output_file_prefix);

	strcat(train_file, train_file_prefix);
	strcat(test_file, test_file_prefix);
	strcat(cross_file, cross_file_prefix);
	strcat(loss_file, loss_file_prefix);
	strcat(result_file, result_file_prefix);

	strcat(train_file, train_file_name);
	strcat(test_file, test_file_name);
	strcat(cross_file, cross_file_name);
	strcat(loss_file, loss_file_name);
	strcat(result_file, result_file_name);

	strcat(train_file, file_postfix);
	strcat(test_file, file_postfix);
	strcat(cross_file, file_postfix);
	strcat(loss_file, file_postfix);
	strcat(result_file, file_postfix);
}

void Matrix_dump(
    char model_file_name[],
    char file_directory[],
    Matrix_t *matrix
) {
	char model_file[FILE_NAME_LENGTH] = {0};
	char file_postfix[] = ".txt";

	strcat(model_file, file_directory);
	strcat(model_file, model_file_name);
	strcat(model_file, file_postfix);

	write_matrix_to_file(model_file, matrix, "w");
}

void Matrix_load(
    char model_file_name[],
    char file_directory[],
    Matrix_t *matrix
) {
	char model_file[FILE_NAME_LENGTH] = {0};
	char file_postfix[] = ".txt";

	strcat(model_file, file_directory);
	strcat(model_file, model_file_name);
	strcat(model_file, file_postfix);

	read_matrix_from_file(model_file, matrix);
}


DataSet_t *read_set_from_file(char *file_name) {
	FILE *pFile = fopen (file_name, "r");
	if (!pFile) {
		printf("%s read error\n", file_name);
		exit(69);
	}

	int num_matrix;
	fscanf(pFile, "%d", &num_matrix);
	printf("Reading %d matrix from %s\n", num_matrix, file_name);

	DataSet_t *train_set = (DataSet_t *) malloc(sizeof(DataSet_t));
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

// Matrix should be allocated already
void read_matrix_from_file(char file[], Matrix_t *matrix) {
	int m, n, r, j;

	FILE *pFile = fopen (file, "r");
	if (!pFile) {
		printf("%s read error\n", file);
		exit(69);
	}
	fscanf(pFile, "%d %d", &m, &n);

	if (matrix->m != m || matrix->n != n) {
		printf("matrix size mismatch\n");
		exit(77);
	} 

	for (r = 0; r < m; ++r) {
		for (j = 0; j < n; ++j) {
			fscanf(pFile, "%lf", &(matrix->data[r][j]));
		}
	}
}

void write_matrix_to_file(char *file_name, Matrix_t *matrix, char *file_modifier) {
	FILE *pFile = fopen (file_name, file_modifier);
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
	fprintf(pFile, "\n");

	fclose(pFile);
}


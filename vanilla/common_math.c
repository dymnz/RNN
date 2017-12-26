#include "common_math.h"


math_t uniform_random_with_seed(
    math_t lower_bound,
    math_t upper_bound,
    unsigned int *seedp
) {
	return ((math_t)rand_r(seedp) / (math_t)RAND_MAX) *
	       (upper_bound - lower_bound + 1) +
	       lower_bound;
}

void matrix_random_with_seed(
	Matrix_t *matrix,
    math_t lower_bound,
    math_t upper_bound,
    unsigned int *seedp
) {
	int m, n;
	for (m = 0; m < matrix->m; ++m)
		for (n = 0; n < matrix->n; ++n)
			matrix->data[m][n] = 
				uniform_random_with_seed(lower_bound, upper_bound, seedp);
}

math_t **create_2d(int m, int n) {
	math_t **data = (math_t **) malloc(m * sizeof(math_t *));
	if (!data) {
		exit(69);
	}

	int i;
	math_t *col_data;
	for (i = 0; i < m; ++i) {
		col_data = (math_t *) malloc(n * sizeof(math_t));
		if (!col_data)
			exit(69);

		data[i] = col_data;
	}

	return data;
}

void clear_2d(math_t **data, int m, int n) {
	int i, r;
	for (i = 0; i < m; ++i) {
		for (r = 0; r < n; ++r) {
			data[i][r] = 0.0;
		}
	}
}

Matrix_t *matrix_create(int m, int n) {
	Matrix_t *matrix = (Matrix_t *) malloc(sizeof(Matrix_t));
	if (!matrix)
		exit(69);

	matrix->m = m;
	matrix->n = n;
	matrix->data = create_2d(m, n);

	return matrix;
}

void matrix_resize(Matrix_t *matrix, int m, int n) {
	if (matrix != NULL && matrix->m == m && matrix->n == n)
		return;

	if (matrix == NULL)
		exit(77);
	
	free_2d(matrix->data, matrix->m);	

	matrix->m = m;
	matrix->n = n;
	matrix->data = create_2d(m, n);
}

void free_2d(math_t **data, int m) {
	int i;
	for (i = 0; i < m; ++i) 
		free(data[i]);

	free(data);
}

void matrix_free(Matrix_t *matrix) {
	if (!matrix || !(matrix->data))
		return;
	free_2d(matrix->data, matrix->m);
	matrix->data = NULL;
	free(matrix);
}

void matrix_print(Matrix_t *matrix) {
	int m, n;
	for (m = 0; m < matrix->m; ++m) {
		for (n = 0; n < matrix->n; ++n) {
			printf("%3.5lf  ", matrix->data[m][n]);
		}
		printf("\n");
	}
}

void softmax(math_t *vector, math_t *result, int dim) {
	int i;

	math_t denom = 0;
	for (i = 0; i < dim; ++i) {
		result[i] = exp(vector[i]);
		denom += result[i];
	}

	for (i = 0; i < dim; ++i)
		result[i] /= denom;
}


// Subtract max value from exp for numerical stability
void stable_softmax(math_t *vector, math_t *result, int dim) {
	int i;
	
	math_t max_val = 0;
	for (i = 0; i < dim; ++i)
		if (vector[i] > max_val)
			max_val = vector[i];

	math_t denom = 0;
	for (i = 0; i < dim; ++i) {
		result[i] = exp(vector[i] - max_val);
		denom += result[i];
	}

	for (i = 0; i < dim; ++i)
		result[i] /= denom;
}
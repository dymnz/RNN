#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef double math_t;
typedef struct
{
	int m;
	int n;
	math_t **data;
} Matrix_t;


math_t uniform_random_with_seed(
    math_t lower_bound,
    math_t upper_bound,
    unsigned int *seedp
);

void matrix_random_with_seed(
	Matrix_t *matrix,
    math_t lower_bound,
    math_t upper_bound,
    unsigned int *seedp
);


Matrix_t *matrix_create(int m, int n);
void matrix_free(Matrix_t *matrix);
void matrix_resize(Matrix_t *matrix, int m, int n);
void matrix_prepare(Matrix_t **m_ptr, int m, int n, math_t *data);


void free_2d(math_t **data, int m);
void clear_2d(math_t **data, int m, int n);
void print_1d(math_t *data, int m);
void print_2d(math_t **data, int m, int n);
math_t **create_2d(int m, int n);


void matrix_print(Matrix_t *matrix);

void softmax(math_t *vector, math_t *result, int dim);
void stable_softmax(math_t *vector, math_t *result, int dim);
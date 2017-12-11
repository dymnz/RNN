#pragma once
#include <stdio.h>
#include <string.h>
#include "RNN.h"
#include "common_math.h"

void file_prepare(
    char train_file[],
    char test_file[],
    char loss_file[],
    char result_file[],
    char train_file_name[],
    char test_file_name[],
    char loss_file_name[],
    char result_file_name[]
);
DataSet_t *read_set_from_file(char *file_name);
void write_matrix_to_file(char *file_name, Matrix_t *matrix, char *file_modifier);

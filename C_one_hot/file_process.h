#pragma once
#include <stdio.h>
#include "RNN.h"
#include "common_math.h"

TrainSet_t *read_set_from_file(char *file_name);
void write_matrix_to_file(char *file_name, Matrix_t *matrix);

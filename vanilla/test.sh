#!/bin/sh

echo "-----------------running seed $1" | tee -a "mult_results_$2_$3.txt"

./rnn $1 | tee -a "mult_results_$2_$3.txt"



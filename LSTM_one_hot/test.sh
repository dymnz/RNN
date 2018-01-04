#!/bin/sh

echo "thread:$1 | hidden:$2" #| tee -a "./benchmark/t$1_h$2.txt"
(time ./rnn reddit_14 reddit_14 $2 10 0.001 100 100 100000 4 $1) 2>&1 | tee -a "./benchmark/batch.txt" #"./benchmark/t$1_h$2.txt"
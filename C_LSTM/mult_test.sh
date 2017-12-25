#!/bin/sh

a=$1
while [ "$a" -le $2 ]
do 

./test.sh $a $1 $2

a=`expr $a + 1`
done

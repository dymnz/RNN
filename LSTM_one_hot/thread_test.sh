#!/bin/sh

thread=$1
hidden=$3
while [ "$thread" -le $2 ]
do 

while [ "$hidden" -le $4 ]
do
screen -dmS test$thread ./test.sh $thread $hidden
while screen -list | grep -q test$thread
do
    sleep 1
done
hidden=`expr $hidden + 10`
done

thread=`expr $thread \* 2`
done
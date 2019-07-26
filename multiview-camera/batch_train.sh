#!/bin/bash
counter=1
while [ $counter -le 60 ]
do
	python baseline_train.py -mn $counter
	sleep 1m
	((counter++))
	echo $counter
done

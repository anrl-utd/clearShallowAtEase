#!/bin/bash
counter=1
while [ $counter -le 50 ]
do
	python activeGuard_train.py -mn $counter
	sleep 5m
	((counter++))
	echo $counter
done
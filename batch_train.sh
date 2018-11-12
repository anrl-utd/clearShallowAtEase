#!/bin/bash
counter=21
while [ $counter -le 40 ]
do
	python fixedGuard_train.py -mn $counter
	sleep 1m
	((counter++))
	echo $counter
done

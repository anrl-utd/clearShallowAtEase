#!/bin/bash
counter=41
while [ $counter -le 60 ]
do
	python fixedGuard_train.py -mn $counter
	sleep 1m
	((counter++))
	echo $counter
done

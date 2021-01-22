#!/bin/env bash

## Variables
div="==="
fin_div="***"
cwd=$(pwd)
test_path="./project_name"

## Functions
run_test () {
	## Header
	echo -e "$div $2 $div\n"
	## evaluate a command string
	eval $1
	## Check exit code and provide information, exit on failure
	if [ $? -eq 0 ]
	then
		echo -e "$fin_div $3 passed $fin_div\n"
	else
		echo -e "$fin_div $3 failed $fin_div\n"
		## Exit on failure to fail on Travis-CI
		exit 1
	fi
}

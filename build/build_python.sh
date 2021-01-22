#!/bin/env bash

if [[ "$1" == "release" ]]; then
	req_filter=$(ls ./requirements* | grep -v "dev")
else
	req_filter=$(ls ./requirements*)
fi

# install all python packages
for req_file in $req_filter
do
	pip install -r $req_file
done

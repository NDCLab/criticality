#!/bin/env bash

if [[ "$1" == "release" ]]; then
	rdep_filter=$(ls ./r_packages* | grep -v "dev")
else
	# dev
	rdep_filter=$(ls ./r_packages*)

	# Necessary for lintr
	apt-get update
	apt-get install -y libssl-dev libcurl4-openssl-dev libxml2-dev
	apt-get clean
fi

# install all R packages
for rdep_file in $rdep_filter
do
	Rscript $rdep_file
done

#!/bin/env bash

# import utility functions
source ./utility.sh

# Switch to testing path
cd $test_path

## Check R code using R lintr
lr_chk="Rscript -e \"library('lintr')\" -e \"lint_dir(path='./project_name', relative_path=TRUE)\""
run_test $lr_chk "Running R lintr on code" "Lint check"

## Return to cwd
cd $cwd

## Successful run
exit 0

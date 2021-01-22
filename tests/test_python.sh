#!/bin/env bash

# Import utility functions
source ./utility.sh

# Switch to testing path
cd $test_path

## Check that code conforms to pep8 and passes linter
run_test "pycodestyle ./" "Checking code style" "Code style check"
run_test "pylint ./" "Running pylint on code" "Lint check"

## Run unit and integration tests
run_test "python -m unittest" "Running python tests" "Tests"

## Return to cwd
cd $cwd

## Successful run
exit 0

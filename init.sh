#!/bin/env bash

project_name=$1

file_paths=$(grep -R project_name 2>/dev/null | cat | grep -v "init" | cut -d ":" -f 1)

for path in $file_paths; do
	sed -i -Ee "s/project_name/"$project_name"/g" $path
done

exit
# Rename project dir
mv project_name $project_name

# Setup .github dir
mkdir .github/
mv github/* .github/
rm -r github/

# Add data dir
mkdir -p data/{input,output}

# Setup git
rm -rf .git/
git init
git add -A
git commit -m "Initial commit"
git branch -M main

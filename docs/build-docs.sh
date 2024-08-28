#!/bin/bash
rm -rf build/

# Move theming elements into the docs folder
mv pytket-docs-theming/_static .
mv pytket-docs-theming/quantinuum-sphinx .
mv pytket-docs-theming/conf.py .

# Get the name of the project
parentdir="$(basename "$(dirname `pwd`)")"

# Build the docs
sphinx-build -b html -D html_title="$parentdir" . build 

# Move the theming elements back after docs are built. 
mv _static pytket-docs-theming
mv quantinuum-sphinx pytket-docs-theming 
mv conf.py pytket-docs-theming
# This ensures reusability and doesn't clutter source control.
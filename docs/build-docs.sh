#!/bin/bash
rm -rf build/

# Move theming elements into the docs folder
mv pytket-docs-theming/_static .
mv pytket-docs-theming/quantinuum-sphinx .
mv pytket-docs-theming/conf.py .

# Get the name of the project
EXTENSION_NAME="$(basename "$(dirname `pwd`)")"

# Build the docs
sphinx-build -b html -D html_title="$EXTENSION_NAME" . build 

# Correct github link in navbar
sed 's#CQCL/tket#CQCL/'$EXTENSION_NAME'#' _static/nav-config.js


# Move the theming elements back after docs are built. 
mv _static pytket-docs-theming
mv quantinuum-sphinx pytket-docs-theming 
mv conf.py pytket-docs-theming
# This ensures reusability and doesn't clutter source control.
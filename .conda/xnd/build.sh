#!/usr/bin/env sh

cd $RECIPE_DIR/../../ || exit 1
$PYTHON setup.py module || exit 1
$PYTHON setup.py module_install || exit 1
mkdir -p $RECIPE_DIR/test && cp python/*.py $RECIPE_DIR/test

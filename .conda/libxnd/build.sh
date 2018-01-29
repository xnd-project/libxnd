#!/usr/bin/env sh

cd $RECIPE_DIR/../../ || exit 1
./configure --without-docs --prefix=$PREFIX || exit 1
make check
make install

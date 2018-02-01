#!/usr/bin/env sh

cd "$RECIPE_DIR/../../" || exit 1
./configure --prefix="$PREFIX" --with-includes="$PREFIX/include" --with-libs="$PREFIX/lib" --without-docs || exit 1
make check || exit 1
make install

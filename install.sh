#!/bin/bash

# fetch ndtypes and gumath
git submodule init
git submodule update

cd ndtypes
# install ndtypes locally in xnd
python3 setup.py install --local=$PWD/../python
# install ndtypes locally in gumath
python3 setup.py install --local=$PWD/../gumath/python
# install ndtypes (globally)
python3 setup.py install
cd ..

# install xnd (globally)
python3 setup.py install

# install xnd locally in gumath
python3 setup.py install --local=$PWD/gumath/python
cd  gumath
# install gumath (globally)
python3 setup.py build
python3 setup.py install
cd ..

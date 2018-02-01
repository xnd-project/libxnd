#!/bin/sh

if [ $TRAVIS_BRANCH == "master" ] && [ $TRAVIS_PULL_REQUEST == "false" ]; then
  anaconda --token $ANACONDA_NDTYPES_TOKEN upload $LIBNDTYPES --user plures --channel plures &&
  anaconda --token $ANACONDA_NDTYPES_TOKEN upload $NDTYPES --user plures --channel plures &&
  anaconda --token $ANACONDA_XND_TOKEN upload $LIBXND --user plures --channel plures &&
  anaconda --token $ANACONDA_XND_TOKEN upload $XND --user plures --channel plures
fi

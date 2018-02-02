#!/bin/sh

if [ X"$TRAVIS_BRANCH" = X"master" ] && [ X"$TRAVIS_PULL_REQUEST" = X"false" ]; then
  anaconda --token $ANACONDA_NDTYPES_TOKEN upload $LIBNDTYPES --user plures --channel plures &&
  anaconda --token $ANACONDA_NDTYPES_TOKEN upload $NDTYPES --user plures --channel plures &&
  anaconda --token $ANACONDA_XND_TOKEN upload $LIBXND --user plures --channel plures &&
  anaconda --token $ANACONDA_XND_TOKEN upload $XND --user plures --channel plures &&
  echo "\nAnaconda uploads successful.\n"
else
  echo "\nAnaconda uploads not triggered.\n"
fi

# Makefile for Sphinx documentation.
SPHINXBUILD = sphinx-build
BUILDDIR = build

default: html

html:
	sphinx-build -b html -d build/doctrees . build/html

doctest:
	sphinx-build -b doctest -d build/doctrees . build/html

clean:
	rm -rf $(BUILDDIR)/*

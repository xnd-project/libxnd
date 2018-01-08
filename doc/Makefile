# Makefile for Sphinx documentation.
SPHINXBUILD = sphinx-build
BUILDDIR = build

default: html

html:
	sphinx-build -b html -d build/doctrees . build/html

clean:
	rm -rf $(BUILDDIR)/*

@ECHO off

cd ..\libxnd
if exist Makefile nmake /nologo distclean

cd tests
if exist Makefile nmake /nologo distclean

cd ..\..\vcbuild
if exist dist64 rd /q /s dist64
if exist dist32 rd /q /s dist32




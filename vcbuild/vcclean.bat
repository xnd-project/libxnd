@ECHO off

cd ..\libxnd
if exist Makefile nmake /nologo clean

cd tests
if exist Makefile nmake /nologo clean

cd ..\..\vcbuild
if exist dist64 rd /q /s dist64
if exist dist32 rd /q /s dist32



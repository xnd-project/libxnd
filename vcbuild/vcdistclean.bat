@ECHO off

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64

cd ..\libxnd
if exist Makefile nmake /nologo distclean

cd tests
if exist Makefile nmake /nologo distclean

cd ..\..\vcbuild
if exist dist64 rd /q /s dist64
if exist dist32 rd /q /s dist32




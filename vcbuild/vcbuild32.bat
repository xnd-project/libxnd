@ECHO off

if not exist dist32 mkdir dist32
if exist dist32\* del /q dist32\*

cd ..\libxnd
copy /y Makefile.vc Makefile

nmake /nologo clean
nmake /nologo

copy /y libxnd-0.1.0.lib ..\vcbuild\dist32
copy /y libxnd-0.1.0.dll ..\vcbuild\dist32
copy /y libxnd-0.1.0.dll.lib ..\vcbuild\dist32
copy /y libxnd-0.1.0.dll.exp ..\vcbuild\dist32
copy /y xnd.h ..\vcbuild\dist32

cd tests
copy /y Makefile.vc Makefile
nmake /nologo clean
nmake /nologo

copy /y runtest.exe ..\..\vcbuild\dist32
copy /y runtest_shared.exe ..\..\vcbuild\dist32

cd ..\..\vcbuild




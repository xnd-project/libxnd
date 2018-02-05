@ECHO OFF
echo.
<nul (set /p x="Running static library tests ... ")
echo.
echo.
dist32\runtest.exe
IF ERRORLEVEL 1 echo FAIL
echo.
<nul (set /p x="Running shared library tests ... ")
echo.
echo.
copy /y ..\ndtypes\libndtypes\libndtypes-0.2.0dev3.dll dist32
dist32\runtest_shared.exe
IF ERRORLEVEL 1 echo FAIL



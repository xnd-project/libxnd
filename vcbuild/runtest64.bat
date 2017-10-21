@ECHO OFF
echo.
<nul (set /p x="Running static library tests ... ")
echo.
echo.
dist64\runtest.exe
IF ERRORLEVEL 1 echo FAIL
echo.
<nul (set /p x="Running shared library tests ... ")
echo.
echo.
copy /y ..\ndtypes\libndtypes\libndtypes-0.1.0.dll dist64
dist64\runtest_shared.exe
IF ERRORLEVEL 1 echo FAIL

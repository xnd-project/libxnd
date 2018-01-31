cd "%RECIPE_DIR%\..\..\vcbuild" || exit 1
call vcbuild64.bat || exit 1
call runtest64.bat || exit 1
copy /y dist64\lib* "%PREFIX%\Library\bin\"
copy /y dist64\ndtypes.h "%PREFIX%\Library\include\"

set "LIBNDTYPESDIR=%PREFIX%\Library\bin"
set "LIBNDTYPESINCLUDE=%PREFIX%\Library\include"
cd "%RECIPE_DIR%\..\..\vcbuild" || exit 1
call vcbuild64.bat || exit 1
call runtest64.bat || exit 1
copy /y dist64\lib* "%PREFIX%\Library\bin\"
copy /y dist64\xnd.h "%PREFIX%\Library\include\"

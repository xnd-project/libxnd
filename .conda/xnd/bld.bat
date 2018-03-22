cd "%RECIPE_DIR%\..\..\" || exit 1
"%PYTHON%" setup.py conda_install || exit 1
if not exist "%SP_DIR%\xnd\contrib" mkdir "%SP_DIR%\xnd\contrib"
copy /y python\contrib\*.py "%SP_DIR%\xnd\contrib"
if not exist "%RECIPE_DIR%\test" mkdir "%RECIPE_DIR%\test"
copy /y python\*.py "%RECIPE_DIR%\test"

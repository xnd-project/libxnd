

libxnd build instructions for Visual Studio
===========================================


   Requirements
   ------------

      - Visual Studio 2015 or later.

      - The path to vcvarsall.bat is hardcoded for VS 2015 in the scripts.


   64-bit build
   ------------

      Run vcbuild64.bat. If successful, the static library, the dynamic
      library, the common header file and two executables for running the
      unit tests should be in the dist64 directory.


   32-bit build
   ------------

      Run vcbuild32.bat. If successful, the static library, the dynamic
      library, the common header file and two executables for running the
      unit tests should be in the dist32 directory.


   Run the unit tests
   ------------------

      Depending on the build, run runtest64.bat or runtest32.bat.





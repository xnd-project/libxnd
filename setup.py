#
# BSD 3-Clause License
#
# Copyright (c) 2017, plures
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from distutils.core import setup, Extension
from glob import glob
import platform
import sys, os
import subprocess
import shutil


DESCRIPTION = """Typed memory container based on libndtypes"""


PY_MAJOR = sys.version_info[0]
PY_MINOR = sys.version_info[1]
ARCH = platform.architecture()[0]
BUILD_ALL = True


if PY_MAJOR < 3:
    raise NotImplementedError(
        "python2 support is not implemented")


def get_module_path():
    pathlist = glob("build/lib.*/")
    if pathlist:
        return pathlist[0]
    raise RuntimeError("cannot find xnd module in build directory")

def copy_ext():
    if sys.platform == "win32":
        pathlist = glob("build/lib.*/xnd/_xnd.*.pyd")
    else:
        pathlist = glob("build/lib.*/xnd/_xnd.*.so")
    if pathlist:
        shutil.copy2(pathlist[0], "python/xnd")


if len(sys.argv) == 2:
    if sys.argv[1] == 'module':
       sys.argv[1] = 'build'
       BUILD_ALL = False
    if sys.argv[1] == 'test':
        module_path = get_module_path()
        python_path = os.getenv('PYTHONPATH')
        path = module_path + ':' + python_path if python_path else module_path
        env = os.environ.copy()
        env['PYTHONPATH'] = path
        ret = subprocess.call([sys.executable, "python/test_xnd.py"], env=env)
        sys.exit(ret)
    elif sys.argv[1] == 'clean':
        shutil.rmtree("build", ignore_errors=True)
        os.chdir("python/xnd")
        shutil.rmtree("__pycache__", ignore_errors=True)
        for f in glob("_xnd*.so"):
            os.remove(f)
        sys.exit(0)
    elif sys.argv[1] == 'distclean':
        if sys.platform == "win32":
            os.chdir("vcbuild")
            os.system("vcdistclean.bat")
        else:
            os.system("make distclean")
        sys.exit(0)
    else:
        pass


def ndtypes_ext():
    include_dirs = ["libxnd", "ndtypes/libndtypes"]
    library_dirs = ["libxnd", "ndtypes/libndtypes"]
    depends = ["libxnd/xnd.h"]
    sources = ["python/xnd/_xnd.c"]

    if sys.platform == "win32":
        libraries = ["libxnd-0.1.0.dll", "libndtypes-0.1.0.dll"]
        extra_compile_args = ["/DIMPORT"]
        extra_link_args = []
        runtime_library_dirs = []

        if BUILD_ALL:
            from distutils.msvc9compiler import MSVCCompiler
            MSVCCompiler().initialize()
            os.chdir("vcbuild")
            if ARCH == "64bit":
                  os.system("vcbuild64.bat")
            else:
                  os.system("vcbuild32.bat")
            os.chdir("..")

    else:
        libraries = ["xnd", "ndtypes"]
        extra_compile_args = ["-Wextra", "-Wno-missing-field-initializers", "-std=c11"]
        if sys.platform == "darwin":
            extra_link_args = ["-Wl,-rpath,@loader_path"]
            runtime_library_dirs = []
        else:
            extra_link_args = []
            runtime_library_dirs = ["$ORIGIN"]

        if BUILD_ALL:
            os.system("./configure --with-ndtypes-path=. && make")

    return Extension (
      "xnd._xnd",
      include_dirs = include_dirs,
      library_dirs = library_dirs,
      depends = depends,
      sources = sources,
      libraries = libraries,
      extra_compile_args = extra_compile_args,
      extra_link_args = extra_link_args,
      runtime_library_dirs = runtime_library_dirs
    )

setup (
    name = "xnd",
    version = "0.1",
    description = DESCRIPTION,
    url = "https://github.com/plures/xnd",
    license = "BSD License",
    keywords = ["xnd", "array computing", "data description"],
    platforms = ["Many"],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: C",
        "Programming Language :: Python :: 3"
    ],
    package_dir = {"": "python"},
    packages = ["xnd"],
    package_data = {"xnd": ["libxnd*"]},
    ext_modules = [ndtypes_ext()],
)

copy_ext()

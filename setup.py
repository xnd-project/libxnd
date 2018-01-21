#
# BSD 3-Clause License
#
# Copyright (c) 2017-2018, plures
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

import sys, os

if "bdist_wheel" in sys.argv:
    from setuptools import setup, Extension
else:
    from distutils.core import setup, Extension

from distutils.sysconfig import get_python_lib
from glob import glob
import platform
import subprocess
import shutil
import warnings


DESCRIPTION = """\
General container that maps a wide range of Python values directly to memory.\
"""

LONG_DESCRIPTION = """
Overview
--------

The ``xnd`` module implements a container type that maps most Python values
relevant for scientific computing directly to typed memory.

Whenever possible, a single, pointer-free memory block is used.

``xnd`` supports ragged arrays, categorical types, indexing, slicing, aligned
memory blocks and type inference.

Operations like indexing and slicing return zero-copy typed views on the data.

Importing PEP-3118 buffers is supported.

Links
-----

* https://github.com/plures/
* http://ndtypes.readthedocs.io/en/latest/
* http://xnd.readthedocs.io/en/latest/

"""

warnings.simplefilter("ignore", UserWarning)

if sys.platform == "darwin":
    LIBNAME = "libxnd.dylib"
    LIBSONAME = "libxnd.0.dylib"
    LIBSHARED = "libxnd.0.2.0b1.dylib"
else:
    LIBNAME = "libxnd.so"
    LIBSONAME = "libxnd.so.0"
    LIBSHARED = "libxnd.so.0.2.0b1"

if "install" in sys.argv or "bdist_wheel" in sys.argv:
    LIBNDTYPESDIR = "%s/ndtypes" % get_python_lib()
    LIBXNDDIR = "%s/xnd" % get_python_lib()
else:
    LIBNDTYPESDIR = "../python/ndtypes"
    LIBXNDDIR = "../python/xnd"

PY_MAJOR = sys.version_info[0]
PY_MINOR = sys.version_info[1]
ARCH = platform.architecture()[0]
BUILD_ALL = \
    "build" in sys.argv or "install" in sys.argv or "bdist_wheel" in sys.argv


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

def make_symlinks():
    if sys.platform == "win32" or "install" not in sys.argv:
        return
    os.chdir(LIBXNDDIR)
    os.chmod(LIBSHARED, 0o755)
    os.system("ln -sf %s %s" % (LIBSHARED, LIBSONAME))
    os.system("ln -sf %s %s" % (LIBSHARED, LIBNAME))


if len(sys.argv) == 2:
    if sys.argv[1] == 'module':
       sys.argv[1] = 'build'
    if sys.argv[1] == 'test':
        module_path = get_module_path()
        python_path = os.getenv('PYTHONPATH')
        path = module_path + ':' + python_path if python_path else module_path
        env = os.environ.copy()
        env['PYTHONPATH'] = path
        ret = subprocess.call([sys.executable, "python/test_xnd.py", "--long"], env=env)
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
    include_dirs = ["libxnd", "ndtypes/python/ndtypes", LIBNDTYPESDIR]
    library_dirs = ["libxnd", "ndtypes/libndtypes", LIBNDTYPESDIR]
    depends = ["libxnd/xnd.h", "python/xnd/util.h"]
    sources = ["python/xnd/_xnd.c"]

    if sys.platform == "win32":
        libraries = ["libxnd-0.2.0b1.dll", "libndtypes-0.2.0b1.dll"]
        extra_compile_args = ["/DIMPORT"]
        extra_link_args = []
        runtime_library_dirs = []

        if BUILD_ALL:
            from distutils.msvc9compiler import MSVCCompiler
            MSVCCompiler().initialize()
            os.chdir("vcbuild")
            os.environ['LIBNDTYPESDIR'] = os.path.normpath(LIBNDTYPESDIR)
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
            os.system("./configure --with-ndtypes='%s' && make" % LIBNDTYPESDIR)

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
    version = "0.2.0b1",
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    url = "https://github.com/plures/xnd",
    author = 'Stefan Krah',
    author_email = 'skrah@bytereef.org',
    license = "BSD License",
    keywords = ["xnd", "array computing", "container", "memory blocks"],
    platforms = ["Many"],
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development"
    ],
    install_requires = ["ndtypes == v0.2.0b1"],
    package_dir = {"": "python"},
    packages = ["xnd"],
    package_data = {"xnd": ["libxnd*", "xnd.h"]},
    ext_modules = [ndtypes_ext()],
)

copy_ext()
make_symlinks()

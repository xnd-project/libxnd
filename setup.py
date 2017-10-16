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
from distutils.cmd import Command
from glob import glob
import sys, os
import subprocess
import shutil


DESCRIPTION = """xnd container"""


def get_module_path():
    pathlist = glob("build/lib.*/")
    if pathlist:
        return pathlist[0]
    raise RuntimeError("cannot find xnd module in build directory")

def copy_ext():
    pathlist = glob("build/lib.*/_xnd.*.so")
    if pathlist:
        shutil.copy2(pathlist[0], "python/")


if len(sys.argv) == 2:
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
        os.chdir("python")
        shutil.rmtree("__pycache__", ignore_errors=True)
        for f in glob("*.so"):
            os.remove(f)
        sys.exit(0)
    else:
        pass


def ndtypes_ext():
    include_dirs = ["libxnd", "ndtypes/libndtypes"]

    depends = [
      "libxnd/xnd.h",
      "ndtypes/libndtypes/attr.h",
      "ndtypes/libndtypes/grammar.h",
      "ndtypes/libndtypes/lexer.h",
      "ndtypes/libndtypes/ndtypes.h",
      "ndtypes/libndtypes/parsefuncs.h",
      "ndtypes/libndtypes/seq.h",
      "ndtypes/libndtypes/symtable.h"
    ]

    sources = [
      "python/_xnd.c",
      "libxnd/xnd.c",
      "ndtypes/libndtypes/alloc.c",
      "ndtypes/libndtypes/attr.c",
      "ndtypes/libndtypes/display.c",
      "ndtypes/libndtypes/display_meta.c",
      "ndtypes/libndtypes/equal.c",
      "ndtypes/libndtypes/grammar.c",
      "ndtypes/libndtypes/lexer.c",
      "ndtypes/libndtypes/match.c",
      "ndtypes/libndtypes/ndtypes.c",
      "ndtypes/libndtypes/parsefuncs.c",
      "ndtypes/libndtypes/parser.c",
      "ndtypes/libndtypes/seq.c",
      "ndtypes/libndtypes/symtable.c",
    ]

    if sys.platform == "win32":
        extra_compile_args = [
          "/wd4200", "/wd4201", "/wd4244", "/wd4267", "/wd4702",
          "/wd4127", "/nologo", "/DYY_NO_UNISTD_H=1", "/D__STDC_VERSION__=199901L"
        ]
    else:
        extra_compile_args = [
           "-Wextra", "-Wno-missing-field-initializers", "-std=c11"
        ]

    return Extension (
      "_xnd",
      include_dirs=include_dirs,
      extra_compile_args = extra_compile_args,
      depends = depends,
      sources = sources
    )

setup (
    name = "_xnd",
    version = "0.1",
    description = DESCRIPTION,
    url = "https://github.com/plures/xnd",
    license = "BSD License",
    keywords = ["xnd", "array computing", "data description"],
    platforms = ["Many"],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
    package_dir = {"": "python"},
    py_modules = ["xnd"],
    ext_modules = [ndtypes_ext()],
)

copy_ext()

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

"""
Xnd implements a container for mapping all Python values relevant for
scientific computing directly to memory.

xnd supports ragged arrays, categorical types, indexing, slicing, aligned
memory blocks and type inference.

Operations like indexing and slicing return zero-copy typed views on the
data.

Importing PEP-3118 buffers is supported.
"""


# Ensure that libndtypes is loaded and initialized.
from ndtypes import ndt, instantiate, MAX_DIM
from ._xnd import Xnd, XndEllipsis
from itertools import accumulate
from .contrib.pretty import pretty

__all__ = ['xnd', 'XndEllipsis', 'typeof', '_typeof']


# ======================================================================
#                              xnd object
# ======================================================================

class xnd(Xnd):
    """General container type for unboxing a wide range of Python values
       to typed memory blocks.

       Operations like indexing and slicing return zero-copy typed views
       on the data.

       Create fixed or ragged arrays:

           >>> xnd([[1,2,3], [4,5,6]])
           xnd([[1, 2, 3], [4, 5, 6]], type="2 * 3 * int64")

           >>> xnd([[1,2,3], [4]])
           xnd([[1, 2, 3], [4]], type="var * var * int64")

       Create a record:

           >>> xnd({'a': "xyz", 'b': [1, 2, 3]})
           xnd({'a': 'xyz', 'b': [1, 2, 3]}, type="{a : string, b : 3 * int64}")

       Create a categorical type:

           >>> xnd(['a', 'b', None, 'a'], levels=['a', 'b', None])
           xnd(['a', 'b', None, 'a'], type="4 * categorical('a', 'b', NA)")

       Create an explicitly typed memory block:

           >>> xnd(100000 * [1], type="100000 * uint8")
          xnd([1, 1, 1, 1, 1, 1, 1, 1, 1, ...], type="100000 * uint8")

       Create an empty (zero initialized) memory block:

           >>> xnd.empty("100000 * uint8")
           xnd([0, 0, 0, 0, 0, 0, 0, 0, 0, ...], type="100000 * uint8")

       Import a memory block from a buffer exporter:

           >>> xnd.from_buffer(b"123")
           xnd([49, 50, 51], type="3 * uint8")
    """

    def __new__(cls, value, *, type=None, dtype=None, levels=None,
                typedef=None, dtypedef=None):
        if (type, dtype, levels, typedef, dtypedef).count(None) < 2:
            raise TypeError(
                "the 'type', 'dtype', 'levels' and 'typedef' arguments are "
                "mutually exclusive")
        if type is not None:
            if isinstance(type, str):
                type = ndt(type)
        elif dtype is not None:
            type = typeof(value, dtype=dtype)
        elif levels is not None:
            args = ', '.join("'%s'" % l if l is not None else 'NA' for l in levels)
            t = "%d * categorical(%s)" % (len(value), args)
            type = ndt(t)
        elif typedef is not None:
            type = ndt(typedef)
            if type.isabstract():
                dtype = type.hidden_dtype
                t = typeof(value, dtype=dtype)
                type = instantiate(typedef, t)
        elif dtypedef is not None:
            dtype = ndt(dtypedef)
            type = typeof(value, dtype=dtype)
        else:
            type = typeof(value)
        return super().__new__(cls, type=type, value=value)

    def __repr__(self):
        value = self.short_value(maxshape=10)
        fmt = pretty((value, "@type='%s'@" % self.type), max_width=120)
        fmt = fmt.replace('"@', "")
        fmt = fmt.replace('@"', "")
        fmt = fmt.replace("\n", "\n   ")
        return "xnd%s" % fmt


# ======================================================================
#                            Type inference
# ======================================================================

def typeof(value, *, dtype=None):
    return ndt(_typeof(value, dtype=dtype))

def _choose_dtype(lst):
    for x in lst:
        if x is not None:
            return _typeof(x)
    return "float64"

def _typeof(value, *, dtype=None):
    """Infer the type of a Python value.  Only a subset of Datashape is
       supported.  In general, types need to be explicitly specified when
       creating xnd objects.
    """
    if isinstance(value, list):
        data, shapes = data_shapes(value)
        opt = None in data

        if dtype is None:
            if not data:
                dtype = 'float64'
            else:
                dtype = _choose_dtype(data)
                for x in data:
                    if x is not None:
                        t = _typeof(x)
                        if t != dtype:
                            raise ValueError("dtype mismatch: have %s and %s" % (dtype, t))

            if opt:
                dtype = '?' + dtype

        t = dtype
        var = any(len(set(lst)) > 1 or None in lst for lst in shapes)
        for lst in shapes:
            opt = None in lst
            lst = [0 if x is None else x for x in lst]
            t = add_dim(opt=opt, shapes=lst, typ=t, use_var=var)

        return t

    elif dtype is not None:
        raise TypeError("dtype argument is only supported for arrays")

    elif isinstance(value, tuple):
        return "(" + ", ".join([_typeof(x) for x in value]) + ")"

    elif isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return "{" + ", ".join(["%s: %s" % (k, _typeof(v)) for k, v in value.items()]) + "}"
        raise ValueError("all dict keys must be strings")

    elif value is None:
        return '?float64'

    elif isinstance(value, float):
        return 'float64'

    elif isinstance(value, complex):
        return 'complex128'

    elif isinstance(value, int):
        return 'int64'

    elif isinstance(value, str):
        return 'string'

    elif isinstance(value, bytes):
        return 'bytes'

    else:
        raise ValueError("cannot infer type for %r" % value)


def add_dim(*, opt=False, shapes=None, typ=None, use_var=False):
    """Construct a new dimension type based on the list of 'shapes' that
       are present in a dimension.
    """
    if use_var:
        offsets = [0] + list(accumulate(shapes))
        return "%svar(offsets=%s) * %s" % ('?' if opt else '', offsets, typ)
    else:
        n = len(set(shapes))
        assert n <= 1 and not None in shapes
        shape = 0 if n == 0 else shapes[0]
        return "%d * %s" % (shape, typ)

def data_shapes(tree):
    """Extract array data and dimension shapes from a nested list. The
       list may contain None for missing data or dimensions.

       >>> data_shapes([[0, 1], [2, 3, 4], [5, 6, 7, 8]])
       ([0, 1, 2, 3, 4, 5, 6, 7, 8], [[2, 3, 4], [3]])
                     ^                    ^       ^
                     |                    |       `--- ndim=2: single shape 3
                     |                    `-- ndim=1: shapes 2, 3, 4
                     `--- ndim=0: extracted array data
    """
    acc = [[] for _ in range(MAX_DIM+1)]
    min_level = MAX_DIM + 1
    max_level = 0

    def search(level, a):
        nonlocal min_level, max_level

        if level > MAX_DIM:
            raise ValueError("too many dimensions")

        current = acc[level]
        if a is None:
            current.append(a)
        elif isinstance(a, list):
            current.append(len(a))
            next_level = level + 1
            max_level = max(next_level, max_level)
            if not a:
                min_level = min(next_level, min_level)
            else:
                for item in a:
                    search(level+1, item)
        else:
            acc[max_level].append(a)
            min_level = min(level, min_level)

    search(max_level, tree)
    if acc[max_level] and all(x is None for x in acc[max_level]):
        pass # min_level is not set in this special case, hence the check.
    elif min_level != max_level:
        raise ValueError("unbalanced tree: min depth: %d max depth: %d" %
                         (min_level, max_level))

    data = acc[max_level]
    shapes = list(reversed(acc[0:max_level]))

    return data, shapes




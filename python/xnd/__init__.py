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
from ._xnd import Xnd, XndEllipsis, data_shapes, _typeof
from .contrib.pretty import pretty

__all__ = ['xnd', 'XndEllipsis', 'typeof']


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

    @classmethod
    def unsafe_from_data(cls, obj=None, type=None):
        """Return an xnd object that obtains memory from 'obj' via the
           buffer protocol.  The buffer protocol's type is overridden by
          'type'.  No safety checks are performed, the user is responsible
           for passing a suitable type.
        """
        if isinstance(type, str):
            type = ndt(type)
        return cls._unsafe_from_data(obj, type)

def typeof(v, dtype=None):
    if isinstance(dtype, str):
        dtype = ndt(dtype)
    return _typeof(v, dtype=dtype)

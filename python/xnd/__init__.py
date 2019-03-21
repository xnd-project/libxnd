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

__all__ = ['xnd', 'array', 'XndEllipsis', 'typeof']


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
                typedef=None, dtypedef=None, device=None):
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

        if device is not None:
            name, no = device.split(":")
            no = -1 if no == "managed" else int(no)
            device = (name, no)

        return super().__new__(cls, type=type, value=value, device=device)

    def __repr__(self):
        value = self.short_value(maxshape=10)
        fmt = pretty((value, "@type='%s'@" % self.type), max_width=120)
        fmt = fmt.replace('"@', "")
        fmt = fmt.replace('@"', "")
        fmt = fmt.replace("\n", "\n   ")
        return "xnd%s" % fmt

    def copy_contiguous(self, dtype=None):
        if isinstance(dtype, str):
            dtype = ndt(dtype)
        return super().copy_contiguous(dtype=dtype)

    def reshape(self, *args, order=None):
        return super()._reshape(args, order=order)

    @classmethod
    def empty(cls, type=None, device=None):
        if device is not None:
            name, no = device.split(":")
            no = -1 if no == "managed" else no
            device = (name, int(no))

        return super(xnd, cls).empty(type, device)

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
    return _typeof(v, dtype=dtype, shortcut=True)


# ======================================================================
#                              array object
# ======================================================================

class array(object):
    """Extended array type that relies on gumath for the array functions."""

    __slots__ = ('_xnd',)

    _functions = None
    _cuda = None

    def __init__(self, obj, dtype=None, levels=None, device=None):
        if isinstance(obj, xnd):
            if dtype is None and levels is None and device is None:
                self._xnd = obj
            else:
                raise TypeError("type(obj) == xnd, but other arguments are given")
        else:
            self._xnd = xnd(obj, dtype=dtype, levels=levels, device=device)

    @property
    def shape(self):
        return self._xnd.type.shape

    @property
    def strides(self):
        return self._xnd.type.strides

    @property
    def dtype(self):
        return self._xnd.dtype

    def __repr__(self):
        value = self._xnd.short_value(maxshape=10)
        fmt = pretty((value, "@type='%s'@" % self._xnd.type), max_width=120)
        fmt = fmt.replace('"@', "")
        fmt = fmt.replace('@"', "")
        fmt = fmt.replace("\n", "\n     ")
        return "array%s" % fmt

    def __getitem__(self, *args, **kwargs):
        x = self._xnd.__getitem__(*args, **kwargs)
        return array(x)

    def __setitem__(self, *args, **kwargs):
        self._xnd.__setitem__(*args, **kwargs)

    def _get_module(self, *devices):
        if all(d == "cuda:managed" for d in devices):
            if array._cuda is None:
                import gumath.cuda
                array._cuda = gumath.cuda
            return array._cuda
        else:
            if array._functions is None:
                import gumath.functions
                array._functions = gumath.functions
            return array._functions

    def _call_unary(self, name):
        m = self._get_module(self._xnd.device)
        x = getattr(m, name)(self._xnd)
        return array(x)

    def _call_binary(self, name, other):
        m = self._get_module(self._xnd.device, other._xnd.device)
        x = getattr(m, name)(self._xnd, other._xnd)
        return array(x)

    def __neg__(self):
        return self._call_unary("negate")

    def __pos__(self):
        raise NotImplementedError("the unary '+' operator is not implemented")

    def __abs__(self):
        raise NotImplementedError("abs() is not implemented")

    def __invert__(self):
        return self._call_unary("invert")

    def __complex__(self):
        raise TypeError("complex() is not supported")

    def __int__(self):
        raise TypeError("int() is not supported")

    def __float__(self):
        raise TypeError("float() is not supported")

    def __index__(self):
        raise TypeError("index() is not supported")

    def __round__(self):
        return self._call_unary("round")

    def __trunc__(self):
        return self._call_unary("trunc")

    def __floor__(self):
        return self._call_unary("floor")

    def __ceil__(self):
        return self._call_unary("ceil")

    def __eq__(self, other):
        return self._call_binary("equal", other)

    def __ne__(self, other):
        return self._call_binary("not_equal", other)

    def __lt__(self, other):
        return self._call_binary("less", other)

    def __le__(self, other):
        return self._call_binary("less_equal", other)

    def __ge__(self, other):
        return self._call_binary("greater_equal", other)

    def __gt__(self, other):
        return self._call_binary("greater", other)

    def __add__(self, other):
        return self._call_binary("add", other)

    def __sub__(self, other):
        return self._call_binary("subtract", other)

    def __mul__(self, other):
        return self._call_binary("multiply", other)

    def __matmul__(self, other):
        raise NotImplementedError("matrix multiplication is not implemented")

    def __truediv__(self, other):
        return self._call_binary("divide", other)

    def __floordiv__(self, other):
        return self._call_binary("floor_divide", other)

    def __mod__(self, other):
        return self._call_binary("remainder", other)

    def __divmod__(self, other):
        return self._call_binary("divmod", other)

    def __pow__(self, other):
        raise NotImplementedError("power is not implemented")

    def __lshift__(self, other):
        raise TypeError("the '<<' operator is not supported")

    def __rshift__(self, other):
        raise TypeError("the '>>' operator is not supported")

    def __and__(self, other):
        return self._call_binary("bitwise_and", other)

    def __or__(self, other):
        return self._call_binary("bitwise_or", other)

    def __xor__(self, other):
        return self._call_binary("bitwise_xor", other)

    def tolist(self):
        return self._xnd.value

    def copy(self):
        return self._call_unary("copy")

    def transpose(self, axes=None):
        x = self._xnd.transpose(permute=axes)
        return array(x)

    def acos(self):
        return self._call_unary("acos")

    def acosh(self):
        return self._call_unary("acosh")

    def asin(self):
        return self._call_unary("asin")

    def asinh(self):
        return self._call_unary("asinh")

    def atan(self):
        return self._call_unary("atan")

    def atanh(self):
        return self._call_unary("atanh")

    def cbrt(self):
        return self._call_unary("cbrt")

    def cos(self):
        return self._call_unary("cos")

    def cosh(self):
        return self._call_unary("cosh")

    def erf(self):
        return self._call_unary("erf")

    def erfc(self):
        return self._call_unary("erfc")

    def exp(self):
        return self._call_unary("exp")

    def exp2(self):
        return self._call_unary("exp2")

    def expm1(self):
        return self._call_unary("expm1")

    def fabs(self):
        return self._call_unary("fabs")

    def lgamma(self):
        return self._call_unary("lgamma")

    def log(self):
        return self._call_unary("log")

    def log10(self):
        return self._call_unary("log10")

    def log1p(self):
        return self._call_unary("log1p")

    def log2(self):
        return self._call_unary("log2")

    def logb(self):
        return self._call_unary("logb")

    def nearbyint(self):
        return self._call_unary("nearbyint")

    def sin(self):
        return self._call_unary("sin")

    def sinh(self):
        return self._call_unary("sinh")

    def sqrt(self):
        return self._call_unary("sqrt")

    def tan(self):
        return self._call_unary("tan")

    def tanh(self):
        return self._call_unary("tanh")

    def tanh(self):
        return self._call_unary("tgamma")

    def equaln(self, other):
        return self._call_binary("equaln", other)

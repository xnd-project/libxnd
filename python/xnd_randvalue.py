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

# Python NDarray and functions for generating test cases.

from itertools import accumulate, count, product
from random import randrange
from xnd_support import R


# ======================================================================
#                             Primitive types 
# ======================================================================

PRIMITIVE = [
    'bool',
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'float32', 'float64',
    'complex64', 'complex128'
]

EMPTY_TEST_CASES = [
    (0, "%s"),
    ([], "0 * %s"),
    ([0], "1 * %s"),
    ([0, 0], "var(offsets=[0, 2]) * %s"),
    (3 * [{"a": 0, "b": 0}], "3 * {a: int64, b: %s}")
]


# ======================================================================
#                             Typed values
# ======================================================================

DTYPE_EMPTY_TEST_CASES = [
   # Tuples
   ((), "()"),
   ((0,), "(int8)"),
   ((0, 0), "(int8, int64)"),
   ((0, (0+0j,)), "(uint16, (complex64))"),
   ((0, (0+0j,)), "(uint16, (complex64), pack=1)"),
   ((0, (0+0j,)), "(uint16, (complex64), pack=2)"),
   ((0, (0+0j,)), "(uint16, (complex64), pack=4)"),
   ((0, (0+0j,)), "(uint16, (complex64), pack=8)"),
   ((0, (0+0j,)), "(uint16, (complex64), align=16)"),
   (([],), "(0 * bytes)"),
   (([], []), "(0 * bytes, 0 * string)"),
   (([b''], 2 * [0.0j], 3 * [""]), "(1 * bytes, 2 * complex128, 3 * string)"),
   (([b''], 2 * [(0.0j, 10 * [2 * [""]])], 3 * [""]), "(1 * bytes, 2 * (complex128, 10 * 2 * string), 3 * string)"),
   ((0, 2 * [3 * [4 * [5 * [0.0]]]]), "(int64, 2 * 3 * Some(4 * 5 * float64))"),
   ((0, 2 * [3 * [4 * [5 * [0.0]]]]), "(int64, 2 * 3 * ref(4 * 5 * float64))"),

   # Optional tuples
   (None, "?()"),
   (None, "?(int8)"),
   (None, "?(int8, int64)"),
   (None, "?(uint16, (complex64))"),
   (None, "?(uint16, (complex64), pack=1)"),
   (None, "?(uint16, (complex64), pack=2)"),
   (None, "?(uint16, (complex64), pack=4)"),
   (None, "?(uint16, (complex64), pack=8)"),
   (None, "?(uint16, (complex64), align=16)"),
   (None, "?(0 * bytes)"),
   (None, "?(0 * bytes, 0 * string)"),
   (None, "?(1 * bytes, 2 * complex128, 3 * string)"),
   (None, "?(1 * bytes, 2 * (complex128, 10 * 2 * string), 3 * string)"),

   # Tuples with optional elements
   ((None,), "(?int8)"),
   ((None, 0), "(?int8, int64)"),
   ((0, None), "(int8, ?int64)"),
   ((None, None), "(?int8, ?int64)"),
   (None, "?(?int8, ?int64)"),

   ((0, None), "(uint16, ?(complex64))"),
   ((0, (None,)), "(uint16, (?complex64))"),
   ((0, None), "(uint16, ?(?complex64))"),

   ((None, (0+0j,)), "(?uint16, (complex64), pack=1)"),
   ((0, None), "(uint16, ?(complex64), pack=1)"),
   ((0, (None,)), "(uint16, (?complex64), pack=1)"),

   (([],), "(0 * ?bytes)"),
   (([None],), "(1 * ?bytes)"),
   ((10 * [None],), "(10 * ?bytes)"),
   (([], []), "(0 * ?bytes, 0 * ?string)"),
   ((5 * [None], 2 * [""]), "(5 * ?bytes, 2 * string)"),
   ((5 * [b''], 2 * [None]), "(5 * bytes, 2 * ?string)"),
   ((5 * [None], 2 * [None]), "(5 * ?bytes, 2 * ?string)"),

   (([None], 2 * [None], 3 * [None]), "(1 * ?bytes, 2 * ?complex128, 3 * ?string)"),

   (([None], 2 * [(0.0j, 10 * [2 * [""]])], 3 * [""]), "(1 * ?bytes, 2 * (complex128, 10 * 2 * string), 3 * string)"),
   (([b''], 2 * [None], 3 * [""]), "(1 * bytes, 2 * ?(complex128, 10 * 2 * string), 3 * string)"),
   (([b''], 2 * [(0.0j, 10 * [2 * [""]])], 3 * [None]), "(1 * bytes, 2 * (complex128, 10 * 2 * string), 3 * ?string)"),
   (([None], 2 * [None], 3 * [""]), "(1 * ?bytes, 2 * ?(complex128, 10 * 2 * string), 3 * string)"),
   (([None], 2 * [(0.0j, 10 * [2 * [""]])], 3 * [None]), "(1 * ?bytes, 2 * (complex128, 10 * 2 * string), 3 * ?string)"),
   (([None], 2 * [None], 3 * [None]), "(1 * ?bytes, 2 * ?(complex128, 10 * 2 * string), 3 * ?string)"),

   (([b''], 2 * [(0.0j, 10 * [2 * [None]])], 3 * [""]), "(1 * bytes, 2 * (complex128, 10 * 2 * ?string), 3 * string)"),
   (([None], 2 * [(0.0j, 10 * [2 * [None]])], 3 * [""]), "(1 * ?bytes, 2 * (complex128, 10 * 2 * ?string), 3 * string)"),
   (([None], 2 * [(None, 10 * [2 * [None]])], 3 * [""]), "(1 * ?bytes, 2 * (?complex128, 10 * 2 * ?string), 3 * string)"),
   (([b''], 2 * [(None, 10 * [2 * [None]])], 3 * [None]), "(1 * bytes, 2 * (?complex128, 10 * 2 * ?string), 3 * ?string)"),

   ((0, 2 * [3 * [4 * [5 * [None]]]]), "(int64, 2 * 3 * Some(4 * 5 * ?float64))"),
   ((0, 2 * [3 * [4 * [5 * [None]]]]), "(int64, 2 * 3 * ref(4 * 5 * ?float64))"),

   # Records
   ({}, "{}"),
   (R['x': 0], "{x: int8}"),
   (R['x': 0, 'y': 0], "{x: int8, y: int64}"),
   (R['x': 0, 'y': R['z': 0+0j]], "{x: uint16, y: {z: complex64}}"),
   (R['x': 0, 'y': R['z': 0+0j]], "{x: uint16, y: {z: complex64}, pack=1}"),
   (R['x': 0, 'y': R['z': 0+0j]], "{x: uint16, y: {z: complex64}, pack=2}"),
   (R['x': 0, 'y': R['z': 0+0j]], "{x: uint16, y: {z: complex64}, pack=4}"),
   (R['x': 0, 'y': R['z': 0+0j]], "{x: uint16, y: {z: complex64}, pack=8}"),
   (R['x': 0, 'y': R['z': 0+0j]], "{x: uint16, y: {z: complex64}, align=16}"),
   (R['x': []], "{x: 0 * bytes}"),
   (R['x': [], 'y': []], "{x: 0 * bytes, y: 0 * string}"),
   (R['x': [b''], 'y': 2 * [0.0j], 'z': 3 * [""]], "{x: 1 * bytes, y: 2 * complex128, z: 3 * string}"),
   (R['x': [b''], 'y': 2 * [R['a': 0.0j, 'b': 10 * [2 * [""]]]], 'z': 3 * [""]], "{x: 1 * bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * string}"),
   (R['x': 0, 'y': 2 * [3 * [4 * [5 * [0.0]]]]], "{x: int64, y: 2 * 3 * Some(4 * 5 * float64)}"),
   (R['x': 0, 'y': 2 * [3 * [4 * [5 * [0.0]]]]], "{x: int64, y: 2 * 3 * ref(4 * 5 * float64)}"),

   # Optional records
   (None, "?{}"),
   (None, "?{x: int8}"),
   (None, "?{x: int8, y: int64}"),
   (None, "?{x: uint16, y: {z: complex64}}"),
   (None, "?{x: uint16, y: {z: complex64}, pack=1}"),
   (None, "?{x: uint16, y: {z: complex64}, pack=2}"),
   (None, "?{x: uint16, y: {z: complex64}, pack=4}"),
   (None, "?{x: uint16, y: {z: complex64}, pack=8}"),
   (None, "?{x: uint16, y: {z: complex64}, align=16}"),
   (None, "?{x: 0 * bytes}"),
   (None, "?{x: 0 * bytes, y: 0 * string}"),
   (None, "?{x: 1 * bytes, y: 2 * complex128, z: 3 * string}"),
   (None, "?{x: 1 * bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * string}"),

   # Records with optional elements
   (R['x': None], "{x: ?int8}"),
   (R['x': None, 'y': 0], "{x: ?int8, y: int64}"),
   (R['x': 0, 'y': None], "{x: int8, y: ?int64}"),
   (R['x': None, 'y': None], "{x: ?int8, y: ?int64}"),
   (None, "?{x: ?int8, y: ?int64}"),

   (R['x': 0, 'y': None], " {x: uint16, y: ?{z: complex64}}"),
   (R['x': 0, 'y': R['z': None]], "{x: uint16, y: {z: ?complex64}}"),
   (R['x': 0, 'y': None], "{x: uint16, y: ?{z: ?complex64}}"),

   (R['x': None, 'y': R['z': 0+0j]], "{x: ?uint16, y: {z: complex64}, pack=1}"),
   (R['x': 0, 'y': None], "{x: uint16, y: ?{z: complex64}, pack=1}"),
   (R['x': 0, 'y': R['z': None]], "{x: uint16, y: {z: ?complex64}, pack=1}"),

   (R['x': []], "{x: 0 * ?bytes}"),
   (R['x': 1 * [None]], "{x: 1 * ?bytes}"),
   (R['x': 10 * [None]], "{x: 10 * ?bytes}"),
   (R['x': [], 'y': []], "{x: 0 * ?bytes, y: 0 * ?string}"),
   (R['x': 5 * [None], 'y': 2 * [""]], "{x: 5 * ?bytes, y: 2 * string}"),
   (R['x': 5 * [b''], 'y': 2 * [None]], "{x: 5 * bytes, y: 2 * ?string}"),
   (R['x': 5 * [None], 'y': 2 * [None]], "{x: 5 * ?bytes, y: 2 * ?string}"),

   (R['x': [None], 'y': 2 * [None], 'z': 3 * [None]], "{x: 1 * ?bytes, y: 2 * ?complex128, z: 3 * ?string}"),
   (R['x': [None], 'y': 2 * [R['a': 0.0j, 'b': 10 * [2 * [""]]]], 'z': 3 * [""]], "{x: 1 * ?bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * string}"),
   (R['x': [b''], 'y': 2 * [None], 'z': 3 * [""]], "{x: 1 * bytes, y: 2 * ?{a: complex128, b: 10 * 2 * string}, z: 3 * string}"),
   (R['x': [b''], 'y': 2 * [R['a': 0.0j, 'b': 10 * [2 * [""]]]], 'z': 3 * [None]], "{x: 1 * bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * ?string}"),
   (R['x': [None], 'y': 2 * [None], 'z': 3 * [""]], "{x: 1 * ?bytes, y: 2 * ?{a: complex128, b: 10 * 2 * string}, z: 3 * string}"),
   (R['x': [None], 'y': 2 * [R['a': 0.0j, 'b': 10 * [2 * [""]]]], 'z': 3 * [None]], "{x: 1 * ?bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * ?string}"),
   (R['x': [None], 'y': 2 * [None], 'z': 3 * [None]], "{x: 1 * ?bytes, y: 2 * ?{a: complex128, b: 10 * 2 * string}, z: 3 * ?string}"),

   (R['x': [b''], 'y': 2 * [R['a': 0.0j, 'b': 10 * [2 * [None]]]], 'z': 3 * [""]], "{x: 1 * bytes, y: 2 * {a: complex128, b: 10 * 2 * ?string}, z: 3 * string}"),
   (R['x': [None], 'y': 2 * [R['a': 0.0j, 'b': 10 * [2 * [None]]]], 'z': 3 * [""]], "{x: 1 * ?bytes, y: 2 * {a: complex128, b: 10 * 2 * ?string}, z: 3 * string}"),
   (R['x': [None], 'y': 2 * [R['a': None, 'b': 10 * [2 * [None]]]], 'z': 3 * [""]], "{x: 1 * ?bytes, y: 2 * {a: ?complex128, b: 10 * 2 * ?string}, z: 3 * string}"),
   (R['x': [b''], 'y': 2 * [R['a': None, 'b': 10 * [2 * [None]]]], 'z': 3 * [None]], "{x: 1 * bytes, y: 2 * {a: ?complex128, b: 10 * 2 * ?string}, z: 3 * ?string}"),

   (R['x': 0, 'y': 2 * [3 * [4 * [5 * [None]]]]], "{x: int64, y: 2 * 3 * Some(4 * 5 * ?float64)}"),
   (R['x': 0, 'y': 2 * [3 * [4 * [5 * [None]]]]], "{x: int64, y: 2 * 3 * ref(4 * 5 * ?float64)}"),

   # Primitive types
   (False, "bool"),
   (0, "bool"),

   (0, "int8"),
   (0, "int16"),
   (0, "int32"),
   (0, "int64"),

   (0, "uint8"),
   (0, "uint16"),
   (0, "uint32"),
   (0, "uint64"),

   (0.0, "float32"),
   (0.0, "float64"),

   (0+0j, "complex64"),
   (0+0j, "complex128"),

   (0+0j, "complex64"),
   (0+0j, "complex128"),

   # Optional primitive types
   (None, "?bool"),

   (None, "?int8"),
   (None, "?int16"),
   (None, "?int32"),
   (None, "?int64"),

   (None, "?uint8"),
   (None, "?uint16"),
   (None, "?uint32"),
   (None, "?uint64"),

   (None, "?float32"),
   (None, "?float64"),

   (None, "?complex64"),
   (None, "?complex128"),

   (None, "?complex64"),
   (None, "?complex128"),

   # References
   (False, "&bool"),
   (0, "&bool"),

   (0, "&int8"),
   (0, "&int16"),
   (0, "&int32"),
   (0, "&int64"),

   (0, "ref(uint8)"),
   (0, "ref(uint16)"),
   (0, "ref(uint32)"),
   (0, "ref(uint64)"),

   (0, "ref(ref(uint8))"),
   (0, "ref(ref(uint16))"),
   (0, "ref(ref(uint32))"),
   (0, "ref(ref(uint64))"),

   (0.0, "ref(float32)"),
   (0.0, "ref(float64)"),

   (0+0j, "ref(complex64)"),
   (0+0j, "ref(complex128)"),

   ([], "ref(0 * bool)"),
   ([0], "ref(1 * int16)"),
   (2 * [0], "ref(2 * int32)"),
   (2 * [3 * [0]], "ref(2 * 3 * int8)"),

   ([], "ref(ref(0 * bool))"),
   ([0], "ref(ref(1 * int16))"),
   (2 * [0], "ref(ref(2 * int32))"),
   (2 * [3 * [0]], "ref(ref(2 * 3 * int8))"),

   ([], "ref(!0 * bool)"),
   ([0], "ref(!1 * int16)"),
   (2 * [0], "ref(!2 * int32)"),
   (2 * [3 * [0]], "ref(!2 * 3 * int8)"),

   ([], "ref(ref(!0 * bool))"),
   ([0], "ref(ref(!1 * int16))"),
   (2 * [0], "ref(ref(!2 * int32))"),
   (2 * [3 * [0]], "ref(ref(!2 * 3 * int8))"),

   # Optional references
   (None, "?&bool"),

   (None, "?&int8"),
   (None, "?&int16"),
   (None, "?&int32"),
   (None, "?&int64"),

   (None, "?ref(uint8)"),
   (None, "?ref(uint16)"),
   (None, "?ref(uint32)"),
   (None, "?ref(uint64)"),

   (None, "?ref(ref(uint8))"),
   (None, "?ref(ref(uint16))"),
   (None, "?ref(ref(uint32))"),
   (None, "?ref(ref(uint64))"),

   (None, "?ref(float32)"),
   (None, "?ref(float64)"),

   (None, "?ref(complex64)"),
   (None, "?ref(complex128)"),

   (None, "?ref(0 * bool)"),
   (None, "?ref(1 * int16)"),
   (None, "?ref(2 * int32)"),
   (None, "?ref(2 * 3 * int8)"),

   (None, "?ref(ref(0 * bool))"),
   (None, "ref(?ref(0 * bool))"),
   (None, "?ref(?ref(0 * bool))"),
   (None, "?ref(ref(1 * int16))"),
   (None, "ref(?ref(1 * int16))"),
   (None, "?ref(?ref(1 * int16))"),
   (None, "?ref(ref(2 * int32))"),

   (None, "?ref(!2 * 3 * int8)"),
   (None, "?ref(ref(!2 * 3 * int32))"),

   # References to types with optional data
   (None, "&?bool"),

   (None, "&?int8"),
   (None, "&?int16"),
   (None, "&?int32"),
   (None, "&?int64"),

   (None, "ref(?uint8)"),
   (None, "ref(?uint16)"),
   (None, "ref(?uint32)"),
   (None, "ref(?uint64)"),

   (None, "ref(ref(?uint8))"),
   (None, "ref(ref(?uint16))"),
   (None, "ref(ref(?uint32))"),
   (None, "ref(ref(?uint64))"),

   (None, "ref(?float32)"),
   (None, "ref(?float64)"),

   (None, "ref(?complex64)"),
   (None, "ref(?complex128)"),

   ([], "ref(0 * ?bool)"),
   ([None], "ref(1 * ?int16)"),
   (2 * [None], "ref(2 * ?int32)"),
   (2 * [3 * [None]], "ref(2 * 3 * ?int8)"),

   ([], "ref(ref(0 * ?bool))"),
   ([None], "ref(ref(1 * ?int16))"),
   (2 * [None], "ref(ref(2 * ?int32))"),

   (2 * [3 * [None]], "ref(!2 * 3 * ?int8)"),
   (2 * [3 * [None]], "ref(ref(!2 * 3 * ?int8))"),

   # Constructors
   (False, "Some(bool)"),
   (0, "Some(bool)"),

   (0, "Some(int8)"),
   (0, "Some(int16)"),
   (0, "Some(int32)"),
   (0, "Some(int64)"),

   (0, "Some(uint8)"),
   (0, "Some(uint16)"),
   (0, "Some(uint32)"),
   (0, "Some(uint64)"),

   (0.0, "Some(float32)"),
   (0.0, "Some(float64)"),

   (0+0j, "Some(complex64)"),
   (0+0j, "Some(complex128)"),

   ([0], "ThisGuy(1 * int16)"),
   (2 * [0], "ThisGuy(2 * int32)"),
   (2 * [3 * [0.0]], "ThisGuy(2 * 3 * float32)"),

   (2 * [3 * [0.0]], "ThisGuy(!2 * 3 * float32)"),

   # Optional constructors
   (None, "?Some(bool)"),

   (None, "?Some(int8)"),
   (None, "?Some(int16)"),
   (None, "?Some(int32)"),
   (None, "?Some(int64)"),

   (None, "?Some(uint8)"),
   (None, "?Some(uint16)"),
   (None, "?Some(uint32)"),
   (None, "?Some(uint64)"),

   (None, "?Some(float32)"),
   (None, "?Some(float64)"),

   (None, "?Some(complex64)"),
   (None, "?Some(complex128)"),

   (None, "?ThisGuy(0 * int16)"),
   (None, "?ThisGuy(1 * int16)"),
   (None, "?ThisGuy(2 * int32)"),
   (None, "?ThisGuy(2 * 3 * float32)"),

   (None, "?ThisGuy(!2 * 3 * float32)"),

   # Constructors with an optional data type argument
   (None, "Some(?bool)"),

   (None, "Some(?int8)"),
   (None, "Some(?int16)"),
   (None, "Some(?int32)"),
   (None, "Some(?int64)"),

   (None, "Some(?uint8)"),
   (None, "Some(?uint16)"),
   (None, "Some(?uint32)"),
   (None, "Some(?uint64)"),

   (None, "Some(?float32)"),
   (None, "Some(?float64)"),

   (None, "Some(?complex64)"),
   (None, "Some(?complex128)"),

   ([], "ThisGuy(0 * ?int16)"),
   ([None], "ThisGuy(1 * ?int16)"),
   (2 * [None], "ThisGuy(2 * ?int32)"),
   (2 * [3 * [None]], "ThisGuy(2 * 3 * ?float32)"),

   (2 * [3 * [None]], "ThisGuy(!2 * 3 * ?float32)"),
]


# ======================================================================
#            Definition of generalized slicing and indexing
# ======================================================================

def maxlevel(lst):
    """Return maximum nesting depth"""
    maxlev = 0
    def f(lst, level):
        nonlocal maxlev
        if isinstance(lst, list):
            level += 1
            maxlev = max(level, maxlev)
            for item in lst:
                f(item, level)
    f(lst, 0)
    return maxlev

def getitem(lst, indices):
    """Definition for multidimensional slicing and indexing on arbitrarily
       shaped nested lists.
    """
    if not indices:
        return lst

    i, indices = indices[0], indices[1:]
    item = list.__getitem__(lst, i)

    if isinstance(i, int):
        return getitem(item, indices)

    if not item:
        # Empty slice: check if all subsequent indices are in range for the
        # full slice, raise IndexError otherwise. This is NumPy's behavior.
        _ = [getitem(x, indices) for x in lst]
        return []

    return [getitem(x, indices) for x in item]

class NDArray(list):
    """A simple wrapper for using generalized slicing/indexing on a list."""
    def __init__(self, value):
        list.__init__(self, value)
        self.maxlevel = maxlevel(value)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) > self.maxlevel: # NumPy
            raise IndexError("too many indices")

        if not all(isinstance(i, (int, slice)) for i in indices):
            raise TypeError(
                "index must be int or slice or a tuple of integers and slices")

        result = getitem(self, indices)
        return NDArray(result) if isinstance(result, list) else result


# ======================================================================
#                          Generate test cases 
# ======================================================================

SUBSCRIPT_FIXED_TEST_CASES = [
  [],
  [[]],
  [[], []],
  [[0], [1]],
  [[0], [1], [2]],
  [[0, 1], [1, 2], [2 ,3]],
  [[[]]],
  [[[0]]],
  [[[], []]],
  [[[0], [1]]],
  [[[0, 1], [2, 3]]],
  [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
  [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
]

SUBSCRIPT_VAR_TEST_CASES = [
  [[[0, 1], [2, 3]], [[4, 5, 6], [7]], [[8, 9]]],
  [[[0, 1], [2, 3]], [[4, 5, None], [None], [7]], [[], [None, 8]], [[9, 10]]],
  [[[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]], [[11, 12, 13, 14], [15, 16, 17], [18, 19]]],
  [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9]], [[10, 11]]]
]

def single_fixed(max_ndim=4, min_shape=1, max_shape=10):
    nat = count()
    shape = [randrange(min_shape, max_shape+1) for _ in range(max_ndim)]

    def f(ndim):
        if ndim == 0:
            return next(nat)
        return [f(ndim-1) for _ in range(shape[ndim-1])]

    return f(max_ndim)

def gen_fixed(max_ndim=4, min_shape=1, max_shape=10):
    assert max_ndim >=0 and min_shape >=0 and min_shape <= max_shape

    for _ in range(30):
        yield single_fixed(max_ndim, min_shape, max_shape)

def single_var(max_ndim=4, min_shape=1, max_shape=10):
    nat = count()

    def f(ndim):
        if ndim == 0:
            return next(nat)
        if ndim == 1:
            shape = randrange(min_shape, max_shape+1)
        else:
            n = 1 if min_shape == 0 else min_shape
            shape = randrange(n, max_shape+1)
        return [f(ndim-1) for _ in range(shape)]

    return f(max_ndim)

def gen_var(max_ndim=4, min_shape=1, max_shape=10):
    assert max_ndim >=0 and min_shape >=0 and min_shape <= max_shape

    for _ in range(30):
        yield single_var(max_ndim, min_shape, max_shape)


def genindices():
    for i in range(4):
        yield (i,)
    for i in range(4):
        for j in range(4):
            yield (i, j)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                yield (i, j, k)

def rslice(ndim):
    start = randrange(0, ndim+1)
    stop = randrange(0, ndim+1)
    step = 0
    while step == 0:
        step = randrange(-ndim-1, ndim+1)
    start = None if randrange(5) == 4 else start
    stop = None if randrange(5) == 4 else stop
    step = None if randrange(5) == 4 else step
    return slice(start, stop, step)

def multislice(ndim):
    return tuple(rslice(ndim) for _ in range(randrange(1, ndim+1)))

def randslices(ndim):
    for i in range(5):
        yield multislice(ndim)

def gen_indices_or_slices():
    for i in range(5):
        if randrange(2):
            yield (randrange(4), randrange(4), randrange(4))
        else:
            yield multislice(3)

def genslices(n):
    """Generate all possible slices for a single dimension."""
    def range_with_none():
        yield None
        yield from range(-n, n+1)

    for t in product(range_with_none(), range_with_none(), range_with_none()):
        s = slice(*t)
        if s.step != 0:
            yield s

def genslices_ndim(ndim, shape):
    """Generate all possible slice tuples for 'shape'."""
    iterables = [genslices(shape[n]) for n in range(ndim)]
    yield from product(*iterables)

def mixed_index(max_ndim):
    ndim = randrange(1, max_ndim+1)
    indices = []
    for i in range(1, ndim+1):
        if randrange(2):
            indices.append(randrange(max_ndim))
        else:
            indices.append(rslice(ndim))
    return tuple(indices)

def mixed_indices(max_ndim):
    for i in range(5):
        yield mixed_index(max_ndim)

def itos(indices):
    return ", ".join(str(i) if isinstance(i, int) else "%s:%s:%s" %
                     (i.start, i.stop, i.step) for i in indices)

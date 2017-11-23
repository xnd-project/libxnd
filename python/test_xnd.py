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

import sys, unittest
from collections import OrderedDict
from functools import partial
from ndtypes import ndt
from xnd import xnd
from randvalue import *


try:
    import numpy as np
except ImportError:
    np = None


# OrderedDict literals hack.
class Record(OrderedDict):
    @staticmethod
    def _kv(s):
        if not isinstance(s, slice):
            raise TypeError("expect key-value pair")
        if s.step is not None:
            raise ValueError("expect key-value pair")
        return s.start, s.stop
    def __getitem__(self, items):
        if not isinstance(items, tuple):
            items = (items,)
        return OrderedDict(list(map(self._kv, items)))

R = Record()


primitive = [
  'bool',
  'int8', 'int16', 'int32', 'int64',
  'uint8', 'uint16', 'uint32', 'uint64',
  'float16', 'float32', 'float64',
  'complex32', 'complex64', 'complex128'
]


class EmptyConstructionTest(unittest.TestCase):

    def test_primitive_empty(self):
        test_cases = [
            '%s', '0 * %s', '1 * %s', 'var(offsets=[0, 2]) * %s',
            '10 * {a: int64, b: %s}'
        ]

        for c in test_cases:
            for p in primitive:
                s = c % p
                t = ndt(s)
                x = xnd.empty(s)
                self.assertEqual(x.type, t)

    def test_fixed_string_empty(self):
        test_cases = [
          'fixed_string(1)',
          'fixed_string(100)',
          "fixed_string(1, 'ascii')",
          "fixed_string(100, 'utf8')",
          "fixed_string(200, 'utf16')",
          "fixed_string(300, 'utf32')",
        ]

        for s in test_cases:
            t = ndt(s)
            x = xnd.empty(s)
            self.assertEqual(x.type, t)

    def test_fixed_bytes_empty(self):
        test_cases = [
          'fixed_bytes(size=1)',
          'fixed_bytes(size=100)',
          'fixed_bytes(size=1, align=2)',
          'fixed_bytes(size=100, align=16)',
        ]

        for s in test_cases:
            t = ndt(s)
            x = xnd.empty(s)
            self.assertEqual(x.type, t)

    def test_string_empty(self):
        test_cases = [
          'string',
          '(string)',
          '10 * 2 * string',
          '10 * 2 * (string, string)',
          '10 * 2 * {a: string, b: string}',
          'var(offsets=[0,3]) * var(offsets=[0,2,7,10]) * {a: string, b: string}'
        ]

        for s in test_cases:
            t = ndt(s)
            x = xnd.empty(s)
            self.assertEqual(x.type, t)

    def test_bytes_empty(self):
        test_cases = [
          'bytes(align=16)',
          '(bytes(align=32))',
          '10 * 2 * bytes',
          '10 * 2 * (bytes, bytes)',
          '10 * 2 * {a: bytes(align=32), b: bytes(align=1)}',
          '10 * 2 * {a: bytes(align=1), b: bytes(align=32)}',
          'var(offsets=[0,3]) * var(offsets=[0,2,7,10]) * {a: bytes(align=32), b: bytes}'
        ]

        for s in test_cases:
            t = ndt(s)
            x = xnd.empty(s)
            self.assertEqual(x.type, t)


class TypeInferenceTest(unittest.TestCase):

    def test_tuple(self):
        d = R['a': (2.0, b"bytes"), 'b': ("str", float('inf'))]
        typeof_d = "{a: (float64, bytes), b: (string, float64)}"

        test_cases = [
          ((), "()"),
          (((),), "(())"),
          (((), ()), "((), ())"),
          ((((),), ()), "((()), ())"),
          ((((),), ((), ())), "((()), ((), ()))"),
          ((1, 2, 3), "(int64, int64, int64)"),
          ((1.0, 2, "str"), "(float64, int64, string)"),
          ((1.0, 2, ("str", b"bytes", d)),
           "(float64, int64, (string, bytes, %s))" % typeof_d)
        ]

        for v, t in test_cases:
            x = xnd(v)
            self.assertEqual(x.type, ndt(t))
            self.assertEqual(x.value, v)

    def test_record(self):
        d = R['a': (2.0, b"bytes"), 'b': ("str", float('inf'))]
        typeof_d = "{a: (float64, bytes), b: (string, float64)}"

        test_cases = [
          ({}, "{}"),
          ({'x': {}}, "{x: {}}"),
          (R['x': {}, 'y': {}], "{x: {}, y: {}}"),
          (R['x': R['y': {}], 'z': {}], "{x: {y: {}}, z: {}}"),
          (R['x': R['y': {}], 'z': R['a': {}, 'b': {}]], "{x: {y: {}}, z: {a: {}, b: {}}}"),
          (d, typeof_d)
        ]

        for v, t in test_cases:
            x = xnd(v)
            self.assertEqual(x.type, ndt(t))
            self.assertEqual(x.value, v)

    def test_float64(self):
        d = R['a': 2.221e100, 'b': float('inf')]
        typeof_d = "{a: float64, b: float64}"

        test_cases = [
          # 'float64' is the default dtype if there is no data at all.
          ([], "0 * float64"),
          ([[]], "1 * 0 * float64"),
          ([[], []], "2 * 0 * float64"),
          ([[[]], [[]]], "2 * 1 * 0 * float64"),
          ([[[]], [[], []]], "var(offsets=[0, 2]) * var(offsets=[0, 1, 3]) * var(offsets=[0, 0, 0, 0]) * float64"),

          ([0.0], "1 * float64"),
          ([0.0, 1.2], "2 * float64"),
          ([[0.0], [1.2]], "2 * 1 * float64"),

          (d, typeof_d),
          ([d] * 2, "2 * %s" % typeof_d),
          ([[d] * 2] * 10, "10 * 2 * %s" % typeof_d)
        ]

        for v, t in test_cases:
            x = xnd(v)
            self.assertEqual(x.type, ndt(t))
            self.assertEqual(x.value, v)

    def test_int64(self):
        t = (1, -2, -3)
        typeof_t = "(int64, int64, int64)"

        test_cases = [
          ([0], "1 * int64"),
          ([0, 1], "2 * int64"),
          ([[0], [1]], "2 * 1 * int64"),

          (t, typeof_t),
          ([t] * 2, "2 * %s" % typeof_t),
          ([[t] * 2] * 10, "10 * 2 * %s" % typeof_t)
        ]

        for v, t in test_cases:
            x = xnd(v)
            self.assertEqual(x.type, ndt(t))
            self.assertEqual(x.value, v)

    def test_string(self):
        t = ("supererogatory", "exiguous")
        typeof_t = "(string, string)"

        test_cases = [
          (["mov"], "1 * string"),
          (["mov", "$0"], "2 * string"),
          ([["cmp"], ["$0"]], "2 * 1 * string"),

          (t, typeof_t),
          ([t] * 2, "2 * %s" % typeof_t),
          ([[t] * 2] * 10, "10 * 2 * %s" % typeof_t)
        ]

        for v, t in test_cases:
            x = xnd(v)
            self.assertEqual(x.type, ndt(t))
            self.assertEqual(x.value, v)

    def test_bytes(self):
        t = (b"lagrange", b"points")
        typeof_t = "(bytes, bytes)"

        test_cases = [
          ([b"L1"], "1 * bytes"),
          ([b"L2", b"L3", b"L4"], "3 * bytes"),
          ([[b"L5"], [b"none"]], "2 * 1 * bytes"),

          (t, typeof_t),
          ([t] * 2, "2 * %s" % typeof_t),
          ([[t] * 2] * 10, "10 * 2 * %s" % typeof_t)
        ]

        for v, t in test_cases:
            x = xnd(v)
            self.assertEqual(x.type, ndt(t))
            self.assertEqual(x.value, v)


class IndexTest(unittest.TestCase):

    def test_indexing(self):
        x = xnd([])
        self.assertRaises(IndexError, x.__getitem__, 0)
        self.assertRaises(IndexError, x.__getitem__, (0, 0))

        x = xnd([0])
        self.assertEqual(x[0], 0)

        self.assertRaises(IndexError, x.__getitem__, 1)
        self.assertRaises(IndexError, x.__getitem__, (0, 1))

        x = xnd([[0,1,2], [3,4,5]])
        self.assertEqual(x[0,0], 0)
        self.assertEqual(x[0,1], 1)
        self.assertEqual(x[0,2], 2)

        self.assertEqual(x[1,0], 3)
        self.assertEqual(x[1,1], 4)
        self.assertEqual(x[1,2], 5)

        self.assertRaises(IndexError, x.__getitem__, (0, 3))
        self.assertRaises(IndexError, x.__getitem__, (2, 0))

        t1 = (1.0, "capricious", (1, 2, 3))
        t2 = (2.0, "volatile", (4, 5, 6))

        x = xnd([t1, t2])
        self.assertEqual(x[0], t1)
        self.assertEqual(x[1], t2)

        self.assertEqual(x[0,0], 1.0)
        self.assertEqual(x[0,1], "capricious")
        self.assertEqual(x[0,2], (1, 2, 3))

        self.assertEqual(x[1,0], 2.0)
        self.assertEqual(x[1,1], "volatile")
        self.assertEqual(x[1,2], (4, 5, 6))

    def test_subview(self):
        # fixed
        x = xnd([["a", "b"], ["c", "d"]])
        self.assertEqual(x[0].value, ["a", "b"])
        self.assertEqual(x[1].value, ["c", "d"])

        # var
        x = xnd([["a", "b"], ["x", "y", "z"]])
        self.assertEqual(x[0].value, ["a", "b"])
        self.assertEqual(x[1].value, ["x", "y", "z"])


class LongIndexSliceTest(unittest.TestCase):

    def log_err(self, lst, indices_stack, depth):
        """Dump the error as a Python script for debugging."""

        sys.stderr.write("\n\nfrom xnd import *\n")
        sys.stderr.write("from test_xnd import NDArray\n")
        sys.stderr.write("lst = %s\n\n" % lst)
        sys.stderr.write("x0 = xnd(lst)\n")
        sys.stderr.write("y0 = NDArray(lst)\n" % lst)

        for i in range(depth+1):
            sys.stderr.write("x%d = x%d[%s]\n" % (i+1, i, itos(indices_stack[i])))
            sys.stderr.write("y%d = y%d[%s]\n" % (i+1, i, itos(indices_stack[i])))

        sys.stderr.write("\n")

    def compare(self, lst, x, definition, indices, indices_stack, depth):
        """Run a single test case."""

        xnd_exc = None
        try:
            xnd_result = x[indices]
        except Exception as e:
            xnd_exc =  e

        definition_exc = None
        try:
            definition_result = definition[indices]
        except Exception as e:
            definition_exc = e

        if xnd_exc or definition_exc:
            if xnd_exc is None and definition_exc.__class__ is IndexError:
                # Example: type = 0 * 0 * int64
                if len(indices) <= x.ndim:
                    return None, None

            if xnd_exc.__class__ is not definition_exc.__class__:
                self.log_err(lst, indices_stack, depth)

            self.assertIs(xnd_exc.__class__, definition_exc.__class__)

            return None, None

        else:
            try:
                if isinstance(xnd_result, xnd):
                    xnd_lst = xnd_result.value
                else:
                    xnd_lst = xnd_result
            except Exception as e:
                self.log_err(lst, indices_stack, depth)
                raise e

            if xnd_lst != definition_result:
                self.log_err(lst, indices_stack, depth)

            self.assertEqual(xnd_lst, definition_result)

            return xnd_result, definition_result

    def doit(self, array=None, tests=None, genindices=None, genlists=None):
        indices_stack = [None] * 8

        def check(lst, x, definition, depth):
            if depth > 3: # adjust for longer tests
                return

            for indices in genindices():
                indices_stack[depth] = indices
                _xnd, _def = self.compare(lst, x, definition, indices,
                                          indices_stack, depth)

                if isinstance(_def, list): # possibly None or scalar
                    check(lst, _xnd, _def, depth+1)

        for lst in tests:
            check(lst, array(lst), NDArray(lst), 0)

        for max_ndim in range(1, 5):
            for min_shape in (0, 1):
                for max_shape in range(1, 8):
                    for lst in genlists(max_ndim, min_shape, max_shape):
                        check(lst, array(lst), NDArray(lst), 0)

    def test_subarray(self):
        # Multidimensional indexing
        self.doit(xnd, FIXED_TEST_CASES, genindices, gen_fixed)
        # self.doit(xnd, VAR_TEST_CASES, genindices, gen_var)

    def test_slices(self):
        # Multidimensional slicing
        genindices = partial(randslices, 3)
        self.doit(xnd, FIXED_TEST_CASES, genindices, gen_fixed)
        # self.doit(xnd, VAR_TEST_CASES, genindices, gen_var)

    def test_chained_indices_slices(self):
        # Multidimensional indexing and slicing, chained
        self.doit(xnd, FIXED_TEST_CASES, gen_indices_or_slices, gen_fixed)
        # self.doit(xnd, VAR_TEST_CASES, gen_indices_or_slices, gen_var)

    def test_mixed_indices_slices(self):
        # Multidimensional indexing and slicing, mixed
        genindices = partial(mixed_indices, 3)
        self.doit(xnd, FIXED_TEST_CASES, genindices, gen_fixed)
        # self.doit(xnd, VAR_TEST_CASES, genindices, gen_var)

    @unittest.skipIf(np is None, "numpy not found")
    def test_array_definition(self):
        # Test the NDArray definition against NumPy
        genindices = partial(mixed_indices, 3)
        self.doit(np.array, FIXED_TEST_CASES, genindices, gen_fixed)

    @unittest.skipIf(True, "very long duration")
    def test_slices_brute_force(self):
        # Test all possible slices for the given ndim and shape
        genindices = partial(genslices_ndim, 3, [3,3,3])
        self.doit(np.array, FIXED_TEST_CASES, genindices, gen_fixed)
        # self.doit(np.array, VAR_TEST_CASES, genindices, gen_var)


if __name__ == '__main__':
    unittest.main(verbosity=2)

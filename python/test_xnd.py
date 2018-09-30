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

import sys, unittest, argparse
from math import isinf, isnan
from ndtypes import ndt, typedef
from xnd import xnd, XndEllipsis
from xnd._xnd import _test_view_subscript, _test_view_new
from xnd_support import *
from xnd_randvalue import *
from _testbuffer import ndarray, ND_WRITABLE


try:
    import numpy as np
except ImportError:
    np = None


SKIP_LONG = True
SKIP_BRUTE_FORCE = True


def check_buffer(x):
    try:
        y = memoryview(x)
    except ValueError:
        return
    with memoryview(x) as y:
        del x
        y.tobytes()


class XndTestCase(unittest.TestCase):

    def assertStrictEqual(self, x, y):
        self.assertTrue(x.strict_equal(y))
        self.assertEqual(x, y)

    def assertNotStrictEqual(self, x, y):
        self.assertFalse(x.strict_equal(y))
        self.assertNotEqual(x, y)


class TestModule(XndTestCase):

    def test_module(self):
        test_cases = [
          "Foo:: 2 * 3 * ?int64",
          "Foo:: 10 * 2 * ?string",
          "Bar:: !10 * 2 * {a: !2 * ?int64}",
          "Quux:: {a: string, b: ?bytes}"
        ]

        for s in test_cases:
            self.assertRaises(ValueError, xnd.empty, s)


class TestFunction(XndTestCase):

    def test_function(self):
        test_cases = [
          "(2 * 3 * ?int64, complex128) -> (T, T)",
          "(2 * 3 * ?int64, {a: float64, b: bytes}) -> bytes",
        ]

        for s in test_cases:
            self.assertRaises(ValueError, xnd.empty, s)


class TestVoid(XndTestCase):

    def test_void(self):
        self.assertRaises(ValueError, xnd.empty, "void")
        self.assertRaises(ValueError, xnd.empty, "10 * 2 * void")


class TestAny(XndTestCase):

    def test_any(self):
        test_cases = [
          "Any",
          "10 * 2 * Any",
          "10 * N * int64",
          "{a: string, b: Any}"
        ]

        for s in test_cases:
            self.assertRaises(ValueError, xnd.empty, s)


class TestFixedDim(XndTestCase):

    def test_fixed_dim_empty(self):
        for v, s in DTYPE_EMPTY_TEST_CASES:
            for vv, ss in [
               (0 * [v], "0 * %s" % s),
               (1 * [v], "1 * %s" % s),
               (2 * [v], "2 * %s" % s),
               (1000 * [v], "1000 * %s" % s),

               (0 * [0 * [v]], "0 * 0 * %s" % s),
               (0 * [1 * [v]], "0 * 1 * %s" % s),
               (1 * [0 * [v]], "1 * 0 * %s" % s),

               (1 * [1 * [v]], "1 * 1 * %s" % s),
               (1 * [2 * [v]], "1 * 2 * %s" % s),
               (2 * [1 * [v]], "2 * 1 * %s" % s),
               (2 * [2 * [v]], "2 * 2 * %s" % s),
               (2 * [3 * [v]], "2 * 3 * %s" % s),
               (3 * [2 * [v]], "3 * 2 * %s" % s),
               (3 * [40 * [v]], "3 * 40 * %s" % s) ]:

                t = ndt(ss)
                x = xnd.empty(ss)
                self.assertEqual(x.type, t)
                self.assertEqual(x.value, vv)
                self.assertEqual(len(x), len(vv))

    def test_fixed_dim_subscript(self):
        test_cases = [
            ([[11.12-2.3j, -1222+20e8j],
              [complex("inf"), -0.00002j],
              [0.201+1j, -1+1e301j]], "3 * 2 * complex128"),
            ([[11.12-2.3j, None],
              [complex("inf"), None],
              [0.201+1j, -1+1e301j]], "3 * 2 * ?complex128")
        ]

        for v, s in test_cases:
            nd = NDArray(v)
            t = ndt(s)
            x = xnd(v, type=t)
            check_buffer(x)

            for i in range(3):
                self.assertEqual(x[i].value, nd[i])

            for i in range(3):
                for k in range(2):
                    self.assertEqual(x[i][k], nd[i][k])
                    self.assertEqual(x[i, k], nd[i][k])

            self.assertEqual(x[:].value, nd[:])

            for start in list(range(-3, 4)) + [None]:
                for stop in list(range(-3, 4)) + [None]:
                    for step in list(range(-3, 0)) + list(range(1, 4)) + [None]:
                        s = slice(start, stop, step)
                        self.assertEqual(x[s].value, nd[s])
                        check_buffer(x[s])

            self.assertEqual(x[:, 0].value, nd[:, 0])
            self.assertEqual(x[:, 1].value, nd[:, 1])

    def test_fixed_dim_assign(self):
        ### Regular data ###
        x = xnd.empty("2 * 4 * float64")
        v = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]

        # Full slice
        x[:] = v
        self.assertEqual(x.value, v)

        # Subarray
        x[0] = v[0] = [1.2, -3e45, float("inf"), -322.25]
        self.assertEqual(x.value, v)

        x[1] = v[1] = [-11.25, 3.355e301, -0.000002, -5000.2]
        self.assertEqual(x.value, v)

        # Single values
        for i in range(2):
            for j in range(4):
                x[i][j] = v[i][j] = 3.22 * i + j
        self.assertEqual(x.value, v)

        # Tuple indexing
        for i in range(2):
            for j in range(4):
                x[i, j] = v[i][j] = -3.002e1 * i + j
        self.assertEqual(x.value, v)


        ### Optional data ###
        x = xnd.empty("2 * 4 * ?float64")
        v = [[10.0, None, 2.0, 100.12], [None, None, 6.0, 7.0]]

        # Full slice
        x[:] = v
        self.assertEqual(x.value, v)

        # Subarray
        x[0] = v[0] = [None, 3e45, float("inf"), None]
        self.assertEqual(x.value, v)

        x[1] = v[1] = [-11.25, 3.355e301, -0.000002, None]
        self.assertEqual(x.value, v)

        # Single values
        for i in range(2):
            for j in range(4):
                x[i][j] = v[i][j] = -325.99 * i + j
        self.assertEqual(x.value, v)

        # Tuple indexing
        for i in range(2):
            for j in range(4):
                x[i, j] = v[i][j] = -8.33e1 * i + j
        self.assertEqual(x.value, v)

    @unittest.skipIf(sys.platform == "darwin",
                     "mach_vm_map message defeats the purpose of this test")
    def test_fixed_dim_overflow(self):
        # Type cannot be created.
        s = "2147483648 * 2147483648 * 2 * uint8"
        self.assertRaises(ValueError, xnd.empty, s)

        if HAVE_64_BIT:
            # Allocation fails.
            s = "2147483648 * 2147483647 * 2 * uint8"
            self.assertRaises(MemoryError, xnd.empty, s)
        else:
            # Allocation fails.
            s = "32768 * 32768 * 2 * uint8"
            self.assertRaises(MemoryError, xnd.empty, s)

    def test_fixed_dim_richcompare(self):

        x = xnd([1,2,3,4])

        self.assertIs(x.__lt__(x), NotImplemented)
        self.assertIs(x.__le__(x), NotImplemented)
        self.assertIs(x.__gt__(x), NotImplemented)
        self.assertIs(x.__ge__(x), NotImplemented)

        self.assertStrictEqual(x, xnd([1,2,3,4]))

        # Different shape and/or data.
        self.assertNotStrictEqual(x, xnd([1,2,3,100]))
        self.assertNotStrictEqual(x, xnd([1,2,3]))
        self.assertNotStrictEqual(x, xnd([1,2,3,4,5]))

        # Different shape.
        self.assertNotStrictEqual(x, xnd([1,2,3]))
        self.assertNotStrictEqual(x, xnd([[1,2,3,4]]))
        self.assertNotStrictEqual(x, xnd([[1,2], [3,4]]))

        # Simple multidimensional arrays.
        x = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
        y = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
        self.assertStrictEqual(x, y)

        for i in range(4):
            for k in range(3):
                v = y[i, k]
                y[i, k] = 100
                self.assertNotStrictEqual(x, y)
                y[i, k] = v

        # C <-> Fortran.
        x = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
        y = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type="!4 * 3 * int64")
        self.assertStrictEqual(x, y)

        for i in range(4):
            for k in range(3):
                v = y[i, k]
                y[i, k] = 100
                self.assertNotStrictEqual(x, y)
                y[i, k] = v

        # Slices.
        x = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
        y = xnd([[1,2,3], [7,8,9]], type="!2 * 3 * int64")
        self.assertStrictEqual(x[::2], y)

        x = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
        y = xnd([1,4,7,10])
        self.assertStrictEqual(x[:, 0], y)

        # Test corner cases and many dtypes.
        for v, t, u, _, _ in EQUAL_TEST_CASES:
            for vv, tt, uu in [
               (0 * [v], "0 * %s" % t, "0 * %s" % u),
               (0 * [0 * [v]], "0 * 0 * %s" % t, "0 * 0 * %s" % u),
               (0 * [1 * [v]], "0 * 1 * %s" % t, "0 * 1 * %s" % u),
               (1 * [0 * [v]], "1 * 0 * %s" % t, "1 * 0 * %s" % u)]:

                ttt = ndt(tt)

                x = xnd(vv, type=ttt)
                y = xnd(vv, type=ttt)
                self.assertStrictEqual(x, y)

                if u is not None:
                    uuu = ndt(uu)
                    y = xnd(vv, type=uuu)
                    self.assertStrictEqual(x, y)

        for v, t, u, w, eq in EQUAL_TEST_CASES:
            for vv, tt, uu, indices in [
               (1 * [v], "1 * %s" % t, "1 * %s" % u, (0,)),
               (2 * [v], "2 * %s" % t, "2 * %s" % u, (1,)),
               (1000 * [v], "1000 * %s" % t, "1000 * %s" % u, (961,)),

               (1 * [1 * [v]], "1 * 1 * %s" % t, "1 * 1 * %s" % u, (0, 0)),
               (1 * [2 * [v]], "1 * 2 * %s" % t, "1 * 2 * %s" % u, (0, 1)),
               (2 * [1 * [v]], "2 * 1 * %s" % t, "2 * 1 * %s" % u, (1, 0)),
               (2 * [2 * [v]], "2 * 2 * %s" % t, "2 * 2 * %s" % u, (1, 1)),
               (2 * [3 * [v]], "2 * 3 * %s" % t, "2 * 3 * %s" % u, (1, 2)),
               (3 * [2 * [v]], "3 * 2 * %s" % t, "3 * 2 * %s" % u, (2, 1)),
               (3 * [40 * [v]], "3 * 40 * %s" % t, "3 * 40 * %s" % u, (1, 32))]:

                ttt = ndt(tt)
                uuu = ndt(uu)

                x = xnd(vv, type=ttt)

                y = xnd(vv, type=ttt)
                if eq:
                    self.assertStrictEqual(x, y)
                else:
                    self.assertNotStrictEqual(x, y)

                if u is not None:
                    y = xnd(vv, type=uuu)
                    if eq:
                        self.assertStrictEqual(x, y)
                    else:
                        self.assertNotStrictEqual(x, y)

                if w is not None:
                    y = xnd(vv, type=ttt)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)

                    y = xnd(vv, type=uuu)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)


class TestFortran(XndTestCase):

    def test_fortran_empty(self):
        for v, s in DTYPE_EMPTY_TEST_CASES:
            for vv, ss in [
               (0 * [v], "!0 * %s" % s),
               (1 * [v], "!1 * %s" % s),
               (2 * [v], "!2 * %s" % s),
               (1000 * [v], "!1000 * %s" % s),

               (0 * [0 * [v]], "!0 * 0 * %s" % s),
               (0 * [1 * [v]], "!0 * 1 * %s" % s),
               (1 * [0 * [v]], "!1 * 0 * %s" % s),

               (1 * [1 * [v]], "!1 * 1 * %s" % s),
               (1 * [2 * [v]], "!1 * 2 * %s" % s),
               (2 * [1 * [v]], "!2 * 1 * %s" % s),
               (2 * [2 * [v]], "!2 * 2 * %s" % s),
               (2 * [3 * [v]], "!2 * 3 * %s" % s),
               (3 * [2 * [v]], "!3 * 2 * %s" % s),
               (3 * [40 * [v]], "!3 * 40 * %s" % s) ]:

                t = ndt(ss)
                x = xnd.empty(ss)
                self.assertEqual(x.type, t)
                self.assertEqual(x.value, vv)
                self.assertEqual(len(x), len(vv))

    def test_fortran_subscript(self):
        test_cases = [
            ([[11.12-2.3j, -1222+20e8j],
              [complex("inf"), -0.00002j],
              [0.201+1j, -1+1e301j]], "!3 * 2 * complex128"),
            ([[11.12-2.3j, None],
              [complex("inf"), None],
              [0.201+1j, -1+1e301j]], "!3 * 2 * ?complex128")
        ]

        for v, s in test_cases:
            nd = NDArray(v)
            t = ndt(s)
            x = xnd(v, type=t)
            check_buffer(x)

            for i in range(3):
                self.assertEqual(x[i].value, nd[i])

            for i in range(3):
                for k in range(2):
                    self.assertEqual(x[i][k], nd[i][k])
                    self.assertEqual(x[i, k], nd[i][k])

            self.assertEqual(x[:].value, nd[:])

            for start in list(range(-3, 4)) + [None]:
                for stop in list(range(-3, 4)) + [None]:
                    for step in list(range(-3, 0)) + list(range(1, 4)) + [None]:
                        s = slice(start, stop, step)
                        self.assertEqual(x[s].value, nd[s])
                        check_buffer(x[s])

            self.assertEqual(x[:, 0].value, nd[:, 0])
            self.assertEqual(x[:, 1].value, nd[:, 1])

    def test_fortran_assign(self):
        ### Regular data ###
        x = xnd.empty("!2 * 4 * float64")
        v = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]

        # Full slice
        x[:] = v
        self.assertEqual(x.value, v)

        # Subarray
        x[0] = v[0] = [1.2, -3e45, float("inf"), -322.25]
        self.assertEqual(x.value, v)

        x[1] = v[1] = [-11.25, 3.355e301, -0.000002, -5000.2]
        self.assertEqual(x.value, v)

        # Single values
        for i in range(2):
            for j in range(4):
                x[i][j] = v[i][j] = 3.22 * i + j
        self.assertEqual(x.value, v)

        # Tuple indexing
        for i in range(2):
            for j in range(4):
                x[i, j] = v[i][j] = -3.002e1 * i + j
        self.assertEqual(x.value, v)


        ### Optional data ###
        x = xnd.empty("!2 * 4 * ?float64")
        v = [[10.0, None, 2.0, 100.12], [None, None, 6.0, 7.0]]

        # Full slice
        x[:] = v
        self.assertEqual(x.value, v)

        # Subarray
        x[0] = v[0] = [None, 3e45, float("inf"), None]
        self.assertEqual(x.value, v)

        x[1] = v[1] = [-11.25, 3.355e301, -0.000002, None]
        self.assertEqual(x.value, v)

        # Single values
        for i in range(2):
            for j in range(4):
                x[i][j] = v[i][j] = -325.99 * i + j
        self.assertEqual(x.value, v)

        # Tuple indexing
        for i in range(2):
            for j in range(4):
                x[i, j] = v[i][j] = -8.33e1 * i + j
        self.assertEqual(x.value, v)

    @unittest.skipIf(sys.platform == "darwin",
                     "mach_vm_map message defeats the purpose of this test")
    def test_fortran_overflow(self):
        # Type cannot be created.
        s = "!2147483648 * 2147483648 * 2 * uint8"
        self.assertRaises(ValueError, xnd.empty, s)

        if HAVE_64_BIT:
            # Allocation fails.
            s = "!2147483648 * 2147483647 * 2 * uint8"
            self.assertRaises(MemoryError, xnd.empty, s)
        else:
            # Allocation fails.
            s = "!32768 * 32768 * 2 * uint8"
            self.assertRaises(MemoryError, xnd.empty, s)

    def test_fortran_richcompare(self):

        x = xnd([1,2,3,4], type="!4 * int64")

        self.assertIs(x.__lt__(x), NotImplemented)
        self.assertIs(x.__le__(x), NotImplemented)
        self.assertIs(x.__gt__(x), NotImplemented)
        self.assertIs(x.__ge__(x), NotImplemented)

        self.assertStrictEqual(x, xnd([1,2,3,4], type="!4 * int64"))

        # Different shape and/or data.
        self.assertNotStrictEqual(x, xnd([1,2,3,100], type="!4 * int64"))
        self.assertNotStrictEqual(x, xnd([1,2,3], type="!3 * int64"))
        self.assertNotStrictEqual(x, xnd([1,2,3,4,5], type="!5 * int64"))

        # Different shape.
        self.assertNotStrictEqual(x, xnd([1,2,3], type="!3 * int64"))
        self.assertNotStrictEqual(x, xnd([[1,2,3,4]], type="!1 * 4 * int64"))
        self.assertNotStrictEqual(x, xnd([[1,2], [3,4]], type="!2 * 2 * int64"))

        # Simple multidimensional arrays.
        x = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type="!4 * 3 * int64")
        y = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type="!4 * 3 * int64")
        self.assertStrictEqual(x, y)

        for i in range(4):
            for k in range(3):
                v = y[i, k]
                y[i, k] = 100
                self.assertNotStrictEqual(x, y)
                y[i, k] = v

        # Fortran <-> C.
        x = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type="!4 * 3 * int64")
        y = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
        self.assertStrictEqual(x, y)

        for i in range(4):
            for k in range(3):
                v = y[i, k]
                y[i, k] = 100
                self.assertNotStrictEqual(x, y)
                y[i, k] = v

        # Slices.
        x = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type="!4 * 3 * int64")
        y = xnd([[1,2,3], [7,8,9]])
        self.assertStrictEqual(x[::2], y)

        x = xnd([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type="!4 * 3 * int64")
        y = xnd([1,4,7,10], type="!4 * int64")
        self.assertStrictEqual(x[:, 0], y)

        # Test corner cases and many dtypes.
        for v, t, u, _, _ in EQUAL_TEST_CASES:
            for vv, tt, uu in [
               (0 * [v], "!0 * %s" % t, "!0 * %s" % u),
               (0 * [0 * [v]], "!0 * 0 * %s" % t, "!0 * 0 * %s" % u),
               (0 * [1 * [v]], "!0 * 1 * %s" % t, "!0 * 1 * %s" % u),
               (1 * [0 * [v]], "!1 * 0 * %s" % t, "!1 * 0 * %s" % u)]:

                ttt = ndt(tt)

                x = xnd(vv, type=ttt)
                y = xnd(vv, type=ttt)
                self.assertStrictEqual(x, y)

                if u is not None:
                    uuu = ndt(uu)
                    y = xnd(vv, type=uuu)
                    self.assertStrictEqual(x, y)

        for v, t, u, w, eq in EQUAL_TEST_CASES:
            for vv, tt, uu, indices in [
               (1 * [v], "!1 * %s" % t, "!1 * %s" % u, (0,)),
               (2 * [v], "!2 * %s" % t, "!2 * %s" % u, (1,)),
               (1000 * [v], "!1000 * %s" % t, "!1000 * %s" % u, (961,)),

               (1 * [1 * [v]], "!1 * 1 * %s" % t, "!1 * 1 * %s" % u, (0, 0)),
               (1 * [2 * [v]], "!1 * 2 * %s" % t, "!1 * 2 * %s" % u, (0, 1)),
               (2 * [1 * [v]], "!2 * 1 * %s" % t, "!2 * 1 * %s" % u, (1, 0)),
               (2 * [2 * [v]], "!2 * 2 * %s" % t, "!2 * 2 * %s" % u, (1, 1)),
               (2 * [3 * [v]], "!2 * 3 * %s" % t, "!2 * 3 * %s" % u, (1, 2)),
               (3 * [2 * [v]], "!3 * 2 * %s" % t, "!3 * 2 * %s" % u, (2, 1)),
               (3 * [40 * [v]], "!3 * 40 * %s" % t, "!3 * 40 * %s" % u, (1, 32))]:

                ttt = ndt(tt)

                x = xnd(vv, type=ttt)
                y = xnd(vv, type=ttt)
                if eq:
                    self.assertStrictEqual(x, y)
                else:
                    self.assertNotStrictEqual(x, y)

                if u is not None:
                    uuu = ndt(uu)
                    y = xnd(vv, type=uuu)
                    if eq:
                        self.assertStrictEqual(x, y)
                    else:
                        self.assertNotStrictEqual(x, y)

                if w is not None:
                    y = xnd(vv, type=ttt)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)

                    y = xnd(vv, type=uuu)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)


class TestVarDim(XndTestCase):

    def test_var_dim_empty(self):
        for v, s in DTYPE_EMPTY_TEST_CASES:
            for vv, ss in [
               (0 * [v], "var(offsets=[0,0]) * %s" % s),
               (1 * [v], "var(offsets=[0,1]) * %s" % s),
               (2 * [v], "var(offsets=[0,2]) * %s" % s),
               (1000 * [v], "var(offsets=[0,1000]) * %s" % s),

               (1 * [0 * [v]], "var(offsets=[0,1]) * var(offsets=[0,0]) * %s" % s),

               ([[v], []], "var(offsets=[0,2]) * var(offsets=[0,1,1]) * %s" % s),
               ([[], [v]], "var(offsets=[0,2]) * var(offsets=[0,0,1]) * %s" % s),

               ([[v], [v]], "var(offsets=[0,2]) * var(offsets=[0,1,2]) * %s" % s),
               ([[v], 2 * [v], 5 * [v]], "var(offsets=[0,3]) * var(offsets=[0,1,3,8]) * %s" % s)]:

                t = ndt(ss)
                x = xnd.empty(ss)
                self.assertEqual(x.type, t)
                self.assertEqual(x.value, vv)
                self.assertEqual(len(x), len(vv))

    def test_var_dim_assign(self):
        ### Regular data ###
        x = xnd.empty("var(offsets=[0,2]) * var(offsets=[0,2,5]) * float64")
        v = [[0.0, 1.0], [2.0, 3.0, 4.0]]

        # Full slice assignment
        x[:] = v
        self.assertEqual(x.value, v)

        # Subarray assignment
        x[0] = v[0] = [1.2, 2.5]
        self.assertEqual(x.value, v)

        x[1] = v[1] = [1.2, 2.5, 3.99]
        self.assertEqual(x.value, v)

        # Individual value assignment
        for i in range(2):
            x[0][i] = v[0][i] = 100.0 * i
        for i in range(3):
            x[1][i] = v[1][i] = 200.0 * i
        self.assertEqual(x.value, v)

        # Tuple indexing assignment
        for i in range(2):
            x[0, i] = v[0][i] = 300.0 * i + 1.222
        for i in range(3):
            x[1, i] = v[1][i] = 400.0 * i + 1.333

        # Optional data
        x = xnd.empty("var(offsets=[0,2]) * var(offsets=[0,2,5]) * ?float64")
        v = [[0.0, None], [None, 3.0, 4.0]]

        # Full slice assignment
        x[:] = v
        self.assertEqual(x.value, v)

        # Subarray assignment
        x[0] = v[0] = [None, 2.0]
        self.assertEqual(x.value, v)

        x[1] = v[1] = [1.22214, None, 10.0]
        self.assertEqual(x.value, v)

        # Individual value assignment
        for i in range(2):
            x[0][i] = v[0][i] = 3.14 * i + 1.222
        for i in range(3):
            x[1][i] = v[1][i] = 23.333 * i
        self.assertEqual(x.value, v)

        # Tuple indexing assignment
        for i in range(2):
            x[0, i] = v[0][i] = -122.5 * i + 1.222
        for i in range(3):
            x[1, i] = v[1][i] = -3e22 * i
        self.assertEqual(x.value, v)

    def test_var_dim_overflow(self):
        s = "var(offsets=[0, 2]) * var(offsets=[0, 1073741824, 2147483648]) * uint8"
        self.assertRaises(ValueError, xnd.empty, s)

    def test_var_dim_match(self):
        def type_equal(t, u):
            return t.match(u) and u.match(t)

        x = xnd([0, 1, 2, 3, 4], type="var(offsets=[0,5]) * complex128")
        sig = ndt("var... * complex128 -> var... * complex128")

        spec = sig.apply([x.type])
        self.assertTrue(type_equal(spec.out_types[0], x.type))

        y = x[::-1]
        spec = sig.apply([y.type])
        self.assertTrue(type_equal(spec.out_types[0], y.type))

        y = x[1:3:-1]
        spec = sig.apply([y.type])
        self.assertTrue(type_equal(spec.out_types[0], y.type))

        sig = ndt("var... * complex128, var... * complex128 -> var... * complex128")
        spec = sig.apply([x.type, x.type])
        self.assertTrue(type_equal(spec.out_types[0], x.type))

        y = x[::-1]
        sig = ndt("var... * complex128, var... * complex128 -> var... * complex128")
        spec = sig.apply([x.type, y.type])
        self.assertTrue(type_equal(spec.out_types[0], x.type))

        x = xnd([[0], [1, 2], [3, 4, 5]], dtype="complex128")
        y = xnd([[3, 4, 5], [1, 2], [0]], dtype="complex128")
        z = y[::-1]
        sig = ndt("var... * complex128, var... * complex128 -> var... * complex128")
        spec = sig.apply([x.type, z.type])
        self.assertTrue(type_equal(spec.out_types[0], x.type))

        x = xnd([[0], [1, 2], [3, 4, 5]], dtype="complex128")
        y = xnd([[5, 4, 3], [2, 1], [0]], dtype="complex128")
        z = y[::-1, ::-1]
        sig = ndt("var... * complex128, var... * complex128 -> var... * complex128")
        spec = sig.apply([x.type, z.type])
        self.assertTrue(type_equal(spec.out_types[0], x.type))

    def test_var_dim_richcompare(self):

        x = xnd([1,2,3,4], type="var(offsets=[0,4]) * int64")

        self.assertIs(x.__lt__(x), NotImplemented)
        self.assertIs(x.__le__(x), NotImplemented)
        self.assertIs(x.__gt__(x), NotImplemented)
        self.assertIs(x.__ge__(x), NotImplemented)

        self.assertStrictEqual(x, xnd([1,2,3,4], type="var(offsets=[0,4]) * int64"))

        # Different shape and/or data.
        self.assertNotStrictEqual(x, xnd([1,2,3,100], type="var(offsets=[0,4]) * int64"))
        self.assertNotStrictEqual(x, xnd([1,2,3], type="var(offsets=[0,3]) * int64"))
        self.assertNotStrictEqual(x, xnd([1,2,3,4,5], type="var(offsets=[0,5]) * int64"))

        # Different shape.
        self.assertNotStrictEqual(x, xnd([1,2,3], type="var(offsets=[0,3]) * int64"))
        self.assertNotStrictEqual(x, xnd([[1,2,3,4]], type="var(offsets=[0,1]) * var(offsets=[0,4]) * int64"))
        self.assertNotStrictEqual(x, xnd([[1,2], [3,4]], type="var(offsets=[0,2]) * var(offsets=[0,2,4]) * int64"))

        # Simple multidimensional arrays.
        x = xnd([[1], [2,3,4,5], [6,7], [8,9,10]])
        y = xnd([[1], [2,3,4,5], [6,7], [8,9,10]])
        self.assertStrictEqual(x, y)

        for i, shape in zip(range(4), (1, 4, 2, 3)):
            for k in range(shape):
                v = y[i, k]
                y[i, k] = 100
                self.assertNotStrictEqual(x, y)
                y[i, k] = v

        y = xnd([[1], [2,3,5], [6,7], [8,9,10]])
        self.assertNotStrictEqual(x, y)

        # Slices.
        x = xnd([[1], [4,5], [6,7,8], [9,10,11,12]])

        y = xnd([[1], [6,7,8]])
        self.assertStrictEqual(x[::2], y)

        y = xnd([[9,10,11,12], [4,5]])
        self.assertStrictEqual(x[::-2], y)

        y = xnd([[12,11,10,9], [5,4]])
        self.assertStrictEqual(x[::-2, ::-1], y)

        # Test corner cases and many dtypes.
        for v, t, u, _, _ in EQUAL_TEST_CASES:
            for vv, tt, uu in [
               (0 * [v], "var(offsets=[0,0]) * %s" % t,
                         "var(offsets=[0,0]) * %s" % u),
               (1 * [0 * [v]], "var(offsets=[0,1]) * var(offsets=[0,0]) * %s" % t,
                               "var(offsets=[0,1]) * var(offsets=[0,0]) * %s" % u)]:

                ttt = ndt(tt)

                x = xnd(vv, type=ttt)
                y = xnd(vv, type=ttt)
                self.assertStrictEqual(x, y)

                if u is not None:
                    uuu = ndt(uu)
                    y = xnd(vv, type=uuu)
                    self.assertStrictEqual(x, y)

        for v, t, u, w, eq in EQUAL_TEST_CASES:
            for vv, tt, uu, indices in [
               (1 * [v], "var(offsets=[0,1]) * %s" % t, "var(offsets=[0,1]) * %s" % u, (0,)),
               (2 * [v], "var(offsets=[0,2]) * %s" % t, "var(offsets=[0,2]) * %s" % u, (1,)),
               (1000 * [v], "var(offsets=[0,1000]) * %s" % t, "var(offsets=[0,1000]) * %s" % u, (961,)),
               ([[v], []], "var(offsets=[0,2]) * var(offsets=[0,1,1]) * %s" % t,
                           "var(offsets=[0,2]) * var(offsets=[0,1,1]) * %s" % u, (0, 0)),
               ([[], [v]], "var(offsets=[0,2]) * var(offsets=[0,0,1]) * %s" % t,
                           "var(offsets=[0,2]) * var(offsets=[0,0,1]) * %s" % u, (1, 0)),
               ([[v], [v]], "var(offsets=[0,2]) * var(offsets=[0,1,2]) * %s" % t,
                            "var(offsets=[0,2]) * var(offsets=[0,1,2]) * %s" % u, (1, 0)),
               ([[v], 2 * [v], 5 * [v]], "var(offsets=[0,3]) * var(offsets=[0,1,3,8]) * %s" % t,
                                         "var(offsets=[0,3]) * var(offsets=[0,1,3,8]) * %s" % u, (2, 3))]:

                ttt = ndt(tt)

                x = xnd(vv, type=ttt)
                y = xnd(vv, type=ttt)
                if eq:
                    self.assertStrictEqual(x, y)
                else:
                    self.assertNotStrictEqual(x, y)

                if u is not None:
                    uuu = ndt(uu)
                    y = xnd(vv, type=uuu)
                    if eq:
                        self.assertStrictEqual(x, y)
                    else:
                        self.assertNotStrictEqual(x, y)

                if w is not None:
                    y = xnd(vv, type=ttt)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)

                    y = xnd(vv, type=uuu)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)


class TestSymbolicDim(XndTestCase):

    def test_symbolic_dim_raise(self):
        for _, s in DTYPE_EMPTY_TEST_CASES:
            for err, ss in [
               (ValueError, "N * %s" % s),
               (ValueError, "10 * N * %s" % s),
               (ValueError, "N * 10 * N * %s" % s),
               (ValueError, "X * 10 * N * %s" % s)]:

                t = ndt(ss)
                self.assertRaises(err, xnd.empty, t)


class TestEllipsisDim(XndTestCase):

    def test_ellipsis_dim_raise(self):
        for _, s in DTYPE_EMPTY_TEST_CASES:
            for err, ss in [
               (ValueError, "... * %s" % s),
               (ValueError, "Dims... * %s" % s),
               (ValueError, "... * 10 * %s" % s),
               (ValueError, "B... *2 * 3 * ref(%s)" % s),
               (ValueError, "A... * 10 * Some(ref(%s))" % s),
               (ValueError, "B... * 2 * 3 * Some(ref(ref(%s)))" % s)]:

                t = ndt(ss)
                self.assertRaises(err, xnd.empty, t)


class TestTuple(XndTestCase):

    def test_tuple_empty(self):
        for v, s in DTYPE_EMPTY_TEST_CASES:
            for vv, ss in [
               ((v,), "(%s)" % s),
               (((v,),), "((%s))" % s),
               ((((v,),),), "(((%s)))" % s),

               ((0 * [v],), "(0 * %s)" % s),
               (((0 * [v],),), "((0 * %s))" % s),
               ((1 * [v],), "(1 * %s)" % s),
               (((1 * [v],),), "((1 * %s))" % s),
               ((3 * [v],), "(3 * %s)" % s),
               (((3 * [v],),), "((3 * %s))" % s)]:

                t = ndt(ss)
                x = xnd.empty(ss)
                self.assertEqual(x.type, t)
                self.assertEqual(x.value, vv)
                self.assertEqual(len(x), len(vv))

    def test_tuple_assign(self):
        ### Regular data ###
        x = xnd.empty("(complex64, bytes, string)")
        v = (1+20j, b"abc", "any")

        x[0] = v[0]
        x[1] = v[1]
        x[2] = v[2]

        self.assertEqual(x.value, v)

        ### Optional data ###
        x = xnd.empty("(complex64, ?bytes, ?string)")
        v = (1+20j, None, "Some")

        x[0] = v[0]
        x[1] = v[1]
        x[2] = v[2]
        self.assertEqual(x.value, v)

        v = (-2.5+125j, None, None)
        x[0] = v[0]
        x[1] = v[1]
        x[2] = v[2]
        self.assertEqual(x.value, v)

        x = xnd([("a", 100, 10.5), ("a", 100, 10.5)])
        x[0][1] = 20000000
        self.assertEqual(x[0][1], 20000000)
        self.assertEqual(x[0, 1], 20000000)

    def test_tuple_overflow(self):
        # Type cannot be created.
        s = "(4611686018427387904 * uint8, 4611686018427387904 * uint8)"
        self.assertRaises(ValueError, xnd.empty, s)

        if HAVE_64_BIT:
            # Allocation fails.
            s = "(4611686018427387904 * uint8, 4611686018427387903 * uint8)"
            self.assertRaises(MemoryError, xnd.empty, s)
        else:
            # Allocation fails.
            s = "(1073741824 * uint8, 1073741823 * uint8)"
            self.assertRaises(MemoryError, xnd.empty, s)

    def test_tuple_optional_values(self):
        lst = [(None, 1, 2), (3, None, 4), (5, 6, None)]
        x = xnd(lst, dtype="(?int64, ?int64, ?int64)")
        self.assertEqual(x.value, lst)

    def test_tuple_richcompare(self):

        # Simple tests.
        x = xnd((1, 2.0, "3", b"123"))

        self.assertIs(x.__lt__(x), NotImplemented)
        self.assertIs(x.__le__(x), NotImplemented)
        self.assertIs(x.__gt__(x), NotImplemented)
        self.assertIs(x.__ge__(x), NotImplemented)

        self.assertStrictEqual(x, xnd((1, 2.0, "3", b"123")))

        self.assertNotStrictEqual(x, xnd((2, 2.0, "3", b"123")))
        self.assertNotStrictEqual(x, xnd((1, 2.1, "3", b"123")))
        self.assertNotStrictEqual(x, xnd((1, 2.0, "", b"123")))
        self.assertNotStrictEqual(x, xnd((1, 2.0, "345", b"123")))
        self.assertNotStrictEqual(x, xnd((1, 2.0, "3", b"")))
        self.assertNotStrictEqual(x, xnd((1, 2.0, "3", b"12345")))

        # Nested structures.
        t = """
            (uint8,
             fixed_string(100, 'utf32'),
              (complex128,
               2 * 3 * (fixed_bytes(size=64, align=32), bytes)),
             ref(string))
            """

        v = (10,
             "\U00001234\U00001001abc",
               (12.1e244+3j,
                [[(b"123", 10 * b"22"),
                  (b"123456", 10 * b"23"),
                  (b"123456789", 10 * b"24")],
                 [(b"1", b"a"),
                  (b"12", b"ab"),
                  (b"123", b"abc")]]),
             "xyz")

        x = xnd(v, type=t)
        y = xnd(v, type=t)
        self.assertStrictEqual(x, y)

        w = y[0].value
        y[0] = 11
        self.assertNotStrictEqual(x, y)
        y[0] = w
        self.assertStrictEqual(x, y)

        w = y[1].value
        y[1] = "\U00001234\U00001001abx"
        self.assertNotStrictEqual(x, y)
        y[1] = w
        self.assertStrictEqual(x, y)

        w = y[2,0].value
        y[2,0] = 12.1e244-3j
        self.assertNotStrictEqual(x, y)
        y[2,0] = w
        self.assertStrictEqual(x, y)

        w = y[2,1,1,2,0].value
        y[2,1,1,2,0] = b"abc"
        self.assertNotStrictEqual(x, y)
        y[2,1,1,2,0] = w
        self.assertStrictEqual(x, y)

        w = y[3].value
        y[3] = ""
        self.assertNotStrictEqual(x, y)
        y[3] = w
        self.assertStrictEqual(x, y)

        # Test corner cases and many dtypes.
        for v, t, u, _, _ in EQUAL_TEST_CASES:
            for vv, tt, uu in [
               ((0 * [v],), "(0 * %s)" % t, "(0 * %s)" % u),
               (((0 * [v],),), "((0 * %s))" % t, "((0 * %s))" % u)]:

                ttt = ndt(tt)

                x = xnd(vv, type=ttt)
                y = xnd(vv, type=ttt)
                self.assertStrictEqual(x, y)

                if u is not None:
                    uuu = ndt(uu)
                    y = xnd(vv, type=uuu)
                    self.assertStrictEqual(x, y)

        for v, t, u, w, eq in EQUAL_TEST_CASES:
            for vv, tt, uu, indices in [
               ((v,), "(%s)" % t, "(%s)" % u, (0,)),
               (((v,),), "((%s))" % t, "((%s))" % u, (0, 0)),
               ((((v,),),), "(((%s)))" % t, "(((%s)))" % u, (0, 0, 0)),

               ((1 * [v],), "(1 * %s)" % t, "(1 * %s)" % u, (0, 0)),
               (((1 * [v],),), "((1 * %s))" % t, "((1 * %s))" % u, (0, 0, 0)),
               ((3 * [v],), "(3 * %s)" % t, "(3 * %s)" % u, (0, 2))]:

                ttt = ndt(tt)

                x = xnd(vv, type=ttt)
                y = xnd(vv, type=ttt)
                if eq:
                    self.assertStrictEqual(x, y)
                else:
                    self.assertNotStrictEqual(x, y)

                if u is not None:
                    uuu = ndt(uu)
                    y = xnd(vv, type=uuu)
                    if eq:
                        self.assertStrictEqual(x, y)
                    else:
                        self.assertNotStrictEqual(x, y)

                if w is not None:
                    y = xnd(vv, type=ttt)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)

                    y = xnd(vv, type=uuu)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)


class TestRecord(XndTestCase):

    def test_record_empty(self):
        for v, s in DTYPE_EMPTY_TEST_CASES:
            for vv, ss in [
               ({'x': v}, "{x: %s}" % s),
               ({'x': {'y': v}}, "{x: {y: %s}}" % s),

               ({'x': 0 * [v]}, "{x: 0 * %s}" % s),
               ({'x': {'y': 0 * [v]}}, "{x: {y: 0 * %s}}" % s),
               ({'x': 1 * [v]}, "{x: 1 * %s}" % s),
               ({'x': 3 * [v]}, "{x: 3 * %s}" % s)]:

                t = ndt(ss)
                x = xnd.empty(ss)
                self.assertEqual(x.type, t)
                self.assertEqual(x.value, vv)
                self.assertEqual(len(x), len(vv))

    def test_record_assign(self):
        ### Regular data ###
        x = xnd.empty("{x: complex64, y: bytes, z: string}")
        v = R['x': 1+20j, 'y': b"abc", 'z': "any"]

        x['x'] = v['x']
        x['y'] = v['y']
        x['z'] = v['z']

        self.assertEqual(x.value, v)

        ### Optional data ###
        x = xnd.empty("{x: complex64, y: ?bytes, z: ?string}")
        v = R['x': 1+20j, 'y': None, 'z': "Some"]

        x['x'] = v['x']
        x['y'] = v['y']
        x['z'] = v['z']
        self.assertEqual(x.value, v)

        v = R['x': -2.5+125j, 'y': None, 'z': None]
        x['x'] = v['x']
        x['y'] = v['y']
        x['z'] = v['z']
        self.assertEqual(x.value, v)

        x = xnd([R['x': "abc", 'y': 100, 'z': 10.5]])
        x[0][1] = 20000000
        self.assertEqual(x[0][1], 20000000)
        self.assertEqual(x[0, 1], 20000000)

    def test_record_overflow(self):
        # Type cannot be created.
        s = "{a: 4611686018427387904 * uint8, b: 4611686018427387904 * uint8}"
        self.assertRaises(ValueError, xnd.empty, s)

        if HAVE_64_BIT:
            # Allocation fails.
            s = "{a: 4611686018427387904 * uint8, b: 4611686018427387903 * uint8}"
            self.assertRaises(MemoryError, xnd.empty, s)
        else:
            # Allocation fails.
            s = "{a: 1073741824 * uint8, b: 1073741823 * uint8}"
            self.assertRaises(MemoryError, xnd.empty, s)

    def test_record_optional_values(self):
        lst = [R['a': None, 'b': 2, 'c': 3], R['a': 4, 'b': None, 'c': 5],
               R['a': 5, 'b': 6, 'c': None]]
        x = xnd(lst, dtype="{a: ?int64, b: ?int64, c: ?int64}")
        self.assertEqual(x.value, lst)

    def test_record_richcompare(self):

        # Simple tests.
        x = xnd(R['a': 1, 'b': 2.0, 'c': "3", 'd': b"123"])

        self.assertIs(x.__lt__(x), NotImplemented)
        self.assertIs(x.__le__(x), NotImplemented)
        self.assertIs(x.__gt__(x), NotImplemented)
        self.assertIs(x.__ge__(x), NotImplemented)

        self.assertStrictEqual(x, xnd(R['a': 1, 'b': 2.0, 'c': "3", 'd': b"123"]))

        self.assertNotStrictEqual(x, xnd(R['z': 1, 'b': 2.0, 'c': "3", 'd': b"123"]))
        self.assertNotStrictEqual(x, xnd(R['a': 2, 'b': 2.0, 'c': "3", 'd': b"123"]))
        self.assertNotStrictEqual(x, xnd(R['a': 1, 'b': 2.1, 'c': "3", 'd': b"123"]))
        self.assertNotStrictEqual(x, xnd(R['a': 1, 'b': 2.0, 'c': "", 'd': b"123"]))
        self.assertNotStrictEqual(x, xnd(R['a': 1, 'b': 2.0, 'c': "345", 'd': "123"]))
        self.assertNotStrictEqual(x, xnd(R['a': 1, 'b': 2.0, 'c': "3", 'd': b""]))
        self.assertNotStrictEqual(x, xnd(R['a': 1, 'b': 2.0, 'c': "3", 'd': b"12345"]))

        # Nested structures.
        t = """
            {a: uint8,
             b: fixed_string(100, 'utf32'),
             c: {x: complex128,
                 y: 2 * 3 * {v: fixed_bytes(size=64, align=32),
                             u: bytes}},
             d: ref(string)}
            """

        v = R['a': 10,
              'b': "\U00001234\U00001001abc",
              'c': R['x': 12.1e244+3j,
                     'y': [[R['v': b"123", 'u': 10 * b"22"],
                            R['v': b"123456", 'u': 10 * b"23"],
                            R['v': b"123456789", 'u': 10 * b"24"]],
                           [R['v': b"1", 'u': b"a"],
                            R['v': b"12", 'u': b"ab"],
                            R['v': b"123", 'u': b"abc"]]]],
              'd': "xyz"]

        x = xnd(v, type=t)
        y = xnd(v, type=t)
        self.assertStrictEqual(x, y)

        w = y[0].value
        y[0] = 11
        self.assertNotStrictEqual(x, y)
        y[0] = w
        self.assertStrictEqual(x, y)

        w = y[1].value
        y[1] = "\U00001234\U00001001abx"
        self.assertNotStrictEqual(x, y)
        y[1] = w
        self.assertStrictEqual(x, y)

        w = y[2,0].value
        y[2,0] = 12.1e244-3j
        self.assertNotStrictEqual(x, y)
        y[2,0] = w
        self.assertStrictEqual(x, y)

        w = y[2,1,1,2,0].value
        y[2,1,1,2,0] = b"abc"
        self.assertNotStrictEqual(x, y)
        y[2,1,1,2,0] = w
        self.assertStrictEqual(x, y)

        w = y[3].value
        y[3] = ""
        self.assertNotStrictEqual(x, y)
        y[3] = w
        self.assertStrictEqual(x, y)

        # Test corner cases and many dtypes.
        for v, t, u, _, _ in EQUAL_TEST_CASES:
            for vv, tt, uu in [
               ({'x': 0 * [v]}, "{x: 0 * %s}" % t, "{x: 0 * %s}" % u),
               ({'x': {'y': 0 * [v]}}, "{x: {y: 0 * %s}}" % t, "{x: {y: 0 * %s}}" % u)]:

                ttt = ndt(tt)

                x = xnd(vv, type=ttt)

                y = xnd(vv, type=ttt)
                self.assertStrictEqual(x, y)

                if u is not None:
                    uuu = ndt(uu)
                    y = xnd(vv, type=uuu)
                    self.assertStrictEqual(x, y)

        for v, t, u, w, eq in EQUAL_TEST_CASES:
            for vv, tt, uu, indices in [
               ({'x': v}, "{x: %s}" % t, "{x: %s}" % u, (0,)),
               ({'x': {'y': v}}, "{x: {y: %s}}" % t, "{x: {y: %s}}" % u, (0, 0)),
               ({'x': 1 * [v]}, "{x: 1 * %s}" % t, "{x: 1 * %s}" % u, (0, 0)),
               ({'x': 3 * [v]}, "{x: 3 * %s}" % t, "{x: 3 * %s}" % u, (0, 2))]:

                ttt = ndt(tt)
                uuu = ndt(uu)

                x = xnd(vv, type=ttt)

                y = xnd(vv, type=ttt)
                if eq:
                    self.assertStrictEqual(x, y)
                else:
                    self.assertNotStrictEqual(x, y)

                if u is not None:
                    y = xnd(vv, type=uuu)
                    if eq:
                        self.assertStrictEqual(x, y)
                    else:
                        self.assertNotStrictEqual(x, y)

                if w is not None:
                    y = xnd(vv, type=ttt)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)


class TestRef(XndTestCase):

    def test_ref_empty(self):
        for v, s in DTYPE_EMPTY_TEST_CASES:
            for vv, ss in [
               (v, "ref(%s)" % s),
               (v, "ref(ref(%s))" % s),
               (v, "ref(ref(ref(%s)))" % s),

               (0 * [v], "ref(0 * %s)" % s),
               (0 * [v], "ref(ref(0 * %s))" % s),
               (0 * [v], "ref(ref(ref(0 * %s)))" % s),
               (1 * [v], "ref(1 * %s)" % s),
               (1 * [v], "ref(ref(1 * %s))" % s),
               (1 * [v], "ref(ref(ref(1 * %s)))" % s),
               (3 * [v], "ref(3 * %s)" % s),
               (3 * [v], "ref(ref(3 * %s))" % s),
               (3 * [v], "ref(ref(ref(3 * %s)))" % s)]:

                t = ndt(ss)
                x = xnd.empty(ss)
                self.assertEqual(x.type, t)
                self.assertEqual(x.value, vv)
                assertEqualWithEx(self, len, x, vv)

    def test_ref_empty_view(self):
        # If a ref is a dtype but contains an array itself, indexing should
        # return a view and not a Python value.
        inner = 4 * [5 * [0+0j]]
        x = xnd.empty("2 * 3 * ref(4 * 5 * complex128)")

        y = x[1][2]
        self.assertIsInstance(y, xnd)
        self.assertEqual(y.value, inner)

        y = x[1, 2]
        self.assertIsInstance(y, xnd)
        self.assertEqual(y.value, inner)

    def test_ref_indexing(self):
        # If a ref is a dtype but contains an array itself, indexing through
        # the ref should work transparently.
        inner = [['a', 'b', 'c', 'd', 'e'],
                 ['f', 'g', 'h', 'i', 'j'],
                 ['k', 'l', 'm', 'n', 'o'],
                 ['p', 'q', 'r', 's', 't']]

        v = 2 * [3 * [inner]]

        x = xnd(v, type="2 * 3 * ref(4 * 5 * string)")

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    for l in range(5):
                        self.assertEqual(x[i][j][k][l], inner[k][l])
                        self.assertEqual(x[i, j, k, l], inner[k][l])

    def test_ref_assign(self):
        # If a ref is a dtype but contains an array itself, assigning through
        # the ref should work transparently.
        inner = [['a', 'b', 'c', 'd', 'e'],
                 ['f', 'g', 'h', 'i', 'j'],
                 ['k', 'l', 'm', 'n', 'o'],
                 ['p', 'q', 'r', 's', 't']]

        v = 2 * [3 * [inner]]

        x = xnd(v, type="2 * 3 * ref(4 * 5 * string)")
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    for l in range(5):
                        x[i][j][k][l] = inner[k][l] = "%d" % (k * 5 + l)

        self.assertEqual(x.value, v)

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    for l in range(5):
                        x[i, j, k, l] = inner[k][l] = "%d" % (k * 5 + l + 1)

        self.assertEqual(x.value, v)

    def test_ref_richcompare(self):

        # Simple tests.
        x = xnd([1,2,3,4], type="ref(4 * float32)")

        self.assertIs(x.__lt__(x), NotImplemented)
        self.assertIs(x.__le__(x), NotImplemented)
        self.assertIs(x.__gt__(x), NotImplemented)
        self.assertIs(x.__ge__(x), NotImplemented)

        self.assertStrictEqual(x, xnd([1,2,3,4], type="ref(4 * float32)"))

        self.assertNotStrictEqual(x, xnd([1,2,3,5], type="ref(4 * float32)"))
        self.assertNotStrictEqual(x, xnd([1,2,3], type="ref(3 * float32)"))
        self.assertNotStrictEqual(x, xnd([1,2,3,4,5], type="ref(5 * float32)"))

        # Test corner cases and many dtypes.
        for v, t, u, _, _ in EQUAL_TEST_CASES:
            for vv, tt, uu in [
               (0 * [v], "ref(0 * %s)" % t, "ref(0 * %s)" % u),
               (0 * [v], "ref(ref(0 * %s))" % t, "ref(ref(0 * %s))" % u),
               (0 * [v], "ref(ref(ref(0 * %s)))" % t, "ref(ref(ref(0 * %s)))" % u)]:

                ttt = ndt(tt)

                x = xnd(vv, type=ttt)

                y = xnd(vv, type=ttt)
                self.assertStrictEqual(x, y)

                if u is not None:
                    uuu = ndt(uu)
                    y = xnd(vv, type=uuu)
                    self.assertStrictEqual(x, y)

        for v, t, u, w, eq in EQUAL_TEST_CASES:
            for vv, tt, uu, indices in [
               (v, "ref(%s)" % t, "ref(%s)" % u, ()),
               (v, "ref(ref(%s))" % t, "ref(ref(%s))" % u, ()),
               (v, "ref(ref(ref(%s)))" % t, "ref(ref(ref(%s)))" % u, ()),
               (1 * [v], "ref(1 * %s)" % t, "ref(1 * %s)" % u, (0,)),
               (3 * [v], "ref(3 * %s)" % t, "ref(3 * %s)" % u, (2,))]:

                ttt = ndt(tt)
                uuu = ndt(uu)

                x = xnd(vv, type=ttt)

                y = xnd(vv, type=ttt)
                if eq:
                    self.assertStrictEqual(x, y)
                else:
                    self.assertNotStrictEqual(x, y)

                if u is not None:
                    y = xnd(vv, type=uuu)
                    if eq:
                        self.assertStrictEqual(x, y)
                    else:
                        self.assertNotStrictEqual(x, y)

                if w is not None:
                    y = xnd(vv, type=ttt)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)


class TestConstr(XndTestCase):

    def test_constr_empty(self):
        for v, s in DTYPE_EMPTY_TEST_CASES:
            for vv, ss in [
               (v, "SomeConstr(%s)" % s),
               (v, "Just(Some(%s))" % s),

               (0 * [v], "Some(0 * %s)" % s),
               (1 * [v], "Some(1 * %s)" % s),
               (3 * [v], "Maybe(3 * %s)" % s)]:

                t = ndt(ss)
                x = xnd.empty(ss)
                self.assertEqual(x.type, t)
                self.assertEqual(x.value, vv)
                assertEqualWithEx(self, len, x, vv)

    def test_constr_empty_view(self):
        # If a constr is a dtype but contains an array itself, indexing should
        # return a view and not a Python value.
        inner = 4 * [5 * [""]]
        x = xnd.empty("2 * 3 * InnerArray(4 * 5 * string)")

        y = x[1][2]
        self.assertIsInstance(y, xnd)
        self.assertEqual(y.value, inner)

        y = x[1, 2]
        self.assertIsInstance(y, xnd)
        self.assertEqual(y.value, inner)

    def test_constr_indexing(self):
        # If a constr is a dtype but contains an array itself, indexing through
        # the constructor should work transparently.
        inner = [['a', 'b', 'c', 'd', 'e'],
                 ['f', 'g', 'h', 'i', 'j'],
                 ['k', 'l', 'm', 'n', 'o'],
                 ['p', 'q', 'r', 's', 't']]

        v = 2 * [3 * [inner]]

        x = xnd(v, type="2 * 3 * InnerArray(4 * 5 * string)")

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    for l in range(5):
                        self.assertEqual(x[i][j][k][l], inner[k][l])
                        self.assertEqual(x[i, j, k, l], inner[k][l])

    def test_constr_assign(self):
        # If a constr is a dtype but contains an array itself, assigning through
        # the constructor should work transparently.
        inner = [['a', 'b', 'c', 'd', 'e'],
                 ['f', 'g', 'h', 'i', 'j'],
                 ['k', 'l', 'm', 'n', 'o'],
                 ['p', 'q', 'r', 's', 't']]

        v = 2 * [3 * [inner]]

        x = xnd(v, type="2 * 3 * InnerArray(4 * 5 * string)")

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    for l in range(5):
                        x[i][j][k][l] = inner[k][l] = "%d" % (k * 5 + l)

        self.assertEqual(x.value, v)

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    for l in range(5):
                        x[i][j][k][l] = inner[k][l] = "%d" % (k * 5 + l + 1)

        self.assertEqual(x.value, v)

    def test_constr_richcompare(self):

        # Simple tests.
        x = xnd([1,2,3,4], type="A(4 * float32)")

        self.assertIs(x.__lt__(x), NotImplemented)
        self.assertIs(x.__le__(x), NotImplemented)
        self.assertIs(x.__gt__(x), NotImplemented)
        self.assertIs(x.__ge__(x), NotImplemented)

        self.assertStrictEqual(x, xnd([1,2,3,4], type="A(4 * float32)"))

        self.assertNotStrictEqual(x, xnd([1,2,3,4], type="B(4 * float32)"))
        self.assertNotStrictEqual(x, xnd([1,2,3,5], type="A(4 * float32)"))
        self.assertNotStrictEqual(x, xnd([1,2,3], type="A(3 * float32)"))
        self.assertNotStrictEqual(x, xnd([1,2,3,4,5], type="A(5 * float32)"))

        # Test corner cases and many dtypes.
        for v, t, u, _, _ in EQUAL_TEST_CASES:
            for vv, tt, uu in [
               (0 * [v], "A(0 * %s)" % t, "A(0 * %s)" % u),
               (0 * [v], "A(B(0 * %s))" % t, "A(B(0 * %s))" % u),
               (0 * [v], "A(B(C(0 * %s)))" % t, "A(B(C(0 * %s)))" % u)]:

                ttt = ndt(tt)

                x = xnd(vv, type=ttt)
                y = xnd(vv, type=ttt)
                self.assertStrictEqual(x, y)

                if u is not None:
                    uuu = ndt(uu)
                    y = xnd(vv, type=uuu)
                    self.assertStrictEqual(x, y)

        for v, t, u, w, eq in EQUAL_TEST_CASES:
            for vv, tt, uu, indices in [
               (v, "A(%s)" % t, "A(%s)" % u, ()),
               (v, "A(B(%s))" % t, "A(B(%s))" % u, ()),
               (v, "A(B(C(%s)))" % t, "A(B(C(%s)))" % u, ()),
               (1 * [v], "A(1 * %s)" % t, "A(1 * %s)" % u, (0,)),
               (3 * [v], "A(3 * %s)" % t, "A(3 * %s)" % u, (2,))]:

                ttt = ndt(tt)
                uuu = ndt(uu)

                x = xnd(vv, type=ttt)

                y = xnd(vv, type=ttt)
                if eq:
                    self.assertStrictEqual(x, y)
                else:
                    self.assertNotStrictEqual(x, y)

                if u is not None:
                    y = xnd(vv, type=uuu)
                    if eq:
                        self.assertStrictEqual(x, y)
                    else:
                        self.assertNotStrictEqual(x, y)

                if w is not None:
                    y = xnd(vv, type=ttt)
                    y[indices] = w
                    self.assertNotStrictEqual(x, y)


class TestNominal(XndTestCase):

    def test_nominal_empty(self):
        c = 0
        for v, s in DTYPE_EMPTY_TEST_CASES:
            typedef("some%d" % c, s)
            typedef("just%d" % c, "some%d" %c)

            for vv, ss in [
               (v, "some%d" % c),
               (v, "just%d" % c)]:

                t = ndt(ss)
                x = xnd.empty(ss)
                self.assertEqual(x.type, t)
                self.assertEqual(x.value, vv)
                assertEqualWithEx(self, len, x, vv)

            c += 1

    def test_nominal_empty_view(self):
        # If a typedef is a dtype but contains an array itself, indexing should
        # return a view and not a Python value.
        typedef("inner_array", "4 * 5 * string")
        inner = 4 * [5 * [""]]
        x = xnd.empty("2 * 3 * inner_array")

        y = x[1][2]
        self.assertIsInstance(y, xnd)
        self.assertEqual(y.value, inner)

        y = x[1, 2]
        self.assertIsInstance(y, xnd)
        self.assertEqual(y.value, inner)

    def test_nominal_indexing(self):
        # If a typedef is a dtype but contains an array itself, indexing through
        # the constructor should work transparently.
        typedef("inner", "4 * 5 * string")
        inner = [['a', 'b', 'c', 'd', 'e'],
                 ['f', 'g', 'h', 'i', 'j'],
                 ['k', 'l', 'm', 'n', 'o'],
                 ['p', 'q', 'r', 's', 't']]

        v = 2 * [3 * [inner]]

        x = xnd(v, type="2 * 3 * inner")

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    for l in range(5):
                        self.assertEqual(x[i][j][k][l], inner[k][l])
                        self.assertEqual(x[i, j, k, l], inner[k][l])

    def test_nominal_assign(self):
        # If a typedef is a dtype but contains an array itself, assigning through
        # the constructor should work transparently.
        typedef("in", "4 * 5 * string")
        inner = [['a', 'b', 'c', 'd', 'e'],
                 ['f', 'g', 'h', 'i', 'j'],
                 ['k', 'l', 'm', 'n', 'o'],
                 ['p', 'q', 'r', 's', 't']]

        v = 2 * [3 * [inner]]

        x = xnd(v, type="2 * 3 * in")

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    for l in range(5):
                        x[i][j][k][l] = inner[k][l] = "%d" % (k * 5 + l)

        self.assertEqual(x.value, v)

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    for l in range(5):
                        x[i][j][k][l] = inner[k][l] = "%d" % (k * 5 + l + 1)

        self.assertEqual(x.value, v)

    def test_nominal_error(self):
        self.assertRaises(ValueError, xnd.empty, "undefined_t")

    def test_nominal_richcompare(self):

        typedef("some1000", "4 * float32")
        typedef("some1001", "4 * float32")

        # Simple tests.
        x = xnd([1,2,3,4], type="some1000")

        self.assertIs(x.__lt__(x), NotImplemented)
        self.assertIs(x.__le__(x), NotImplemented)
        self.assertIs(x.__gt__(x), NotImplemented)
        self.assertIs(x.__ge__(x), NotImplemented)

        self.assertStrictEqual(x, xnd([1,2,3,4], type="some1000"))

        self.assertNotStrictEqual(x, xnd([1,2,3,4], type="some1001"))
        self.assertNotStrictEqual(x, xnd([1,2,3,5], type="some1000"))


class TestScalarKind(XndTestCase):

    def test_scalar_kind(self):
        self.assertRaises(ValueError, xnd.empty, "Scalar")


class TestCategorical(XndTestCase):

    def test_categorical_empty(self):
        # Categorical values are stored as indices into the type's categories.
        # Since empty xnd objects are initialized to zero, the value of an
        # empty categorical entry is always the value of the first category.
        # This is safe, since categorical types must have at least one entry.
        r = R['a': "", 'b': 1.2]
        rt = "{a: string, b: categorical(1.2, 10.0, NA)}"

        test_cases = [
          ("January", "categorical('January')"),
          ((None,), "(categorical(NA, 'January', 'August'))"),
          (10 * [2 * [1.2]], "10 * 2 * categorical(1.2, 10.0, NA)"),
          (10 * [2 * [100]], "10 * 2 * categorical(100, 'mixed')"),
          (10 * [2 * [r]], "10 * 2 * %s" % rt),
          ([2 * [r], 5 * [r], 3 * [r]], "var(offsets=[0,3]) * var(offsets=[0,2,7,10]) * %s" % rt)
        ]

        for v, s in test_cases:
            t = ndt(s)
            x = xnd.empty(s)
            self.assertEqual(x.type, t)
            self.assertEqual(x.value, v)

    def test_categorical_assign(self):
        s = """2 * categorical(
                     NA, 'January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December'
               )
            """

        x = xnd([None, None], type=s)
        x[0] = 'August'
        x[1] = 'December'

        self.assertEqual(x.value, ['August', 'December'])

        x[0] = None
        self.assertEqual(x.value, [None, 'December'])

    def test_categorical_richcompare(self):
        t = "3 * categorical(NA, 'January', 'August')"
        x = xnd(['August', 'January', 'January'], type=t)

        y = xnd(['August', 'January', 'January'], type=t)
        self.assertStrictEqual(x, y)

        y = xnd(['August', 'January', 'August'], type=t)
        self.assertNotStrictEqual(x, y)

        x = xnd(['August', None, 'August'], type=t)
        y = xnd(['August', None, 'August'], type=t)
        self.assertNotStrictEqual(x, y)


class TestFixedStringKind(XndTestCase):

    def test_fixed_string_kind(self):
        self.assertRaises(ValueError, xnd.empty, "FixedString")


class TestFixedString(XndTestCase):

    def test_fixed_string_empty(self):
        test_cases = [
          ("fixed_string(1)", ""),
          ("fixed_string(3)", 3 * ""),
          ("fixed_string(1, 'ascii')", ""),
          ("fixed_string(3, 'utf8')", 3 * ""),
          ("fixed_string(3, 'utf16')", 3 * ""),
          ("fixed_string(3, 'utf32')", 3 * ""),
          ("2 * fixed_string(3, 'utf32')", 2 * [3 * ""]),
        ]

        for s, v in test_cases:
            t = ndt(s)
            x = xnd.empty(s)
            self.assertEqual(x.type, t)
            self.assertEqual(x.value, v)

    def test_fixed_string(self):
        t = "2 * fixed_string(3, 'utf16')"
        v = ["\u1111\u2222\u3333", "\u1112\u2223\u3334"]
        x = xnd(v, type=t)
        self.assertEqual(x.value, v)


        t = "2 * fixed_string(3, 'utf32')"
        v = ["\U00011111\U00022222\U00033333", "\U00011112\U00022223\U00033334"]
        x = xnd(v, type=t)
        self.assertEqual(x.value, v)

    def test_fixed_string_assign(self):
        t = "2 * fixed_string(3, 'utf32')"
        v = ["\U00011111\U00022222\U00033333", "\U00011112\U00022223\U00033334"]
        x = xnd(v, type=t)

        x[0] = "a"
        self.assertEqual(x.value, ["a", "\U00011112\U00022223\U00033334"])

        x[0] = "a\x00\x00"
        self.assertEqual(x.value, ["a", "\U00011112\U00022223\U00033334"])

        x[1] = "b\x00c"
        self.assertEqual(x.value, ["a", "b\x00c"])

    def test_fixed_string_overflow(self):
        # Type cannot be created.
        for s in ["fixed_string(9223372036854775808)",
                  "fixed_string(4611686018427387904, 'utf16')",
                  "fixed_string(2305843009213693952, 'utf32')"]:
            self.assertRaises(ValueError, xnd.empty, s)

        if HAVE_64_BIT:
            # Allocation fails.
            s = "fixed_string(4611686018427387903, 'utf16')"
            self.assertRaises(MemoryError, xnd.empty, s)
        else:
            # Allocation fails.
            s = "fixed_string(1073741824, 'utf16')"
            self.assertRaises(MemoryError, xnd.empty, s)

    def test_fixed_string_richcompare(self):
        test_cases = [
          ("fixed_string(1)", "", "x"),
          ("fixed_string(3)", 3 * "y", "yyz"),
          ("fixed_string(1, 'ascii')", "a", "b"),
          ("fixed_string(3, 'utf8')", 3 * "a", "abc"),
          ("fixed_string(3, 'utf16')", 3 * "\u1234", "\u1234\u1235\u1234"),
          ("fixed_string(3, 'utf32')", 3 * "\U00001234", "\U00001234\U00001234\U00001235"),
        ]

        for t, v, w in test_cases:
            x = xnd(v, type=t)
            y = xnd(v, type=t)
            self.assertStrictEqual(x, y)
            y[()] = w
            self.assertNotStrictEqual(x, y)


class TestFixedBytesKind(XndTestCase):

    def test_fixed_bytes_kind(self):
        self.assertRaises(ValueError, xnd.empty, "FixedBytes")


class TestFixedBytes(XndTestCase):

    def test_fixed_bytes_empty(self):
        r = R['a': 3 * b'\x00', 'b': 10 * b'\x00']

        test_cases = [
          (b'\x00', 'fixed_bytes(size=1)'),
          (100 * b'\x00', 'fixed_bytes(size=100)'),
          (4 * b'\x00', 'fixed_bytes(size=4, align=2)'),
          (128 * b'\x00', 'fixed_bytes(size=128, align=16)'),
          (r, '{a: fixed_bytes(size=3), b: fixed_bytes(size=10)}'),
          (2 * [3 * [r]], '2 * 3 * {a: fixed_bytes(size=3), b: fixed_bytes(size=10)}')
        ]

        for v, s in test_cases:
            t = ndt(s)
            x = xnd.empty(s)
            self.assertEqual(x.type, t)
            self.assertEqual(x.value, v)

    def test_fixed_bytes_assign(self):
        t = "2 * fixed_bytes(size=3, align=1)"
        v = [b"abc", b"123"]
        x = xnd(v, type=t)

        x[0] = b"xyz"
        self.assertEqual(x.value, [b"xyz", b"123"])


        t = "2 * fixed_bytes(size=3, align=1)"
        v = [b"abc", b"123"]
        x = xnd(v, type=t)

        x[0] = b"xyz"
        self.assertEqual(x.value, [b"xyz", b"123"])

    def test_fixed_bytes_overflow(self):
        # Type cannot be created.
        s = "fixed_bytes(size=9223372036854775808)"
        self.assertRaises(ValueError, xnd.empty, s)

        if HAVE_64_BIT:
            # Allocation fails.
            s = "fixed_bytes(size=9223372036854775807)"
            self.assertRaises(MemoryError, xnd.empty, s)
        else:
            # Allocation fails.
            s = "fixed_bytes(size=2147483648)"
            self.assertRaises(MemoryError, xnd.empty, s)

    def test_fixed_bytes_richcompare(self):
        test_cases = [
          (b"a", "fixed_bytes(size=1)", b"b"),
          (100 * b"a", "fixed_bytes(size=100)", 99 * b"a" + b"b"),
          (4 * b"a", "fixed_bytes(size=4, align=2)", 2 * b"a" + 2 * b"b"),
        ]

        for v, t, w in test_cases:
            x = xnd(v, type=t)
            y = xnd(v, type=t)
            self.assertStrictEqual(x, y)
            y[()] = w
            self.assertNotStrictEqual(x, y)

        x = xnd(128 * b"a", type="fixed_bytes(size=128, align=16)")
        y = xnd(128 * b"a", type="fixed_bytes(size=128, align=4)")
        self.assertStrictEqual(x, y)

        x = xnd(128 * b"a", type="fixed_bytes(size=128, align=16)")
        y = xnd(128 * b"a", type="fixed_bytes(size=128, align=4)")
        self.assertStrictEqual(x, y)


class TestString(XndTestCase):

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

        t = ndt('string')
        x = xnd.empty(t)
        self.assertEqual(x.type, t)
        self.assertEqual(x.value, '')

        t = ndt('10 * string')
        x = xnd.empty(t)
        self.assertEqual(x.type, t)
        for i in range(10):
            self.assertEqual(x[i], '')

    def test_string(self):
        t = '2 * {a: complex128, b: string}'
        x = xnd([R['a': 2+3j, 'b': "thisguy"], R['a': 1+4j, 'b': "thatguy"]], type=t)

        self.assertEqual(x[0]['b'], "thisguy")
        self.assertEqual(x[1]['b'], "thatguy")

    def test_string_assign(self):
        t = '2 * {a: complex128, b: string}'
        x = xnd([R['a': 2+3j, 'b': "thisguy"], R['a': 1+4j, 'b': "thatguy"]], type=t)

        x[0] = R['a': 220j, 'b': 'y']
        x[1] = R['a': -12j, 'b': 'z']
        self.assertEqual(x.value, [R['a': 220j, 'b': 'y'], R['a': -12j, 'b': 'z']])

    def test_string_richcompare(self):

        x = xnd("abc")

        self.assertStrictEqual(x, xnd("abc"))
        self.assertStrictEqual(x, xnd("abc\0\0"))

        self.assertNotStrictEqual(x, xnd("acb"))


class TestBytes(XndTestCase):

    def test_bytes_empty(self):
        r = R['a': b'', 'b': b'']

        test_cases = [
          (b'', 'bytes(align=16)'),
          ((b'',), '(bytes(align=32))'),
          (3 * [2 * [b'']], '3 * 2 * bytes'),
          (10 * [2 * [(b'', b'')]], '10 * 2 * (bytes, bytes)'),
          (10 * [2 * [r]], '10 * 2 * {a: bytes(align=32), b: bytes(align=1)}'),
          (10 * [2 * [r]], '10 * 2 * {a: bytes(align=1), b: bytes(align=32)}'),
          ([2 * [r], 5 * [r], 3 * [r]], 'var(offsets=[0,3]) * var(offsets=[0,2,7,10]) * {a: bytes(align=32), b: bytes}')
        ]

        for v, s in test_cases:
            t = ndt(s)
            x = xnd.empty(s)
            self.assertEqual(x.type, t)
            self.assertEqual(x.value, v)

    def test_bytes_assign(self):
        t = "2 * SomeByteArray(3 * bytes)"
        inner = [b'a', b'b', b'c']
        v = 2 * [inner]

        x = xnd(v, type=t)
        for i in range(2):
            for k in range(3):
                x[i, k] = inner[k] = bytes(chr(ord('x') + k), "ascii")

        self.assertEqual(x.value, v)


class TestChar(XndTestCase):

    def test_char(self):
        # Semantics need to be evaluated (we already have fixed_string
        # with different encodings).
        self.assertRaises(NotImplementedError, xnd.empty, "char('utf8')")
        self.assertRaises(NotImplementedError, xnd, 1, type="char('utf8')")


class TestBool(XndTestCase):

    def test_bool(self):
        # From bool.
        x = xnd(True, type="bool")
        self.assertIs(x.value, True)

        x = xnd(False, type="bool")
        self.assertIs(x.value, False)

        # From int.
        x = xnd(1, type="bool")
        self.assertIs(x.value, True)

        x = xnd(0, type="bool")
        self.assertIs(x.value, False)

        # From object (for numpy compat: np.bool([1,2,3]))
        x = xnd([1,2,3], type="bool")
        self.assertIs(x.value, True)

        x = xnd(None, type="?bool")
        self.assertIs(x.value, None)

        self.assertRaises(ValueError, xnd, None, type="bool")

        # Test broken input.
        b = BoolMemoryError()
        self.assertRaises(MemoryError, xnd, b, type="bool")

        # Test len.
        x = xnd(True, type="bool")
        self.assertRaises(TypeError, len, x)

    def test_bool_richcompare(self):

        self.assertStrictEqual(xnd(True), xnd(True))
        self.assertStrictEqual(xnd(False), xnd(False))
        self.assertNotStrictEqual(xnd(True), xnd(False))
        self.assertNotStrictEqual(xnd(False), xnd(True))


class TestSignedKind(XndTestCase):

    def test_signed_kind(self):
        self.assertRaises(ValueError, xnd.empty, "Signed")


class TestSigned(XndTestCase):

    def test_signed(self):
        # Test bounds.
        for n in (8, 16, 32, 64):
            t = "int%d" % n

            v = -2**(n-1)
            x = xnd(v, type=t)
            self.assertEqual(x.value, v)
            self.assertRaises((ValueError, OverflowError), xnd, v-1, type=t)

            v = 2**(n-1) - 1
            x = xnd(v, type=t)
            self.assertEqual(x.value, v)
            self.assertRaises((ValueError, OverflowError), xnd, v+1, type=t)

        # Test index.
        i = Index()
        for n in (8, 16, 32, 64):
            t = "int%d" % n
            x = xnd(i, type=t)
            self.assertEqual(x.value, 10)

        # Test broken input.
        for n in (8, 16, 32, 64):
            t = "int%d" % n
            i = IndexMemoryError()
            self.assertRaises(MemoryError, xnd, i, type=t)
            i = IndexTypeError()
            self.assertRaises(TypeError, xnd, i, type=t)

        # Test len.
        x = xnd(10, type="int16")
        self.assertRaises(TypeError, len, x)

    def test_signed_richcompare(self):

        for t in "int8", "int16", "int32", "int64":
            self.assertStrictEqual(xnd(-10, type=t), xnd(-10, type=t))
            self.assertNotStrictEqual(xnd(-10, type=t), xnd(100, type=t))


class TestUnsignedKind(XndTestCase):

    def test_unsigned_kind(self):
        self.assertRaises(ValueError, xnd.empty, "Unsigned")


class TestUnsigned(XndTestCase):

    def test_unsigned(self):
        # Test bounds.
        for n in (8, 16, 32, 64):
            t = "uint%d" % n

            v = 0
            x = xnd(v, type=t)
            self.assertEqual(x.value, v)
            self.assertRaises((ValueError, OverflowError), xnd, v-1, type=t)

            v = 2**n - 1
            x = xnd(v, type=t)
            self.assertEqual(x.value, v)
            self.assertRaises((ValueError, OverflowError), xnd, v+1, type=t)

        # Test index.
        i = Index()
        for n in (8, 16, 32, 64):
            t = "uint%d" % n
            x = xnd(i, type=t)
            self.assertEqual(x.value, 10)

        # Test broken input.
        for n in (8, 16, 32, 64):
            t = "uint%d" % n
            i = IndexMemoryError()
            self.assertRaises(MemoryError, xnd, i, type=t)
            i = IndexTypeError()
            self.assertRaises(TypeError, xnd, i, type=t)

        # Test len.
        x = xnd(10, type="uint64")
        self.assertRaises(TypeError, len, x)

    def test_unsigned_richcompare(self):

        for t in "uint8", "uint16", "uint32", "uint64":
            self.assertStrictEqual(xnd(10, type=t), xnd(10, type=t))
            self.assertNotStrictEqual(xnd(10, type=t), xnd(100, type=t))


class TestFloatKind(XndTestCase):

    def test_float_kind(self):
        self.assertRaises(ValueError, xnd.empty, "Float")


class TestFloat(XndTestCase):

    @requires_py36
    def test_float16(self):
        fromhex = float.fromhex

        # Test creation and initialization of empty xnd objects.
        for value, type_string in EMPTY_TEST_CASES:
            ts = type_string % "float16"
            x = xnd.empty(ts)
            self.assertEqual(x.value, value)
            self.assertEqual(x.type, ndt(ts))

        # Test bounds.
        DENORM_MIN = fromhex("0x1p-24")
        LOWEST = fromhex("-0x1.ffcp+15")
        MAX = fromhex("0x1.ffcp+15")
        INF = fromhex("0x1.ffep+15")

        x = xnd(DENORM_MIN, type="float16")
        self.assertEqual(x.value, DENORM_MIN)

        x = xnd(LOWEST, type="float16")
        self.assertEqual(x.value, LOWEST)

        x = xnd(MAX, type="float16")
        self.assertEqual(x.value, MAX)

        self.assertRaises(OverflowError, xnd, INF, type="float16")
        self.assertRaises(OverflowError, xnd, -INF, type="float16")

        # Test special values.
        x = xnd(float("inf"), type="float16")
        self.assertTrue(isinf(x.value))

        x = xnd(float("nan"), type="float16")
        self.assertTrue(isnan(x.value))

        # Richcompare.
        self.assertStrictEqual(xnd(1.2e3, type="float16"), xnd(1.2e3, type="float16"))
        self.assertStrictEqual(xnd(float("inf"), type="float16"), xnd(float("inf"), type="float16"))
        self.assertStrictEqual(xnd(float("-inf"), type="float16"), xnd(float("-inf"), type="float16"))

        self.assertNotStrictEqual(xnd(1.2e3, type="float16"), xnd(-1.2e3, type="float16"))
        self.assertNotStrictEqual(xnd(float("inf"), type="float16"), xnd(float("-inf"), type="float16"))
        self.assertNotStrictEqual(xnd(float("-inf"), type="float16"), xnd(float("inf"), type="float16"))
        self.assertNotStrictEqual(xnd(float("nan"), type="float16"), xnd(float("nan"), type="float16"))

    def test_float32(self):
        fromhex = float.fromhex

        # Test bounds.
        DENORM_MIN = fromhex("0x1p-149")
        LOWEST = fromhex("-0x1.fffffep+127")
        MAX = fromhex("0x1.fffffep+127")
        INF = fromhex("0x1.ffffffp+127")

        x = xnd(DENORM_MIN, type="float32")
        self.assertEqual(x.value, DENORM_MIN)

        x = xnd(LOWEST, type="float32")
        self.assertEqual(x.value, LOWEST)

        x = xnd(MAX, type="float32")
        self.assertEqual(x.value, MAX)

        self.assertRaises(OverflowError, xnd, INF, type="float32")
        self.assertRaises(OverflowError, xnd, -INF, type="float32")

        # Test special values.
        x = xnd(float("inf"), type="float32")
        self.assertTrue(isinf(x.value))

        x = xnd(float("nan"), type="float32")
        self.assertTrue(isnan(x.value))

        # Richcompare.
        self.assertStrictEqual(xnd(1.2e7, type="float32"), xnd(1.2e7, type="float32"))
        self.assertStrictEqual(xnd(float("inf"), type="float32"), xnd(float("inf"), type="float32"))
        self.assertStrictEqual(xnd(float("-inf"), type="float32"), xnd(float("-inf"), type="float32"))

        self.assertNotStrictEqual(xnd(1.2e7, type="float32"), xnd(-1.2e7, type="float32"))
        self.assertNotStrictEqual(xnd(float("inf"), type="float32"), xnd(float("-inf"), type="float32"))
        self.assertNotStrictEqual(xnd(float("-inf"), type="float32"), xnd(float("inf"), type="float32"))
        self.assertNotStrictEqual(xnd(float("nan"), type="float32"), xnd(float("nan"), type="float32"))

    def test_float64(self):
        fromhex = float.fromhex

        # Test bounds.
        DENORM_MIN = fromhex("0x0.0000000000001p-1022")
        LOWEST = fromhex("-0x1.fffffffffffffp+1023")
        MAX = fromhex("0x1.fffffffffffffp+1023")

        x = xnd(DENORM_MIN, type="float64")
        self.assertEqual(x.value, DENORM_MIN)

        x = xnd(LOWEST, type="float64")
        self.assertEqual(x.value, LOWEST)

        x = xnd(MAX, type="float64")
        self.assertEqual(x.value, MAX)

        # Test special values.
        x = xnd(float("inf"), type="float64")
        self.assertTrue(isinf(x.value))

        x = xnd(float("nan"), type="float64")
        self.assertTrue(isnan(x.value))

        # Richcompare.
        self.assertStrictEqual(xnd(1.2e7, type="float64"), xnd(1.2e7, type="float64"))
        self.assertStrictEqual(xnd(float("inf"), type="float64"), xnd(float("inf"), type="float64"))
        self.assertStrictEqual(xnd(float("-inf"), type="float64"), xnd(float("-inf"), type="float64"))

        self.assertNotStrictEqual(xnd(1.2e7, type="float64"), xnd(-1.2e7, type="float64"))
        self.assertNotStrictEqual(xnd(float("inf"), type="float64"), xnd(float("-inf"), type="float64"))
        self.assertNotStrictEqual(xnd(float("-inf"), type="float64"), xnd(float("inf"), type="float64"))
        self.assertNotStrictEqual(xnd(float("nan"), type="float64"), xnd(float("nan"), type="float64"))


class TestComplexKind(XndTestCase):

    def test_complex_kind(self):
n        self.assertRaises(ValueError, xnd.empty, "Complex")


class TestComplex(XndTestCase):

    @requires_py36
    def test_complex32(self):
        fromhex = float.fromhex

        # Test creation and initialization of empty xnd objects.
        for value, type_string in EMPTY_TEST_CASES:
            ts = type_string % "complex32"
            x = xnd.empty(ts)
            self.assertEqual(x.value, value)
            self.assertEqual(x.type, ndt(ts))

        # Test bounds.
        DENORM_MIN = fromhex("0x1p-24")
        LOWEST = fromhex("-0x1.ffcp+15")
        MAX = fromhex("0x1.ffcp+15")
        INF = fromhex("0x1.ffep+15")

        v = complex(DENORM_MIN, DENORM_MIN)
        x = xnd(v, type="complex32")
        self.assertEqual(x.value, v)

        v = complex(LOWEST, LOWEST)
        x = xnd(v, type="complex32")
        self.assertEqual(x.value, v)

        v = complex(MAX, MAX)
        x = xnd(v, type="complex32")
        self.assertEqual(x.value, v)

        v = complex(INF, INF)
        self.assertRaises(OverflowError, xnd, v, type="complex32")

        v = complex(-INF, -INF)
        self.assertRaises(OverflowError, xnd, v, type="complex32")

        # Test special values.
        x = xnd(complex("inf"), type="complex32")
        self.assertTrue(isinf(x.value.real))
        self.assertEqual(x.value.imag, 0.0)

        x = xnd(complex("nan"), type="complex32")
        self.assertTrue(isnan(x.value.real))
        self.assertEqual(x.value.imag, 0.0)

        # Richcompare.
        t = "complex32"
        c = [DENORM_MIN, LOWEST, MAX, float("inf"), float("-inf"), float("nan")]
        for r in c:
            for s in c:
                for i in c:
                    for j in c:
                        x = xnd(complex(r, i), type=t)
                        y = xnd(complex(s, j), type=t)
                        if r == s and i == j:
                            self.assertStrictEqual(x, y)
                        else:
                            self.assertNotStrictEqual(x, y)

    def test_complex64(self):
        fromhex = float.fromhex

        # Test bounds.
        DENORM_MIN = fromhex("0x1p-149")
        LOWEST = fromhex("-0x1.fffffep+127")
        MAX = fromhex("0x1.fffffep+127")
        INF = fromhex("0x1.ffffffp+127")

        v = complex(DENORM_MIN, DENORM_MIN)
        x = xnd(v, type="complex64")
        self.assertEqual(x.value, v)

        v = complex(LOWEST, LOWEST)
        x = xnd(v, type="complex64")
        self.assertEqual(x.value, v)

        v = complex(MAX, MAX)
        x = xnd(v, type="complex64")
        self.assertEqual(x.value, v)

        v = complex(INF, INF)
        self.assertRaises(OverflowError, xnd, INF, type="complex64")

        v = complex(-INF, -INF)
        self.assertRaises(OverflowError, xnd, -INF, type="complex64")

        # Test special values.
        x = xnd(complex("inf"), type="complex64")
        self.assertTrue(isinf(x.value.real))
        self.assertEqual(x.value.imag, 0.0)

        x = xnd(complex("nan"), type="complex64")
        self.assertTrue(isnan(x.value.real))
        self.assertEqual(x.value.imag, 0.0)

        # Richcompare.
        t = "complex64"
        c = [DENORM_MIN, LOWEST, MAX, float("inf"), float("-inf"), float("nan")]
        for r in c:
            for s in c:
                for i in c:
                    for j in c:
                        x = xnd(complex(r, i), type=t)
                        y = xnd(complex(s, j), type=t)
                        if r == s and i == j:
                            self.assertStrictEqual(x, y)
                        else:
                            self.assertNotStrictEqual(x, y)

    def test_complex128(self):
        fromhex = float.fromhex

        # Test bounds.
        DENORM_MIN = fromhex("0x0.0000000000001p-1022")
        LOWEST = fromhex("-0x1.fffffffffffffp+1023")
        MAX = fromhex("0x1.fffffffffffffp+1023")

        v = complex(DENORM_MIN, DENORM_MIN)
        x = xnd(v, type="complex128")
        self.assertEqual(x.value, v)

        v = complex(LOWEST, LOWEST)
        x = xnd(v, type="complex128")
        self.assertEqual(x.value, v)

        v = complex(MAX, MAX)
        x = xnd(v, type="complex128")
        self.assertEqual(x.value, v)

        # Test special values.
        x = xnd(complex("inf"), type="complex128")
        self.assertTrue(isinf(x.value.real))
        self.assertEqual(x.value.imag, 0.0)

        x = xnd(complex("nan"), type="complex128")
        self.assertTrue(isnan(x.value.real))
        self.assertEqual(x.value.imag, 0.0)

        # Richcompare.
        t = "complex128"
        c = [DENORM_MIN, LOWEST, MAX, float("inf"), float("-inf"), float("nan")]
        for r in c:
            for s in c:
                for i in c:
                    for j in c:
                        x = xnd(complex(r, i), type=t)
                        y = xnd(complex(s, j), type=t)
                        if r == s and i == j:
                            self.assertStrictEqual(x, y)
                        else:
                            self.assertNotStrictEqual(x, y)


class TestPrimitive(XndTestCase):

    def test_primitive_empty(self):
        # Test creation and initialization of empty xnd objects.

        for value, type_string in EMPTY_TEST_CASES:
            for p in PRIMITIVE:
                ts = type_string % p
                x = xnd.empty(ts)
                self.assertEqual(x.value, value)
                self.assertEqual(x.type, ndt(ts))


class TestTypevar(XndTestCase):

    def test_typevar(self):
        self.assertRaises(ValueError, xnd.empty, "T")
        self.assertRaises(ValueError, xnd.empty, "2 * 10 * T")
        self.assertRaises(ValueError, xnd.empty, "{a: 2 * 10 * T, b: bytes}")


class TestTypeInference(XndTestCase):

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

    def test_complex128(self):
        d = R['a': 3.123+10j, 'b': complex('inf')]
        typeof_d = "{a: complex128, b: complex128}"

        test_cases = [
          ([1+3e300j], "1 * complex128"),
          ([-2.2-5j, 1.2-10j], "2 * complex128"),
          ([-2.2-5j, 1.2-10j, None], "3 * ?complex128"),
          ([[-1+3j], [-3+5j]], "2 * 1 * complex128"),

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

    def test_optional(self):
        test_cases = [
          (None, "?float64"),
          ([None], "1 * ?float64"),
          ([None, None], "2 * ?float64"),
          ([None, 10], "2 * ?int64"),
          ([None, b'abc'], "2 * ?bytes"),
          ([None, 'abc'], "2 * ?string")
        ]

        for v, t in test_cases:
            x = xnd(v)
            self.assertEqual(x.type, ndt(t))
            self.assertEqual(x.value, v)

        # Optional dimensions are not implemented.
        not_implemented = [
          [None, []],
          [[], None],
          [None, [10]],
          [[None, [0, 1]], [[2, 3]]]
        ]

        for v in not_implemented:
            self.assertRaises(NotImplementedError, xnd, v)


class TestIndexing(XndTestCase):

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
        self.assertEqual(x[0].value, t1)
        self.assertEqual(x[1].value, t2)

        self.assertEqual(x[0,0], 1.0)
        self.assertEqual(x[0,1], "capricious")
        self.assertEqual(x[0,2].value, (1, 2, 3))

        self.assertEqual(x[1,0], 2.0)
        self.assertEqual(x[1,1], "volatile")
        self.assertEqual(x[1,2].value, (4, 5, 6))

    def test_subview(self):
        # fixed
        x = xnd([["a", "b"], ["c", "d"]])
        self.assertEqual(x[0].value, ["a", "b"])
        self.assertEqual(x[1].value, ["c", "d"])

        # var
        x = xnd([["a", "b"], ["x", "y", "z"]])
        self.assertEqual(x[0].value, ["a", "b"])
        self.assertEqual(x[1].value, ["x", "y", "z"])


class TestSequence(XndTestCase):

    def test_sequence(self):
        for v, s in DTYPE_EMPTY_TEST_CASES:
            for vv, ss in [
               (1 * [1 * [v]], "!1 * 1 * %s" % s),
               (1 * [2 * [v]], "!1 * 2 * %s" % s),
               (2 * [1 * [v]], "!2 * 1 * %s" % s),
               (2 * [2 * [v]], "2 * 2 * %s" % s),
               (2 * [3 * [v]], "2 * 3 * %s" % s),
               (3 * [2 * [v]], "3 * 2 * %s" % s)]:

                x = xnd(vv, type=ss)

                lst = [v for v in x]

                for i, z in enumerate(x):
                    self.assertEqual(z.value, lst[i].value)


class TestAPI(XndTestCase):

    def test_hash(self):
        x = xnd(1000)
        self.assertRaises(TypeError, hash, x)

        x = xnd([1, 2, 3])
        self.assertRaises(TypeError, hash, x)

    def test_short_value(self):
        x = xnd([1, 2])
        self.assertEqual(x.short_value(0), [])
        self.assertEqual(x.short_value(1), [XndEllipsis])
        self.assertEqual(x.short_value(2), [1, XndEllipsis])
        self.assertEqual(x.short_value(3), [1, 2])
        self.assertRaises(ValueError, x.short_value, -1)

        x = xnd([[1, 2], [3]])
        self.assertEqual(x.short_value(0), [])
        self.assertEqual(x.short_value(1), [XndEllipsis])
        self.assertEqual(x.short_value(2), [[1, XndEllipsis], XndEllipsis])
        self.assertEqual(x.short_value(3), [[1, 2], [3]])
        self.assertRaises(ValueError, x.short_value, -1)

        x = xnd((1, 2))
        self.assertEqual(x.short_value(0), ())
        self.assertEqual(x.short_value(1), (XndEllipsis,))
        self.assertEqual(x.short_value(2), (1, XndEllipsis))
        self.assertEqual(x.short_value(3), (1, 2))
        self.assertRaises(ValueError, x.short_value, -1)

        x = xnd(R['a': 1, 'b': 2])
        self.assertEqual(x.short_value(0), {})
        self.assertEqual(x.short_value(1), {XndEllipsis: XndEllipsis})
        self.assertEqual(x.short_value(2), R['a': 1, XndEllipsis: XndEllipsis])
        self.assertEqual(x.short_value(3), R['a': 1, 'b': 2])
        self.assertRaises(ValueError, x.short_value, -1)


class TestRepr(XndTestCase):

    def test_repr(self):
        lst = 10 * [19 * [23 * [{'a': 100, 'b': "xyz", 'c': ['abc', 'uvw']}]]]
        x = xnd(lst)
        r = repr(x)
        self.assertLess(len(r), 100000)


class TestBuffer(XndTestCase):

    @unittest.skipIf(np is None, "numpy not found")
    def test_from_buffer(self):
        x = np.array([[[0,1,2],
                       [3,4,5]],
                     [[6,7,8],
                      [9,10,11]]])

        y = xnd.from_buffer(x)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    self.assertEqual(y[i,j,k], x[i,j,k])

        x = np.array([(1000, 400.25, 'abc'), (-23, -1e10, 'cba')],
                     dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'S3')])
        y = xnd.from_buffer(x)

        for i in range(2):
            for k in ['x', 'y', 'z']:
                self.assertEqual(y[i][k], x[i][k])

    @unittest.skipIf(np is None, "numpy not found")
    def test_unsafe_from_buffer(self):
        x = np.array([[[0,1,2],
                       [3,4,5]],
                     [[6,7,8],
                      [9,10,11]]], dtype="int64")

        y = xnd.unsafe_from_data(obj=x, type="12 * int64")
        np.testing.assert_equal(y, x.reshape(12))

    @unittest.skipIf(np is None, "numpy not found")
    def test_endian(self):
        standard = [
            '?',
            'c', 'b', 'B',
            'h', 'i', 'l', 'q',
            'H', 'I', 'L', 'Q',
            'f', 'd',
        ]

        if HAVE_PYTHON_36:
            standard += 'e'

        modifiers = ['', '<', '>']

        for fmt in standard:
            for mod in modifiers:
                f = mod + fmt
                x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=f)
                y = xnd.from_buffer(x)
                for i in range(10):
                    self.assertEqual(y[i], x[i])

        # XXX T{>i:x:f:y:3s:z:} does not work.
        x = np.array([(1000, 400.25, 'abc'), (-23, -1e10, 'cba')],
                     dtype=[('x', '<i4'), ('y', '>f4'), ('z', 'S3')])
        y = xnd.from_buffer(x)

        for i in range(2):
            for k in ['x', 'y', 'z']:
                self.assertEqual(y[i][k], x[i][k])

    def test_readonly(self):
        x = ndarray([1,2,3], shape=[3], format="L")
        y = xnd.from_buffer(x)
        self.assertRaises(TypeError, y.__setitem__, 0, 1000)

        x = ndarray([1,2,3], shape=[3], format="L", flags=ND_WRITABLE)
        y = xnd.from_buffer(x)
        y[:] = [1000, 2000, 3000]
        self.assertEqual(x.tolist(), [1000, 2000, 3000])


class TestSplit(XndTestCase):

    def test_split(self):
        for _ in range(1):
            for lst in gen_fixed():
                x = xnd(lst)
                for n in range(1, 100):
                    try:
                        a = split_xnd(x, n)
                    except ValueError:
                        continue
                    b = xnd.split(x, n)
                    self.assertEqual(a, b)

    def test_split_limit_outer(self):
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        for lst in gen_fixed():
            x = xnd(lst)
            for n in range(1, 10):
                for m in range(x.type.ndim + 2):
                    try:
                        a = split_xnd(x, n, max_outer=m)
                    except ValueError:
                        continue
                    b = xnd.split(x, n, max_outer=m)
                    self.assertEqual(a, b)


class TestView(XndTestCase):

    def test_view_subscript(self):
        x = xnd([[1,2,3], [4,5,6]])
        y = _test_view_subscript(x, key=(0, 1))
        self.assertEqual(y, xnd(2))

        x = xnd([[1,2,3], [4,5,6]])
        y = _test_view_subscript(x, key=(1, slice(None, None, -1)))
        self.assertEqual(y, xnd([6,5,4]))

    def test_view_new(self):
        x = _test_view_new()
        self.assertEqual(x, xnd([1.1, 2.2, 3.3]))


class TestSpec(XndTestCase):

    def __init__(self, *, constr,
                 values, value_generator,
                 indices_generator, indices_generator_args):
        super().__init__()
        self.constr = constr
        self.values = values
        self.value_generator = value_generator
        self.indices_generator = indices_generator
        self.indices_generator_args = indices_generator_args
        self.indices_stack = [None] * 8

    def log_err(self, value, depth):
        """Dump an error as a Python script for debugging."""

        sys.stderr.write("\n\nfrom xnd import *\n")
        sys.stderr.write("from test_xnd import NDArray\n")
        sys.stderr.write("lst = %s\n\n" % value)
        sys.stderr.write("x0 = xnd(lst)\n")
        sys.stderr.write("y0 = NDArray(lst)\n" % value)

        for i in range(depth+1):
            sys.stderr.write("x%d = x%d[%s]\n" % (i+1, i, itos(self.indices_stack[i])))
            sys.stderr.write("y%d = y%d[%s]\n" % (i+1, i, itos(self.indices_stack[i])))

        sys.stderr.write("\n")

    def run_single(self, nd, d, indices):
        """Run a single test case."""

        self.assertEqual(len(nd), len(d))

        nd_exception = None
        try:
            nd_result = nd[indices]
        except Exception as e:
            nd_exception =  e

        def_exception = None
        try:
            def_result = d[indices]
        except Exception as e:
            def_exception = e

        if nd_exception or def_exception:
            if nd_exception is None and def_exception.__class__ is IndexError:
                # Example: type = 0 * 0 * int64
                if len(indices) <= nd.ndim:
                    return None, None

            self.assertIs(nd_exception.__class__, def_exception.__class__)
            return None, None

        if isinstance(nd_result, xnd):
            nd_value = nd_result.value
        elif np is not None and isinstance(nd_result, np.ndarray):
            nd_value = nd_result.tolist()
        else:
            nd_value = nd_result

        self.assertEqual(nd_value, def_result)
        return nd_result, def_result

    def run(self):
        def check(nd, d, value, depth):
            if depth > 3: # adjust for longer tests
                return

            g = self.indices_generator(*self.indices_generator_args)

            for indices in g:
                self.indices_stack[depth] = indices

                try:
                    next_nd, next_d = self.run_single(nd, d, indices)
                except Exception as e:
                    self.log_err(value, depth)
                    raise e

                if isinstance(next_d, list): # possibly None or scalar
                    check(next_nd, next_d, value, depth+1)

        for value in self.values:
            nd = self.constr(value)
            d = NDArray(value)
            check(nd, d, value, 0)
            check_buffer(nd)

        for max_ndim in range(1, 5):
            for min_shape in (0, 1):
                for max_shape in range(1, 8):
                    for value in self.value_generator(max_ndim, min_shape, max_shape):
                        nd = self.constr(value)
                        d = NDArray(value)
                        check(nd, d, value, 0)
                        check_buffer(nd)


class LongIndexSliceTest(XndTestCase):

    def test_subarray(self):
        # Multidimensional indexing
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=genindices,
                     indices_generator_args=())
        t.run()

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=genindices,
                     indices_generator_args=())
        t.run()

    def test_slices(self):
        # Multidimensional slicing
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=randslices,
                     indices_generator_args=(3,))
        t.run()

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=randslices,
                     indices_generator_args=(3,))
        t.run()

    def test_chained_indices_slices(self):
        # Multidimensional indexing and slicing, chained
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=gen_indices_or_slices,
                     indices_generator_args=())
        t.run()


        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=gen_indices_or_slices,
                     indices_generator_args=())
        t.run()

    def test_fixed_mixed_indices_slices(self):
        # Multidimensional indexing and slicing, mixed
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=mixed_indices,
                     indices_generator_args=(3,))
        t.run()

    def test_var_mixed_indices_slices(self):
        # Multidimensional indexing and slicing, mixed
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        x = xnd([[1], [2, 3], [4, 5, 6]])

        indices = (0, slice(0,1,1))
        self.assertRaises(IndexError, x.__getitem__, indices)

        indices = (slice(0,1,1), 0)
        self.assertRaises(IndexError, x.__getitem__, indices)

    def test_slices_brute_force(self):
        # Test all possible slices for the given ndim and shape
        skip_if(SKIP_BRUTE_FORCE, "use --all argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=genslices_ndim,
                     indices_generator_args=(3, [3,3,3]))
        t.run()

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=genslices_ndim,
                     indices_generator_args=(3, [3,3,3]))
        t.run()

    @unittest.skipIf(np is None, "numpy not found")
    def test_array_definition(self):
        # Test the NDArray definition against NumPy
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=np.array,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=mixed_indices,
                     indices_generator_args=(3,))
        t.run()


ALL_TESTS = [
  TestModule,
  TestFunction,
  TestVoid,
  TestAny,
  TestFixedDim,
  TestFortran,
  TestVarDim,
  TestSymbolicDim,
  TestEllipsisDim,
  TestTuple,
  TestRecord,
  TestRef,
  TestConstr,
  TestNominal,
  TestScalarKind,
  TestCategorical,
  TestFixedStringKind,
  TestFixedString,
  TestFixedBytesKind,
  TestFixedBytes,
  TestString,
  TestBytes,
  TestChar,
  TestBool,
  TestSignedKind,
  TestSigned,
  TestUnsignedKind,
  TestUnsigned,
  TestFloatKind,
  TestFloat,
  TestComplexKind,
  TestComplex,
  TestPrimitive,
  TestTypevar,
  TestTypeInference,
  TestIndexing,
  TestSequence,
  TestAPI,
  TestRepr,
  TestBuffer,
  TestSplit,
  TestView,
  LongIndexSliceTest,
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--failfast", action="store_true",
                        help="stop the test run on first error")
    parser.add_argument('--long', action="store_true", help="run long slice tests")
    parser.add_argument('--all', action="store_true", help="run brute force tests")
    args = parser.parse_args()
    SKIP_LONG = not (args.long or args.all)
    SKIP_BRUTE_FORCE = not args.all

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for case in ALL_TESTS:
        s = loader.loadTestsFromTestCase(case)
        suite.addTest(s)

    runner = unittest.TextTestRunner(failfast=args.failfast, verbosity=2)
    result = runner.run(suite)
    ret = not result.wasSuccessful()

    sys.exit(ret)

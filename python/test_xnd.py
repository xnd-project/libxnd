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

import unittest
from ndtypes import ndt
from xnd import xnd


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


class TypeInferenceTest(unittest.TestCase):

    def test_float64(self):
        d = {'a': 2.221e100, 'b': float('inf')}
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


unittest.main(verbosity=2)



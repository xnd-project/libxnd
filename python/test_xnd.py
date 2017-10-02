from _pyxnd import Array, data_shapes
from itertools import accumulate, count, product
from functools import partial
from random import randrange
import sys, unittest

try:
    import numpy as np
except ImportError:
    np = None


# ======================================================================
#            Definition of generalized slicing and indexing
# ======================================================================

def maxlevel(lst):
    maxlevel = 0
    def f(lst, level):
        nonlocal maxlevel
        if isinstance(lst, list):
            level += 1
            maxlevel = max(level, maxlevel)
            for item in lst:
                f(item, level)
    f(lst, 0)
    return maxlevel

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

class SimpleArray(list):
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
        return SimpleArray(result) if isinstance(result, list) else result


# ======================================================================
#                          Generate test cases 
# ======================================================================

FIXED_TEST_CASES = [
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

VAR_TEST_CASES = [
  [[[0, 1], [2, 3]], [[4, 5, 6], [7]], [[8, 9]]],
  # [[[0, 1], [2, 3]], [[4, 5, None], None, [7]], None, [[8, 9]]],
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
    ret = []
    for t in product(range(-n, n+1), range(-n, n+1), range(-n, n+1)):
        s = slice(*t)
        if s.step != 0:
            ret.append(s)
    return ret

def genslices_ndim(ndim, shape):
    """Generate all possible slice tuples for 'shape'."""
    iterables = [genslices(shape[n]) for n in range(ndim)]
    return product(*iterables)

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
    for i in range(20):
        yield mixed_index(max_ndim)

def get_ndim(a):
    if isinstance(a, Array):
        return a.typ.ndim
    else: # NumPy
        return a.ndim

def itos(indices):
    return ", ".join(str(i) if isinstance(i, int) else "%d:%d:%d" %
                     (i.start, i.stop, i.step) for i in indices)

class IndicesTest(unittest.TestCase):


    def log_err(self, lst, indices_stack, depth):
        """Dump the error as a Python script for debugging."""

        sys.stderr.write("\n\nfrom xnd import *\n")
        sys.stderr.write("from test_xnd import SimpleArray\n")
        sys.stderr.write("lst = %s\n\n" % lst)
        sys.stderr.write("x0 = Array(lst)\n")
        sys.stderr.write("y0 = SimpleArray(lst)\n" % lst)

        for i in range(depth+1):
            sys.stderr.write("x%d = x%d[%s]\n" % (i+1, i, itos(indices_stack[i])))
            sys.stderr.write("y%d = y%d[%s]\n" % (i+1, i, itos(indices_stack[i])))

        sys.stderr.write("\n")

    def compare(self, lst, xnd, definition, indices, indices_stack, depth):
        """Run a single test case."""

        xnd_exc = None
        try:
            xnd_result = xnd[indices]
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
                if len(indices) <= get_ndim(xnd):
                    return None, None

            if xnd_exc.__class__ is IndexError and definition_exc is None:
                if "technical difficulties" in xnd_exc.args[0]:
                    return None, None

            if xnd_exc.__class__ is not definition_exc.__class__:
                self.log_err(lst, indices_stack, depth)

            self.assertIs(xnd_exc.__class__, definition_exc.__class__)

            return None, None

        else:
            try:
                xnd_lst = xnd_result.tolist()
            except Exception as e:
                self.log_err(lst, indices_stack, depth)
                raise e

            if xnd_lst != definition_result:
                self.log_err(lst, indices_stack, depth)

            self.assertEqual(xnd_lst, definition_result)

            return xnd_result, definition_result

    def doit(self, array=None, tests=None, genindices=None, genlists=None):
        indices_stack = [None] * 8

        def check(lst, xnd, definition, depth):
            if depth > 3: # adjust for longer tests
                return

            for indices in genindices():
                indices_stack[depth] = indices
                _xnd, _def = self.compare(lst, xnd, definition, indices,
                                          indices_stack, depth)

                if isinstance(_def, list): # possibly None or scalar
                    check(lst, _xnd, _def, depth+1)

        for lst in tests:
            check(lst, array(lst), SimpleArray(lst), 0)

        for max_ndim in range(1, 5):
            for min_shape in (0, 1):
                for max_shape in range(1, 8):
                    for lst in genlists(max_ndim, min_shape, max_shape):
                        check(lst, array(lst), SimpleArray(lst), 0)

    def test_subarray(self):
        # Multidimensional indexing
        self.doit(Array, FIXED_TEST_CASES, genindices, gen_fixed)
        self.doit(Array, VAR_TEST_CASES, genindices, gen_var)

    def test_slices(self):
        # Multidimensional slicing
        genindices = partial(randslices, 3)
        self.doit(Array, FIXED_TEST_CASES, genindices, gen_fixed)
        self.doit(Array, VAR_TEST_CASES, genindices, gen_var)

    def test_chained_indices_slices(self):
        # Multidimensional indexing and slicing, chained
        self.doit(Array, FIXED_TEST_CASES, gen_indices_or_slices, gen_fixed)
        self.doit(Array, VAR_TEST_CASES, gen_indices_or_slices, gen_var)

    def test_mixed_indices_slices(self):
        # Multidimensional indexing and slicing, mixed
        genindices = partial(mixed_indices, 3)
        self.doit(Array, FIXED_TEST_CASES, genindices, gen_fixed)
        self.doit(Array, VAR_TEST_CASES, genindices, gen_var)

    @unittest.skipIf(np is None, "numpy not found")
    def test_simple_array(self):
        # Test the SimpleArray definition against NumPy
        genindices = partial(mixed_indices, 3)
        self.doit(np.array, FIXED_TEST_CASES, genindices, gen_fixed)

    @unittest.skipIf(True, "very long duration")
    def test_slices_brute_force(self):
        # Test all possible slices for the given ndim and shape
        genindices = partial(genslices_ndim, 3, [3,3,3])
        self.doit(np.array, FIXED_TEST_CASES, genindices, gen_fixed)
        self.doit(np.array, VAR_TEST_CASES, genindices, gen_var)


if __name__ == '__main__':
    unittest.main(verbosity=2)

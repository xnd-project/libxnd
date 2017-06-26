from xnd import Array, FixedDim, typeof
import numpy as np
from random import randrange
from itertools import product
import unittest


# Numpy arrays
NUMPY_TEST_CASES = [
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

# Variable arrays
VAR_TEST_CASES = [
  [[[0, 1], [2, 3]], [[4, 5, 6], [7]], [[8, 9]]],
  # [[[0, 1], [2, 3]], [[4, 5, None], None, [7]], None, [[8, 9]]],
  [[[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]], [[11, 12, 13, 14], [15, 16, 17], [18, 19]]],
  [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9]], [[10, 11]]]
]

TEST_CASES = NUMPY_TEST_CASES + VAR_TEST_CASES


# ======================================================================
#            Definition of generalized slicing and indexing
# ======================================================================

def getitem(lst, indices):
    """The shortest definition for multidimensional slicing and indexing
       on arbitrarily shaped nested lists.
    """
    if not indices:
        return lst
    i, indices = indices[0], indices[1:]
    item = list.__getitem__(lst, i)
    if isinstance(i, int):
        return getitem(item, indices)
    if any(isinstance(i, int) for i in indices):
        raise IndexError("out of bounds")
    return [getitem(x, indices) for x in item]

class SimpleArray(list):
    """A simple wrapper for using slicing/indexing syntax on a nested list."""
    def __init__(self, value):
        list.__init__(self, value)
        self.typ = typeof(self)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) > self.typ.ndim:
            raise IndexError("too many indices")
        if not all(isinstance(x, int) or isinstance(x, slice) for x in indices):
            raise TypeError(
                "index must be int or slice or a tuple of integers and slices")
        return getitem(self, indices)


# ======================================================================
#                           Helper functions
# ======================================================================

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
    for i in range(20):
        yield multislice(ndim)

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


class IndexTest(unittest.TestCase):

    def test_subarray(self):
        # Test multidimensional indexing (also chained).

        indices_stack = [0] * 4
        test_stack = [0] * 4

        def f(lst, nd, depth):
            if depth > 3:
                return

            test_stack[depth] = lst

            for indices in genindices():
                indices_stack[depth] = indices

                err_lst = None
                try:
                    subarray_lst = lst[indices]
                except Exception as e:
                    err_lst = e.__class__

                err_nd = None
                try:
                    subarray_nd = nd[indices]
                except Exception as e:
                    err_nd = e.__class__

                if err_lst or err_nd:
                    self.assertIs(err_nd, err_lst)
                else:
                    self.assertEqual(subarray_nd.tolist(), subarray_lst)
                    if isinstance(subarray_lst, list): # can also be a scalar
                        f(SimpleArray(subarray_lst), subarray_nd, depth+1)

        for t in TEST_CASES:
            f(SimpleArray(t), Array(t), 0)

    def test_rand_slices(self):
        # Test random slices and chained slicing.

        slice_stack = [0] * 4

        def f(primary, lst, nd, depth):
            if depth > 3:
                return

            for slices in randslices(3):
                slice_stack[depth] = slices

                if isinstance(nd.typ, FixedDim) and nd.typ.shape <= 1 and \
                   len(slices) > 1: # slicing may work on nd but not on lst
                    continue

                err_lst = None
                try:
                    sliced_lst = lst[slices]
                except Exception as e:
                    err_lst = e

                err_nd = None
                try:
                    sliced_nd = nd[slices]
                except Exception as e:
                    err_nd = e

                if err_lst or err_nd:
                    if err_nd.__class__ is not err_lst.__class__:
                        if err_nd is None:
                            if err_lst.__class__ is IndexError:
                                continue
                            raise(err_lst)
                        elif err_lst is None:
                            if err_nd.__class__ is TypeError:
                                continue
                            raise(err_nd)
                        else:
                            self.assertIs(err_nd.__class__, err_lst.__class__)
                else:
                    self.assertEqual(sliced_nd.tolist(), sliced_lst)
                    f(primary, SimpleArray(sliced_lst), sliced_nd, depth+1)

        for t in TEST_CASES:
            f(t, SimpleArray(t), Array(t), 0)

    def test_mixed(self):
        # Mixed random indices and slices.

        indices_stack = [0] * 4

        def f(primary, lst, nd, depth):
            if depth > 3:
                return

            for indices in mixed_indices(4):
                indices_stack[depth] = indices

                if isinstance(nd.typ, FixedDim) and nd.typ.shape <= 1 and \
                   len(indices) > 1: # slicing may work on nd but not on lst
                    continue

                err_lst = None
                try:
                    result_lst = lst[indices]
                except Exception as e:
                    err_lst = e

                err_nd = None
                try:
                    result_nd = nd[indices]
                except Exception as e:
                    err_nd = e

                if err_lst or err_nd:
                    if err_nd.__class__ is not err_lst.__class__:
                        if err_nd is None:
                            if err_lst.__class__ is IndexError:
                                continue
                            raise(err_lst)
                        elif err_lst is None:
                            if err_nd.__class__ is TypeError:
                                continue
                            raise(err_nd)
                        else:
                            self.assertIs(err_nd.__class__, err_lst.__class__)
                else:
                    self.assertEqual(result_nd.tolist(), result_lst)
                    if isinstance(result_lst, list):
                        f(primary, SimpleArray(result_lst), result_nd, depth+1)

        for t in TEST_CASES:
            f(t, SimpleArray(t), Array(t), 0)

    def test_numpy(self):
        # Test the SimpleArray definition against NumPy.

        indices_stack = [0] * 4

        def f(primary, lst, nd, depth):
            if depth > 3:
                return

            for indices in mixed_indices(4):
                indices_stack[depth] = indices

                err_lst = None
                try:
                    result_lst = lst[indices]
                except Exception as e:
                    err_lst = e

                err_nd = None
                try:
                    result_nd = nd[indices]
                except Exception as e:
                    err_nd = e

                if err_lst or err_nd:
                    if err_nd.__class__ is not err_lst.__class__:
                        if err_nd is None:
                            if err_lst.__class__ is IndexError:
                                continue
                            raise(err_lst)
                        elif err_lst is None:
                            if err_nd.__class__ is TypeError:
                                continue
                            raise(err_nd)
                        else:
                            self.assertIs(err_nd.__class__, err_lst.__class__)
                else:
                    self.assertEqual(result_nd.tolist(), result_lst)
                    if isinstance(result_lst, list):
                        f(primary, SimpleArray(result_lst), result_nd, depth+1)

        for t in NUMPY_TEST_CASES:
            f(t, SimpleArray(t), np.array(t), 0)

    @unittest.skipIf(True, "brute force test: very long duration")
    def test_all_slices(self):
        # Test all possible slices for the given ndim and shape.

        for t in TEST_CASES:
            lst = SimpleArray(t)
            nd = Array(t)

            for slices in genslices_ndim(3, [3,3,3]):
                err_lst = None
                try:
                    sliced_lst = lst[slices]
                except Exception as e:
                    err_lst = e.__class__

                err_nd = None
                try:
                    sliced_nd = nd[slices]
                except Exception as e:
                    err_nd = e.__class__

                if err_lst or err_nd:
                    self.assertIs(err_nd, err_lst)
                else:
                    self.assertEqual(sliced_nd.tolist(), sliced_lst)


if __name__ == '__main__':
    unittest.main(verbosity=2)

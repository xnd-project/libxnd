from xnd import Array
from random import randrange
from itertools import product
import unittest



TEST_CASES = [
  [[[0, 1], [2, 3]], [[4, 5, 6], [7]], [[8, 9]]],
  [[[0, 1], [2, 3]], [[4, 5, None], None, [7]], None, [[8, 9]]],
  [[[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]], [[11, 12, 13, 14], [15, 16, 17], [18, 19]]],
  [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9]], [[10, 11]]]
]


# ======================================================================
#            Definition of generalized slicing and indexing
# ======================================================================

def getitem(lst, indices):
    """The shortest definition for multidimensional slicing and indexing
       on arbitrarily shaped nested lists.
    """
    if not indices:
        return lst
    if lst is None:
        return lst
    i, indices = indices[0], indices[1:]
    item = list.__getitem__(lst, i)
    if isinstance(i, int):
        return getitem(item, indices)
    return [getitem(x, indices) for x in item]

class SimpleArray(list):
    """A simple wrapper for using slicing/indexing syntax on a nested list."""
    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if not all(isinstance(x, int) or isinstance(x, slice) for x in indices):
            raise TypeError(
                "index must be int or slice or a tuple of integers and slices")
        return getitem(self, indices)


# ======================================================================
#                           Helper functions
# ======================================================================

def typeof(lst):
    """Infer the type of a nested list."""
    _, dim_shapes = data_shapes(lst)

    opt = None in lst
    t = Int64(opt)

    for lst in dim_shapes:
        opt = None in lst
        lst = replace_none(lst)
        t = choose_dim_type(t, lst, opt)

    return t

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


class SliceTest(unittest.TestCase):

    def test_rand_slices(self):
        # Test random slices and chained slicing.

        slice_stack = [0] * 4

        def f(lst, nd, depth):
            if depth > 3:
                return

            for slices in randslices(3):
                slice_stack[depth] = slices

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
                    f(SimpleArray(sliced_lst), sliced_nd, depth+1)

        for t in TEST_CASES:
            f(SimpleArray(t), Array(t), 0)

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

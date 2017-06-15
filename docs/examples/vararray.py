from itertools import accumulate, product
from random import randrange
from copy import copy
from _testbuffer import slice_indices
from test.test_buffer import genslices_ndim, rslices_ndim


########################################################################
#                   Python model of the xnd array type                 #
########################################################################


def is_value_list(lst):
    return all(not isinstance(v, list) for v in lst)

def replace_none(lst):
    return [0 if v is None else v for v in lst]

def opt_repr(b):
    return '?' if b else ''


# ======================================================================
#          Mini Datashape with just Int64, FixedDim and VarDim
# ======================================================================

class Datashape(object):
    pass

class Int64(Datashape):
    """Example dtype"""
    def __init__(self, opt=False):
        self.ndim = 0
        self.opt = opt
        self.size = 1

    def __repr__(self):
        return "%sint64" % opt_repr(self.opt)

    def __copy__(self):
        return Int64(self.opt)

    shape_repr = __repr__
    offset_shape_repr = __repr__

class FixedDim(Datashape):
    def __init__(self, shape, start=0, step=None, opt=False, typ=None):
        self.ndim = typ.ndim + 1
        self.start = start
        self.shape = shape
        if step is None:
            step = 1 if isinstance(typ, VarDim) else typ.size
        self.step = step
        self.size = shape * typ.size
        self.typ = typ
        self.opt = opt

    def __repr__(self):
        return "%s%d * %r" % (opt_repr(self.opt), self.shape, self.typ)

    def __copy__(self):
        typ = self.typ.__copy__()
        return FixedDim(self.shape, self.start, self.step, self.opt, typ)

    def shape_repr(self):
        return "%s%d * %s" % (opt_repr(self.opt), self.shape, self.typ.shape_repr())

    def offset_shape_repr(self):
        return "%s%d * %s" % (opt_repr(self.opt), self.shape, self.typ.offset_shape_repr())


class VarDim(Datashape):
    def __init__(self, shapes, start=0, step=1, opt=False, typ=None):
        self.ndim = typ.ndim + 1
        self.nshapes = len(shapes)
        self.shapes = shapes
        self.offsets = [0] + list(accumulate(shapes))
        self.typ = typ
        self.start = start
        self.step = step
        self.size = 1
        self.opt = opt

    def __repr__(self):
        return "%svar * %r" % (opt_repr(self.opt), self.typ)

    def __copy__(self):
        typ = self.typ.__copy__()
        return VarDim(self.shapes, self.start, self.step, self.opt, typ)

    def shape_repr(self):
        """shape arguments: var(1,2,3,...) * t"""
        def shapes():
            return ",".join(str(v) for v in self.shapes)

        return "%svar(%s) * %s" % (opt_repr(self.opt), shapes(), self.typ.shape_repr())

    def offset_shape_repr(self):
        """offset:shape arguments: var(1:3, 7:4, ...) * t"""
        def offsets_shapes():
            return ",".join("%d:%d" % (v[0], v[1]) for v in zip(self.offsets, self.shapes+[0]))

        return "%svar(%s) * %s" % (opt_repr(self.opt), offsets_shapes(), self.typ.offset_shape_repr())


# ======================================================================
#                       Arrow compatible bitmap
# ======================================================================

class Bitmap(object):
    """Validity bitmap (Arrow compatible)"""
    def __init__(self, lst):
        self.shape = len(lst)
        self.data = bytearray((self.shape + 7) // 8)

        for n, v in enumerate(lst):
            if v is not None:
                self.set_valid(n)

    def set_valid(self, n):
        if n >= self.shape:
            raise ValueError("out of range")
        self.data[n // 8] |= (1 << (n % 8))

    def is_valid(self, n):
        return int(bool(self.data[n // 8] & (1 << (n % 8))))

    def __repr__(self):
        lst = [self.is_valid(n) for n in range(self.shape)]
        return "Bitmap(%r)" % lst


# ======================================================================
#                      Model of the xnd array type
# ======================================================================

def choose_dim_type(typ, shapes, opt=False):
    """Choose a dimension type based on the list of 'shapes' that are present
       in a dimension. If all 'shapes' are equal, the more efficient FixedDim
       type suffices.
    """
    n = len(set(shapes))
    if n <= 1:
        return FixedDim(0 if n == 0 else shapes[0], opt=opt, typ=typ)
    return VarDim(shapes, opt=opt, typ=typ)

def data_shapes(lst):
    """Extract array data and dimension shapes from a nested list. The
       list may contain None for missing data or dimensions.

       >>> data_shapes([[0, 1], [2, 3, 4], [5, 6, 7, 8]])
       ([0, 1, 2, 3, 4, 5, 6, 7, 8], [[2, 3, 4], [3]])
                     ^                    ^       ^
                     |                    |       `--- ndim=2: single shape 3
                     |                    `-- ndim=1: shapes 2, 3, 4
                     `--- ndim=0: extracted array data
    """
    # These two are just function-internal tags used to classify data
    # extracted from the untyped list, so an error can be raised for
    # unbalanced trees.
    class Shape(list): pass
    class Data(list): pass

    acc = []
    def f(cls, depth, array):
        if depth == len(acc): # new dimension or data
            acc.append(cls([]))
        if not isinstance(acc[depth], cls):
            raise ValueError("unbalanced tree")
        if isinstance(array, list):
            acc[depth].append(len(array))
            cls = Data if is_value_list(array) else Shape
            for item in array:
                f(cls, depth+1, item)
        else: # None or int
            assert array is None or isinstance(array, int)
            acc[depth].append(array)
    f(Shape, 0, lst)
    acc.reverse()
    return acc[0], acc[1:]

def array_info(array_data, dim_shapes):
    """Construct the xnd array info from the actual array data and the
       list of dimension shapes. See the Array docstring for further
       information.
    """
    opt = None in array_data
    t = Int64(opt)

    bitmaps = [Bitmap(array_data)] if opt else [Bitmap([])]
    offsets = [[]]
    shapes = [[]]

    for lst in dim_shapes:
        opt = None in lst
        if opt:
            bitmaps.append(Bitmap(lst))
        else:
            bitmaps.append(Bitmap([]))

        lst = replace_none(lst)
        t = choose_dim_type(t, lst, opt)
        if isinstance(t, FixedDim):
            offsets.append([])
            shapes.append([])
        else: # VarDim
            offsets.append([0] + list(accumulate(lst)))
            shapes.append(lst)

    return t, offsets, shapes, bitmaps

class Array(object):
    """Python model of the XND array.

    An xnd array consists of the actual data array and three metadata
    arrays that are indexed by ndim. 

       1) offsets[[], [ndim1_data], [ndim2_data], ...]

          "offsets" are indices into the next dimension data. In the
          above example, ndim2_data is a list of indices into ndim1_data,
          and ndim1_data is a list of indices into the actual array data
          (ndim0_data).

          This holds if ndim1 and ndim2 are both variable dimensions.
          For fixed dimensions the ndimX_data list is empty.

          If only variable dimensions are present, this representation
          is compatible with the Arrow list type.

          If only fixed dimensions are present, all lists are empty
          and the offsets are unused. The array is de facto a NumPy
          ndarray and can be treated as such.

          If mixed fixed/var dimensions are present, the fixed dimensions
          are optimized by ommitting the (potentially very long) index
          arrays.  This is not Arrow compatible, but in the real libxnd
          variable dimensions can be forced by ommitting fixed dimensions
          in the type, so there is effectively always a way to force Arrow.

      2) shapes[[], [ndim1_shapes], [ndim2_shapes], ...]

         "shapes" contain all shapes in a variable dimension. This
         is redundant for unsliced Arrow arrays, since the shapes
         are also encoded as the difference between two offsets.

         For sliced arrays, however, separate shapes are needed for
         full generality.

      3) bitmaps[[ndim0_bitmap], [ndim1_bitmap], [ndim2_bitmap], ...]

         Bitmaps are Arrow compatible. 'ndim0_bitmap' tracks which values
         in the array data are missing.  'ndimX_bitmap' (X > 0) tracks
         which dimensions are missing.

         If ndimX does not contain missing values, the bitmaps are empty.

    """

    def __init__(self, lst=None, typ=None, data=None, offsets=None,
                 shapes=None, bitmaps=None):
        """
        Initialize the array either

          a) From a multi-dimensional nested list that contains integers
             or 'None' (missing values) in the last dimension. The list
             must be a balanced tree, except that missing dimensions are
             permitted.

                >>> Array(lst=[[1, None], None, [3, 4, 5]])
                Array(
                  type="(fixed(start=0, step=1, shape=3) * ?var(start=0, step=1) * ?int64",
                  type_with_shapes="3 * ?var(2,0,3) * ?int64",
                  type_with_offsets_shapes="3 * ?var(0:2,2:0,2:3,5:0) * ?int64",
                  data=[1, None, 3, 4, 5],
                  offsets=[[], [0, 2, 2, 5], []],
                  shapes=[[], [2, 0, 3], []],
                  bitmaps=[Bitmap([1, 0, 1, 1, 1]), Bitmap([1, 0, 1]), Bitmap([])],
                )


          b) From explicit array info (all parameters except 'lst').
             In this case, the array info is not checked and assumed
             to be consistent.

        """

        if lst:
            if not (typ is data is offsets is shapes is bitmaps is None):
                raise TypeError("the 'lst' argument precludes other arguments")

            data, shapes = data_shapes(lst)
            self.data = data
            self.typ, self.offsets, self.shapes, self.bitmaps = array_info(data, shapes)

            assert len(self.offsets) == len(self.shapes) == len(self.bitmaps) == self.typ.ndim + 1

        else:
            if lst is not None or None in (typ, data, offsets, shapes, bitmaps):
                raise TypeError("invalid argument combination")

            self.typ = typ
            self.data = data
            self.offsets = offsets
            self.shapes = shapes
            self.bitmaps = bitmaps

    def tolist(self):
        """Return the xnd array as an untyped list."""

        def f(t, linear_index):
            if t.opt and not self.bitmaps[t.ndim].is_valid(linear_index):
                return None

            if isinstance(t, FixedDim):
                linear_index += t.start
                shape = t.shape
                return [f(t.typ, linear_index + k*t.step) for k in range(shape)]

            elif isinstance(t, VarDim):
                shape = self.shapes[t.ndim][linear_index]
                linear_index = self.offsets[t.ndim][linear_index]
                linear_index += t.start
                return [f(t.typ, linear_index + k*t.step) for k in range(shape)]

            elif isinstance(t, Int64):
                assert t.ndim == 0
                return self.data[linear_index]

            else:
                raise TypeError("unexpected type tag")

        return f(self.typ, 0)

    def _getitem(self, indices):
        """Return a multi-dimensional slice of the xnd array. Multi-dimensional
           indices are not yet supported."""

        t = typ = copy(self.typ)

        data = self.data
        offsets = copy(self.offsets)
        shapes = copy(self.shapes)
        bitmaps = self.bitmaps

        dimensions = []
        while (t.ndim > 0): 
            dimensions.append(t)
            t = t.typ

        linear_index = 0
        for i, s in enumerate(indices):
            t = dimensions[i]
            if isinstance(t, FixedDim):
                start, stop, step, shape = slice_indices(s, t.shape)
                linear_index += start
                t.start = linear_index
                t.shape = shape
                t.step *= step
            elif isinstance(t, VarDim):
                _offsets = copy(t.offsets)
                _shapes = [0] * t.nshapes
                for k, shape in enumerate(t.shapes):
                    start, stop, step, shape = slice_indices(s, shape)
                    _offsets[k] = _offsets[k] + start
                    _shapes[k] = shape
                offsets[t.ndim] = _offsets
                shapes[t.ndim] = _shapes
                t.offsets = _offsets
                t.shapes = _shapes
                t.step = step
            else:
                raise RuntimeError("unexpected type")

        return Array(typ=typ, data=data, offsets=offsets, shapes=shapes, bitmaps=bitmaps)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if not all(isinstance(x, slice) for x in indices):
            raise TypeError("index must be a slice or a tuple of slices")

        return self._getitem(indices)

    def __repr__(self):
        return """\
Array(
  type="%s",
  type_with_shapes="%s",
  type_with_offsets_shapes="%s",
  data=%s,
  offsets=%s,
  shapes=%s,
  bitmaps=%s,
)""" % (self.typ, self.typ.shape_repr(), self.typ.offset_shape_repr(),
        self.data, self.offsets, self.shapes, self.bitmaps)


# ======================================================================
#                 For debugging: simple pythonic "array"
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
#                                 Tests
# ======================================================================

class TestError(Exception):
    pass

def test():
    lst = [[[0, 1], [2, 3]], [[4, 5, None], None, [7]], None, [[8, 9]]]
    x = SimpleArray(lst)
    y = Array(lst)
    for slices in rslices_ndim(3, [3,3,3], 100000):
        xerr = None
        try:
            xres = x[slices]
        except Exception as e:
            xerr = e.__class__

        yerr = None
        try:
            yres = y[slices].tolist()
        except Exception as e:
            yerr = e.__class__

        if (xerr or yerr) and yerr is not xerr:
            raise TestError("\nslice: %s\nxerr: %s\nyerr: %s\n%d  %d\n" % (slices, xerr, yerr, id(xerr), id(yerr)))
        if yres != xres:
            raise TestError("%s\n%s\n%s\n" % (slices,  y[slices].tolist(), x[slices]))


if __name__ == '__main__':
    lst = [[[0, 1], [2, 3]], [[4, 5, None], None, [7]], None, [[8, 9]]]
    a = Array(lst)

    print("Input: %s\n" % lst)
    print("%s\n" % a)
    print("Roundtrip: %s\n" % a.tolist())

    b = a[1:2, ::2, ::-1]
    print("%s\n" % b)
    print("slice: %s\n" % b.tolist())

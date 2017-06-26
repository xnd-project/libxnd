from itertools import accumulate, product
from random import randrange, seed
from copy import copy, deepcopy


########################################################################
#                   Python model of the xnd array type                 #
########################################################################

MAX_DIM = 128

def slice_spec(s, shape):
    indices = s.indices(shape)
    length = len(range(*indices))
    return indices + (length,)

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
    def __init__(self, *, opt=False):
        self.opt = opt
        self.target = 0
        self.ndim = 0
        # concrete
        self.size = 1 # 8 in ndtypes (steps vs. strides issue)
        self.index = 0

    def __repr__(self):
        return "%sint64" % opt_repr(self.opt)

    def copy(self):
        return Int64(opt=self.opt)

    concrete_repr = __repr__

class FixedDim(Datashape):
    def __init__(self, *, opt=False, shape=None, step=None, typ=None,
                 index=None):
        # abstract
        self.opt = opt
        self.ndim = typ.ndim + 1
        self.shape = shape
        self.target = typ.ndim if isinstance(typ, VarDim) else typ.target
        self.typ = typ
        # concrete
        if step is None:
            self.step = 1 if isinstance(typ, VarDim) else typ.size
        else:
            self.step = step
        self.size = shape * self.step
        self.index = self.ndim if index is None else index

    def copy(self):
        typ = self.typ.copy()
        return FixedDim(opt=self.opt, shape=self.shape, step=self.step, typ=typ)

    def __repr__(self):
        return "%s%d * %r" % (opt_repr(self.opt), self.shape, self.typ)

    def concrete_repr(self):
        return "%sfixed(shape=%d, step=%d, ndim=%d, index=%d, target=%d) * %s" % \
            (opt_repr(self.opt), self.shape, self.step, self.ndim, self.target,
             self.index, self.typ.concrete_repr())


class VarDim(Datashape):
    def __init__(self, *, opt=False, step=None, typ=None, index=None):
        # abstract
        self.opt = opt
        self.ndim = typ.ndim + 1
        self.target = typ.ndim if isinstance(typ, VarDim) else typ.target
        self.typ = typ
        # concrete
        if step is None:
            self.step = 1 if isinstance(typ, VarDim) else typ.size
        else:
            self.step = step
        self.size = 1
        self.index = self.ndim if index is None else index

    def copy(self):
        typ = self.typ.copy()
        return VarDim(opt=self.opt, step=self.step, typ=typ)

    def __repr__(self):
        return "%svar * %r" % (opt_repr(self.opt), self.typ)

    def concrete_repr(self):
        return "%svar(ndim=%d, index=%s, target=%d, step=%d, size=%d) * %s" % \
            (opt_repr(self.opt), self.ndim, self.index, self.target, self.step,
             self.size, self.typ.concrete_repr())

def dims_as_list(t):
    dimensions = []
    while (t.ndim > 0): 
        dimensions.append(t)
        t = t.typ
    return dimensions

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

def choose_dim_type(*, opt=False, typ=None, shapes=None):
    """Choose a dimension type based on the list of 'shapes' that are present
       in a dimension. If all 'shapes' are equal, the more efficient FixedDim
       type suffices.
    """
    n = len(set(shapes))
    if n <= 1:
        return FixedDim(opt=opt, typ=typ, shape=0 if n == 0 else shapes[0])
    return VarDim(opt=opt, typ=typ)

def data_shapes(tree):
    """Extract array data and dimension shapes from a nested list. The
       list may contain None for missing data or dimensions.

       >>> data_shapes([[0, 1], [2, 3, 4], [5, 6, 7, 8]])
       ([0, 1, 2, 3, 4, 5, 6, 7, 8], [[2, 3, 4], [3]])
                     ^                    ^       ^
                     |                    |       `--- ndim=2: single shape 3
                     |                    `-- ndim=1: shapes 2, 3, 4
                     `--- ndim=0: extracted array data
    """
    # Classify data and shapes extracted from the untyped list, so an error
    # can be raised for unbalanced trees.
    class Unknown(list): pass
    class Data(list): pass
    class Shape(list): pass

    acc = [Unknown() for _ in range(MAX_DIM+1)]
    max_level = 0

    def search(level, a):
        nonlocal max_level

        if level >= MAX_DIM:
            raise ValueError("too many dimensions")

        current = acc[level]
        if isinstance(current, Unknown):
            if a is None:
                current.append(a)
            elif isinstance(a, int):
                current.append(a)
                acc[level] = Data(current)
            elif isinstance(a, list):
                current.append(len(a))
                acc[level] = Shape(current)
                max_level = level + 1
                for item in a:
                    search(level+1, item)
            else:
                raise TypeError("unsupported type '%s'", type(a))

        elif isinstance(current, Data):
            if a is None:
                current.append(a)
            elif isinstance(a, int):
                current.append(a)
            else:
                raise TypeError(
                    "unbalanced tree, expected int or None, got '%s'" % type(a))

        elif isinstance(current, Shape):
            if a is None:
                current.append(a)
            elif isinstance(current, list):
                current.append(len(a))
                max_level = level + 1
                for item in a:
                    search(level+1, item)
            else:
                raise TypeError(
                    "unbalanced tree, expected list or None, got '%s'" % type(a))
        else:
            raise TypeError(
                "expected Unknown, Data or Shape, got '%s'" % type(a))

    search(0, tree)

    lst = acc[max_level]
    if isinstance(lst, Unknown): # No data found
        acc[max_level] = Data(lst)

    acc.reverse()
    data_shapes = [v for v in acc if not isinstance(v, Unknown)]

    return data_shapes[0], data_shapes[1:]

def typeof(lst):
    """Infer the type of a nested list."""
    data, shapes = data_shapes(lst)

    opt = None in data
    t = Int64(opt=opt)

    for l in shapes:
        opt = None in l
        lst = replace_none(l)
        t = choose_dim_type(opt=opt, typ=t, shapes=l)

    return t

def array_info(array_data, dim_shapes):
    """Construct the xnd array info from the actual array data and the
       list of dimension shapes. See the Array docstring for further
       information.
    """
    opt = None in array_data
    t = Int64(opt=opt)

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
        t = choose_dim_type(opt=opt, typ=t, shapes=lst)
        if isinstance(t, FixedDim):
            offsets.append([])
            shapes.append([])
        else: # VarDim
            offsets.append([0] + list(accumulate([x * t.step for x in lst])))
            shapes.append(lst)

    return t, offsets, shapes, bitmaps

class Array(object):
    """Python model of the XND array.

    An xnd array consists of the actual data array and four metadata
    arrays that are indexed by ndim. The data array and bitmaps are
    shared.  Offsets, suboffsets and shapes are per-view copies.

       1) bitmaps[[ndim0_bitmap], [ndim1_bitmap], [ndim2_bitmap], ...]

          Bitmaps are Arrow compatible. 'ndim0_bitmap' tracks which values
          in the array data are missing.  'ndimX_bitmap' (X > 0) tracks
          which dimensions are missing.

          If ndimX does not contain missing values, the bitmaps are empty.

       2) offsets[[], [ndim1_data], [ndim2_data], ...]

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

       3) suboffsets[ndim0_suboffset, ndim1_suboffset, ...]

          Suboffsets are used here to store a linear index component that
          is always present as a result of slicing.  ndim0_suboffset is the
          suboffset of the array data. For fixed dimensions the suboffset
          is always 0.  For variable dimensions the suboffset targets the
          offset arrays.

          In NumPy there is only one data array (the actual array data), so
          suboffsets can be replaced by moving around the start pointer
          on the array data.

          Variable dimensions introduce additional data (index/offset)
          arrays, so some form of suboffsets needs to be stored in the
          C-xnd array as well.

       4) shapes[[], [ndim1_shapes], [ndim2_shapes], ...]

          "shapes" contain all shapes in a variable dimension. This
          is redundant for unsliced Arrow arrays, since the shapes
          are also encoded as the difference between two offsets.

          For sliced arrays, however, separate shapes are needed for
          full generality.


    """

    def __init__(self, lst=None, *, typ=None, data=None, bitmaps=None,
                 offsets=None, suboffsets=None, shapes=None):
        """
        Initialize the array either

          a) From a multi-dimensional nested list that contains integers
             or 'None' (missing values) in the last dimension. The list
             must be a balanced tree, except that missing dimensions are
             permitted.

                >>> Array(lst=[[1, None], None, [3, 4, 5]])
                Array(
                  type="3 * ?var * ?int64",
                  type_with_meta="fixed(shape=3, step=1, ndim=2, target=1) * ?var(ndim=1, target=0, step=1, size=1) * ?int64",
                  data=[1, None, 3, 4, 5],
                  bitmaps=[Bitmap([1, 0, 1, 1, 1]), Bitmap([1, 0, 1]), Bitmap([])],
                  offsets=[[], [0, 2, 2, 5], []],
                  suboffsets=[0, 0, 0],
                  shapes=[[], [2, 0, 3], []],
                )


          b) From explicit array info (all parameters except 'lst').
             In this case, the array info is not checked and assumed
             to be consistent.

        """

        if lst is not None:
            if not (typ is data is offsets is suboffsets is shapes is
                    bitmaps is None):
                raise TypeError("the 'lst' argument precludes other arguments")

            data, shapes = data_shapes(lst)
            self.data = data
            self.typ, self.offsets, self.shapes, self.bitmaps = array_info(data, shapes)
            assert len(self.offsets) == len(self.shapes) == len(self.bitmaps) == self.typ.ndim + 1

            self.suboffsets = [0] * (self.typ.ndim + 1)

        else:
            if lst is not None or None in (typ, data, offsets, suboffsets,
                                           shapes, bitmaps):
                raise TypeError("invalid argument combination")

            # view-specific
            self.typ = typ

            # shared
            self.data = data
            self.bitmaps = bitmaps

            # view-specific
            self.offsets = offsets
            self.suboffsets = suboffsets
            self.shapes = shapes

    def tolist(self):
        """Return the xnd array as an untyped list."""

        def f(t, linear_index):
            if isinstance(t, FixedDim):
                if t.opt and not self.bitmaps[t.index].is_valid(linear_index):
                    return None
                shape = t.shape
                ret = []
                for k in range(shape):
                    x = f(t.typ, linear_index + k*t.step)
                    ret.append(x)
                return ret

            elif isinstance(t, VarDim):
                linear_index += self.suboffsets[t.index]
                if t.opt and not self.bitmaps[t.index].is_valid(linear_index):
                    return None
                shape = self.shapes[t.index][linear_index]
                linear_index = self.offsets[t.index][linear_index]
                ret = []
                for k in range(shape):
                    x = f(t.typ, linear_index + k*t.step)
                    ret.append(x)
                return ret

            elif isinstance(t, Int64):
                linear_index += self.suboffsets[t.index]
                return self.data[linear_index]

            else:
                raise TypeError("unexpected type tag")

        return f(self.typ, 0)

    def getitem(self, indices):
        """Return multi-dimensional subarrays or slices of an xnd array."""

        data = self.data
        bitmaps = self.bitmaps

        offsets = deepcopy(self.offsets)
        suboffsets = deepcopy(self.suboffsets)
        shapes = deepcopy(self.shapes)

        def dispatch(t, indices):
            if not indices:
                return t.copy()

            i, rest = indices[0], indices[1:]
            if isinstance(i, int):
                return subarray(t, i, rest)
            elif isinstance(i, slice):
                return subslice(t, i, rest)
            else:
                raise TypeError("indices must be int or slice, got '%s'" % i)

        def subarray(t, i, rest):
            if t.opt and not bitmaps[t.index].is_valid(suboffsets[t.index]):
                raise TypeError("missing value or dimension is not indexable")

            if isinstance(t, FixedDim):
                suboffsets[t.target] += i * t.step
                shape = t.shape

            elif isinstance(t, VarDim):
                linear_index = suboffsets[t.index]
                offset = offsets[t.index][linear_index]
                shape = shapes[t.index][linear_index]
                suboffsets[t.target] = offset + i * t.step

            else:
                raise TypeError("type not indexable")

            if i < 0 or i >= shape:
                raise IndexError("index out of range")

            return dispatch(t.typ, rest)

        def subslice(t, s, rest):
            if t.opt and not bitmaps[t.index].is_valid(suboffsets[t.index]):
                raise TypeError("missing value or dimension is not sliceable")

            if isinstance(t, FixedDim):
                start, stop, step, shape = slice_spec(s, t.shape)
                suboffsets[t.target] += t.step * start
                typ = dispatch(t.typ, rest)
                return FixedDim(opt=t.opt, shape=shape, step=t.step*step, typ=typ)

            elif isinstance(t, VarDim):
                for k, shape in enumerate(self.shapes[t.index]):
                    start, stop, step, shape = slice_spec(s, shape)
                    offsets[t.index][k] += start * t.step
                    shapes[t.index][k] = shape
                typ = dispatch(t.typ, rest)
                return VarDim(opt=t.opt, step=t.step*step, index=t.index, typ=typ)
            else:
                raise TypeError("unexpected type")

        typ = dispatch(self.typ, indices)

        return Array(typ=typ, data=data, bitmaps=bitmaps, offsets=offsets,
                     suboffsets=suboffsets, shapes=shapes)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) > self.typ.ndim:
            raise IndexError("too many indices")
        if all(isinstance(x, (int, slice)) for x in indices):
            return self.getitem(indices)
        else:
            raise NotImplemented("mixed slices and indices not yet implemented")

    def __repr__(self):
        return """\
Array(
  type="%s",
  type_with_meta="%s",
  data=%s,
  bitmaps=%s,
  offsets=%s,
  suboffsets=%s,
  shapes=%s,
)""" % (self.typ, self.typ.concrete_repr(), self.data, self.bitmaps,
        self.offsets, self.suboffsets, self.shapes)


if __name__ == '__main__':
    lst = [[[0, 1], [2, 3]], [[4, 5, None], None, [7]], None, [[8, 9]]]
    a = Array(lst)

    print("Input: %s\n" % lst)
    print("%s\n" % a)
    print("Roundtrip: %s\n" % a.tolist())

    b = a[1:2, ::2, ::-1]
    print("%s\n" % b)
    print("slice: %s\n" % b.tolist())

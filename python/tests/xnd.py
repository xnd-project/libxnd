from itertools import accumulate, product
from random import randrange, seed
from copy import copy, deepcopy


########################################################################
#                   Python model of the xnd array type                 #
########################################################################

def slice_indices(s, shape):
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
        self.size = 1 # 8 in ndtypes (steps vs. strides issue)

    def __repr__(self):
        return "%sint64" % opt_repr(self.opt)

    def copy(self):
        return Int64(opt=self.opt)

    concrete_repr = __repr__

class FixedDim(Datashape):
    def __init__(self, *, opt=False, shape=None, step=None, typ=None):
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

    def copy(self):
        typ = self.typ.copy()
        return FixedDim(opt=self.opt, shape=self.shape, step=self.step, typ=typ)

    def __repr__(self):
        return "%s%d * %r" % (opt_repr(self.opt), self.shape, self.typ)

    def concrete_repr(self):
        return "%sfixed(shape=%d, step=%d, ndim=%d, target=%d) * %s" % \
            (opt_repr(self.opt), self.shape, self.step, self.ndim, self.target,
             self.typ.concrete_repr())


class VarDim(Datashape):
    def __init__(self, *, opt=False, step=None, typ=None):
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

    def copy(self):
        typ = self.typ.copy()
        return VarDim(opt=self.opt, step=self.step, typ=typ)

    def __repr__(self):
        return "%svar * %r" % (opt_repr(self.opt), self.typ)

    def concrete_repr(self):
        return "%svar(ndim=%d, target=%d, step=%d, size=%d) * %s" % \
            (opt_repr(self.opt), self.ndim, self.target, self.step, self.size, self.typ.concrete_repr())

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
    if not lst:
        return [[], [[0]]]

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

      2) suboffsets[ndim0_suboffset, ndim1_suboffset, ...]

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

      3) shapes[[], [ndim1_shapes], [ndim2_shapes], ...]

         "shapes" contain all shapes in a variable dimension. This
         is redundant for unsliced Arrow arrays, since the shapes
         are also encoded as the difference between two offsets.

         For sliced arrays, however, separate shapes are needed for
         full generality.

      4) bitmaps[[ndim0_bitmap], [ndim1_bitmap], [ndim2_bitmap], ...]

         Bitmaps are Arrow compatible. 'ndim0_bitmap' tracks which values
         in the array data are missing.  'ndimX_bitmap' (X > 0) tracks
         which dimensions are missing.

         If ndimX does not contain missing values, the bitmaps are empty.

    """

    def __init__(self, lst=None, *, typ=None, data=None, offsets=None,
                 suboffsets=None, shapes=None, bitmaps=None):
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
                  offsets=[[], [0, 2, 2, 5], []],
                  suboffsets=[0, 0, 0],
                  shapes=[[], [2, 0, 3], []],
                  bitmaps=[Bitmap([1, 0, 1, 1, 1]), Bitmap([1, 0, 1]), Bitmap([])],
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

            self.typ = typ
            self.data = data
            self.offsets = offsets
            self.suboffsets = suboffsets
            self.shapes = shapes
            self.bitmaps = bitmaps

    def tolist(self):
        """Return the xnd array as an untyped list."""

        def f(t, linear_index):
            if isinstance(t, FixedDim):
                if t.opt and not self.bitmaps[t.ndim].is_valid(linear_index):
                    return None
                shape = t.shape
                ret = []
                for k in range(shape):
                    x = f(t.typ, linear_index + k*t.step)
                    ret.append(x)
                return ret

            elif isinstance(t, VarDim):
                linear_index += self.suboffsets[t.ndim]
                if t.opt and not self.bitmaps[t.ndim].is_valid(linear_index):
                    return None
                shape = self.shapes[t.ndim][linear_index]
                linear_index = self.offsets[t.ndim][linear_index]
                ret = []
                for k in range(shape):
                    x = f(t.typ, linear_index + k*t.step)
                    ret.append(x)
                return ret

            elif isinstance(t, Int64):
                linear_index += self.suboffsets[t.ndim]
                return self.data[linear_index]

            else:
                raise TypeError("unexpected type tag")

        return f(self.typ, 0)

    def subarray(self, indices):
        """Return a subarray of the xnd array. Actual values at ndim=0 are a
           special case of subarrays."""

        data = self.data
        bitmaps = self.bitmaps

        offsets = self.offsets
        suboffsets = self.suboffsets[:]
        shapes = self.shapes

        def f(indices):
            if not indices:
                return self

            i, indices = indices[0], indices[1:]
            t = self.typ

            if t.opt and not self.bitmaps[t.ndim].is_valid(suboffsets[t.ndim]):
                raise TypeError("missing value or dimension is not indexable")

            if isinstance(t, FixedDim):
                suboffsets[t.target] += i * t.step
                shape = t.shape

            elif isinstance(t, VarDim):
                linear_index = suboffsets[t.ndim]
                offset = self.offsets[t.ndim][linear_index]
                shape = self.shapes[t.ndim][linear_index]
                suboffsets[t.target] = offset + i * t.step

            else:
                raise TypeError("type not indexable")

            if i < 0 or i >= shape:
                raise IndexError("index out of range")

            a = Array(typ=t.typ, data=data, offsets=offsets, suboffsets=suboffsets,
                      shapes=shapes, bitmaps=bitmaps)

            return a.subarray(indices)

        return f(indices)

    def getslice(self, indices):
        """Return a multi-dimensional slice of the xnd array."""

        data = self.data
        bitmaps = self.bitmaps

        offsets = deepcopy(self.offsets)
        suboffsets = deepcopy(self.suboffsets)
        shapes = deepcopy(self.shapes)

        def f(t, indices):
            if not indices:
                return t.copy()

            i, indices = indices[0], indices[1:]

            if t.opt and not self.bitmaps[t.ndim].is_valid(suboffsets[t.ndim]):
                raise TypeError("missing value or dimension is not sliceable")

            if isinstance(t, FixedDim):
                start, stop, step, shape = slice_indices(i, t.shape)
                suboffsets[t.target] += t.step * start
                typ = f(t.typ, indices)
                return FixedDim(opt=t.opt, shape=shape, step=t.step*step, typ=typ)

            elif isinstance(t, VarDim):
                for k, shape in enumerate(self.shapes[t.ndim]):
                    start, stop, step, shape = slice_indices(i, shape)
                    offsets[t.ndim][k] += start * t.step
                    shapes[t.ndim][k] = shape
                typ = f(t.typ, indices)
                return VarDim(opt=t.opt, step=t.step*step, typ=typ)
            else:
                raise TypeError("unexpected type")

        typ = f(self.typ, indices)

        return Array(typ=typ, data=data, offsets=offsets,
                     suboffsets=suboffsets, shapes=shapes,
                     bitmaps=bitmaps)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if all(isinstance(x, slice) for x in indices):
            return self.getslice(indices)
        elif all(isinstance(x, int) for x in indices):
            return self.subarray(indices)
        else:
            raise NotImplemented("mixed slices and indices not yet implemented")

    def __repr__(self):
        return """\
Array(
  type="%s",
  type_with_meta="%s",
  data=%s,
  offsets=%s,
  suboffsets=%s,
  shapes=%s,
  bitmaps=%s,
)""" % (self.typ, self.typ.concrete_repr(), self.data, self.offsets,
        self.suboffsets, self.shapes, self.bitmaps)


if __name__ == '__main__':
    lst = [[[0, 1], [2, 3]], [[4, 5, None], None, [7]], None, [[8, 9]]]
    a = Array(lst)

    print("Input: %s\n" % lst)
    print("%s\n" % a)
    print("Roundtrip: %s\n" % a.tolist())

    b = a[1:2, ::2, ::-1]
    print("%s\n" % b)
    print("slice: %s\n" % b.tolist())

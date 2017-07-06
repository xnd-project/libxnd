from itertools import accumulate, product
from random import randrange, seed
from copy import copy, deepcopy


########################################################################
#                   Python model of the xnd array type                 #
########################################################################

MAX_DIM = 128

def adjust_indices_shape(s, shape):
    """Calculate resulting start, stop, step, shape from a slice and a shape."""
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
    def __init__(self, *, opt=False, shape=None, step=None, index=None, target=None, typ=None):
        # abstract
        self.opt = opt
        self.ndim = typ.ndim + 1
        self.shape = shape
        if target is not None:
            self.target = target
        else:
            self.target = typ.index if isinstance(typ, (VarDim, Ref)) else typ.target
        self.typ = typ
        # concrete
        if step is None:
            self.step = 1 if isinstance(typ, (VarDim, Ref)) else typ.size
        else:
            self.step = step
        self.size = shape * self.step
        self.index = self.ndim if index is None else index

    def copy(self):
        typ = self.typ.copy()
        return FixedDim(opt=self.opt, shape=self.shape, step=self.step, index=self.index, target=self.target, typ=typ)

    def __repr__(self):
        return "%s%d * %r" % (opt_repr(self.opt), self.shape, self.typ)

    def concrete_repr(self):
        return "%sfixed(shape=%d, step=%d, ndim=%d, index=%d, target=%d) * %s" % \
            (opt_repr(self.opt), self.shape, self.step, self.ndim, self.index,
             self.target, self.typ.concrete_repr())

class VarDim(Datashape):
    def __init__(self, *, opt=False, nshapes=None, step=None, index=None, target=None, typ=None):
        # abstract
        self.opt = opt
        self.ndim = typ.ndim + 1
        if target is not None:
            self.target = target
        else:
            self.target = typ.index if isinstance(typ, (VarDim, Ref)) else typ.target
        self.typ = typ
        # concrete
        self.nshapes = nshapes
        if step is None:
            self.step = 1 if isinstance(typ, VarDim) else typ.size
        else:
            self.step = step
        self.size = 1
        self.index = self.ndim if index is None else index

    def copy(self):
        typ = self.typ.copy()
        return VarDim(opt=self.opt, step=self.step, nshapes=self.nshapes,
                      index=self.index, target=self.target, typ=self.typ)

    def __repr__(self):
        return "%svar * %r" % (opt_repr(self.opt), self.typ)

    def concrete_repr(self):
        return "%svar(ndim=%d, nshapes=%d, index=%s, target=%d, step=%d, size=%d) * %s" % \
            (opt_repr(self.opt), self.ndim, self.nshapes, self.index, self.target, self.step,
             self.size, self.typ.concrete_repr())

class Ref(Datashape):

    def __init__(self, ndim=None, index=None, target=None, typ=None, *, opt=False, step=None):
        # abstract
        self.opt = typ.opt
        self.ndim = 0 if ndim is None else ndim
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
        return Ref(opt=self.opt, step=self.step, index=self.index, ndim=self.ndim, typ=self.typ)

    def __repr__(self):
        return "%sref(%r)" % (opt_repr(self.opt), self.typ)

    def concrete_repr(self):
        return "%sref(ndim=%d, index=%d, target=%d, step=%s, typ=%s)" % \
            (opt_repr(self.opt), self.ndim, self.index, self.target,
             self.step, self.typ.concrete_repr())

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
    return VarDim(opt=opt, nshapes=len(shapes), typ=typ)

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
    acc = [[] for _ in range(MAX_DIM+1)]
    min_level = MAX_DIM + 1
    max_level = 0

    def search(level, a):
        nonlocal min_level, max_level

        if level > MAX_DIM:
            raise ValueError("too many dimensions")

        current = acc[level]
        if a is None:
            current.append(a)
        elif isinstance(a, int):
            acc[max_level].append(a)
            min_level = min(level, min_level)
        elif isinstance(a, list):
            current.append(len(a))
            next_level = level + 1
            max_level = max(next_level, max_level)
            if not a:
                min_level = min(next_level, min_level)
            else:
                for item in a:
                    search(level+1, item)
        else:
            raise TypeError("unsupported type '%s'", type(a))

    search(max_level, tree)
    if acc[max_level] and all(x is None for x in acc[max_level]):
        pass # min_level is not set in this special case, hence the check.
    elif min_level != max_level:
        raise ValueError("unbalanced tree: min_level: %d max_level: %d list: %s" %
                         (min_level, max_level, tree))

    data = acc[max_level]
    shapes = reversed(acc[0:max_level])

    return data, shapes

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
    shared.  Offsets, bases and shapes are per-view copies.

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

       3) bases[ndim0_base, ndim1_base, ...]

          "bases" are used to store a linear index component that is always
          present as a result of indexing or slicing.  ndim0_base is the base
          of the array data.  For fixed dimensions the base is always 0.  For
          variable dimensions the base targets the offset arrays.

          In NumPy there is only one data array (the actual array data), so
          bases can be replaced by moving around the start pointer
          on the array data (in fact, adjusting bases[0] is equivalent to
          moving around the 'buf' pointer in PEP-3118).

          Variable dimensions introduce additional data (index/offset)
          arrays, so some form of bases (either in pointer or index form)
          need to be stored in the C-xnd array as well.

       4) shapes[[], [ndim1_shapes], [ndim2_shapes], ...]

          "shapes" contain all shapes in a variable dimension. This
          is redundant for unsliced Arrow arrays, since the shapes
          are also encoded as the difference between two offsets.

          For sliced arrays, however, separate shapes are needed for
          full generality.


    """

    def __init__(self, lst=None, *, typ=None, data=None, bitmaps=None,
                 bases=None, offsets=None, shapes=None):
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
                  bases=[0, 0, 0],
                  shapes=[[], [2, 0, 3], []],
                )


          b) From explicit array info (all parameters except 'lst').
             In this case, the array info is not checked and assumed
             to be consistent.

        """

        if lst is not None:
            if not (typ is data is bitmaps is bases is offsets is shapes is None):
                raise TypeError("the 'lst' argument precludes other arguments")

            data, shapes = data_shapes(lst)
            self.data = data
            self.typ, self.offsets, self.shapes, self.bitmaps = array_info(data, shapes)
            assert len(self.offsets) == len(self.shapes) == len(self.bitmaps) == self.typ.ndim + 1

            self.bases = [0] * (self.typ.ndim + 1)

        else:
            if lst is not None or None in (typ, data, bitmaps, bases, offsets, shapes):
                raise TypeError("invalid argument combination: %s" % ((typ,
                                data, bitmaps, bases, offsets, shapes),))

            # view-specific
            self.typ = typ

            # shared
            self.data = data
            self.bitmaps = bitmaps

            # view-specific
            self.offsets = offsets
            self.bases = bases
            self.shapes = shapes

    def tolist(self):
        """Represent the xnd array as an untyped list."""

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
                # n: index into the offset array
                n = self.bases[t.index] + linear_index
                # check if the nth list is a missing dimension
                if t.opt and not self.bitmaps[t.index].is_valid(n):
                    return None
                # shape of the nth list in the offset array
                shape = self.shapes[t.index][n]
                # get the linear index into the next target array
                linear_index = self.offsets[t.index][n]
                ret = []
                for k in range(shape):
                    x = f(t.typ, linear_index + k*t.step)
                    ret.append(x)
                return ret

            elif isinstance(t, Ref):
                # n: index into the offset array
                n = self.bases[t.index] + linear_index
                # get the linear index into the next target array
                linear_index = self.offsets[t.index][n]
                return f(t.typ, linear_index)

            elif isinstance(t, Int64):
                # n: index into the data array
                n = self.bases[t.index] + linear_index
                return self.data[n]

            else:
                raise TypeError("unexpected type")

        return f(self.typ, 0)

    def getitem(self, indices):
        """Return multi-dimensional subarrays or slices of an xnd array."""

        data = self.data
        bitmaps = self.bitmaps

        bases = deepcopy(self.bases)
        offsets = deepcopy(self.offsets)
        shapes = deepcopy(self.shapes)

        have_slice = False

        def dispatch(t, indices):
            nonlocal have_slice

            if not indices:
                return t.copy()

            i, rest = indices[0], indices[1:]
            if isinstance(i, int):
                if have_slice:
                    return subindex(t, i, rest)
                else:
                    return subarray(t, i, rest)
            elif isinstance(i, slice):
                have_slice = True
                return subslice(t, i, rest)
            else:
                raise TypeError("indices must be int or slice, got '%s'" % i)

        def subarray(t, i, rest):
            """Return a pure subarray (no slices) of the xnd array."""
            assert not have_slice and isinstance(i, int)

            if t.opt and not bitmaps[t.index].is_valid(bases[t.index]):
                raise TypeError("missing value or dimension is not indexable")

            if isinstance(t, FixedDim):
                shape = t.shape
                # accumulate the linear index in the target array
                bases[t.target] += i * t.step

            ### This works if the VarDim is eliminated entirely.
            elif isinstance(t, VarDim):
                # n: index into the offset array
                n = bases[t.index]
                # shape of the nth list in the offset array
                shape = shapes[t.index][n]
                # get the linear index into the next target array
                linear_index = offsets[t.index][n] + i * t.step
                bases[t.target] += linear_index

            else:
                raise TypeError("type not indexable")

            if i < 0 or i >= shape:
                raise IndexError("index out of range (index=%d, shape=%d)" % (i, shape))

            return dispatch(t.typ, rest)

        def subindex(t, i, rest):
            if t.opt and not bitmaps[t.index].is_valid(bases[t.index]):
                raise TypeError("missing value or dimension is not indexable")

            if isinstance(t, FixedDim):
                shape = t.shape
                bases[t.target] += i * t.step
                if i < 0 or i >= shape:
                    raise IndexError("index out of range")

                return dispatch(t.typ, rest)

            elif isinstance(t, VarDim):
                if isinstance(t.typ, FixedDim):
                    step = t.typ.step
                    shape = t.typ.shape

                typ = dispatch(t.typ, rest)

                if isinstance(typ, FixedDim):
                    for n in range(t.nshapes):
                        offsets[t.index][n] = offsets[t.index][n] + shape * i * step
                        shapes[t.index][n] = typ.shape
                    return VarDim(nshapes=t.nshapes, step=typ.step, index=t.index, typ=typ.typ)

                elif isinstance(typ, VarDim):
                    for n in range(t.nshapes):
                        if i < 0 or i >= shapes[t.index][n]:
                            raise IndexError(
                                "out of range: subindices currently must be in "
                                "range for all var shapes due to technical "
                                "difficulties");
                        linear_index = offsets[t.index][n] + i * t.step
                        offsets[t.index][n] = offsets[typ.index][linear_index] + bases[typ.index]
                        shapes[t.index][n] = shapes[typ.index][linear_index]

                    return VarDim(nshapes=t.nshapes, step=typ.step, index=t.index, typ=typ.typ)

                elif isinstance(typ, Ref):
                    for n in range(t.nshapes):
                        if i < 0 or i >= shapes[t.index][n]:
                            raise IndexError(
                                "out of range: subindices currently must be in "
                                "range for all var shapes due to technical "
                                "difficulties");
                        linear_index = offsets[t.index][n] + i * t.step
                        offsets[t.index][n] = offsets[typ.index][linear_index] + bases[typ.index]
                        shapes[t.index][n] = shapes[typ.index][linear_index]

                    return Ref(step=typ.step, index=t.index, target=0, typ=typ.typ)

                elif isinstance(typ, Int64):
                    for n in range(t.nshapes):
                        offsets[t.index][n] += i * t.step
                        shapes[t.index][n] = 1
                    return Ref(ndim=typ.ndim, index=t.index, typ=typ)

                else:
                    raise TypeError("unexpected type")

            else:
                raise TypeError("type not indexable")

        def subslice(t, s, rest):
            if t.opt and not bitmaps[t.index].is_valid(bases[t.index]):
                raise TypeError("missing value or dimension is not sliceable")

            if isinstance(t, FixedDim):
                # get the resulting list layout from a slice and a shape
                start, stop, step, shape = adjust_indices_shape(s, t.shape)
                # adjust the linear index in the target array
                if shape > 0:
                    bases[t.target] += t.step * start
                # get the result type
                typ = dispatch(t.typ, rest)
                return FixedDim(opt=t.opt, shape=shape, step=t.step*step, index=t.index, typ=typ)

            elif isinstance(t, VarDim):
                for n, shape in enumerate(self.shapes[t.index]):
                    # get the resulting list layout from a slice and a shape
                    start, stop, step, shape = adjust_indices_shape(s, shape)
                    if shape > 0:
                        offsets[t.index][n] += start * t.step
                    shapes[t.index][n] = shape
                typ = dispatch(t.typ, rest)
                return VarDim(opt=t.opt, step=t.step*step, nshapes=t.nshapes, index=t.index, typ=typ)

            else:
                raise TypeError("unexpected type")

        typ = dispatch(self.typ, indices)

        return Array(typ=typ, data=data, bitmaps=bitmaps, bases=bases,
                     offsets=offsets, shapes=shapes)

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
  bases=%s,
  offsets=%s,
  shapes=%s,
)""" % (self.typ, self.typ.concrete_repr(), self.data, self.bitmaps,
        self.bases, self.offsets, self.shapes)


if __name__ == '__main__':
    lst = [[[0, 1], [2, 3]], [[4, 5, None], None, [7]], None, [[8, 9]]]
    a = Array(lst)

    print("Input: %s\n" % lst)
    print("%s\n" % a)
    print("Roundtrip: %s\n" % a.tolist())

    b = a[1:2, ::2, ::-1]
    print("%s\n" % b)
    print("slice: %s\n" % b.tolist())

# Ensure that libndtypes is loaded and initialized.
from ndtypes import ndt, MAX_DIM
from ._xnd import _xnd
from itertools import accumulate

__all__ = ['xnd', 'typeof', '_typeof']


# ======================================================================
#                              xnd object
# ======================================================================

class xnd(_xnd):
    def __new__(cls, value=None, type=None, levels=None):
        if type is None:
            if levels is not None:
                args = ', '.join("'%s'" % l if l is not None else 'NA' for l in levels)
                t = "%d * categorical(%s)" % (len(value), args)
                type = ndt(t)
            else:
                type = typeof(value)
        else:
            if levels is not None:
                raise TypeError(
                    "the 'type' and 'levels' arguments are mutually exclusive")
            elif isinstance(type, str):
                type = ndt(type)
        return _xnd(value, type)

    @classmethod
    def empty(cls, t):
        if isinstance(t, str):
            t = ndt(t)
        return cls(value=None, type=t)



# ======================================================================
#                            Type inference
# ======================================================================

def typeof(value):
    return ndt(_typeof(value))

def _typeof(value):
    """Infer the type of a Python value.  Only a subset of Datashape is
       supported.  In general, types need to be explicitly specified when
       creating xnd objects.
    """
    if isinstance(value, list):
        data, shapes = data_shapes(value)
        opt = None in data

        if not data:
            dtype = 'float64'
        else:
            dtype = _typeof(data[0])
            for x in data[1:]:
                t = _typeof(x)
                if t != dtype:
                    raise ValueError("dtype mismatch: have %s and %s" % (dtype, t))

        if opt:
            dtype = '?' + dtype

        t = dtype
        var = False
        for lst in shapes:
            opt = None in lst
            lst = [0 if x is None else x for x in lst]
            var, t = choose_dim_type(opt=opt, shapes=lst, typ=t, have_var=var)

        return t

    elif isinstance(value, tuple):
        return "(" + ", ".join([_typeof(x) for x in value]) + ")"

    elif isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return "{" + ", ".join(["%s: %s" % (k, _typeof(v)) for k, v in value.items()]) + "}"
        raise ValueError("all dict keys must be strings")

    elif isinstance(value, int):
        return 'int64'

    elif isinstance(value, float):
        return 'float64'

    elif isinstance(value, str):
        return 'string'

    elif isinstance(value, bytes):
        return 'bytes'

def choose_dim_type(*, opt=False, shapes=None, typ=None, have_var=False):
    """Choose a dimension type based on the list of 'shapes' that are present
       in a dimension. If all 'shapes' are equal, the more efficient FixedDim
       type suffices.
    """
    opt = '?' if opt else ''
    n = len(set(shapes))
    if not have_var and not opt and n <= 1:
        shape = 0 if n == 0 else shapes[0]
        return False, "%s%d * %s" % (opt, shape, typ)
    offsets = [0] + list(accumulate(shapes))
    return True, "%svar(offsets=%s) * %s" % (opt, offsets, typ)

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
            acc[max_level].append(a)
            min_level = min(level, min_level)

    search(max_level, tree)
    if acc[max_level] and all(x is None for x in acc[max_level]):
        pass # min_level is not set in this special case, hence the check.
    elif min_level != max_level:
        raise ValueError("unbalanced tree: min depth: %d max depth: %d" %
                         (min_level, max_level))

    data = acc[max_level]
    shapes = reversed(acc[0:max_level])

    return data, shapes




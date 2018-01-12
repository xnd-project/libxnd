from collections import OrderedDict
import unittest
import sys


# Support for test_xnd.py.


# OrderedDict literals hack.
class Record(OrderedDict):
    @staticmethod
    def _kv(s):
        if not isinstance(s, slice):
            raise TypeError("expect key-value pair")
        if s.step is not None:
            raise ValueError("expect key-value pair")
        return s.start, s.stop
    def __getitem__(self, items):
        if not isinstance(items, tuple):
            items = (items,)
        return OrderedDict(list(map(self._kv, items)))

R = Record()


# Broken classes.
class BoolMemoryError(object):
    def __bool__(self):
        raise MemoryError

class Index(object):
    def __index__(self):
        return 10

class IndexMemoryError(object):
    def __index__(self):
        raise MemoryError

class IndexTypeError(object):
    def __index__(self):
        return ""

def assertEqualWithEx(self, func, x, y):
    if x.value is y is None:
        return

    xerr = None
    try:
        xres = func(x)
    except Exception as e:
        xerr = e.__class__

    yerr = None
    try:
        yres = func(y)
    except Exception as e:
        yerr = e.__class__

    if xerr is None is yerr:
        self.assertEqual(xres, yres, msg="x: %s y: %s" % (x.type, y))
    else:
        self.assertEqual(xerr, yerr, msg="x: %s y: %s" % (x.type, y))

def skip_if(condition, reason):
    if condition:
        raise unittest.SkipTest(reason)

HAVE_64_BIT = sys.maxsize == 2**63-1

HAVE_PYTHON_36 = sys.version_info >= (3, 6, 0)

requires_py36 = unittest.skipUnless(
    sys.version_info > (3, 6),
    "test requires Python 3.6 or greater")

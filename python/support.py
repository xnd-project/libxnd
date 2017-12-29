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

def skip_if(condition, reason):
    if condition:
        raise unittest.SkipTest(reason)

HAVE_PYTHON_36 = sys.version_info >= (3, 6, 0)

requires_py36 = unittest.skipUnless(
    sys.version_info > (3, 6),
    "test requires Python 3.6 or greater")

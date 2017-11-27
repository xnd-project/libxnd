from collections import OrderedDict


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

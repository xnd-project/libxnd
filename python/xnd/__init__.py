# Make sure libndtypes.so is loaded and initialized by _ndtypes.so.
import ndtypes
del ndtypes

from ._xnd import *



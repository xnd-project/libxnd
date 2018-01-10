
xnd
===

libxnd
------

libxnd is a lightweight library for initializing and managing typed memory
blocks.  It relies on the libndtypes library for typing and traversing
memory.

http://xnd.readthedocs.io/en/latest/


xnd (Python module)
-------------------

The xnd Python module implements a container type that maps most Python
values relevant for scientific computing directly to typed memory.

Whenever possible, a single, pointer-free memory block is used.

xnd supports ragged arrays, categorical types, indexing, slicing, aligned
memory blocks and type inference.


Operations like indexing and slicing return zero-copy typed views on the
data.


http://xnd.readthedocs.io/en/latest/

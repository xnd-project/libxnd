
xnd
===

.. image:: https://mybinder.org/badge.svg
   :target: http://mybinder.org/v2/gh/plures/xnd/master?filepath=notebooks/python-tutorial.ipynb

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


Authors
-------

libxnd/xnd was created by `Stefan Krah <https://github.com/skrah>`_.
The funding that made this project possible came from `Anaconda Inc. <https://www.anaconda.com/>`_
and currently from `Quansight LLC <https://www.quansight.com/>`_.


.. image:: https://badges.gitter.im/Plures/xnd.svg
   :alt: Join the chat at https://gitter.im/Plures/xnd
   :target: https://gitter.im/Plures/xnd?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

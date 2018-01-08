.. meta::
   :robots: index, follow
   :description: xnd documentation
   :keywords: libxnd, xnd, C, Python, array computing

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


xnd
===

xnd is a package for typed memory blocks.


Libxnd
------

C library.

.. toctree::
   :maxdepth: 1

   libxnd/index.rst


Xnd
---

A Python module based on libxnd/libndtypes that implements a container
type for unboxing most Python values used in scientific computing.

The unboxed memory is typed using the Python ndtypes module. Indexing
and slicing return zero-copy views that are still correctly typed.



.. toctree::
   :maxdepth: 1

   xnd/index.rst


Releases
--------

.. toctree::
   :maxdepth: 1

   releases/index.rst

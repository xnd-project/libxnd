.. meta::
   :robots: index, follow
   :description: xnd documentation
   :keywords: memory blocks, unboxed values, array computing, Python

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


xnd
---

xnd is a Python module based on libxnd/libndtypes that implements a container
type for unboxing most Python values used in scientific computing.

The unboxed memory is typed using the Python ndtypes module. Indexing
and slicing return zero-copy views that are still correctly typed.


.. toctree::
   :maxdepth: 1

   types.rst
   align-pack.rst
   quickstart.rst

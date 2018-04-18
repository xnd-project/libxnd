.. meta::
   :robots: index,follow
   :description: xnd container
   :keywords: xnd, alignment, packing

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Alignment and packing
=====================

The xnd memory allocators support explicit alignment.  Alignment is specified
in the types.


Tuples and records
------------------

Tuples and records have the *align* and *pack* keywords that have the same
purpose as gcc's *aligned* and *packed* struct attributes.


Field alignment
~~~~~~~~~~~~~~~

The *align* keyword can be used to specify an alignment that is greater
than the natural alignment of a field:

.. doctest::

   >>> from xnd import *
   >>> s = "(uint8, uint64 |align=32|, uint64)"
   >>> x = xnd.empty(s)
   >>> x.align
   32
   >>> x.type.datasize
   64



Field packing
~~~~~~~~~~~~~

The *pack* keyword can be used to specify an alignment that is smaller
than the natural alignment of a field:

.. doctest::

   >>> s = "(uint8, uint64 |pack=2|, uint64)"
   >>> x = xnd.empty(s)
   >>> x.align
   8
   >>> x.type.datasize
   24



Struct packing
~~~~~~~~~~~~~~

The *pack* and *align* keywords can be applied to the entire struct:

.. doctest::

   >>> s = "(uint8, uint64, uint64, pack=1)"
   >>> x = xnd.empty(s)
   >>> x.align
   1
   >>> x.type.datasize
   17


Individual field and struct directives are mutually exclusive:

.. doctest::

   >>> s = "2 * (uint8 |align=16|, uint64, pack=1)"
   >>> x = xnd.empty(s)
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   ValueError: cannot have 'pack' tuple attribute and field attributes


Array alignment
~~~~~~~~~~~~~~~

An array has the same alignment as its elements:

.. doctest::

   >>> s = "2 * (uint8, uint64, pack=1)"
   >>> x = xnd.empty(s)
   >>> x.align
   1
   >>> x.type.datasize
   18

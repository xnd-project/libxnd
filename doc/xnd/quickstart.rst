.. meta::
   :robots: index,follow
   :description: xnd quickstart
   :keywords: xnd examples

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Quick Start
===========

The xnd object is a container that maps a wide range of Python values directly
to memory.  xnd unpacks complex types of arbitary nesting depth to a single
memory block.

Pointers only occur in explicit pointer types like *Ref* (reference), *Bytes*
and *String*, but not in the general case.


Install
-------

Prerequisites
~~~~~~~~~~~~~

Python2 is not supported. If not already present, install the Python3
development packages:

.. code-block:: sh

   # Debian, Ubuntu:
   sudo apt-get install gcc make
   sudo apt-get install python3-dev

   # Fedora, RedHat:
   sudo yum install gcc make
   sudo yum install python3-devel

   # openSUSE:
   sudo zypper install gcc make
   sudo zypper install python3-devel

   # BSD:
   # You know what to do.

   # Mac OS X:
   # Install Xcode and Python 3 headers.


Install
~~~~~~~

If `pip <http://pypi.python.org/pypi/pip>`_ is present on the system, installation
should be as easy as:

.. code-block:: sh

   pip install xnd


Otherwise:

.. code-block:: sh

   tar xvzf xnd.2.0b1.tar.gz
   cd xnd.2.0b1
   python3 setup.py install


Windows
~~~~~~~

Refer to the instructions in the *vcbuild* directory in the source distribution.


Type inference
--------------

If no explicit type is given, xnd supports type inference by assuming
types for the most common Python values.


Fixed arrays
~~~~~~~~~~~~

.. code-block:: py

   >>> x = xnd([[0,1,2], [3,4,5]])
   >>> x.type
   ndt("2 * 3 * int64")
   >>> x.value
   [[0, 1, 2], [3, 4, 5]]

As expected, lists are mapped to ndarrays and integers to int64.  Indexing and
slicing works the usual way.  For performance reasons these operations return
zero-copy views whenever possible:

.. code-block:: py

   >>> x[0][1]
   1

   >>> y = x[:, ::-1]
   >>> y
   <xnd.xnd object at 0x141a798>
   >>> y.type
   ndt("2 * 3 * int64")
   >>> y.value
   [[2, 1, 0], [5, 4, 3]]


Subarrays are views and properly typed:

.. code-block:: py

   >>> y = x[1]
   >>> y.type
   ndt("3 * int64")
   >>> y.value
   [3, 4, 5]


Ragged arrays
~~~~~~~~~~~~~

Ragged arrays are stored in the Arrow list representation. The data is
pointer-free, addressing the elements works by having one offset array
per dimension.

.. code-block:: py

   >>> x = xnd([[0.1j], [3+2j, 4+5j, 10j]])
   >>> x.type
   ndt("var * var * complex128")
   >>> x.value
   [[0.1j], [(3+2j), (4+5j), 10j]]
   >>>

Indexing and slicing works as usual, returning properly typed views or
values in the case of scalars:

.. code-block:: py

   >>> x = xnd([[0.1j], [3+2j, 4+5j, 10j]])
   >>> x[1, 2]
   10j

   >>> y = x[1]
   >>> y
   <xnd.xnd object at 0x1e8b848>
   >>> y.type
   ndt("var * complex128")
   >>> y.value
   [(3+2j), (4+5j), 10j]


Eliminating dimensions through mixed slicing and indexing is not supported
because it would require copying and adjusting potentially huge offset arrays:

.. code-block:: py

   >>> y = x[:, 1]
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   IndexError: mixed indexing and slicing is not supported for var dimensions


Records (structs)
~~~~~~~~~~~~~~~~~

From Python 3.6 on, dicts retain their order, so they can be used directly
for initializing C structs.

.. code-block:: py

   >>> x = xnd({'a': 'foo', 'b': 10.2})
   >>> x.type
   ndt("{a : string, b : float64}")
   >>> x.value
   {'a': 'foo', 'b': 10.2}



Tuples
~~~~~~

Python tuples are directly translated to the libndtypes tuple type:

.. code-block:: py

   >>> x = xnd(('foo', b'bar', [None, 10.0, 20.0]))
   >>> x.type
   ndt("(string, bytes(), 3 * ?float64)")
   >>> x.value
   ('foo', b'bar', [None, 10.0, 20.0])


Nested arrays in structs
~~~~~~~~~~~~~~~~~~~~~~~~

xnd seamlessly supports nested values of arbitrary depth:

.. code-block:: py

   >>> lst = [{'name': 'John', 'internet_points': [1, 2, 3]},
   ...        {'name': 'Jane', 'internet_points': [4, 5, 6]}]
   >>> x = xnd(lst)
   >>> x.type
   ndt("2 * {name : string, internet_points : 3 * int64}")
   >>> x.value
   [{'name': 'John', 'internet_points': [1, 2, 3]}, {'name': 'Jane', 'internet_points': [4, 5, 6]}]



Optional data (missing values)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional data is currently specified using *None*.  It is under debate if
a separate *NA* singeton object would be more suitable.

.. code-block:: py

   >>> lst = [0, 1, None, 2, 3, None, 5, 10]
   >>> x = xnd(lst)
   >>> x.type
   ndt("8 * ?int64")
   >>> x.value
   [0, 1, None, 2, 3, None, 5, 10]



Categorical data
~~~~~~~~~~~~~~~~

Type inference would be ambiguous, so it cannot work directly. xnd supports
the *levels* argument that is internally translated to the type.

.. code-block:: py

   >>> levels = ['January', 'August', 'December', None]
   >>> x = xnd(['January', 'January', None, 'December', 'August', 'December', 'December'], levels=levels)
   >>> x.type
   ndt("7 * categorical('January', 'August', 'December', NA)")
   >>> x.value
   ['January', 'January', None, 'December', 'August', 'December', 'December']


The above is equivalent to specifying the type directly:

.. code-block:: py

   >>> from ndtypes import *
   >>> t = ndt("7 * categorical('January', 'August', 'December', NA)")
   >>> x = xnd(['January', 'January', None, 'December', 'August', 'December', 'December'], type=t)
   >>> x.type
   ndt("7 * categorical('January', 'August', 'December', NA)")
   >>> x.value
   ['January', 'January', None, 'December', 'August', 'December', 'December']


Explicit types
--------------

While type inference is well-defined, it necessarily makes assumptions about
the programmer's intent.

There are two cases where types should be given:


Different types are intended
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: py

   >>> from ndtypes import *
  >>> x = xnd([[0,1,2], [3,4,5]], type="2 * 3 * uint8")
  >>> x.type
  ndt("2 * 3 * uint8")
  >>> x.value
  [[0, 1, 2], [3, 4, 5]]

Here, type inference would deduce :macro:`int64`, so :macro:`uint8` needs
to be passed explicitly.


Performance
~~~~~~~~~~~

For large arrays, explicit types are significantly faster.  Type inference
supports arbitrary nesting depth, is complex and still implemented in pure
Python. Compare:

.. code-block:: py

   >>> lst = [1] * 1000000
   >>> x = xnd(lst) # inference
   >>>
   >>> x = xnd(lst, type="1000000 * int64") # explicit

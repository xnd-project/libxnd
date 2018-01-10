.. meta::
   :robots: index,follow
   :description: xnd container
   :keywords: xnd, types, examples

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Types
=====

The xnd object is a container that maps a wide range of Python values directly
to memory.  xnd unpacks complex types of arbitrary nesting depth to a single
memory block.

Pointers only occur in explicit pointer types like *Ref* (reference), *Bytes*
and *String*, but not in the general case.


Type inference
--------------

If no explicit type is given, xnd supports type inference by assuming
types for the most common Python values.


Fixed arrays
~~~~~~~~~~~~

.. code-block:: py

   >>> from xnd import *
   >>> x = xnd([[0, 1, 2], [3, 4, 5]])
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

Ragged arrays are compatible with the Arrow list representation. The data
is pointer-free, addressing the elements works by having one offset array
per dimension.

.. code-block:: py

   >>> x = xnd([[0.1j], [3+2j, 4+5j, 10j]])
   >>> x.type
   ndt("var * var * complex128")
   >>> x.value
   [[0.1j], [(3+2j), (4+5j), 10j]]


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
   ndt("(string, bytes, 3 * ?float64)")
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
a separate *NA* singleton object would be more suitable.

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


All supported types
-------------------

Fixed arrays
~~~~~~~~~~~~

Fixed arrays are similar to NumPy's ndarray. One difference is that internally
xnd used steps instead of strides. One step is the amount of indices required
to move the linear index from one dimension element to the next.

This facilitates optional data, whose bitmaps need to be addressed by the
linear index.  The equation *stride = step * itemsize* always holds.


.. code-block:: py

   >>> lst = [[[1,2], [None, 3]], [[4, None], [5, 6]]]
   >>> x.type
   ndt("2 * 2 * 2 * ?int64")
   >>> x.value
   [[[1, 2], [None, 3]], [[4, None], [5, 6]]]

This is a fixed array with optional data.


.. code-block:: py

   >>> lst = [(1,2.0,3j), (4,5.0,6j)]
   >>> x = xnd(lst)
   >>> x.type
   ndt("2 * (int64, float64, complex128)")
   >>> x.value
   [(1, 2.0, 3j), (4, 5.0, 6j)]

An array with tuple elements.


Fortran order
~~~~~~~~~~~~~

Fortran order is specified by prefixing the dimensions with *!*:

.. code-block:: py

   >>> lst = [[1, 2, 3], [4, 5, 6]]
   >>> x = xnd(lst, type="!2 * 3 * uint16")
   >>> 
   >>> x.type.shape
   (2, 3)
   >>> x.type.strides
   (2, 4)


Alternatively, steps can be passed as arguments to the fixed dimension type:

.. code-block:: py

   >>> from ndtypes import *
   >>> lst = [[1, 2, 3], [4, 5, 6]]
   >>> t = ndt("fixed(shape=2, step=1) * fixed(shape=3, step=2) * uint16")
   >>> x = xnd(lst, type=t)
   >>> x.type.shape
   (2, 3)
   >>> x.type.strides
   (2, 4)


Ragged arrays
~~~~~~~~~~~~~

Ragged arrays with explicit types are easiest to construct using the *dtype*
argument to the xnd constructor.

.. code-block:: py

   >>> lst = [[0], [1, 2], [3, 4, 5]]
   >>> x = xnd(lst, dtype="int32")
   >>> lst = [[0], [1, 2], [3, 4, 5]]
   >>> x = xnd(lst, dtype="int32")
   >>> x.type
   ndt("var * var * int32")
   >>> x.value
   [[0], [1, 2], [3, 4, 5]]


Alternatively, offsets can be passed as arguments to the var dimension type:

.. code-block:: py

   >>> from ndtypes import ndt
   >>> t = ndt("var(offsets=[0,3]) * var(offsets=[0,1,3,6]) * int32")
   >>> x = xnd(lst, type=t)
   >>> x.type
   ndt("var * var * int32")
   >>> x.value
   [[0], [1, 2], [3, 4, 5]]


Tuples
~~~~~~

In memory, tuples are the same as C structs.

.. code-block:: py

   >>> x = xnd(("foo", 1.0))
   >>> x.type
   ndt("(string, float64)")
   >>> x.value
   ('foo', 1.0)


Indexing works the same as for arrays:

.. code-block:: py

   >>> x[0]
   'foo'


Nested tuples are more general than ragged arrays. They can a) hold different
data types and b) the trees they represent may be unbalanced.

They do not allow slicing though and are probably less efficient.

This is an example of an unbalanced tree that cannot be represented as a
ragged array:

.. code-block:: py

   >>> unbalanced_tree = (((1.0, 2.0), (3.0)), 4.0, ((5.0, 6.0, 7.0), ()))
   >>> x = xnd(unbalanced_tree)
   >>> x.type
   ndt("(((float64, float64), float64), float64, ((float64, float64, float64), ()))")
   >>> x.value
   (((1.0, 2.0), 3.0), 4.0, ((5.0, 6.0, 7.0), ()))
   >>>
   >>> x[0]
   ((1.0, 2.0), 3.0)
   >>> x[0][0]
   (1.0, 2.0)

Note that the data in the above tree example is packed into a single contiguous
memory block.


Records
~~~~~~~

In memory, records are C structs. The field names are only stored in the type.

The following examples use Python-3.6, which keeps the dict initialization
order.

.. code-block:: py

   >>> x = xnd({'a': b'123', 'b': {'x': 1.2, 'y': 100+3j}})
   >>> x.type
   ndt("{a : bytes, b : {x : float64, y : complex128}}")
   >>> x.value
   {'a': b'123', 'b': {'x': 1.2, 'y': (100+3j)}}


Indexing works the same as for arrays. Additionally, fields can be indexed
by name:

.. code-block:: py

   >>> x[0]
   b'123'
   >>> x['a']
   b'123'
   >>> x['b']
   {'x': 1.2, 'y': (100+3j)}


The nesting depth is arbitrary.  In the following example, the data -- except
for strings, which are pointers -- is packed into a single contiguous memory
block:

.. code-block:: py

   >>> item = {
   ...   "id": 1001,
   ...   "name": "cyclotron",
   ...   "price": 5998321.99,
   ...   "tags": ["connoisseur", "luxury"],
   ...   "stock": {
   ...     "warehouse": 722,
   ...     "retail": 20
   ...   }
   ... }
   >>> x = xnd(item)
   >>> print(x.type.pretty())
   {
     id : int64,
     name : string,
     price : float64,
     tags : 2 * string,
     stock : {
       warehouse : int64,
       retail : int64
     }
   }
   >>> x.value
   {'id': 1001, 'name': 'cyclotron', 'price': 5998321.99, 'tags': ['connoisseur', 'luxury'], 'stock': {'warehouse': 722, 'retail': 20}}


Strings can be embedded into the array by specifying the fixed string type.
In this case, the memory block is pointer-free.

.. code-block:: py

   >>> from ndtypes import ndt
   >>> 
   >>> s = """
   ...   { id : int64,
   ...     name : fixed_string(30),
   ...     price : float64,
   ...     tags : 2 * fixed_string(30),
   ...     stock : {warehouse : int64, retail : int64} 
   ...   }
   ... """
   >>> 
   >>> x = xnd(item, type=t)
   >>> print(x.type.pretty())
   {
     id : int64,
     name : fixed_string(30),
     price : float64,
     tags : 2 * fixed_string(30),
     stock : {
       warehouse : int64,
       retail : int64
     }
   }


Record of arrays
~~~~~~~~~~~~~~~~

Often it is more memory efficient to store an array of records as a record of
arrays.  This example with columnar data is from the Arrow homepage:

.. code-block:: py

   >>> data = {'session_id': [1331247700, 1331247702, 1331247709, 1331247799],
   ...         'timestamp': [1515529735.4895875, 1515529746.2128427, 1515529756.4485607, 1515529766.2181058],
   ...         'source_ip': ['8.8.8.100', '100.2.0.11', '99.101.22.222', '12.100.111.200']}
   x = xnd(data)
   >>> x.type
   ndt("{session_id : 4 * int64, timestamp : 4 * float64, source_ip : 4 * string}")



References
~~~~~~~~~~

References are transparent pointers to new memory blocks (meaning a new
data pointer, not a whole new xnd buffer).

For example, this is an array of pointer to array:

.. code-block:: py

   >>> t = ndt("3 * ref(4 * uint64)")
   >>> lst = [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
   >>> x = xnd(lst, type=t)
   >>> x.type
   ndt("3 * ref(4 * uint64)")
   >>> x.value
   [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

The user sees no difference to a regular 3 by 4 array, but internally
the outer dimension consists of three pointers to the inner arrays.

For memory blocks generated by xnd itself the feature is not so useful --
after all, it is usually better to have a single memory block than one
with additional pointers.


However, suppose that in the above columnar data example another application
represents the arrays inside the record with pointers.  Using the *ref* type,
data structures borrowed from such an application can be properly typed:

.. code-block:: py

   >>> t = ndt("{session_id : &4 * int64, timestamp : &4 * float64, source_ip : &4 * string}")
   >>> x = xnd(data, type=t)
   >>> x.type
   ndt("{session_id : ref(4 * int64), timestamp : ref(4 * float64), source_ip : ref(4 * string)}")

The ampersand is the shorthand for "ref".



Constructors
~~~~~~~~~~~~

Constructors are xnd's way of creating distinct named types. The constructor
argument is a regular type.

Constructors open up a new dtype, so named arrays can be the dtype of
other arrays.  Type inference currently isn't aware of constructors,
so types have to be provided.

.. code-block:: py

   >>> t = ndt("3 * SomeMatrix(2 * 2 * float32)")
   >>> lst = [[[1,2], [3,4]], [[5,6], [7,8]], [[9,10], [11,12]]]
   >>> x = xnd(lst, type=t)
   >>> x.type
   ndt("3 * SomeMatrix(2 * 2 * float32)")
   >>> x.value
   [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]
   >>> x[0].type
   ndt("SomeMatrix(2 * 2 * float32)")


Categorical
~~~~~~~~~~~

Categorical types contain values.  The data stored in xnd buffers are indices
(:macro:`int64`) into the type's categories.

.. code-block:: py

   >>> t = ndt("categorical('a', 'b', 'c', NA)")
   >>> data = ['a', 'a', 'b', 'a', 'a', 'a', 'foo', 'c']
   >>> x = xnd(data, dtype=t)
   >>> x.value
   ['a', 'a', 'b', 'a', 'a', 'a', None, 'c']


Fixed String
~~~~~~~~~~~~

Fixed strings are embedded into arrays.  Supported encodings are 'ascii',
'utf8', 'utf16' and 'utf32'.  The maximum string size is the number of
code points rather than bytes.

.. code-block:: py

   >>> t = ndt("10 * fixed_string(3, 'utf32')")
   >>> x = xnd.empty(t)
   >>> x.value
   ['', '', '', '', '', '', '', '', '', '']
   >>> x[3] = "\U000003B1\U000003B2\U000003B3"
   >>> x.value
   ['', '', '', 'αβγ', '', '', '', '', '', '']


Fixed Bytes
~~~~~~~~~~~

Fixed bytes are embedded into arrays.

.. code-block:: py

   >>> t = ndt("3 * fixed_bytes(size=3)")
   >>> x = xnd.empty(t)
   >>> x[2] = b'123'
   >>> x.value
   [b'\x00\x00\x00', b'\x00\x00\x00', b'123']
   >>> x.align
   1

Alignment can be requested with the requirement that size is a multiple of
alignment:

.. code-block:: py

   >>> t = ndt("3 * fixed_bytes(size=32, align=16)")
   >>> x = xnd.empty(t)
   >>> x.align
   16


String
~~~~~~

Strings are pointers to :macro:`NUL`-terminated UTF-8 strings.

.. code-block:: py

   >>> x = xnd.empty("10 * string")
   >>> x.value
   ['', '', '', '', '', '', '', '', '', '']
   >>> x[0] = "abc"
   >>> x.value
   ['abc', '', '', '', '', '', '', '', '', '']



Bytes
~~~~~

Internally, bytes are structs with a size field and a pointer to the data.

.. code-block:: py

   >>> x = xnd([b'123', b'45678'])
   >>> x.type
   ndt("2 * bytes")
   >>> x.value
   [b'123', b'45678']


The bytes constructor takes an optional *align* argument that specifies the
alignment of the allocated data:

.. code-block:: py

   >>> x = xnd([b'abc', b'123'], type="2 * bytes(align=64)")
   >>> x.value
   [b'abc', b'123']
   >>> x.align
   8

Note that *x.align* is the alignment of the array.  The embedded pointers
to the bytes data are aligned at *64*.


Primitive types
~~~~~~~~~~~~~~~

As a short example, here is a tuple that contains all primitive types:

.. code-block:: py

   >>> s = """
   ...    (bool,
   ...     int8, int16, int32, int64,
   ...     uint8, uint16, uint32, uint64,
   ...     float16, float32, float64,
   ...     complex32, complex64, complex128)
   ... """
   >>> x = xnd.empty(s)
   >>> x.value
   (False, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0j, 0j, 0j)

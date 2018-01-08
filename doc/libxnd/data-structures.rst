.. meta::
   :robots: index,follow
   :description: libxnd documentation

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Data structures
===============

libxnd is a lightweight container library that leaves most of the work to
libndtypes. The underlying idea is to have a small struct that contains
bitmaps, a linear index, a type and a data pointer.

The type contains all low level access information.

Since the struct is small, it can easily be passed around by value when
it serves as a view on the data.

The owner of the data is a master buffer with some additional bookkeeping
fields.


Bitmaps
-------

.. topic:: bitmaps

.. code-block:: c

   typedef struct xnd_bitmap xnd_bitmap_t;

   struct xnd_bitmap {
       uint8_t *data;      /* bitmap */
       int64_t size;       /* number of subtree bitmaps in the "next" array */
       xnd_bitmap_t *next; /* array of bitmaps for subtrees */
   };

libxnd supports data with optional values. Any type can have a bitmap for its
data.  Bitmaps are addressed using the linear index in the :macro:`xnd_t` struct.

For a scalar, the bitmap contains just one addressible bit.

For fixed and variable arrays, the bitmap contains one bit for each item.

Containers dtypes like tuples and records have new bitmaps for each of their
fields if any of the field subtrees contains optional data.

These field bitmaps are in the *next* array.


View
----

.. topic:: xnd_view

.. code-block:: c


   /* Typed memory block, usually a view. */
   typedef struct xnd {
       xnd_bitmap_t bitmap; /* bitmap tree */
       int64_t index;       /* linear index for var dims */
       const ndt_t *type;   /* type of the data */
       char *ptr;           /* data */
   } xnd_t;

This is the xnd view, with a typed data pointer, the current linear index
and a bitmap.

When passing the view around, the linear index needs to be maintained because
it is required to determine if a value is missing (NA).

The convention is to apply the linear index (adjust the *ptr* and set it
to *0*) only when a non-optional elemnent at *ndim == 0* is actually
accessed.

This happens for example when getting the value of an element or when
descending into a record.


Flags
-----

.. topic:: flags

.. code-block:: c

   #define XND_OWN_TYPE     0x00000001U /* type pointer */
   #define XND_OWN_DATA     0x00000002U /* data pointer */
   #define XND_OWN_STRINGS  0x00000004U /* embedded string pointers */
   #define XND_OWN_BYTES    0x00000008U /* embedded bytes pointers */
   #define XND_OWN_POINTERS 0x00000010U /* embedded pointers */

   #define XND_OWN_ALL (XND_OWN_TYPE |    \
                        XND_OWN_DATA |    \
                        XND_OWN_STRINGS | \
                        XND_OWN_BYTES |   \
                        XND_OWN_POINTERS)

   #define XND_OWN_EMBEDDED (XND_OWN_DATA |    \
                             XND_OWN_STRINGS | \
                             XND_OWN_BYTES |   \
                             XND_OWN_POINTERS)


The ownership flags for the xnd master buffer (see below).  Like libndtypes,
libxnd itself has no notion of how many exported views a master buffer has.

This is deliberately done in order to prevent two different memory management
schemes from getting in each other's way.

However, for deallocating a master buffer the flags must be set correctly.

:macro:`XND_OWN_TYPE` is set if the master buffer owns the :macro:`ndt_t`.

:macro:`XND_OWN_DATA` is set if the master buffer owns the data pointer.


The *string*, *bytes* and *ref* types have pointers that are embedded in the
data.  Usually, these are owned and deallocated by libxnd.

For strings, the Python bindings use the convention that :macro:`NULL` strings
are interpreted as the empty string. Once a string pointer is initialized it
belongs to the master buffer.


Macros
------

.. topic:: macros

.. code-block:: c

   /* Convenience macros to extract embedded values. */
   #define XND_POINTER_DATA(ptr) (*((char **)ptr))
   #define XND_BYTES_SIZE(ptr) (((ndt_bytes_t *)ptr)->size)
   #define XND_BYTES_DATA(ptr) (((ndt_bytes_t *)ptr)->data)

These macros should be used to extract embedded *ref*, *string* and *bytes*
data.



Master buffer
-------------

.. topic:: xnd_master

.. code-block:: c

   /* Master memory block. */
   typedef struct xnd_master {
       uint32_t flags; /* ownership flags */
       xnd_t master;   /* typed memory */
   } xnd_master_t;

This is the master buffer.  *flags* are explained above, the *master* buffer
should be considered constant.

For traversing memory, copy a new view buffer by value.

.. meta::
   :robots: index,follow
   :description: libndtypes documentation

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Functions
=========

Create typed memory blocks
--------------------------

The main use case for libxnd is to create and manage typed memory blocks.
These blocks are fully initialized to *0*.  References to additional memory
blocks are allocated and initialized recursively.

*bytes* and *string* types are initialized to :macro:`NULL`, since their
actual length is not known yet.


.. topic:: xnd_empty_from_string

.. code-block:: c

   xnd_master_t *xnd_empty_from_string(const char *s, uint32_t flags, ndt_context_t *ctx);

Return a new master buffer according to the type string in *s*.  *flags*
must include :macro:`XND_OWN_TYPE`.


.. topic:: xnd_empty_from_type

.. code-block:: c

   xnd_master_t *xnd_empty_from_type(const ndt_t *t, uint32_t flags, ndt_context_t *ctx);


Return a new master buffer according to *type*.  *flags* must not include
:macro:`XND_OWN_TYPE`, i.e. the type is externally managed.

This is the case in the Python bindings, where the ndtypes module creates
and manages types.


Delete typed memory blocks
--------------------------

.. topic:: xnd_del

.. code-block:: c

   void xnd_del(xnd_master_t *x);

Delete the master buffer according to its flags. *x* may be :macro:`NULL`.
*x->master.ptr* and *x->master.type* may be :macro:`NULL`.

The latter situation should only arise when breaking up reference cycles.
This is used in the Python module.


Bitmaps
-------

.. topic:: xnd_bitmap_next

.. code-block:: c

   xnd_bitmap_t xnd_bitmap_next(const xnd_t *x, int64_t i, ndt_context_t *ctx);

Get the next bitmap for the *Tuple*, *Record*, *Ref* and *Constr* types.

This is a convenience function that checks if the types have optional
subtrees.

If yes, return the bitmap at index *i*.  If not, it return an empty bitmap
that must not be accessed.


.. topic:: xnd_set_valid

.. code-block:: c

   void xnd_set_valid(xnd_t *x);

Set the validity bit at *x->index*.  *x* must have an optional type.


.. topic:: xnd_set_na

.. code-block:: c

   void xnd_set_na(xnd_t *x);

Clear the validity bit at *x->index*.  *x* must have an optional type.



.. topic:: xnd_is_valid

.. code-block:: c

   int xnd_is_valid(const xnd_t *x);

Check if the element at *x->index* is valid.  If *x* does not have an optional
type, return *1*.  Otherwise, return the validity bit (zero or nonzero).


.. topic:: xnd_is_na

.. code-block:: c

   int xnd_is_na(const xnd_t *x);

Check if the element at *x->index* is valid.  If *x* does not have an optional
type, return *0*.  Otherwise, return the negation of the validity bit.

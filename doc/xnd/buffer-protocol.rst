.. meta::
   :robots: index,follow
   :description: xnd container
   :keywords: xnd, buffer protocol

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Buffer protocol
===============

xnd supports importing PEP-3118 buffers.


From NumPy
----------

Import a simple ndarray:

.. doctest::

   >>> import numpy as np
   >>> from xnd import *
   >>> x = np.array([[[0,1,2], [3,4,5]], [[6,7,8], [9,10,11]]])
   >>> y = xnd.from_buffer(x)
   >>> y.type
   ndt("2 * 2 * 3 * int64")
   >>> y.value
   [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]


Import an ndarray with a struct dtype:

.. doctest::

   >>> x = np.array([(1000, 400.25, 'abc'), (-23, -1e10, 'cba')],
   ...              dtype=[('x', '<i4'), ('y', '>f4'), ('z', 'S3')])
   >>> y = xnd.from_buffer(x)
   >>> y.type
   ndt("2 * {x : int32, y : >float32, z : fixed_bytes(size=3)}")
   >>> y.value
   [{'x': 1000, 'y': 400.25, 'z': b'abc'}, {'x': -23, 'y': -10000000000.0, 'z': b'cba'}]

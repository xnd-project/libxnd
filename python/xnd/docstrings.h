/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2017-2018, plures
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "pymacro.h"


/******************************************************************************/
/*                                Module                                      */
/******************************************************************************/


PyDoc_STRVAR(doc_module, "xnd module");



/******************************************************************************/
/*                             xnd properties                                 */
/******************************************************************************/

PyDoc_STRVAR(doc_value,
"\n\
Convert the in-memory value to a Python value.\n\
\n\
   >>> x = xnd([1, 2, 3, 4, 5])\n\
   >>> x.value\n\
   [1, 2, 3, 4, 5]\n\
\n");

PyDoc_STRVAR(doc_type,
"\n\
Return the type of the in-memory value.\n\
\n\
   >>> x = xnd([[1, 2, 3], [4, 5]])\n\
   >>> x.type\n\
   ndt(\"var * var * int64\")\n\
\n");

PyDoc_STRVAR(doc_ndim,
"\n\
Return the number of dimensions of the in-memory value.\n\
\n\
   >>> x = xnd([[(1, \"a\"), (2, \"b\")], [(3, \"x\"), (4, \"y\")]])\n\
   >>> x.ndim\n\
   2\n\
\n");

PyDoc_STRVAR(doc_align,
"\n\
Return the alignment of the memory block.\n\
\n\
   >>> t = \"2 * {a: uint8, b: float64, align=512}\"\n\
   >>> x = xnd([{'a': 1, 'b': 2.0}, {'a': 2, 'b': 3.0}], type=t)\n\
   >>> x.align\n\
   512\n\
\n");


/******************************************************************************/
/*                                xnd methods                                 */
/******************************************************************************/

PyDoc_STRVAR(doc_short_value,
"short_value($self, /, maxshape=None)\n--\n\n\
Convert the in-memory value to an abbreviated Python value.  Container types\n\
are converted to maxshape-1 actual values followed by an Ellipsis.  If maxshape\n\
is None, the representation is not abbreviated.\n\
\n\
    >>> x = xnd([1, 2, 3, 4, 5])\n\
    >>> x.short_value(maxshape=3)\n\
    [1, 2, ...]\n\
\n");

PyDoc_STRVAR(doc_empty,
"empty($type, type, /)\n--\n\n\
Class method that constructs a new xnd container according to the type\n\
argument.  All values are initialized to zero.\n\
\n\
    >>> xnd.empty(\"10 * 2 * (int64, string, complex128)\")\n\
    xnd([[(0, '', 0j), (0, '', 0j)], [(0, '', 0j), (0, '', 0j)],\n\
         [(0, '', 0j), (0, '', 0j)], [(0, '', 0j), (0, '', 0j)],\n\
         [(0, '', 0j), (0, '', 0j)], [(0, '', 0j), (0, '', 0j)],\n\
         [(0, '', 0j), (0, '', 0j)], [(0, '', 0j), (0, '', 0j)],\n\
         [(0, '', 0j), (0, '', 0j)], ...], type=\"10 * 2 * (int64, string, complex128)\")\n\
\n");

PyDoc_STRVAR(doc_from_buffer,
"from_buffer($type, obj, /)\n--\n\n\
Class method that constructs a new xnd container from an object that supports\n\
the buffer protocol.\n\
\n\
    >>> xnd.from_buffer(b\"123\")\n\
    xnd([49, 50, 51], type=\"3 * uint8\")\n\
\n\
    >>> import numpy as np\n\
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])\n\
    >>> xnd.from_buffer(a)\n\
    xnd([[1, 2, 3], [4, 5, 6]], type=\"2 * 3 * int64\")\n\
\n\
\n");

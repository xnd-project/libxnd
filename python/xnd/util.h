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


#ifndef UTIL_H
#define UTIL_H


#include <Python.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include "ndtypes.h"


static inline Py_ssize_t
safe_downcast(int64_t size)
{
#if SIZE_MAX < INT64_MAX
    if (size > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError,
            "sizes should never exceed INT32_MAX on 32-bit platforms");
        return -1;
    }
#endif
    return size;
}

static inline bool
check_invariants(const ndt_t *t)
{
#if SIZE_MAX < INT64_MAX
    return safe_downcast(t->datasize) >= 0;
#else
    (void)t;
    return 1;
#endif
}

static inline PyObject *
list_new(int64_t size)
{
#if SIZE_MAX < INT64_MAX
    Py_ssize_t n = safe_downcast(size);
    return n < 0 ? NULL : PyList_New(n);
#else
    return PyList_New(size);
#endif
}

static PyObject *
tuple_new(int64_t size)
{
#if SIZE_MAX < INT64_MAX
    Py_ssize_t n = safe_downcast(size);
    return n < 0 ? NULL : PyTuple_New(n);
#else
    return PyTuple_New(size);
#endif
}

static inline PyObject *
unicode_from_kind_and_data(int kind, const void *buffer, int64_t size)
{
#if SIZE_MAX < INT64_MAX
    Py_ssize_t n = safe_downcast(size);
    return n < 0 ? NULL : PyUnicode_FromKindAndData(kind, buffer, n);
#else
    return PyUnicode_FromKindAndData(kind, buffer, size);
#endif
}

static inline PyObject *
bytes_from_string_and_size(const char *str, int64_t size)
{
#if SIZE_MAX < INT64_MAX
    Py_ssize_t n = safe_downcast(size);
    return n < 0 ? NULL : PyBytes_FromStringAndSize(str, n);
#else
    return PyBytes_FromStringAndSize(str, size);
#endif
}

static inline int
py_slice_get_indices_ex(PyObject *key, int64_t length,
                        int64_t *start, int64_t *stop, int64_t *step,
                        int64_t *slicelength)
{
#if SIZE_MAX < INT64_MAX
    Py_ssize_t n, _start, _stop, _step, _slicelength;
    int ret;

    n = safe_downcast(length);
    if (n < 0) {
        return -1;
    }

    ret = PySlice_GetIndicesEx(key, n, &_start, &_stop, &_step, &_slicelength);
    *start = _start;
    *stop = _stop;
    *step = _step;
    *slicelength = _slicelength;

    return ret;
#else
    return PySlice_GetIndicesEx(key, length, start, stop, step, slicelength);
#endif
}

/* PSF copyright: Written by Jim Hugunin and Chris Chase. */
int
pyslice_unpack(PyObject *_r,
               Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step)
{
    PySliceObject *r = (PySliceObject*)_r;
    /* this is harder to get right than you might think */

    assert(PY_SSIZE_T_MIN + 1 <= -PY_SSIZE_T_MAX);

    if (r->step == Py_None) {
        *step = 1;
    }
    else {
        if (!_PyEval_SliceIndex(r->step, step)) return -1;
        if (*step == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "slice step cannot be zero");
            return -1;
        }
        /* Here *step might be -PY_SSIZE_T_MAX-1; in this case we replace it
         * with -PY_SSIZE_T_MAX.  This doesn't affect the semantics, and it
         * guards against later undefined behaviour resulting from code that
         * does "step = -step" as part of a slice reversal.
         */
        if (*step < -PY_SSIZE_T_MAX)
            *step = -PY_SSIZE_T_MAX;
    }

    if (r->start == Py_None) {
        *start = *step < 0 ? PY_SSIZE_T_MAX : 0;
    }
    else {
        if (!_PyEval_SliceIndex(r->start, start)) return -1;
    }

    if (r->stop == Py_None) {
        *stop = *step < 0 ? PY_SSIZE_T_MIN : PY_SSIZE_T_MAX;
    }
    else {
        if (!_PyEval_SliceIndex(r->stop, stop)) return -1;
    }

    return 0;
}


#endif /* UTIL_H */

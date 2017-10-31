/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2017, plures
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


#include <Python.h>
#include "complexobject.h"
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>
#include "ndtypes.h"
#include "xnd.h"


#ifdef _MSC_VER
  #ifndef UNUSED
    #define UNUSED
  #endif
#else
  #if defined(__GNUC__) && !defined(__INTEL_COMPILER)
    #define UNUSED __attribute__((unused))
  #else
    #define UNUSED
  #endif
#endif

typedef struct {
    PyObject_HEAD
    ndt_t *ndt;
} NdtObject;

#undef NDT
#define NDT(v) (((NdtObject *)v)->ndt)


typedef struct {
    PyObject_HEAD
    PyObject *ndt;
    xnd_t xnd;
} XndObject;

#undef XND
static PyTypeObject Xnd_Type;
#define Xnd_CheckExact(v) (Py_TYPE(v) == &Xnd_Type)
#define Xnd_Check(v) PyObject_TypeCheck(v, &Xnd_Type)
#define NDT_REF(v) (((XndObject *)v)->ndt)
#define XND(v) (((XndObject *)v)->xnd)
#define TYP(v) (((XndObject *)v)->xnd.type)
#define INDEX(v) (((XndObject *)v)->xnd.index)
#define PTR(v) (((XndObject *)v)->xnd.ptr)


#if PY_LITTLE_ENDIAN
static const int little_endian = 1;
#else
static const int little_endian = 0;
#endif

static PyObject *
seterr(ndt_context_t *ctx)
{
    PyObject *exc = PyExc_RuntimeError;

    switch (ctx->err) {
    case NDT_Success: /* should never be set on error */
        exc = PyExc_RuntimeError;
        break;
    case NDT_ValueError:
        exc = PyExc_ValueError;
        break;
    case NDT_TypeError:
        exc = PyExc_TypeError;
        break;
    case NDT_InvalidArgumentError:
        exc = PyExc_ValueError;
        break;
    case NDT_NotImplementedError:
        exc = PyExc_NotImplementedError;
        break;
    case NDT_LexError: case NDT_ParseError:
        exc = PyExc_ValueError;
        break;
    case NDT_OSError:
        exc = PyExc_OSError;
        break;
    case NDT_RuntimeError:
        exc = PyExc_RuntimeError;
        break;
    case NDT_MemoryError:
        exc = PyExc_MemoryError;
        break;
    }

    PyErr_SetString(exc, ndt_context_msg(ctx));
    ndt_context_del(ctx);

    return NULL;
};


static PyObject *
pyxnd_alloc(PyTypeObject *type)
{
    XndObject *x;

    if (type == &Xnd_Type) {
        x = PyObject_New(XndObject, &Xnd_Type);
    }
    else {
        x = (XndObject *)type->tp_alloc(type, 0);
    }
    if (x == NULL) {
        return NULL;
    }
 
    TYP(x) = NULL;
    INDEX(x) = 0;
    PTR(x) = NULL;
    NDT_REF(x) = NULL;

    return (PyObject *)x;
}

static void
pyxnd_dealloc(PyObject *x)
{
    if PTR(x) ndt_free(PTR(x));
    Py_CLEAR(NDT_REF(x));

    Py_TYPE(x)->tp_free(x);
}


#define PACK_SINGLE(ptr, src, type) \
    do {                                     \
        type _x;                             \
        _x = (type)src;                      \
        memcpy(ptr, (char *)&_x, sizeof _x); \
    } while (0)

#define UNPACK_SINGLE(dest, ptr, type) \
    do {                                     \
        type _x;                             \
        memcpy((char *)&_x, ptr, sizeof _x); \
        dest = _x;                           \
    } while (0)


PyObject *
dict_get_item(PyObject *v, const char *key)
{
    PyObject *k, *r;

    k = PyUnicode_FromString(key);
    if (k == NULL) {
        return NULL;
    }

    r =  PyDict_GetItemWithError(v, k);
    Py_DECREF(k);
    return r;
}

static int64_t
get_int(PyObject *v, int64_t min, int64_t max)
{
    PyObject *tmp;
    int64_t x;

    tmp = PyNumber_Index(v);
    if (tmp == NULL) {
        return -1;
    }

    x = PyLong_AsLongLong(tmp);
    Py_DECREF(tmp);

    if (x == -1 && PyErr_Occurred()) {
        return -1;
    }

    if (x < min || x > max) {
        PyErr_Format(PyExc_ValueError,
            "out of range: %" PRIi64, x);
        return -1;
    }

    return x;
}

static uint64_t
get_uint(PyObject *v, uint64_t max)
{
    PyObject *tmp;
    unsigned long long x;

    tmp = PyNumber_Index(v);
    if (tmp == NULL) {
        return max;
    }

    x = PyLong_AsUnsignedLongLong(tmp);
    Py_DECREF(tmp);

    if (x == (unsigned long long)-1 && PyErr_Occurred()) {
        return max;
    }

    if (x > max) {
        PyErr_Format(PyExc_ValueError,
            "out of range: %" PRIu64, x);
        return max;
    }

    return x;
}

static int
pyxnd_init(xnd_t x, PyObject *v)
{
    NDT_STATIC_CONTEXT(ctx);
    const ndt_t *t = x.type;
    int64_t shape, i;
    xnd_t next;

    if (ndt_is_abstract(t)) {
        PyErr_SetString(PyExc_TypeError, "xnd has abstract type");
        return -1;
    }

    /* Add the linear index from var dimensions. For a chain of fixed
       dimensions, x.index is zero. */
    if (t->ndim == 0) {
        x.ptr += x.index * t->data_size;
    }

    switch (t->tag) {
    case FixedDim: {
        assert(x.index == 0);

        if (!PyList_Check(v)) {
            PyErr_Format(PyExc_TypeError,
                "xnd: expected list, not '%.200s'", Py_TYPE(v)->tp_name);
            return -1;
        }

        shape = t->FixedDim.shape;
        if (PyList_GET_SIZE(v) != shape) {
            PyErr_Format(PyExc_ValueError,
                "xnd: expected list with size %" PRIi64, shape);
            return -1;
        }

        next.type = t->FixedDim.type;
        next.index = 0;

        for (i = 0; i < shape; i++) {
            next.ptr = x.ptr + i * t->Concrete.FixedDim.stride;
            if (pyxnd_init(next, PyList_GET_ITEM(v, i)) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case VarDim: {
        const int32_t noffsets = t->Concrete.VarDim.noffsets;
        int32_t start, stop;

        if (!PyList_Check(v)) {
            PyErr_Format(PyExc_TypeError,
                "xnd: expected list, not '%.200s'", Py_TYPE(v)->tp_name);
            return -1;
        }

        if (x.index < 0 || x.index+1 >= noffsets) {
            PyErr_Format(PyExc_RuntimeError,
                "xnd: offset index out of range: index=%" PRIi32, " noffsets=%" PRIi32,
                x.index, noffsets);
            return -1;
        }

        start = t->Concrete.VarDim.offsets[x.index];
        stop = t->Concrete.VarDim.offsets[x.index+1];

        shape = stop - start;
        if (PyList_GET_SIZE(v) != shape) {
            PyErr_Format(PyExc_ValueError,
                "xnd: expected list with size %" PRIi64, shape);
            return -1;
        }

        next.type = t->VarDim.type;
        next.ptr = x.ptr;

        for (i = 0; i < shape; i++) {
            next.index =  start + i;
            if (pyxnd_init(next, PyList_GET_ITEM(v, i)) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case Tuple: {
        if (!PyTuple_Check(v)) {
            PyErr_Format(PyExc_TypeError,
                "xnd: expected tuple, not '%.200s'", Py_TYPE(v)->tp_name);
            return -1;
        }

        shape = t->Tuple.shape;
        if (PyTuple_GET_SIZE(v) != shape) {
            PyErr_Format(PyExc_ValueError,
                "xnd: expected tuple with size %" PRIi64, shape);
            return -1;
        }

        for (i = 0; i < shape; i++) {
            next.type = t->Tuple.types[i];
            next.index = 0;
            next.ptr = x.ptr + t->Concrete.Tuple.offset[i];

            if (pyxnd_init(next, PyTuple_GET_ITEM(v, i)) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case Record: {
        PyObject *tmp;

        if (!PyDict_Check(v)) {
            PyErr_Format(PyExc_TypeError,
                "xnd: expected dict, not '%.200s'", Py_TYPE(v)->tp_name);
            return -1;
        }

        shape = t->Record.shape;
        if (PyDict_Size(v) != shape) {
            PyErr_Format(PyExc_ValueError,
                "xnd: expected dict with size %" PRIi64, shape);
            return -1;
        }

        for (i = 0; i < shape; i++) {
            next.type = t->Record.types[i];
            next.index = 0;
            next.ptr = x.ptr + t->Concrete.Record.offset[i];

            tmp = dict_get_item(v, t->Record.names[i]);
            if (tmp == NULL) {
                if (!PyErr_Occurred()) {
                    PyErr_Format(PyExc_ValueError,
                        "xnd: key not found %s", t->Record.names[i]);
                }
                return -1;
            }

            if (pyxnd_init(next, tmp) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case Pointer: {
        next.type = t->Pointer.type;
        next.index = 0;
        next.ptr = XND_POINTER_DATA(x.ptr);
        return pyxnd_init(next, v);
    }

    case Constr: {
        next.type = t->Constr.type;
        next.index = 0;
        next.ptr = x.ptr;
        return pyxnd_init(next, v);
    }

    case Nominal: {
        PyErr_SetString(PyExc_NotImplementedError,
            "the 'nominal' type is opaque and only useful on the C level");
        return -1;
    }

    case Bool: {
        int tmp = PyObject_IsTrue(v);
        bool b;

        tmp = PyObject_IsTrue(v);
        if (tmp < 0) {
            return -1;
        }
        b = (bool)tmp;

        PACK_SINGLE(x.ptr, b, bool);
        return 0;
    }

    case Int8: {
        int8_t tmp = (int8_t)get_int(v, INT8_MIN, INT8_MAX);
        if (tmp == -1 && PyErr_Occurred()) {
            return -1;
        }
        PACK_SINGLE(x.ptr, tmp, int8_t);
        return 0;
    }

    case Int16: {
        int16_t tmp = (int16_t)get_int(v, INT16_MIN, INT16_MAX);
        if (tmp == -1 && PyErr_Occurred()) {
            return -1;
        }
        PACK_SINGLE(x.ptr, tmp, int16_t);
        return 0;
    }

    case Int32: {
        int32_t tmp = (int32_t)get_int(v, INT32_MIN, INT32_MAX);
        if (tmp == -1 && PyErr_Occurred()) {
            return -1;
        }
        PACK_SINGLE(x.ptr, tmp, int32_t);
        return 0;
    }

    case Int64: {
        int64_t tmp = get_int(v, INT64_MIN, INT64_MAX);
        if (tmp == -1 && PyErr_Occurred()) {
            return -1;
        }
        PACK_SINGLE(x.ptr, tmp, int64_t);
        return 0;
    }

    case Uint8: {
        uint8_t tmp = (uint8_t)get_uint(v, UINT8_MAX);
        if (tmp == UINT8_MAX && PyErr_Occurred()) {
            return -1;
        }
        PACK_SINGLE(x.ptr, tmp, uint8_t);
        return 0;
    }

    case Uint16: {
        uint16_t tmp = (uint16_t)get_uint(v, UINT16_MAX);
        if (tmp == UINT16_MAX && PyErr_Occurred()) {
            return -1;
        }
        PACK_SINGLE(x.ptr, tmp, uint16_t);
        return 0;
    }

    case Uint32: {
        uint32_t tmp = (uint32_t)get_uint(v, UINT32_MAX);
        if (tmp == UINT32_MAX && PyErr_Occurred()) {
            return -1;
        }
        PACK_SINGLE(x.ptr, tmp, uint32_t);
        return 0;
    }

    case Uint64: {
        uint64_t tmp = get_uint(v, UINT64_MAX);
        if (tmp == UINT64_MAX && PyErr_Occurred()) {
            return -1;
        }
        PACK_SINGLE(x.ptr, tmp, uint64_t);
        return 0;
    }

    case Float16: {
#if PY_VERSION_HEX >= 0x03060000
        double tmp = PyFloat_AsDouble(v);
        if (tmp == -1 && PyErr_Occurred()) {
            return -1;
        }
        return _PyFloat_Pack2(tmp, (unsigned char *)x.ptr, little_endian);
#else
        PyErr_SetString(PyExc_NotImplementedError,
            "half-float not implemented in Python versions < 3.6");
        return -1;
#endif
    }

    case Float32: {
        double tmp = PyFloat_AsDouble(v);
        if (tmp == -1 && PyErr_Occurred()) {
            return -1;
        }
        return _PyFloat_Pack4(tmp, (unsigned char *)x.ptr, little_endian);
    }

    case Float64: {
        double tmp = PyFloat_AsDouble(v);
        if (tmp == -1 && PyErr_Occurred()) {
            return -1;
        }
        return _PyFloat_Pack8(tmp, (unsigned char *)x.ptr, little_endian);
    }

    case Complex32: {
#if PY_VERSION_HEX >= 0x03060000
        Py_complex c = PyComplex_AsCComplex(v);
        if (c.real == -1.0 && PyErr_Occurred()) {
            return -1;
        }
        if (_PyFloat_Pack2(c.real, (unsigned char *)x.ptr, little_endian) < 0) {
            return -1;
        }
        return _PyFloat_Pack2(c.imag, (unsigned char *)(x.ptr+2), little_endian);
#else
        PyErr_SetString(PyExc_NotImplementedError,
            "half-float not implemented in Python versions < 3.6");
        return -1;
#endif
    }

    case Complex64: {
        Py_complex c = PyComplex_AsCComplex(v);
        if (c.real == -1.0 && PyErr_Occurred()) {
            return -1;
        }
        if (_PyFloat_Pack4(c.real, (unsigned char *)x.ptr, little_endian) < 0) {
            return -1;
        }
        return _PyFloat_Pack4(c.imag, (unsigned char *)(x.ptr+4), little_endian);
    }

    case Complex128: {
        Py_complex c = PyComplex_AsCComplex(v);
        if (c.real == -1.0 && PyErr_Occurred()) {
            return -1;
        }
        if (_PyFloat_Pack8(c.real, (unsigned char *)x.ptr, little_endian) < 0) {
            return -1;
        }
        return _PyFloat_Pack8(c.imag, (unsigned char *)(x.ptr+8), little_endian);
    }

    case FixedString: {
        int64_t len;

        if (!PyUnicode_Check(v)) {
            PyErr_SetString(PyExc_TypeError, "expected unicode object");
            return -1;
        }

        if (PyUnicode_READY(v) < 0) {
            return -1;
        }

        len = PyUnicode_GET_LENGTH(v);

        switch (t->FixedString.encoding) {
        case Ascii: {
            if (!PyUnicode_IS_ASCII(v)) {
                PyErr_SetString(PyExc_ValueError,
                    "expected ascii string");
                return -1;
            }

            if (len >= (int64_t)t->FixedString.size) {
                PyErr_Format(PyExc_ValueError,
                    "maximum string size is %" PRIi64, t->FixedString.size);
                return -1;
            }

            memcpy(x.ptr, PyUnicode_1BYTE_DATA(v), len);
            return 0;
        }

        case Utf8: {
            if (PyUnicode_KIND(v) != PyUnicode_1BYTE_KIND) {
                PyErr_SetString(PyExc_ValueError,
                    "expected utf8 string");
                return -1;
            }

            // XXX
            if (len >= (int64_t)t->FixedString.size) {
                PyErr_Format(PyExc_ValueError,
                    "maximum string size is %" PRIi64, t->FixedString.size);
                return -1;
            }

            memcpy(x.ptr, PyUnicode_1BYTE_DATA(v), len);
            return 0;
        }

        case Utf16: {
            if (PyUnicode_KIND(v) != PyUnicode_2BYTE_KIND) {
                PyErr_SetString(PyExc_ValueError,
                    "expected utf16 string");
                return -1;
            }

            if (len >= (int64_t)t->FixedString.size) {
                PyErr_Format(PyExc_ValueError,
                    "maximum string size is %" PRIi64, t->FixedString.size);
                return -1;
            }

            memcpy(x.ptr, PyUnicode_2BYTE_DATA(v), len * sizeof(uint16_t));
            return 0;
        }

        case Utf32: {
            if (PyUnicode_KIND(v) != PyUnicode_4BYTE_KIND) {
                PyErr_SetString(PyExc_ValueError,
                    "expected utf32 string");
                return -1;
            }

            if (len >= (int64_t)t->FixedString.size) {
                PyErr_Format(PyExc_ValueError,
                    "maximum string size is %" PRIi64, t->FixedString.size);
                return -1;
            }

            memcpy(x.ptr, PyUnicode_4BYTE_DATA(v), len * sizeof(uint64_t));
            return 0;
        }

        case Ucs2:
            PyErr_SetString(PyExc_NotImplementedError,
                "ucs2 encoding not implemented");
            return -1;

        default:
            PyErr_SetString(PyExc_RuntimeError, "invalid encoding");
            return -1;
        }
    }

    case FixedBytes: {
        int64_t len;

        if (!PyBytes_Check(v)) {
            PyErr_SetString(PyExc_TypeError, "expected bytes object");
            return -1;
        }

        len = PyBytes_GET_SIZE(v);

        if (len >= (int64_t)t->FixedBytes.size) {
            PyErr_Format(PyExc_ValueError,
                "maximum bytes size is %" PRIi64, t->FixedBytes.size);
            return -1;
        }

        memcpy(x.ptr, PyBytes_AS_STRING(v), len);
        return 0;
    }

    case String: {
        Py_ssize_t size;
        const char *cp;
        char*s;

        cp = PyUnicode_AsUTF8AndSize(v, &size);
        if (cp == NULL) {
            return -1;
        }

        s = ndt_strdup(cp, &ctx);
        if (s == NULL) {
            (void)seterr(&ctx);
            return -1;
        }

        XND_POINTER_DATA(x.ptr) = s;
        return 0;
    }

    case Bytes: {
        Py_ssize_t size;
        char *cp;
        char*s;

        if (PyBytes_AsStringAndSize(v, &cp, &size) < 0) {
            return -1;
        }

        s = ndt_aligned_alloc(t->Bytes.target_align, size, 1);
        if (s == NULL) {
            PyErr_NoMemory();
            return -1;
        }

        XND_BYTES_SIZE(x.ptr) = size;
        XND_BYTES_DATA(x.ptr) = s;
        return 0;
    }

    case Categorical: {
        size_t k;

        if (PyBool_Check(v)) {
            int tmp = PyObject_IsTrue(v);
            if (tmp < 0) {
                return -1;
            }

            for (k = 0; k < t->Categorical.ntypes; k++) {
                if (t->Categorical.types[k].tag == ValBool &&
                    tmp == t->Categorical.types[k].ValBool) {
                    PACK_SINGLE(x.ptr, k, size_t);
                    return 0;
                }
            }
            goto not_found;
        }

        else if (PyLong_Check(v)) {
            int64_t tmp = get_int(v, INT64_MIN, INT64_MAX);
            if (tmp == -1 && PyErr_Occurred()) {
                return -1;
            }

            for (k = 0; k < t->Categorical.ntypes; k++) {
                if (t->Categorical.types[k].tag == ValInt64 &&
                    tmp == t->Categorical.types[k].ValInt64) {
                    PACK_SINGLE(x.ptr, k, size_t);
                    return 0;
                }
            }
            goto not_found;
        }

        else if (PyFloat_Check(v)) {
            double tmp = PyFloat_AsDouble(v);
            if (tmp == -1 && PyErr_Occurred()) {
                return -1;
            }

            for (k = 0; k < t->Categorical.ntypes; k++) {
                /* XXX: DBL_EPSILON? */
                if (t->Categorical.types[k].tag == ValFloat64 &&
                    tmp == t->Categorical.types[k].ValFloat64) {
                    PACK_SINGLE(x.ptr, k, size_t);
                    return 0;
                }
            }
            goto not_found;
        }

        else if (PyUnicode_Check(v)) {
            const char *tmp = PyUnicode_AsUTF8(v);
            if (tmp == NULL) {
                return -1;
            }

            for (k = 0; k < t->Categorical.ntypes; k++) {
                if (t->Categorical.types[k].tag == ValString &&
                    strcmp(tmp, t->Categorical.types[k].ValString) == 0) {
                    PACK_SINGLE(x.ptr, k, size_t);
                    return 0;
                }
            }
            goto not_found;
        }

    not_found:
        for (k = 0; k < t->Categorical.ntypes; k++) {
            if (t->Categorical.types[k].tag == ValNA) {
                PACK_SINGLE(x.ptr, k, size_t);
                return 0;
            }
        }

        PyErr_Format(PyExc_ValueError, "category not found for: %.200R", v);
        return -1;
    }

    case Char:
        PyErr_SetString(PyExc_NotImplementedError,
            "'Char' type semantics need to be defined");
        return -1;

    case Option: case OptionItem:
        PyErr_SetString(PyExc_NotImplementedError,
            "'Option' type not implemented");
        return -1;

    case Module:
        PyErr_SetString(PyExc_NotImplementedError,
            "'Module' type not implemented");
        return -1;

    /* NOT REACHED: intercepted by ndt_is_abstract(). */
    case AnyKind: case SymbolicDim: case EllipsisDim: case Typevar:
    case ScalarKind: case SignedKind: case UnsignedKind: case FloatKind:
    case ComplexKind: case FixedStringKind: case FixedBytesKind: case Field:
    case Void: case Function:
        PyErr_SetString(PyExc_RuntimeError, "unexpected abstract type");
        return -1;
    }

    /* NOT REACHED: tags should be exhaustive */
    PyErr_SetString(PyExc_RuntimeError, "invalid type tag");
    return -1;
}

static PyObject *Ndt;
static PyObject *
pyxnd_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"value", "type", NULL};
    NDT_STATIC_CONTEXT(ctx);
    PyObject *v, *t;
    PyObject *x;
    int is_ndt;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &v, &t)) {
        return NULL;
    }

    is_ndt = PyObject_IsInstance(t, Ndt);
    if (is_ndt <= 0) {
        if (is_ndt == 0) {
            PyErr_SetString(PyExc_TypeError, "expected ndt");
        }
        return NULL;
    }

    x = pyxnd_alloc(type);
    if (x == NULL) {
        return NULL;
    }

    PTR(x) = xnd_new(NDT(t), true, &ctx);
    if (PTR(x) == NULL) {
        Py_DECREF(x);
        return seterr(&ctx);
    }

    Py_INCREF(t);
    NDT_REF(x) = t;
    TYP(x) = NDT(t);

    if (v != Py_None) {
        if (pyxnd_init(XND(x), v) < 0) {
            Py_DECREF(x);
            return NULL;
        }
    }

    return x;
}

/******************************************************************************/
/*                                 xnd methods                                */
/******************************************************************************/

static int
dict_set_item(PyObject *dict, const char *k, PyObject *value)
{
    PyObject *key;
    int ret;

    key = PyUnicode_FromString(k);
    if (key == NULL) {
        return -1;
    }

    ret = PyDict_SetItem(dict, key, value);
    Py_DECREF(key);

    return ret;
}

static PyObject *
_pyxnd_value(xnd_t x)
{
    const ndt_t *t = x.type;
    xnd_t next;

    assert(ndt_is_concrete(t));

    /* Add the linear index from var dimensions. For a chain of fixed
       dimensions, x.index is zero. */
    if (t->ndim == 0) {
        x.ptr += x.index * t->data_size;
    }

    switch (t->tag) {
    case FixedDim: {
        PyObject *lst, *v;
        int64_t shape, i;

        assert(x.index == 0);

        shape = t->FixedDim.shape;
        lst = PyList_New(shape);
        if (lst == NULL) {
            return NULL;
        }

        next.type = t->FixedDim.type;
        next.index = 0;

        for (i = 0; i < shape; i++) {
            next.ptr = x.ptr + i * t->Concrete.FixedDim.stride;
            v = _pyxnd_value(next);
            if (v == NULL) {
                Py_DECREF(lst);
                return NULL;
            }
            PyList_SET_ITEM(lst, i, v);
        }

        return lst;
    }

    case VarDim: {
        PyObject *lst, *v;
        int32_t start, stop, shape, i;

        assert(0 <= x.index && x.index+1 < t->Concrete.VarDim.noffsets);

        start = t->Concrete.VarDim.offsets[x.index];
        stop = t->Concrete.VarDim.offsets[x.index+1];
        shape = stop - start;

        lst = PyList_New(shape);
        if (lst == NULL) {
            return NULL;
        }

        next.type = t->VarDim.type;
        next.ptr = x.ptr;

        for (i = 0; i < shape; i++) {
            next.index =  start + i;
            v = _pyxnd_value(next);
            if (v == NULL) {
                Py_DECREF(lst);
                return NULL;
            }
            PyList_SET_ITEM(lst, i, v);
        }

        return lst;
    }

    case Tuple: {
        PyObject *tuple, *v;
        int64_t shape, i;

        shape = t->Tuple.shape;
        tuple = PyTuple_New(shape);
        if (tuple == NULL) {
            return NULL;
        }

        for (i = 0; i < shape; i++) {
            next.type = t->Tuple.types[i];
            next.index = 0;
            next.ptr = x.ptr + t->Concrete.Tuple.offset[i];

            v = _pyxnd_value(next);
            if (v == NULL) {
                Py_DECREF(tuple);
                return NULL;
            }
            PyTuple_SET_ITEM(tuple, i, v);
        }

        return tuple;
    }

    case Record: {
        PyObject *dict, *v;
        int64_t shape, i;
        int ret;

        shape = t->Record.shape;
        dict = PyDict_New();
        if (dict == NULL) {
            return NULL;
        }

        for (i = 0; i < shape; i++) {
            next.type = t->Record.types[i];
            next.index = 0;
            next.ptr = x.ptr + t->Concrete.Record.offset[i];

            v = _pyxnd_value(next);
            if (v == NULL) {
                Py_DECREF(dict);
                return NULL;
            }

            ret = dict_set_item(dict, t->Record.names[i], v);
            Py_DECREF(v);
            if (ret < 0) {
                Py_DECREF(dict);
                return NULL;
            }
        }

        return dict;
    }

    case Pointer: {
        next.type = t->Pointer.type;
        next.index = 0;
        next.ptr = XND_POINTER_DATA(x.ptr);
        return _pyxnd_value(next);
    }

    case Constr: {
        next.type = t->Constr.type;
        next.index = 0;
        next.ptr = x.ptr;
        return _pyxnd_value(next);
    }

    case Nominal: {
        PyErr_SetString(PyExc_NotImplementedError,
            "the 'nominal' type is opaque and only useful on the C level");
        return NULL;
    }

    case Bool: {
        bool tmp;
        UNPACK_SINGLE(tmp, x.ptr, bool);
        return PyBool_FromLong(tmp);
    }

    case Int8: {
        int8_t tmp;
        UNPACK_SINGLE(tmp, x.ptr, int8_t);
        return PyLong_FromLong(tmp);
    }

    case Int16: {
        int16_t tmp;
        UNPACK_SINGLE(tmp, x.ptr, int16_t);
        return PyLong_FromLong(tmp);
    }

    case Int32: {
        int32_t tmp;
        UNPACK_SINGLE(tmp, x.ptr, int32_t);
        return PyLong_FromLong(tmp);
    }

    case Int64: {
        int64_t tmp;
        UNPACK_SINGLE(tmp, x.ptr, int64_t);
        return PyLong_FromLongLong(tmp);
    }

    case Uint8: {
        uint8_t tmp;
        UNPACK_SINGLE(tmp, x.ptr, uint8_t);
        return PyLong_FromUnsignedLong(tmp);
    }

    case Uint16: {
        uint16_t tmp;
        UNPACK_SINGLE(tmp, x.ptr, uint16_t);
        return PyLong_FromUnsignedLong(tmp);
    }

    case Uint32: {
        uint32_t tmp;
        UNPACK_SINGLE(tmp, x.ptr, uint32_t);
        return PyLong_FromUnsignedLong(tmp);
    }

    case Uint64: {
        uint64_t tmp;
        UNPACK_SINGLE(tmp, x.ptr, uint64_t);
        return PyLong_FromUnsignedLongLong(tmp);
    }

    case Float16: {
#if PY_VERSION_HEX >= 0x03060000
        double tmp = _PyFloat_Unpack2((unsigned char *)x.ptr, little_endian);
        if (tmp == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        return PyFloat_FromDouble(tmp);
#else
        PyErr_SetString(PyExc_NotImplementedError,
            "half-float not implemented in Python versions < 3.6");
        return NULL;
#endif
    }

    case Float32: {
        double tmp = _PyFloat_Unpack4((unsigned char *)x.ptr, little_endian);
        if (tmp == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        return PyFloat_FromDouble(tmp);
    }

    case Float64: {
        double tmp = _PyFloat_Unpack8((unsigned char *)x.ptr, little_endian);
        if (tmp == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        return PyFloat_FromDouble(tmp);
    }

    case Complex32: {
#if PY_VERSION_HEX >= 0x03060000
        Py_complex c;
        c.real = _PyFloat_Unpack2((unsigned char *)x.ptr, little_endian);
        if (c.real == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        c.imag = _PyFloat_Unpack2((unsigned char *)x.ptr, little_endian);
        if (c.imag == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        return PyComplex_FromCComplex(c);
#else
        PyErr_SetString(PyExc_NotImplementedError,
            "half-float not implemented in Python versions < 3.6");
        return -1;
#endif
    }

    case Complex64: {
        Py_complex c;
        c.real = _PyFloat_Unpack4((unsigned char *)x.ptr, little_endian);
        if (c.real == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        c.imag = _PyFloat_Unpack4((unsigned char *)x.ptr, little_endian);
        if (c.imag == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        return PyComplex_FromCComplex(c);
    }

    case Complex128: {
        Py_complex c;
        c.real = _PyFloat_Unpack8((unsigned char *)x.ptr, little_endian);
        if (c.real == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        c.imag = _PyFloat_Unpack8((unsigned char *)x.ptr, little_endian);
        if (c.imag == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        return PyComplex_FromCComplex(c);
    }

    case FixedString: {
        switch (t->FixedString.encoding) {
        case Ascii:
            return PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, x.ptr,
                                             t->FixedString.size);

        case Utf8:
            return PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, x.ptr,
                                             t->FixedString.size);

        case Utf16:
            return PyUnicode_FromKindAndData(PyUnicode_2BYTE_KIND, x.ptr,
                                             t->FixedString.size/2);

        case Utf32:
            return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, x.ptr,
                                             t->FixedString.size/4);

        case Ucs2:
            PyErr_SetString(PyExc_NotImplementedError,
                "ucs2 encoding not implemented");
            return NULL;

        default:
            PyErr_SetString(PyExc_RuntimeError, "invalid encoding");
            return NULL;
        }
    }

    case FixedBytes: {
        return PyBytes_FromStringAndSize(x.ptr, t->FixedBytes.size);
    }

    case String: {
        return PyUnicode_FromString(XND_POINTER_DATA(x.ptr));
    }

    case Bytes: {
        return PyBytes_FromStringAndSize(XND_BYTES_DATA(x.ptr), XND_BYTES_SIZE(x.ptr));
    }

    case Categorical: {
        size_t k;

        UNPACK_SINGLE(k, x.ptr, size_t);

        switch (t->Categorical.types[k].tag) {
        case ValBool: {
            bool tmp = t->Categorical.types[k].ValBool;
            return PyBool_FromLong(tmp);
        }

        case ValInt64: {
            int64_t tmp = t->Categorical.types[k].ValInt64;
            return PyLong_FromLongLong(tmp);
        }

        case ValFloat64: {
            double tmp = t->Categorical.types[k].ValFloat64;
            return PyFloat_FromDouble(tmp);
        }

        case ValString: {
            const char *tmp = t->Categorical.types[k].ValString;
            return PyUnicode_FromString(tmp);
        }

        case ValNA: {
            Py_RETURN_NONE;
        }

        }

        PyErr_SetString(PyExc_RuntimeError, "unexpected category tag");
        return NULL;
    }

    case Char:
        PyErr_SetString(PyExc_NotImplementedError,
            "'Char' type semantics need to be defined");
        return NULL;

    case Option: case OptionItem:
        PyErr_SetString(PyExc_NotImplementedError,
            "'Option' type not implemented");
        return NULL;

    case Module:
        PyErr_SetString(PyExc_NotImplementedError,
            "'Module' type not implemented");
        return NULL;

    /* NOT REACHED: intercepted by ndt_is_abstract(). */
    case AnyKind: case SymbolicDim: case EllipsisDim: case Typevar:
    case ScalarKind: case SignedKind: case UnsignedKind: case FloatKind:
    case ComplexKind: case FixedStringKind: case FixedBytesKind: case Field:
    case Void: case Function:
        PyErr_SetString(PyExc_RuntimeError, "unexpected abstract type");
        return NULL;
    }

    /* NOT REACHED: tags should be exhaustive */
    PyErr_SetString(PyExc_RuntimeError, "invalid type tag");
    return NULL;
}

static PyObject *
pyxnd_type(PyObject *xnd, PyObject *args UNUSED)
{
    Py_INCREF(NDT_REF(xnd));
    return NDT_REF(xnd);
}

static PyObject *
pyxnd_value(PyObject *xnd, PyObject *args UNUSED)
{
    return _pyxnd_value(XND(xnd));
}

static PyGetSetDef pyxnd_getsets [] =
{
  { "type", (getter)pyxnd_type, NULL, NULL, NULL},
  { "value", (getter)pyxnd_value, NULL, NULL, NULL},
  {NULL}
};


static PyMethodDef pyxnd_methods [] =
{
  { NULL, NULL, 1 }
};

static PyTypeObject Xnd_Type =
{
    PyVarObject_HEAD_INIT(NULL, 0)
    "xnd._xnd",                             /* tp_name */
    sizeof(XndObject),                      /* tp_basicsize */
    0,                                      /* tp_itemsize */
    (destructor) pyxnd_dealloc,             /* tp_dealloc */
    0,                                      /* tp_print */
    (getattrfunc) 0,                        /* tp_getattr */
    (setattrfunc) 0,                        /* tp_setattr */
    0,                                      /* tp_reserved */
    (reprfunc) 0,                           /* tp_repr */
    0,                                      /* tp_as_number */
    0,                                      /* tp_as_sequence */
    0,                                      /* tp_as_mapping */
    0,                                      /* tp_hash */
    0,                                      /* tp_call */
    (reprfunc) 0,                           /* tp_str */
    (getattrofunc) PyObject_GenericGetAttr, /* tp_getattro */
    (setattrofunc) 0,                       /* tp_setattro */
    (PyBufferProcs *) 0,                    /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT|
     Py_TPFLAGS_BASETYPE),                  /* tp_flags */
    0, // xnd_doc,                          /* tp_doc */
    0,                                      /* tp_traverse */
    0,                                      /* tp_clear */
    0,                                      /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    0,                                      /* tp_iter */
    0,                                      /* tp_iternext */
    pyxnd_methods,                          /* tp_methods */
    0,                                      /* tp_members */
    pyxnd_getsets,                          /* tp_getset */
    0,                                      /* tp_base */
    0,                                      /* tp_dict */
    0,                                      /* tp_descr_get */
    0,                                      /* tp_descr_set */
    0,                                      /* tp_dictoffset */
    0,                                      /* tp_init */
    0,                                      /* tp_alloc */
    pyxnd_new,                              /* tp_new */
    PyObject_Del,                           /* tp_free */
};

static struct PyModuleDef xnd_module = {
    PyModuleDef_HEAD_INIT,        /* m_base */
    "_xnd",                       /* m_name */
    NULL,                         /* m_doc */
    -1,                           /* m_size */
    NULL,                         /* m_methods */
    NULL,                         /* m_slots */
    NULL,                         /* m_traverse */
    NULL,                         /* m_clear */
    NULL                          /* m_free */
};


PyMODINIT_FUNC
PyInit__xnd(void)
{
    PyObject *m = NULL;
    PyObject *ndtypes = NULL;

    Xnd_Type.tp_base = &PyBaseObject_Type;
    if (PyType_Ready(&Xnd_Type) < 0) {
        goto error;
    }

    ndtypes = PyImport_ImportModule("ndtypes");
    if (ndtypes == NULL) {
        goto error;
    }
    Ndt = PyObject_GetAttrString(ndtypes, "ndt");
    Py_CLEAR(ndtypes);
    if (Ndt == NULL) {
        goto error;
    }

    m = PyModule_Create(&xnd_module);
    if (m == NULL) {
        goto error;
    }

    Py_INCREF(&Xnd_Type);
    if (PyModule_AddObject(m, "_xnd", (PyObject *)&Xnd_Type) < 0) {
        goto error;
    }

    return m;

error:
    Py_CLEAR(m);
    return NULL;
}



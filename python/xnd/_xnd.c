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
#include "pyndtypes.h"
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

#if PY_LITTLE_ENDIAN
static const int little_endian = 1;
#else
static const int little_endian = 0;
#endif


/****************************************************************************/
/*                               Error handling                             */
/****************************************************************************/

static PyObject *
seterr(ndt_context_t *ctx)
{
    return Ndt_SetError(ctx);
}


/****************************************************************************/
/*                           MemoryBlock Object                             */
/****************************************************************************/

/* This object owns the memory that is shared by several xnd objects. It is
   never exposed to the Python level.

   The memory block is created by the primary xnd object on initialization.
   Sub-views, slices etc. share the memory block.

   At a later stage, the object will potentially need to communicate with
   Arrow or other formats in order to acquire and manage external memory
   blocks. */


typedef struct {
    PyObject_HEAD
    PyObject *type;    /* type owner */
    xnd_master_t *xnd; /* memblock owner */
} MemoryBlockObject;

static int mblock_init(xnd_t x, PyObject *v);
static PyTypeObject MemoryBlock_Type;


static MemoryBlockObject *
mblock_alloc(void)
{
    MemoryBlockObject *self;

    self = PyObject_GC_New(MemoryBlockObject, &MemoryBlock_Type);
    if (self == NULL) {
        return NULL;
    }
 
    self->type = NULL;
    self->xnd = NULL;

    PyObject_GC_Track(self);
    return self;
}

static int
mblock_traverse(MemoryBlockObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->type);
    return 0;
}

static void
mblock_dealloc(MemoryBlockObject *self)
{
    PyObject_GC_UnTrack(self);
    xnd_del(self->xnd);
    Py_CLEAR(self->type);
    PyObject_GC_Del(self);
}

static MemoryBlockObject *
mblock_from_typed_value(PyObject *type, PyObject *value)
{
    NDT_STATIC_CONTEXT(ctx);
    MemoryBlockObject *self;

    if (!Ndt_Check(type)) {
        PyErr_SetString(PyExc_TypeError, "expected ndt object");
        return NULL;
    }

    self = mblock_alloc();
    if (self == NULL) {
        return NULL;
    }

    self->xnd = xnd_empty_from_type(CONST_NDT(type), XND_OWN_EMBEDDED, &ctx);
    if (self->xnd == NULL) {
        Py_DECREF(self);
        seterr(&ctx);
        return NULL;
    }
    Py_INCREF(type);
    self->type = type;

    if (value != Py_None) {
        if (mblock_init(self->xnd->master, value) < 0) {
            Py_DECREF(self);
            return NULL;
        }
    }

    return self;
}

static PyTypeObject MemoryBlock_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_xnd.memblock",
    sizeof(MemoryBlockObject),
    0,
    (destructor)mblock_dealloc,              /* tp_dealloc */
    0,                                       /* tp_print */
    0,                                       /* tp_getattr */
    0,                                       /* tp_setattr */
    0,                                       /* tp_reserved */
    0,                                       /* tp_repr */
    0,                                       /* tp_as_number */
    0,                                       /* tp_as_sequence */
    0,                                       /* tp_as_mapping */
    0,                                       /* tp_hash */
    0,                                       /* tp_call */
    0,                                       /* tp_str */
    PyObject_GenericGetAttr,                 /* tp_getattro */
    0,                                       /* tp_setattro */
    0,                                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags */
    NULL,                                    /* tp_doc */
    (traverseproc)mblock_traverse,           /* tp_traverse */
    NULL,                                    /* tp_clear */
};


/****************************************************************************/
/*                      MemoryBlock Object Initialization                   */
/****************************************************************************/

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
mblock_init(xnd_t x, PyObject *v)
{
    NDT_STATIC_CONTEXT(ctx);
    const ndt_t *t = x.type;
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
        int64_t shape, i;

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

        assert(x.index == 0);
        next.index = x.index;
        next.type = t->FixedDim.type;

        for (i = 0; i < shape; i++) {
            next.ptr = x.ptr + i * t->Concrete.FixedDim.stride;
            if (mblock_init(next, PyList_GET_ITEM(v, i)) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case VarDim: {
        const int32_t noffsets = t->Concrete.VarDim.noffsets;
        int32_t start, stop;
        int64_t shape, i;

        if (!PyList_Check(v)) {
            PyErr_Format(PyExc_TypeError,
                "xnd: expected list, not '%.200s'", Py_TYPE(v)->tp_name);
            return -1;
        }

        if (x.index < 0 || x.index+1 >= noffsets) {
            PyErr_Format(PyExc_RuntimeError,
                "xnd: offset index out of range: index=%" PRIi32 ", noffsets=%" PRIi32,
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
            if (mblock_init(next, PyList_GET_ITEM(v, i)) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case Tuple: {
        int64_t shape, i;

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

        next.index = 0;

        for (i = 0; i < shape; i++) {
            next.type = t->Tuple.types[i];
            next.ptr = x.ptr + t->Concrete.Tuple.offset[i];

            if (mblock_init(next, PyTuple_GET_ITEM(v, i)) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case Record: {
        PyObject *tmp;
        int64_t shape, i;
        int ret;

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

        next.index = 0;

        for (i = 0; i < shape; i++) {
            next.type = t->Record.types[i];
            next.ptr = x.ptr + t->Concrete.Record.offset[i];

            tmp = PyMapping_GetItemString(v, t->Record.names[i]);
            if (tmp == NULL) {
                if (!PyErr_Occurred()) {
                    PyErr_Format(PyExc_ValueError,
                        "xnd: key not found %s", t->Record.names[i]);
                }
                return -1;
            }

            ret = mblock_init(next, tmp);
            Py_DECREF(tmp);
            if (ret < 0) {
                return -1;
            }
        }

        return 0;
    }

    case Pointer: {
        next.index = 0;
        next.type = t->Pointer.type;
        next.ptr = XND_POINTER_DATA(x.ptr);
        return mblock_init(next, v);
    }

    case Constr: {
        next.index = 0;
        next.type = t->Constr.type;
        next.ptr = x.ptr;
        return mblock_init(next, v);
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
        char *s;

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
        char *s;

        if (PyBytes_AsStringAndSize(v, &cp, &size) < 0) {
            return -1;
        }

        s = ndt_aligned_calloc(t->Bytes.target_align, size);
        if (s == NULL) {
            PyErr_NoMemory();
            return -1;
        }

        memcpy(s, cp, size);

        XND_BYTES_SIZE(x.ptr) = size;
        XND_BYTES_DATA(x.ptr) = (uint8_t *)s;
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


/****************************************************************************/
/*                                 xnd object                               */
/****************************************************************************/

typedef struct {
    PyObject_HEAD
    MemoryBlockObject *mblock; /* owner of the primary type and memory block */
    PyObject *type;            /* owner of the current type */
    xnd_t xnd;                 /* typed view, does not own anything */
} XndObject;

static PyTypeObject Xnd_Type;
#define Xnd_CheckExact(v) (Py_TYPE(v) == &Xnd_Type)
#define Xnd_Check(v) PyObject_TypeCheck(v, &Xnd_Type)

#define TYPE_OWNER(v) (((XndObject *)v)->type)
#define XND(v) (((XndObject *)v)->xnd)
#define XND_INDEX(v) (((XndObject *)v)->xnd.index)
#define XND_TYPE(v) (((XndObject *)v)->xnd.type)
#define XND_PTR(v) (((XndObject *)v)->xnd.ptr)


static XndObject *
pyxnd_alloc(PyTypeObject *type)
{
    XndObject *self;

    self = (XndObject *)type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
 
    self->mblock = NULL;
    self->type = NULL;
    self->xnd.index = 0;
    self->xnd.type  = NULL;
    self->xnd.ptr = NULL;

    return self;
}

static int
pyxnd_traverse(XndObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->mblock);
    Py_VISIT(self->type);
    return 0;
}

static void
pyxnd_dealloc(XndObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->mblock);
    Py_CLEAR(self->type);
    Py_TYPE(self)->tp_free(self);
}

static PyObject *
pyxnd_new(PyTypeObject *tp, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"type", "value", NULL};
    PyObject *type, *value = Py_None;
    MemoryBlockObject *mblock;
    XndObject *self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &type,
        &value)) {
        return NULL;
    }

    mblock = mblock_from_typed_value(type, value);
    if (mblock == NULL) {
        return NULL;
    }

    self = pyxnd_alloc(tp);
    if (self == NULL) {
        Py_DECREF(mblock);
        return NULL;
    }

    Py_INCREF(mblock->type);
    self->mblock = mblock;
    self->type = mblock->type;

    self->xnd.index = 0;
    self->xnd.type = mblock->xnd->master.type;
    self->xnd.ptr = mblock->xnd->master.ptr;

    return (PyObject *)self;
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

        shape = t->FixedDim.shape;
        lst = PyList_New(shape);
        if (lst == NULL) {
            return NULL;
        }

        assert(x.index == 0);
        next.index = x.index;
        next.type = t->FixedDim.type;

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

        next.index = 0;

        for (i = 0; i < shape; i++) {
            next.type = t->Tuple.types[i];
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

        next.index = 0;

        for (i = 0; i < shape; i++) {
            next.type = t->Record.types[i];
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
        next.index = 0;
        next.type = t->Pointer.type;
        next.ptr = XND_POINTER_DATA(x.ptr);
        return _pyxnd_value(next);
    }

    case Constr: {
        next.index = 0;
        next.type = t->Constr.type;
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
        return NULL;
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
        return PyBytes_FromStringAndSize((char *)XND_BYTES_DATA(x.ptr),
                                         XND_BYTES_SIZE(x.ptr));
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


/******************************************************************************/
/*                                xnd subscript                               */
/******************************************************************************/

static PyObject *
pyxnd_view_copy_type(const XndObject *src, const xnd_t x)
{
    XndObject *view;
    PyObject *type;

    type = Ndt_CopySubtree(src->type, x.type);
    if (type == NULL) {
        return NULL;
    }

    view = pyxnd_alloc(Py_TYPE(src));
    if (view == NULL) {
        Py_DECREF(type);
        return NULL;
    }

    Py_INCREF(src->mblock);
    view->mblock = src->mblock;
    view->type = type;
    view->xnd = x;
    view->xnd.type = CONST_NDT(type);

    return (PyObject *)view;
}

static PyObject *
pyxnd_view_move_type(const XndObject *src, xnd_t x)
{
    XndObject *view;
    PyObject *type;

    type = Ndt_MoveSubtree(src->type, (ndt_t *)x.type);
    if (type == NULL) {
        return NULL;
    }

    view = pyxnd_alloc(Py_TYPE(src));
    if (view == NULL) {
        Py_DECREF(type);
        return NULL;
    }

    Py_INCREF(src->mblock);
    view->mblock = src->mblock;
    view->type = type;
    view->xnd = x;

    return (PyObject *)view;
}

static Py_ssize_t
get_index(PyObject *key, Py_ssize_t shape)
{
    Py_ssize_t i = PyNumber_AsSsize_t(key, PyExc_IndexError);
    if (i == -1 && PyErr_Occurred()) {
        return -1;
    }

    if (i < 0) {
        i += shape;
    }

    if (i < 0 || i >= shape) {
        PyErr_SetString(PyExc_IndexError, "index out of bounds");
        return -1;
    }

    return i;
}

static Py_ssize_t
get_index_record(const ndt_t *t, PyObject *key)
{
    assert(t->tag == Record);

    if (PyUnicode_Check(key)) {
        Py_ssize_t i;

        for (i = 0; i < t->Record.shape; i++) {
            if (PyUnicode_CompareWithASCIIString(
                    key, t->Record.names[i]) == 0) {
                return i;
            }
        }

        PyErr_SetString(PyExc_KeyError, "key not found");
        return -1;
    }

    return get_index(key, t->Record.shape);
}

/*
 * Return a zero copy view of an xnd object.  If a dtype is indexable,
 * descend into the dtype.
 */
static xnd_t
pyxnd_subtree(xnd_t x, PyObject *indices[], int len)
{
    const ndt_t *t = x.type;
    xnd_t next;
    PyObject *key;

    assert(ndt_is_concrete(t));

    /* Add the linear index from var dimensions. For a chain of fixed
       dimensions, x.index is zero. */
    if (t->ndim == 0) {
        x.ptr += x.index * t->data_size;
    }

    if (len == 0) {
        if (ndt_is_optional(t)) {
            PyErr_SetString(PyExc_NotImplementedError,
                "option type is temporarily disabled");
            return xnd_error;
        }
        return x;
    }

    key = indices[0];

    switch (t->tag) {
    case FixedDim: {
        Py_ssize_t i = get_index(key, t->FixedDim.shape);
        if (i < 0) {
            return xnd_error;
        }

        next.index = x.index;
        next.type = t->FixedDim.type;
        next.ptr = x.ptr + i * t->Concrete.FixedDim.stride;

        break;
    }

    case VarDim: {
        int32_t start, stop, shape;
        Py_ssize_t i;

        if (ndt_is_optional(t)) {
            PyErr_SetString(PyExc_NotImplementedError,
                "optional dimensions are temporarily disabled");
            return xnd_error;
        }

        assert(0 <= x.index && x.index+1 < t->Concrete.VarDim.noffsets);

        start = t->Concrete.VarDim.offsets[x.index];
        stop = t->Concrete.VarDim.offsets[x.index+1];
        shape = stop - start;

        i = get_index(key, shape);
        if (i < 0) {
            return xnd_error;
        }

        next.index = start + i;
        next.type = t->VarDim.type;
        next.ptr = x.ptr;

        break;
    }

    case Tuple: {
        Py_ssize_t i = get_index(key, t->Tuple.shape);
        if (i < 0) {
            return xnd_error;
        }

        next.index = 0;
        next.type = t->Tuple.types[i];
        next.ptr = x.ptr + t->Concrete.Tuple.offset[i];

        break;
    }

    case Record: {
        Py_ssize_t i = get_index_record(t, key);
        if (i < 0) {
            return xnd_error;
        }

        next.index = 0;
        next.type = t->Record.types[i];
        next.ptr = x.ptr + t->Concrete.Record.offset[i];

        break;
    }

    default:
        PyErr_SetString(PyExc_IndexError, "type not indexable");
        return xnd_error;
    }

    return pyxnd_subtree(next, indices+1, len-1);
}

static xnd_t pyxnd_index(xnd_t x, PyObject *indices[], int len);
static xnd_t pyxnd_slice(xnd_t x, PyObject *indices[], int len);

static xnd_t
pyxnd_multikey(xnd_t x, PyObject *indices[], int len)
{
    NDT_STATIC_CONTEXT(ctx);
    const ndt_t *t = x.type;
    PyObject *key;
    xnd_t next;

    assert(len >= 0);
    assert(ndt_is_concrete(t));
    assert(x.ptr != NULL);

    if (len > t->ndim) {
        PyErr_SetString(PyExc_IndexError, "too many indices");
        return xnd_error;
    }

    if (len == 0) {
        if (ndt_is_optional(t)) {
            PyErr_SetString(PyExc_NotImplementedError,
                "option type is temporarily disabled");
            return xnd_error;
        }
        next.index = x.index;
        next.type = ndt_copy(t, &ctx);
        if (next.type == NULL) {
            seterr(&ctx);
            return xnd_error;
        }
        next.ptr = x.ptr;

        return next;
    }

    key = indices[0];

    if (PyIndex_Check(key)) {
        return pyxnd_index(x, indices, len);
    }
    else if (PySlice_Check(key)) {
        return pyxnd_slice(x, indices, len);
    }

    PyErr_SetString(PyExc_RuntimeError,
        "multikey: internal error: key must be index or slice");

    return xnd_error;
}

/*
 * Return a view with a copy of the type.  Indexing into the dtype is
 * not permitted.
 */
static xnd_t
pyxnd_index(xnd_t x, PyObject *indices[], int len)
{
    const ndt_t *t = x.type;
    xnd_t next;
    PyObject *key;

    assert(len > 0);
    assert(PyIndex_Check(indices[0]));
    assert(ndt_is_concrete(t));
    assert(x.ptr != NULL);

    /* Add the linear index from var dimensions. For a chain of fixed
       dimensions, x.index is zero. */
    if (t->ndim == 0) {
        x.ptr += x.index * t->data_size;
    }

    key = indices[0];

    switch (t->tag) {
    case FixedDim: {
        Py_ssize_t i = get_index(key, t->FixedDim.shape);
        if (i < 0) {
            return xnd_error;
        }

        next.index = x.index;
        next.type = t->FixedDim.type;
        next.ptr = x.ptr + i * t->Concrete.FixedDim.stride;

        break;
    }

    case VarDim: {
        int32_t start, stop, shape;
        Py_ssize_t i;

        if (ndt_is_optional(t)) {
            PyErr_SetString(PyExc_NotImplementedError,
                "optional dimensions are temporarily disabled");
            return xnd_error;
        }

        assert(0 <= x.index && x.index+1 < t->Concrete.VarDim.noffsets);

        start = t->Concrete.VarDim.offsets[x.index];
        stop = t->Concrete.VarDim.offsets[x.index+1];
        shape = stop - start;

        i = get_index(key, shape);
        if (i < 0) {
            return xnd_error;
        }

        next.index = start + i;
        next.type = t->VarDim.type;
        next.ptr = x.ptr;

        break;
    }

    default:
        PyErr_SetString(PyExc_IndexError,
            "cannot index the dtype or a non-indexable type");
        return xnd_error;
    }

    return pyxnd_multikey(next, indices+1, len-1);
}

static xnd_t
pyxnd_slice(xnd_t x, PyObject *indices[], int len)
{
    NDT_STATIC_CONTEXT(ctx);
    const ndt_t *t = x.type;
    xnd_t next;
    PyObject *key;

    assert(len > 0);
    assert(PySlice_Check(indices[0]));
    assert(ndt_is_concrete(t));
    assert(x.ptr != NULL);

    key = indices[0];

    switch (t->tag) {
    case FixedDim: {
        Py_ssize_t start, stop, step, shape;
        if (PySlice_GetIndicesEx(key, t->FixedDim.shape,
                                  &start, &stop, &step, &shape) < 0) {
            return xnd_error;
        }

        next.index = x.index;
        next.type = t->FixedDim.type;
        next.ptr = x.ptr + start * t->Concrete.FixedDim.stride;

        next = pyxnd_multikey(next, indices+1, len-1);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        x.index = next.index;
        x.type = ndt_fixed_dim((ndt_t *)next.type, shape,
                               t->Concrete.FixedDim.stride * step,
                               ndt_order(t), &ctx);
        if (x.type == NULL) {
            seterr(&ctx);
            return xnd_error;
        }
        x.ptr = next.ptr;

        return x;
    }

    case VarDim: {
        PyErr_SetString(PyExc_NotImplementedError,
            "slicing var dimensions is not implemented");
        return xnd_error;
    }

    case Tuple: {
        PyErr_SetString(PyExc_NotImplementedError,
            "slicing tuples is not supported");
        return xnd_error;
    }

    case Record: {
        PyErr_SetString(PyExc_NotImplementedError,
            "slicing records is not supported");
        return xnd_error;
    }

    default:
        PyErr_SetString(PyExc_IndexError, "type not sliceable");
        return xnd_error;
    }
}

static int
is_multiindex(const PyObject *key)
{
    Py_ssize_t size, i;

    if (!PyTuple_Check(key)) {
        return 0;
    }

    size = PyTuple_GET_SIZE(key);
    for (i = 0; i < size; i++) {
        PyObject *x = PyTuple_GET_ITEM(key, i);
        if (!PyIndex_Check(x)) {
            return 0;
        }
    }

    return 1;
}

static int
is_multikey(const PyObject *key)
{
    Py_ssize_t size, i;

    if (!PyTuple_Check(key)) {
        return 0;
    }

    size = PyTuple_GET_SIZE(key);
    for (i = 0; i < size; i++) {
        PyObject *x = PyTuple_GET_ITEM(key, i);
        if (!PyIndex_Check(x) && !PySlice_Check(x)) {
            return 0;
        }
    }

    return 1;
}

static PyObject *
value_or_view_copy(XndObject *self, xnd_t x)
{
    if (x.ptr == NULL) {
        return NULL;
    }

    if (x.type->ndim == 0) {
        return _pyxnd_value(x);
    }

    return pyxnd_view_copy_type(self, x);
}

static PyObject *
value_or_view_move(XndObject *self, xnd_t x)
{
    if (x.ptr == NULL) {
        return NULL;
    }

    if (x.type->ndim == 0) {
        return _pyxnd_value(x);
    }

    return pyxnd_view_move_type(self, x);
}

static PyObject *
pyxnd_subscript(XndObject *self, PyObject *key)
{
    xnd_t x;

    if (PyIndex_Check(key)) {
        PyObject *indices[1] = {key};
        x = pyxnd_subtree(self->xnd, indices, 1);
        return value_or_view_copy(self, x);
    }
    else if (is_multiindex(key)) {
        PyObject **indices = &PyTuple_GET_ITEM(key, 0);
        x = pyxnd_subtree(self->xnd, indices, PyTuple_GET_SIZE(key));
        return value_or_view_copy(self, x);
    }
    else if (PySlice_Check(key)) {
        PyObject *indices[1] = {key};
        if (self->xnd.type->tag == FixedDim) {
            x = pyxnd_multikey(self->xnd, indices, 1);
            return value_or_view_move(self, x);
        }
        goto not_implemented_or_type_error;
    }
    else if (is_multikey(key)) {
        PyObject **indices = &PyTuple_GET_ITEM(key, 0);
        Py_ssize_t n = PyTuple_GET_SIZE(key);
        if (self->xnd.type->tag == FixedDim) {
            x = pyxnd_multikey(self->xnd, indices, n);
            return value_or_view_move(self, x);
        }
        goto not_implemented_or_type_error;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "invalid subscript key");
        return NULL;
    }

not_implemented_or_type_error:
    if (self->xnd.type->tag == VarDim) {
        PyErr_SetString(PyExc_NotImplementedError,
            "slicing is not implemented for variable dimensions");
        return NULL;
    }

    PyErr_SetString(PyExc_TypeError, "type does not support slicing");
    return NULL;
}

static PyObject *
pyxnd_type(PyObject *self, PyObject *args UNUSED)
{
    Py_INCREF(TYPE_OWNER(self));
    return TYPE_OWNER(self);
}

static PyObject *
pyxnd_ndim(PyObject *self, PyObject *args UNUSED)
{
    int ndim = XND_TYPE(self)->ndim;
    return PyLong_FromLong(ndim);
}

static PyObject *
pyxnd_value(PyObject *self, PyObject *args UNUSED)
{
    return _pyxnd_value(XND(self));
}

static PyObject *
pyxnd_align(PyObject *self, PyObject *args UNUSED)
{
    uint16_t align = XND_TYPE(self)->data_align;
    return PyLong_FromUnsignedLong(align);
}


static PyGetSetDef pyxnd_getsets [] =
{
  { "type", (getter)pyxnd_type, NULL, NULL, NULL},
  { "value", (getter)pyxnd_value, NULL, NULL, NULL},
  { "align", (getter)pyxnd_align, NULL, NULL, NULL},
  { "ndim", (getter)pyxnd_ndim, NULL, NULL, NULL},
  {NULL}
};

static PyMappingMethods pyxnd_as_mapping = {
    (lenfunc)NULL,                 /* mp_length */
    (binaryfunc)pyxnd_subscript,   /* mp_subscript */
    (objobjargproc)NULL,           /* mp_ass_subscript */
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
    &pyxnd_as_mapping,                      /* tp_as_mapping */
    0,                                      /* tp_hash */
    0,                                      /* tp_call */
    (reprfunc) 0,                           /* tp_str */
    (getattrofunc) PyObject_GenericGetAttr, /* tp_getattro */
    (setattrofunc) 0,                       /* tp_setattro */
    (PyBufferProcs *) 0,                    /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT|
     Py_TPFLAGS_BASETYPE|
     Py_TPFLAGS_HAVE_GC),                   /* tp_flags */
    0,                                      /* tp_doc */
    (traverseproc)pyxnd_traverse,           /* tp_traverse */
    NULL,                                   /* tp_clear */
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
    PyType_GenericAlloc,                    /* tp_alloc */
    pyxnd_new,                              /* tp_new */
    PyObject_GC_Del,                        /* tp_free */
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
    static int initialized = 0;

    if (!initialized) {
        if (import_ndtypes() < 0) {
            return NULL;
        }
        initialized = 1;
    }

    if (PyType_Ready(&MemoryBlock_Type) < 0) {
        return NULL;
    }

    Xnd_Type.tp_base = &PyBaseObject_Type;
    if (PyType_Ready(&Xnd_Type) < 0) {
        return NULL;
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



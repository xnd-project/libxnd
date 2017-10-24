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
#define PTR(v) (((XndObject *)v)->xnd.ptr)


#if PY_LITTLE_ENDIAN
static const int litte_endian = 1;
#else
static const int litte_endian = 0;
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
    PTR(x) = NULL;
    NDT_REF(x) = NULL;

    return (PyObject *)x;
}

static void
pyxnd_dealloc(PyObject *x)
{
    ndt_free(PTR(x));
    Py_CLEAR(NDT_REF(x));

    Py_TYPE(x)->tp_free(x);
}


#define PACK_SINGLE(ptr, src, type) \
    do {                                     \
        type _x;                             \
        _x = (type)src;                      \
        memcpy(ptr, (char *)&_x, sizeof _x); \
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
pyxnd_init(const xnd_t x, PyObject *v)
{
    NDT_STATIC_CONTEXT(ctx);
    const ndt_t *t = x.type;
    int64_t shape, i;
    xnd_t next;

    if (ndt_is_abstract(t)) {
        PyErr_SetString(PyExc_TypeError, "xnd has abstract type");
        return -1;
    }

    switch (t->tag) {
    case FixedDim: {
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
        for (i = 0; i < shape; i++) {
            next.ptr = x.ptr + i * t->Concrete.FixedDim.stride;
            if (pyxnd_init(next, PyList_GET_ITEM(v, i)) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case VarDim: 
        PyErr_Format(PyExc_NotImplementedError,
            "xnd: initialization from var dim not implemented");
        return -1;

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
        return _PyFloat_Pack2(tmp, (unsigned char *)x.ptr, litte_endian);
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
        return _PyFloat_Pack4(tmp, (unsigned char *)x.ptr, litte_endian);
    }

    case Float64: {
        double tmp = PyFloat_AsDouble(v);
        if (tmp == -1 && PyErr_Occurred()) {
            return -1;
        }
        return _PyFloat_Pack8(tmp, (unsigned char *)x.ptr, litte_endian);
    }

    case Complex32: {
#if PY_VERSION_HEX >= 0x03060000
        Py_complex c = PyComplex_AsCComplex(v);
        if (c.real == -1.0 && PyErr_Occurred()) {
            return -1;
        }
        if (_PyFloat_Pack2(c.real, (unsigned char *)x.ptr, litte_endian) < 0) {
            return -1;
        }
        return _PyFloat_Pack2(c.imag, (unsigned char *)(x.ptr+2), litte_endian);
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
        if (_PyFloat_Pack4(c.real, (unsigned char *)x.ptr, litte_endian) < 0) {
            return -1;
        }
        return _PyFloat_Pack4(c.imag, (unsigned char *)(x.ptr+4), litte_endian);
    }

    case Complex128: {
        Py_complex c = PyComplex_AsCComplex(v);
        if (c.real == -1.0 && PyErr_Occurred()) {
            return -1;
        }
        if (_PyFloat_Pack8(c.real, (unsigned char *)x.ptr, litte_endian) < 0) {
            return -1;
        }
        return _PyFloat_Pack8(c.imag, (unsigned char *)(x.ptr+8), litte_endian);
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

        s = ndt_aligned_alloc(x.type->Bytes.target_align, size, 1);
        if (s == NULL) {
            PyErr_NoMemory();
            return -1;
        }

        XND_BYTES_SIZE(x.ptr) = size;
        XND_BYTES_DATA(x.ptr) = s;
        return 0;
    }

#if 0
    case Char:
        return 0;
#endif

    default:
        PyErr_Format(PyExc_NotImplementedError,
            "packing type '%s' not implemented", ndt_tag_as_string(t->tag));
        return -1;
    }
}

static PyObject *Ndt;
static PyObject *
pyxnd_new(PyTypeObject *type, PyObject *args, PyObject *kwds UNUSED)
{
    NDT_STATIC_CONTEXT(ctx);
    PyObject *v = NULL;
    PyObject *x, *t;
    int is_ndt;

    if (!PyArg_ParseTuple(args, "O|O", &t, &v)) {
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

    PTR(x) = xnd_new(NDT(t), false, &ctx);
    if (PTR(x) == NULL) {
        return seterr(&ctx);
    }

    Py_INCREF(t);

    NDT_REF(x) = t;
    TYP(x) = NDT(t);

    if (v && pyxnd_init(XND(x), v) < 0) {
        Py_DECREF(x);
        return NULL;
    }

    return x;
}

/******************************************************************************/
/*                                 Ndt methods                                */
/******************************************************************************/

static PyMethodDef pyxnd_methods [] =
{
  { NULL, NULL, 1 }
};

static PyTypeObject Xnd_Type =
{
    PyVarObject_HEAD_INIT(NULL, 0)
    "xnd.xnd",                              /* tp_name */
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
    0,                                      /* tp_getset */
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
    if (PyModule_AddObject(m, "xnd", (PyObject *)&Xnd_Type) < 0) {
        goto error;
    }

    return m;

error:
    Py_CLEAR(m);
    return NULL;
}



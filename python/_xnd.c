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
#include <stdlib.h>
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
#define TYP(v) (((XndObject *)v)->xnd.type)
#define PTR(v) (((XndObject *)v)->xnd.ptr)


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

static PyObject *Ndt;
static PyObject *
pyxnd_new(PyTypeObject *type, PyObject *args, PyObject *kwds UNUSED)
{
    NDT_STATIC_CONTEXT(ctx);
    PyObject *x;
    PyObject *v;
    int is_ndt;

    if (!PyArg_ParseTuple(args, "O", &v)) {
        return NULL;
    }

    is_ndt = PyObject_IsInstance(v, Ndt);
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

    PTR(x) = xnd_new(NDT(v), false, &ctx);
    if (PTR(x) == NULL) {
        return seterr(&ctx);
    }

    Py_INCREF(v);

    NDT_REF(x) = v;
    TYP(x) = NDT(v);

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



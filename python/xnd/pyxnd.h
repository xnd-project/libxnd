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


#ifndef PYXND_H
#define PYXND_H
#ifdef __cplusplus
extern "C" {
#endif


#include "ndtypes.h"
#include "xnd.h"


#define Xnd_CheckExact_INDEX 0
#define Xnd_CheckExact_RETURN int
#define Xnd_CheckExact_ARGS (const PyObject *)

#define Xnd_Check_INDEX 1
#define Xnd_Check_RETURN int
#define Xnd_Check_ARGS (const PyObject *)

#define CONST_XND_INDEX 2
#define CONST_XND_RETURN const xnd_t *
#define CONST_XND_ARGS (const PyObject *)

#define Xnd_EmptyFromType_INDEX 3
#define Xnd_EmptyFromType_RETURN PyObject *
#define Xnd_EmptyFromType_ARGS (PyTypeObject *, ndt_t *t)

#define Xnd_ViewMoveNdt_INDEX 4
#define Xnd_ViewMoveNdt_RETURN PyObject *
#define Xnd_ViewMoveNdt_ARGS (const PyObject *, ndt_t *t)

#define Xnd_FromXnd_INDEX 5
#define Xnd_FromXnd_RETURN PyObject *
#define Xnd_FromXnd_ARGS (PyTypeObject *, xnd_t *x)

#define XND_MAX_API 6


#ifdef XND_MODULE
static Xnd_CheckExact_RETURN Xnd_CheckExact Xnd_CheckExact_ARGS;
static Xnd_Check_RETURN Xnd_Check Xnd_Check_ARGS;
static CONST_XND_RETURN CONST_XND CONST_XND_ARGS;
static Xnd_EmptyFromType_RETURN Xnd_EmptyFromType Xnd_EmptyFromType_ARGS;
static Xnd_ViewMoveNdt_RETURN Xnd_ViewMoveNdt Xnd_ViewMoveNdt_ARGS;
static Xnd_FromXnd_RETURN Xnd_FromXnd Xnd_FromXnd_ARGS;
#else
static void **_xnd_api;

#define Xnd_CheckExact \
    (*(Xnd_CheckExact_RETURN (*)Xnd_CheckExact_ARGS) _xnd_api[Xnd_CheckExact_INDEX])

#define Xnd_Check \
    (*(Xnd_Check_RETURN (*)Xnd_Check_ARGS) _xnd_api[Xnd_Check_INDEX])

#define CONST_XND \
    (*(CONST_XND_RETURN (*)CONST_XND_ARGS) _xnd_api[CONST_XND_INDEX])

#define Xnd_EmptyFromType \
    (*(Xnd_EmptyFromType_RETURN (*)Xnd_EmptyFromType_ARGS) _xnd_api[Xnd_EmptyFromType_INDEX])

#define Xnd_ViewMoveNdt \
    (*(Xnd_ViewMoveNdt_RETURN (*)Xnd_ViewMoveNdt_ARGS) _xnd_api[Xnd_ViewMoveNdt_INDEX])

#define Xnd_FromXnd \
    (*(Xnd_FromXnd_RETURN (*)Xnd_FromXnd_ARGS) _xnd_api[Xnd_FromXnd_INDEX])

static int
import_xnd(void)
{
    _xnd_api = (void **)PyCapsule_Import("xnd._xnd._API", 0);
    if (_xnd_api == NULL) {
        return -1;
    }

    return 0;
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* PYXND_H */

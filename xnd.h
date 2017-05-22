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


#ifndef XND_H
#define XND_H


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include "ndtypes.h"


/* mem.ptr special value to indicate NA */
#define ND_MISSING ((void *)0x1)

/* convenience macro to extract a pointer value */
#define ND_POINTER_DATA(ptr) (((ndt_pointer_t *)ptr)->data)


typedef struct _nd_array_t nd_array_t;

/* typed memory */
struct _nd_array_t {
    const nd_array_t *base; /* changes if a new array is discovered */
    const ndt_t *type;      /* type of the data */
    char *ptr;              /* data */
};



/*****************************************************************************/
/*              Inline functions to access array data and metadata           */
/*****************************************************************************/

/* Return the data of the current dimension. */
static inline char *
ND_DIM_DATA(const nd_array_t *mem)
{
    const ndt_t *a = mem->base->type;
    const ndt_t *t = mem->type;

    assert(ndt_is_concrete(a));
    assert(a->tag == Array);
    assert(ndt_is_concrete(t));
    assert(t->ndim == 0 || t->tag == VarDim);

    return mem->base->ptr + a->Concrete.Array.data[t->ndim];
}

static inline int64_t
ND_DIM_ITEMSIZE(const nd_array_t *mem)
{
    const ndt_t *t = mem->type;

    assert(ndt_is_concrete(t));

    switch (t->tag) {
    case FixedDim:
        return t->Concrete.FixedDim.itemsize;
    case VarDim:
        return t->Concrete.VarDim.itemsize;
    default:
        return t->Concrete.size;
    }
}

/*
 * Return the linear index of mem.ptr. Linear indices are meaningful in
 * two cases:
 *
 *   a) The type of the argument is a var dimension, so mem.ptr points
 *      into an index array.
 *
 *   b) ndim == 0 and mem.ptr points to a single element of the actual
 *      array data.
 *
 * Otherwise the behavior of the function is undefined.
 */
static inline int64_t
ND_LINEAR_INDEX(const nd_array_t *mem)
{
    const ndt_t *t = mem->type;
    ptrdiff_t d;
    int64_t i;

    assert(ndt_is_concrete(t));
    assert(t->ndim == 0 || t->tag == VarDim);

    d = mem->ptr - ND_DIM_DATA(mem);
    i = d / ND_DIM_ITEMSIZE(mem);
    return i;
}

static inline int64_t
ND_VAR_SHAPE(const nd_array_t *mem)
{
    const ndt_t *a = mem->base->type;
    const ndt_t *t = mem->type;
    enum ndt_dim dim_type = a->Concrete.Array.dim_type;

    assert(ndt_is_concrete(t));
    assert(t->tag == VarDim);

    switch (dim_type) {
    case DimUint8: {
        const uint8_t *p = (const uint8_t *)mem->ptr;
        return (int64_t)(p[1] - p[0]);
    }
    case DimUint16: {
        const uint16_t *p = (const uint16_t *)mem->ptr;
        return (int64_t)(p[1] - p[0]);
    }
    case DimUint32: {
        const uint32_t *p = (const uint32_t *)mem->ptr;
        return (int64_t)(p[1] - p[0]);
    }
    case DimInt32: {
        const int32_t *p = (const int32_t *)mem->ptr;
        return (int64_t)(p[1] - p[0]);
    }
    case DimInt64: {
        const int64_t *p = (const int64_t *)mem->ptr;
        return (int64_t)(p[1] - p[0]);
    }
    default:
        abort(); /* NOT REACHED */
    }
}

static inline char *
ND_NEXT_DIM(const nd_array_t *mem)
{
    const ndt_t *a = mem->base->type;
    const ndt_t *t = mem->type;
    enum ndt_dim dim_type = a->Concrete.Array.dim_type;
    char *next_dim = mem->base->ptr + a->Concrete.Array.data[t->ndim-1];
    int64_t i;

    assert(ndt_is_concrete(t));
    assert(t->tag == VarDim);

    switch (dim_type) {
    case DimUint8: {
        const uint8_t *p = (const uint8_t *)mem->ptr;
        i = (int64_t)p[0];
        break;
    }
    case DimUint16: {
        const uint16_t *p = (const uint16_t *)mem->ptr;
        i = (int64_t)p[0];
        break;
    }
    case DimUint32: {
        const uint32_t *p = (const uint32_t *)mem->ptr;
        i = (int64_t)p[0];
        break;
    }
    case DimInt32: {
        const int32_t *p = (const int32_t *)mem->ptr;
        i = (int64_t)p[0];
        break;
    }
    case DimInt64: {
        const int64_t *p = (const int64_t *)mem->ptr;
        i = (int64_t)p[0];
        break;
    }
    default:
        abort(); /* NOT REACHED */
    }

    return next_dim + i * t->Concrete.VarDim.itemsize;
}

static inline uint64_t *
ND_DIM_BITMAP(const nd_array_t *mem)
{
    const ndt_t *a = mem->base->type;
    const ndt_t *t = mem->type;
    char *bitmap;

    assert(ndt_is_concrete(a));
    assert(a->tag == Array);

    bitmap = mem->base->ptr + a->Concrete.Array.bitmaps[t->ndim];
    return (uint64_t *)bitmap;
}

static inline uint64_t *
ND_BITMAP(const nd_array_t *mem, int ndim)
{
    const ndt_t *a = mem->base == NULL ? mem->type : mem->base->type;
    char *bitmap;

    assert(ndt_is_concrete(a));
    assert(a->tag == Array);

    bitmap = mem->ptr + a->Concrete.Array.bitmaps[ndim];
    return (uint64_t *)bitmap;
}

static inline bool
ND_FIXED_DIM_IS_VALID(const nd_array_t *mem, int64_t i)
{
    const ndt_t *a = mem->base->type;
    const ndt_t *t = mem->type;
    uint64_t *bitmap;

    assert(ndt_is_concrete(a));
    assert(a->tag == Array);
    assert(ndt_is_concrete(t));
    assert(t->tag == FixedDim);

    if (!ndt_is_optional(t)) {
        return true;
    }

    bitmap = (uint64_t *)(mem->base->ptr + a->Concrete.Array.bitmaps[t->ndim]);
    return bitmap[i / 8] & (1 << (i % 8));
}

static inline bool
ND_DATA_IS_VALID(const nd_array_t *mem)
{
    const ndt_t *a = mem->base->type;
    const ndt_t *t = mem->type;
    uint64_t *bitmap;
    int64_t i;

    assert(ndt_is_concrete(a));
    assert(a->tag == Array);
    assert(ndt_is_concrete(t));
    assert(t->ndim == 0 || t->tag == VarDim);

    if (!ndt_is_optional(t)) {
        return true;
    }

    i = ND_LINEAR_INDEX(mem);
    bitmap = (uint64_t *)(mem->base->ptr + a->Concrete.Array.bitmaps[t->ndim]);
    return bitmap[i / 8] & (1 << (i % 8));
}

static inline void
ND_FIXED_DIM_SET_VALID(nd_array_t *mem, int64_t i)
{
    const ndt_t *a = mem->base->type;
    const ndt_t *t = mem->type;
    uint64_t *bitmap;

    assert(ndt_is_concrete(a));
    assert(a->tag == Array);
    assert(ndt_is_concrete(t));

    if (!ndt_is_optional(t)) {
        return;
    }

    bitmap = (uint64_t *)(mem->base->ptr + a->Concrete.Array.bitmaps[t->ndim]);
    bitmap[i / 8] |= (1 << (i % 8));
}

static inline void
ND_DATA_SET_VALID(nd_array_t *mem)
{
    const ndt_t *a = mem->base->type;
    const ndt_t *t = mem->type;
    uint64_t *bitmap;
    int64_t i;

    assert(ndt_is_concrete(a));
    assert(a->tag == Array);
    assert(ndt_is_concrete(t));
    assert(t->ndim == 0 || t->tag == VarDim);

    if (!ndt_is_optional(t)) {
        return;
    }

    i = ND_LINEAR_INDEX(mem);
    bitmap = (uint64_t *)(mem->base->ptr + a->Concrete.Array.bitmaps[t->ndim]);
    bitmap[i / 8] |= (1 << (i % 8));
}


/*****************************************************************************/
/*                               API functions                               */
/*****************************************************************************/

char *nd_new(const ndt_t *t, bool alloc_pointers, ndt_context_t *ctx);
int nd_init(char *ptr, const ndt_t *t, bool alloc_pointers, ndt_context_t *ctx);
void nd_del(nd_array_t a);
nd_array_t nd_empty(const char *datashape, ndt_context_t *ctx);
int nd_subarray_set_valid(nd_array_t a, const int64_t *indices, int len, ndt_context_t *ctx);
nd_array_t nd_subarray(const nd_array_t a, const int64_t *indices, int len, ndt_context_t *ctx);


#endif /* XND_H */

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


#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include "ndtypes.h"
#include "xnd.h"
#include "inline.h"


static int xnd_init(xnd_t *x, uint32_t flags, ndt_context_t *ctx);
static void xnd_clear(xnd_t *x, uint32_t flags);


/*****************************************************************************/
/*                              Error handling                               */
/*****************************************************************************/

/* error return value */
const xnd_t xnd_error = {
  .bitmap = {.data=NULL, .size=0, .next=NULL},
  .index = 0,
  .type = NULL,
  .ptr = NULL
};

int
xnd_err_occurred(const xnd_t *x)
{
    return x->ptr == NULL;
}


/*****************************************************************************/
/*                  Create and initialize a new master buffer                */
/*****************************************************************************/

/* Create and initialize memory with type 't'. */
static char *
xnd_new(const ndt_t * const t, const uint32_t flags, ndt_context_t *ctx)
{
    xnd_t x;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError,
            "cannot create xnd container from abstract type");
        return NULL;
    }

    x.index = 0;
    x.type = t;

    x.ptr = ndt_aligned_calloc(t->align, t->datasize);
    if (x.ptr == NULL) {
        ndt_memory_error(ctx);
        return NULL;
    }

    if (xnd_init(&x, flags, ctx) < 0) {
        ndt_aligned_free(x.ptr);
        return NULL;
    }

    return x.ptr;
}

/*
 * Initialize typed memory. If the XND_OWN_POINTERS flag is set, allocate
 * memory for all ref subtypes and initialize that memory. Otherwise, set
 * refs to NULL.
 *
 * Ref subtypes include any type of the form "Ref(t)".
 *
 * Never allocated are (sizes are not known):
 *   - "string" type (pointer to NUL-terminated UTF8 string)
 *   - data of the "bytes" type: {size: size_t, data: uint8_t *bytes}
 *
 * At all times the data pointers must be NULL or pointers to valid memory.
 */
static int
xnd_init(xnd_t * const x, const uint32_t flags, ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError,
            "cannot initialize concrete memory from abstract type");
        return -1;
    }

    switch (t->tag) {
    case FixedDim: {
        int64_t i;

        for (i = 0; i < t->FixedDim.shape; i++) {
            xnd_t next = _fixed_dim_next(x, i);
            if (xnd_init(&next, flags, ctx) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case VarDim: {
        int64_t start, step, shape;
        int64_t i;

        shape = ndt_var_indices(&start, &step, t, x->index, ctx);
        if (shape < 0) {
            return -1;
        }

        for (i = 0; i < shape; i++) {
            xnd_t next = _var_dim_next(x, start, step, i);
            if (xnd_init(&next, flags, ctx) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case Tuple: {
        int64_t i;

        for (i = 0; i < t->Tuple.shape; i++) {
            xnd_t next = _tuple_next(x, i);
            if (xnd_init(&next, flags, ctx) < 0) {
                xnd_clear(&next, flags);
                return -1;
            }
        }

        return 0;
    }

    case Record: {
        int64_t i;

        for (i = 0; i < t->Record.shape; i++) {
            xnd_t next = _record_next(x, i);
            if (xnd_init(&next, flags, ctx) < 0) {
                xnd_clear(&next, flags);
                return -1;
            }
        }

        return 0;
    }

    /*
     * Ref represents a pointer to an explicit type. If XND_OWN_POINTERS
     * is set, allocate memory for that type and set the pointer.
     */
    case Ref: {
        if (flags & XND_OWN_POINTERS) {
            const ndt_t *u = t->Ref.type;
            void *ref;

            ref = ndt_aligned_calloc(u->align, u->datasize);
            if (ref == NULL) {
                ndt_err_format(ctx, NDT_MemoryError, "out of memory");
                return -1;
            }
            XND_POINTER_DATA(x->ptr) = ref;

            xnd_t next = _ref_next(x);
            if (xnd_init(&next, flags, ctx) < 0) {
                xnd_clear(&next, flags);
                return -1;
            }
        }

        return 0;
    }

    /* Constr is a named explicit type. */
    case Constr: {
        xnd_t next = _constr_next(x);
        if (xnd_init(&next, flags, ctx) < 0) {
            xnd_clear(&next, flags);
            return -1;
        }

        return 0;
    }

    /* Nominal is a globally unique typedef. */
    case Nominal: {
        xnd_t next = _nominal_next(x);
        if (xnd_init(&next, flags, ctx) < 0) {
            xnd_clear(&next, flags);
            return -1;
        }

        return 0;
    }

    /* Categorical is already initialized by calloc(). */
    case Categorical:
        return 0;

    case Char:
        ndt_err_format(ctx, NDT_NotImplementedError, "char not implemented");
        return -1;

    /* Primitive types are already initialized by calloc(). */
    case Bool:
    case Int8: case Int16: case Int32: case Int64:
    case Uint8: case Uint16: case Uint32: case Uint64:
    case Float16: case Float32: case Float64:
    case Complex32: case Complex64: case Complex128:
    case FixedString: case FixedBytes:
    case String: case Bytes:
        return 0;

    case Module: case Function:
    case AnyKind: case SymbolicDim: case EllipsisDim: case Typevar:
    case ScalarKind: case SignedKind: case UnsignedKind: case FloatKind:
    case ComplexKind: case FixedStringKind: case FixedBytesKind:
        /* NOT REACHED: intercepted by ndt_is_abstract(). */
        ndt_err_format(ctx, NDT_RuntimeError, "unexpected abstract type");
        return -1;
   }

   /* NOT REACHED: tags should be exhaustive */
   ndt_err_format(ctx, NDT_RuntimeError, "invalid type tag");
   return -1;
}

/*
 * Create a type from a string and return a new master buffer for that type.
 * Any combination of flags that include XND_OWN_TYPE can be passed.
 */
xnd_master_t *
xnd_empty_from_string(const char * const s, const uint32_t flags, ndt_context_t *ctx)
{
    xnd_bitmap_t b = {.data=NULL, .size=0, .next=NULL};
    xnd_master_t *x;
    ndt_t *t;
    char *ptr;

    if (!(flags & XND_OWN_TYPE)) {
        ndt_err_format(ctx, NDT_InvalidArgumentError,
            "xnd_empty_from_string: XND_OWN_TYPE must be set");
        return NULL;
    }

    x = ndt_alloc(1, sizeof *x);
    if (x == NULL) {
        return ndt_memory_error(ctx);
    }

    t = ndt_from_string(s, ctx);
    if (t == NULL) {
        ndt_free(x);
        return NULL;
    }

    if (!ndt_is_concrete(t)) {
        ndt_err_format(ctx, NDT_ValueError, "type must be concrete");
        ndt_del(t);
        ndt_free(x);
        return NULL;
    }

    if (xnd_bitmap_init(&b, t,ctx) < 0) {
        ndt_del(t);
        ndt_free(x);
        return NULL;
    }

    ptr = xnd_new(t, flags, ctx);
    if (ptr == NULL) {
        xnd_bitmap_clear(&b);
        ndt_del(t);
        ndt_free(x);
        return NULL;
    }

    x->flags = flags;
    x->master.bitmap = b;
    x->master.index = 0;
    x->master.type = t;
    x->master.ptr = ptr;

    return x;
}

/*
 * Return a new master buffer. Any combination of flags except for XND_OWN_TYPE
 * can be passed. 't' must be kept valid as long as the master buffer is valid.
 */
xnd_master_t *
xnd_empty_from_type(const ndt_t * const t, const uint32_t flags, ndt_context_t *ctx)
{
    xnd_bitmap_t b = {.data=NULL, .size=0, .next=NULL};
    xnd_master_t *x;
    char *ptr;

    if (flags & XND_OWN_TYPE) {
        ndt_err_format(ctx, NDT_InvalidArgumentError,
            "xnd_empty_from_type: XND_OWN_TYPE must not be set");
        return NULL;
    }

    if (!ndt_is_concrete(t)) {
        ndt_err_format(ctx, NDT_ValueError, "type must be concrete");
        return NULL;
    }

    x = ndt_alloc(1, sizeof *x);
    if (x == NULL) {
        return ndt_memory_error(ctx);
    }

    if (xnd_bitmap_init(&b, t, ctx) < 0) {
        ndt_free(x);
        return NULL;
    }

    ptr = xnd_new(t, flags, ctx);
    if (ptr == NULL) {
        xnd_bitmap_clear(&b);
        ndt_free(x);
        return NULL;
    }

    x->flags = flags;
    x->master.bitmap = b;
    x->master.index = 0;
    x->master.type = t;
    x->master.ptr = ptr;

    return x;
}


/*****************************************************************************/
/*                     Deallocate and clear a master buffer                  */
/*****************************************************************************/

/* Clear an embedded pointer. */
static void
xnd_clear_ref(xnd_t *x, const uint32_t flags)
{
    assert(x->type->tag == Ref);

    if (flags & XND_OWN_POINTERS) {
        ndt_aligned_free(XND_POINTER_DATA(x->ptr));
        XND_POINTER_DATA(x->ptr) = NULL;
    }
}

/* Strings must always be allocated by non-aligned allocators. */
static void
xnd_clear_string(xnd_t *x, const uint32_t flags)
{
    assert(x->type->tag == String);

    if (flags & XND_OWN_STRINGS) {
        ndt_free(XND_POINTER_DATA(x->ptr));
        XND_POINTER_DATA(x->ptr) = NULL;
    }
}

/* Bytes must always be allocated by aligned allocators. */
static void
xnd_clear_bytes(xnd_t *x, const uint32_t flags)
{
    assert(x->type->tag == Bytes);

    if (flags & XND_OWN_BYTES) {
        ndt_aligned_free(XND_BYTES_DATA(x->ptr));
        XND_BYTES_DATA(x->ptr) = NULL;
    }
}

/* Clear embedded pointers in the data according to flags. */
static void
xnd_clear(xnd_t * const x, const uint32_t flags)
{
    NDT_STATIC_CONTEXT(ctx);
    const ndt_t * const t = x->type;

    assert(ndt_is_concrete(t));

    switch (t->tag) {
    case FixedDim: {
        int64_t i;

        for (i = 0; i < t->FixedDim.shape; i++) {
            xnd_t next = _fixed_dim_next(x, i);
            xnd_clear(&next, flags);
        }

        return;
    }

    case VarDim: {
        int64_t start, step, shape;
        int64_t i;

        shape = ndt_var_indices(&start, &step, t, x->index, &ctx);
        if (shape < 0) {
            /* This cannot happen: indices are checked in xnd_init() and
             * should remain constant. */
            ndt_context_del(&ctx);
            fprintf(stderr, "xnd_clear: internal error: var indices changed\n");
            return;
        }

        for (i = 0; i < shape; i++) {
            xnd_t next = _var_dim_next(x, start, step, i);
            xnd_clear(&next, flags);
        }

        return;
    }

    case Tuple: {
        int64_t i;

        for (i = 0; i < t->Tuple.shape; i++) {
            xnd_t next = _tuple_next(x, i);
            xnd_clear(&next, flags);
        }

        return;
    }

    case Record: {
        int64_t i;

        for (i = 0; i < t->Record.shape; i++) {
            xnd_t next = _record_next(x, i);
            xnd_clear(&next, flags);
        }

        return;
    }

    case Ref: {
        if (flags & XND_OWN_POINTERS) {
            xnd_t next = _ref_next(x);
            xnd_clear(&next, flags);
            xnd_clear_ref(x, flags);
        }

        return;
    }

    case Constr: {
        xnd_t next = _constr_next(x);
        xnd_clear(&next, flags);
        return;
    }

    case Nominal: {
        xnd_t next = _nominal_next(x);
        xnd_clear(&next, flags);
        return;
    }

    case Bool:
    case Int8: case Int16: case Int32: case Int64:
    case Uint8: case Uint16: case Uint32: case Uint64:
    case Float16: case Float32: case Float64:
    case Complex32: case Complex64: case Complex128:
    case FixedString: case FixedBytes:
        return;

    case String:
        xnd_clear_string(x, flags);
        return;

    case Bytes:
        xnd_clear_bytes(x, flags);
        return;

    case Categorical:
        /* Categorical values are just indices into the categories. */
        return;

    case Char:
        /* Just a scalar. */
        return;

    case Module:
        /* XXX Not implemented. */
        return;

    /* NOT REACHED: intercepted by ndt_is_abstract(). */
    case AnyKind: case SymbolicDim: case EllipsisDim: case Typevar:
    case ScalarKind: case SignedKind: case UnsignedKind: case FloatKind:
    case ComplexKind: case FixedStringKind: case FixedBytesKind:
    case Function:
        return;
    }
}

/*
 * Delete the master buffer. The type and embedded pointers are deallocated
 * according to x->flags.
 */
void
xnd_del(xnd_master_t *x)
{
    if (x != NULL) {
        if (x->master.ptr != NULL && x->master.type != NULL) {
            if (x->flags & XND_OWN_DATA) {
                xnd_clear(&x->master, x->flags);
            }

            if (x->flags & XND_OWN_TYPE) {
                ndt_del((ndt_t *)x->master.type);
            }

            if (x->flags & XND_OWN_DATA) {
                ndt_aligned_free(x->master.ptr);
            }
        }

        if (x->flags & XND_OWN_DATA) {
            xnd_bitmap_clear(&x->master.bitmap);
        }
        ndt_free(x);
    }
}


/*****************************************************************************/
/*                 Subtrees (single elements are a special case)             */
/*****************************************************************************/

/* Return a typed subtree of a memory block */
xnd_t
xnd_subtree(const xnd_t * const x, const int64_t *indices, const int len,
            ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;

    assert(ndt_is_concrete(t));

    if (t->ndim > 0 && ndt_is_optional(t)) {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "optional dimensions are not supported");
        return xnd_error;
    }

    if (len == 0) {
        return *x;
    }

    const int64_t i = indices[0];

    switch (t->tag) {
    case FixedDim: {
        if (i < 0 || i >= t->FixedDim.shape) {
            ndt_err_format(ctx, NDT_ValueError,
                "fixed dim index out of bounds");
            return xnd_error;
        }

        const xnd_t next = xnd_fixed_dim_next(x, i);
        return xnd_subtree(&next, indices+1, len-1, ctx);
    }

    case VarDim: {
        int64_t start, step, shape;

        shape = ndt_var_indices(&start, &step, t, x->index, ctx);
        if (shape < 0) {
            return xnd_error;
        }

        if (i < 0 || i >= shape) {
            ndt_err_format(ctx, NDT_ValueError, "var dim index out of bounds");
            return xnd_error;
        }

        const xnd_t next = xnd_var_dim_next(x, start, step, i);
        return xnd_subtree(&next, indices+1, len-1, ctx);
    }

    case Tuple: {
        if (i < 0 || i >= t->Tuple.shape) {
            ndt_err_format(ctx, NDT_ValueError, "tuple index out of bounds");
            return xnd_error;
        }

        const xnd_t next = xnd_tuple_next(x, i, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return xnd_subtree(&next, indices+1, len-1, ctx);
    }

    case Record: {
        if (i < 0 || i >= t->Record.shape) {
            ndt_err_format(ctx, NDT_ValueError, "record index out of bounds");
            return xnd_error;
        }

        const xnd_t next = xnd_record_next(x, i, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return xnd_subtree(&next, indices+1, len-1, ctx);
    }

    case Ref: {
        const xnd_t next = xnd_ref_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return xnd_subtree(&next, indices, len, ctx);
    }

    case Constr: {
        const xnd_t next = xnd_constr_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return xnd_subtree(&next, indices, len, ctx);
    }

   case Nominal: {
        const xnd_t next = xnd_nominal_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return xnd_subtree(&next, indices, len, ctx);
    }

    default:
        ndt_err_format(ctx, NDT_ValueError, "type not indexable");
        return xnd_error;
    }
}


/*****************************************************************************/
/*                               Unstable API                                */
/*****************************************************************************/

/* For gumath -- likely to change */

int
xnd_as_ndarray(xnd_ndarray_t *a, const xnd_t *x, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    int i;

    assert(t->ndim <= NDT_MAX_DIM);

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_TypeError, "type is not an ndarray");
        return -1;
    }

    if (!ndt_is_ndarray(t)) {
        if (t->ndim == 0) {
            a->ndim = t->ndim;
            a->itemsize = t->datasize;
            a->nelem = 1;
            a->ptr = x->ptr + x->index * t->datasize;
            return 0;
        }
        ndt_err_format(ctx, NDT_TypeError, "type is not an ndarray");
        return -1;
    }

    a->ndim = t->ndim;
    a->itemsize = t->Concrete.FixedDim.itemsize;
    a->nelem = t->datasize / t->Concrete.FixedDim.itemsize;
    a->ptr = x->ptr + x->index * a->itemsize;

    for (i=0; t->ndim > 0; i++, t=t->FixedDim.type) {
        a->shape[i] = t->FixedDim.shape;
        a->strides[i] = t->Concrete.FixedDim.step * a->itemsize;
    }

    return 0;
}

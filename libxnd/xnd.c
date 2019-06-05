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
#include <string.h>
#include <math.h>
#include <assert.h>
#include "ndtypes.h"
#include "xnd.h"
#include "inline.h"
#include "contrib.h"
#include "contrib/bfloat16.h"
#include "cuda/cuda_memory.h"
#ifndef _MSC_VER
#include "config.h"
#endif


static int xnd_init(xnd_t * const x, const uint32_t flags, ndt_context_t *ctx);


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

static bool
requires_init(const ndt_t * const t)
{
    return !ndt_is_ref_free(t);
}

static bool
is_primary_type(const ndt_t * const t, ndt_context_t *ctx)
{
    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError,
            "cannot create xnd container from abstract type");
        return false;
    }

    if (t->flags & NDT_CHAR) {
        ndt_err_format(ctx, NDT_NotImplementedError, "char is not implemented");
        return false;
    }

    switch (t->tag) {
    case FixedDim: {
        if (!ndt_is_c_contiguous(t) && !ndt_is_f_contiguous(t)) {
            ndt_err_format(ctx, NDT_ValueError,
                "cannot create xnd container from non-contiguous type");
            return false;
        }
        return true;
    }
    case VarDim: case VarDimElem: {
        if (!ndt_is_var_contiguous(t)) {
            ndt_err_format(ctx, NDT_ValueError,
                "cannot create xnd container from non-contiguous type");
            return false;
        }
        return true;
    }
    case Array: {
        if (requires_init(t)) {
            ndt_err_format(ctx, NDT_ValueError,
                "flexible arrays cannot have dtypes that require "
                "initialization");
            return false;
        }
        return true;
    }
    default:
        return true;
    }

    ndt_err_format(ctx, NDT_ValueError,
        "cannot create xnd container from non-contiguous type");
    return false;
}


/* Create and initialize memory with type 't'. */
#ifdef HAVE_CUDA
static char *
xnd_cuda_new(const ndt_t * const t, ndt_context_t *ctx)
{
    void *ptr;

    if (!is_primary_type(t, ctx)) {
        return NULL;
    }

    if (!ndt_is_pointer_free(t)) {
        ndt_err_format(ctx, NDT_ValueError,
            "only pointer-free types are supported on cuda");
        return NULL;
    }

    ptr = xnd_cuda_calloc_managed(t->align, t->datasize, ctx);
    if (ptr == NULL) {
        return NULL;
    }

    return ptr;
}
#else
static char *
xnd_cuda_new(const ndt_t * const t, ndt_context_t *ctx)
{
    (void)t;

    ndt_err_format(ctx, NDT_ValueError, "xnd compiled without cuda support");
    return NULL;
}
#endif

static char *
xnd_new(const ndt_t * const t, const uint32_t flags, ndt_context_t *ctx)
{
    xnd_t x;

    if (flags & XND_CUDA_MANAGED) {
        return xnd_cuda_new(t, ctx);
    }

    if (!is_primary_type(t, ctx)) {
        return NULL;
    }

    x.index = 0;
    x.type = t;

    x.ptr = ndt_aligned_calloc(t->align, t->datasize);
    if (x.ptr == NULL) {
        ndt_memory_error(ctx);
        return NULL;
    }

    if (requires_init(t) && xnd_init(&x, flags, ctx) < 0) {
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

    if (flags & XND_CUDA_MANAGED) {
        ndt_err_format(ctx, NDT_RuntimeError,
            "internal error: cannot initialize cuda memory with a type "
            "that contains pointers");
        return -1;
    }

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
        for (int64_t i = 0; i < t->Tuple.shape; i++) {
            xnd_t next = _tuple_next(x, i);
            if (xnd_init(&next, flags, ctx) < 0) {
                xnd_clear(&next, flags);
                return -1;
            }
        }

        return 0;
    }

    case Record: {
        for (int64_t i = 0; i < t->Record.shape; i++) {
            xnd_t next = _record_next(x, i);
            if (xnd_init(&next, flags, ctx) < 0) {
                xnd_clear(&next, flags);
                return -1;
            }
        }

        return 0;
    }

    case Union: {
        xnd_t next = _union_next(x);
        if (xnd_init(&next, flags, ctx) < 0) {
            xnd_clear(&next, flags);
            return -1;
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

    case VarDimElem:
        ndt_err_format(ctx, NDT_ValueError,
           "cannot initialize var elem dimension");
        return -1;

    case Char:
        ndt_err_format(ctx, NDT_NotImplementedError, "char not implemented");
        return -1;

    /* Primitive types are already initialized by calloc(). */
    case Bool:
    case Int8: case Int16: case Int32: case Int64:
    case Uint8: case Uint16: case Uint32: case Uint64:
    case BFloat16: case Float16: case Float32: case Float64:
    case BComplex32: case Complex32: case Complex64: case Complex128:
    case FixedString: case FixedBytes:
    case String: case Bytes:
    case Array:
        return 0;

    /* NOT REACHED: intercepted by ndt_is_abstract(). */
    case Module: case Function:
    case AnyKind: case SymbolicDim: case EllipsisDim: case Typevar:
    case ScalarKind: case SignedKind: case UnsignedKind: case FloatKind:
    case ComplexKind: case FixedStringKind: case FixedBytesKind:
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
xnd_empty_from_string(const char *s, uint32_t flags, ndt_context_t *ctx)
{
    xnd_bitmap_t b = {.data=NULL, .size=0, .next=NULL};
    xnd_master_t *x;
    const ndt_t *t;
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
        ndt_decref(t);
        ndt_free(x);
        return NULL;
    }

    if (xnd_bitmap_init(&b, t,ctx) < 0) {
        ndt_decref(t);
        ndt_free(x);
        return NULL;
    }

    ptr = xnd_new(t, flags, ctx);
    if (ptr == NULL) {
        xnd_bitmap_clear(&b);
        ndt_decref(t);
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
xnd_empty_from_type(const ndt_t *t, uint32_t flags, ndt_context_t *ctx)
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

/*
 * Create master buffer from an existing xnd_t.  Ownership of bitmaps, type,
 * ptr is transferred to the master buffer.
 *
 * 'flags' are the master buffer's flags after the transfer.  The flags of
 * 'src' are always assumed to be XND_OWN_ALL.
 *
 * This is a convenience function that should only be used if the xnd_t src
 * owns everything and its internals have not been exposed to other views.
 */
xnd_master_t *
xnd_from_xnd(xnd_t *src, uint32_t flags, ndt_context_t *ctx)
{
    xnd_master_t *x;

    /* XXX xnd_from_xnd() will probably be replaced. */
    assert(!(flags & XND_CUDA_MANAGED));

    x = ndt_alloc(1, sizeof *x);
    if (x == NULL) {
        xnd_clear(src, XND_OWN_ALL);
        ndt_decref(src->type);
        ndt_aligned_free(src->ptr);
        xnd_bitmap_clear(&src->bitmap);
        return ndt_memory_error(ctx);
    }

    x->flags = flags;
    x->master = *src;

    return x;
}


/*****************************************************************************/
/*                     Deallocate and clear a master buffer                  */
/*****************************************************************************/

static bool
requires_clear(const ndt_t * const t)
{
    const ndt_t *dtype = ndt_dtype(t);

    switch (dtype->tag) {
    case Categorical:
    case Bool:
    case Int8: case Int16: case Int32: case Int64:
    case Uint8: case Uint16: case Uint32: case Uint64:
    case BFloat16: case Float16: case Float32: case Float64:
    case BComplex32: case Complex32: case Complex64: case Complex128:
    case FixedString: case FixedBytes:
        return false;
    default:
        return true;
    }
}

/* Clear an embedded pointer. */
static void
xnd_clear_ref(xnd_t *x, const uint32_t flags)
{
    assert(x->type->tag == Ref);
    assert(!(flags & XND_CUDA_MANAGED));

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
    assert(!(flags & XND_CUDA_MANAGED));

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
    assert(!(flags & XND_CUDA_MANAGED));

    if (flags & XND_OWN_BYTES) {
        ndt_aligned_free(XND_BYTES_DATA(x->ptr));
        XND_BYTES_SIZE(x->ptr) = 0;
        XND_BYTES_DATA(x->ptr) = NULL;
    }
}

/* Flexible 1D array data must always be allocated by aligned allocators. */
static void
xnd_clear_array(xnd_t *x, const uint32_t flags)
{
    assert(x->type->tag == Array);
    assert(!(flags & XND_CUDA_MANAGED));

    if (flags & XND_OWN_ARRAYS) {
        ndt_aligned_free(XND_ARRAY_DATA(x->ptr));
        XND_ARRAY_SHAPE(x->ptr) = 0;
        XND_ARRAY_DATA(x->ptr) = NULL;
    }
}

/* Clear embedded pointers in the data according to flags. */
void
xnd_clear(xnd_t * const x, const uint32_t flags)
{
    NDT_STATIC_CONTEXT(ctx);
    const ndt_t * const t = x->type;

    assert(ndt_is_concrete(t));
    assert(!(flags & XND_CUDA_MANAGED));

    switch (t->tag) {
    case FixedDim: {
        for (int64_t i = 0; i < t->FixedDim.shape; i++) {
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

    case VarDimElem: {
        fprintf(stderr, "xnd_clear: internal error: unexpected var elem dimension\n");
        return;
    }

    case Tuple: {
        for (int64_t i = 0; i < t->Tuple.shape; i++) {
            xnd_t next = _tuple_next(x, i);
            xnd_clear(&next, flags);
        }

        return;
    }

    case Record: {
        for (int64_t i = 0; i < t->Record.shape; i++) {
            xnd_t next = _record_next(x, i);
            xnd_clear(&next, flags);
        }

        return;
    }

    case Union: {
        xnd_t next = _union_next(x);
        xnd_clear(&next, flags);
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
    case BFloat16: case Float16: case Float32: case Float64:
    case BComplex32: case Complex32: case Complex64: case Complex128:
    case FixedString: case FixedBytes:
        return;

    case String:
        xnd_clear_string(x, flags);
        return;

    case Bytes:
        xnd_clear_bytes(x, flags);
        return;

    case Array:
        xnd_clear_array(x, flags);
        return;

    case Categorical:
        /* Categorical values are just indices into the categories. */
        return;

    case Char:
        /* Just a scalar. */
        return;

    /* NOT REACHED: intercepted by ndt_is_abstract(). */
    case Module: case Function:
    case AnyKind: case SymbolicDim: case EllipsisDim: case Typevar:
    case ScalarKind: case SignedKind: case UnsignedKind: case FloatKind:
    case ComplexKind: case FixedStringKind: case FixedBytesKind:
        return;
    }
}

/*
 * Delete an xnd_t buffer according to 'flags'. Outside xnd_del(), this
 * function should only be used if an xnd_t owns all its members.
 */
void
xnd_del_buffer(xnd_t *x, uint32_t flags)
{
    if (x != NULL) {
        if (x->ptr != NULL && x->type != NULL) {
            if ((flags&XND_OWN_DATA) && requires_clear(x->type)) {
                xnd_clear(x, flags);
            }

            if (flags & XND_OWN_TYPE) {
                ndt_decref(x->type);
            }

            if (flags & XND_OWN_DATA) {
                if (flags & XND_CUDA_MANAGED) {
                #ifdef HAVE_CUDA
                    xnd_cuda_free(x->ptr);
                #else
                    fprintf(stderr,
                        "xnd_del_buffer: internal error: XND_CUDA_MANAGED set "
                        "without cuda support\n");
                #endif
                }
                else {
                    ndt_aligned_free(x->ptr);
                }
            }
        }

        if (flags & XND_OWN_DATA) {
            xnd_bitmap_clear(&x->bitmap);
        }
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
        xnd_del_buffer(&x->master, x->flags);
        ndt_free(x);
    }
}


/*****************************************************************************/
/*                                Index checks                               */
/*****************************************************************************/

static int64_t
get_index(const xnd_index_t *key, int64_t shape, ndt_context_t *ctx)
{
    switch (key->tag) {
    case Index:
        return adjust_index(key->Index, shape, ctx);

    case FieldName:
        ndt_err_format(ctx, NDT_ValueError,
            "expected integer index, got field name: '%s'", key->FieldName);
        return -1;

    case Slice:
        ndt_err_format(ctx, NDT_ValueError,
            "expected integer index, got slice");
        return -1;
    }

    /* NOT REACHED: tags should be exhaustive */
    ndt_err_format(ctx, NDT_RuntimeError, "invalid index tag");
    return -1;
}

/*
 * Ragged arrays have multiple shapes in a single dimension that are not known
 * when a VarDimElem is created.  Adjusting the index must be done when the
 * VarDimElem is accessed and the slices have been applied.
 */
static int64_t
get_index_var_elem(const xnd_index_t *key, ndt_context_t *ctx)
{
    switch (key->tag) {
    case Index: {
        int64_t i = key->Index;

        if (i < INT32_MIN || i > INT32_MAX) {
            ndt_err_format(ctx, NDT_IndexError,
                "index with value %" PRIi64 " out of bounds", key->Index);
            return INT64_MIN;
        }

        return i;
    }

    case FieldName:
        ndt_err_format(ctx, NDT_ValueError,
            "expected integer index, got field name: '%s'", key->FieldName);
        return INT64_MIN;

    case Slice:
        ndt_err_format(ctx, NDT_ValueError,
            "expected integer index, got slice");
        return INT64_MIN;
    }

    /* NOT REACHED: tags should be exhaustive */
    ndt_err_format(ctx, NDT_RuntimeError, "invalid index tag");
    return INT64_MIN;
}

static int64_t
get_index_record(const ndt_t *t, const xnd_index_t *key, ndt_context_t *ctx)
{
    assert(t->tag == Record);

    switch (key->tag) {
    case FieldName: {
        int64_t i;

        for (i = 0; i < t->Record.shape; i++) {
            if (strcmp(key->FieldName, t->Record.names[i]) == 0) {
                return i;
            }
        }

        ndt_err_format(ctx, NDT_ValueError,
            "invalid field name '%s'", key->FieldName);
        return -1;
    }
    case Index: case Slice:
        return get_index(key, t->Record.shape, ctx);
    }

    /* NOT REACHED: tags should be exhaustive */
    ndt_err_format(ctx, NDT_RuntimeError, "invalid index tag");
    return -1;
}

static int64_t
get_index_union(const ndt_t *t, const xnd_index_t *key, ndt_context_t *ctx)
{
    assert(t->tag == Union);

    switch (key->tag) {
    case FieldName: {
        int64_t i;

        for (i = 0; i < t->Union.ntags; i++) {
            if (strcmp(key->FieldName, t->Union.tags[i]) == 0) {
                return i;
            }
        }

        ndt_err_format(ctx, NDT_ValueError,
            "invalid field name '%s'", key->FieldName);
        return -1;
    }
    case Index: case Slice:
        return get_index(key, t->Union.ntags, ctx);
    }

    /* NOT REACHED: tags should be exhaustive */
    ndt_err_format(ctx, NDT_RuntimeError, "invalid index tag");
    return -1;
}

static void
set_index_exception(bool indexable, ndt_context_t *ctx)
{
    if (indexable) {
        ndt_err_format(ctx, NDT_IndexError, "too many indices");
    }
    else {
        ndt_err_format(ctx, NDT_TypeError, "type not indexable");
    }
}


/*****************************************************************************/
/*                               Stored indices                              */
/*****************************************************************************/

bool
have_stored_index(const ndt_t *t)
{
    return t->tag == VarDimElem;
}

int64_t
get_stored_index(const ndt_t *t)
{
    return t->VarDimElem.index;
}

/* skip stored indices */
xnd_t
apply_stored_index(const xnd_t *x, ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;
    int64_t start, step, shape;

    if (t->tag != VarDimElem) {
        ndt_err_format(ctx, NDT_RuntimeError,
                       "apply_stored_index: need VarDimElem");
        return xnd_error;
    }

    shape = ndt_var_indices(&start, &step, t, x->index, ctx);
    if (shape < 0) {
        return xnd_error;
    }

    const int64_t i = adjust_index(t->VarDimElem.index, shape, ctx);
    if (i < 0) {
        return xnd_error;
    }

    return xnd_var_dim_next(x, start, step, i);
}

xnd_t
apply_stored_indices(const xnd_t *x, ndt_context_t *ctx)
{
    xnd_t tl = *x;

    while (tl.type->tag == VarDimElem) {
        tl = apply_stored_index(&tl, ctx);
    }

    return tl;
}


/*****************************************************************************/
/*                 Subtrees (single elements are a special case)             */
/*****************************************************************************/

/* Return a typed subtree of a memory block */
static xnd_t
_xnd_subtree_index(const xnd_t *x, const int64_t *indices, int len, ndt_context_t *ctx)
{
    APPLY_STORED_INDICES_XND(x)
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
        const int64_t k = adjust_index(i, t->FixedDim.shape, ctx);
        if (k < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_fixed_dim_next(x, k);
        return _xnd_subtree_index(&next, indices+1, len-1, ctx);
    }

    case VarDim: {
        int64_t start, step, shape;

        shape = ndt_var_indices(&start, &step, t, x->index, ctx);
        if (shape < 0) {
            return xnd_error;
        }

        const int64_t k = adjust_index(i, shape, ctx);
        if (k < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_var_dim_next(x, start, step, k);
        return _xnd_subtree_index(&next, indices+1, len-1, ctx);
    }

    case Tuple: {
        const int64_t k = adjust_index(i, t->Tuple.shape, ctx);
        if (k < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_tuple_next(x, k, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree_index(&next, indices+1, len-1, ctx);
    }

    case Record: {
        const int64_t k = adjust_index(i, t->Record.shape, ctx);
        if (k < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_record_next(x, k, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree_index(&next, indices+1, len-1, ctx);
    }

    case Union: {
        const int64_t k = adjust_index(i, t->Union.ntags, ctx);
        if (k < 0) {
            return xnd_error;
        }

        const uint8_t l = XND_UNION_TAG(x->ptr);
        if (k != l) {
            ndt_err_format(ctx, NDT_ValueError,
                "tag mismatch in union addressing: expected '%s', got '%s'",
                t->Union.tags[l], t->Union.tags[k]);
            return xnd_error;
        }

        const xnd_t next = xnd_union_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree_index(&next, indices+1, len-1, ctx);
    }

    case Array: {
        const int64_t shape = XND_ARRAY_SHAPE(x->ptr);
        const int64_t k = adjust_index(i, shape, ctx);
        if (k < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_array_next(x, k);
        return _xnd_subtree_index(&next, indices+1, len-1, ctx);
    }

    case Ref: {
        const xnd_t next = xnd_ref_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree_index(&next, indices, len, ctx);
    }

    case Constr: {
        const xnd_t next = xnd_constr_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree_index(&next, indices, len, ctx);
    }

    case Nominal: {
        const xnd_t next = xnd_nominal_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree_index(&next, indices, len, ctx);
    }

    default:
        ndt_err_format(ctx, NDT_ValueError, "type not indexable");
        return xnd_error;
    }
}

xnd_t
xnd_subtree_index(const xnd_t *x, const int64_t *indices, int len, ndt_context_t *ctx)
{
    if (len < 0 || len > NDT_MAX_DIM) {
        ndt_err_format(ctx, NDT_IndexError, "too many indices");
        return xnd_error;
    }

    return _xnd_subtree_index(x, indices, len, ctx);
}

/*
 * Return a zero copy view of an xnd object.  If a dtype is indexable,
 * descend into the dtype.
 */
static xnd_t
_xnd_subtree(const xnd_t *x, const xnd_index_t indices[], int len, bool indexable,
             ndt_context_t *ctx)
{
    APPLY_STORED_INDICES_XND(x)
    const ndt_t *t = x->type;
    const xnd_index_t *key;

    assert(ndt_is_concrete(t));

    if (t->ndim > 0 && ndt_is_optional(t)) {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "optional dimensions are not supported");
        return xnd_error;
    }

    if (len == 0) {
        return *x;
    }

    key = &indices[0];

    switch (t->tag) {
    case FixedDim: {
        const int64_t i = get_index(key, t->FixedDim.shape, ctx);
        if (i < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_fixed_dim_next(x, i);
        return _xnd_subtree(&next, indices+1, len-1, true, ctx);
    }

    case VarDim: {
        int64_t start, step, shape;

        shape = ndt_var_indices(&start, &step, t, x->index, ctx);
        if (shape < 0) {
            return xnd_error;
        }

        const int64_t i = get_index(key, shape, ctx);
        if (i < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_var_dim_next(x, start, step, i);
        return _xnd_subtree(&next, indices+1, len-1, true, ctx);
    }

    case Tuple: {
        const int64_t i = get_index(key, t->Tuple.shape, ctx);
        if (i < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_tuple_next(x, i, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree(&next, indices+1, len-1, true, ctx);
    }

    case Record: {
        const int64_t i = get_index_record(t, key, ctx);
        if (i < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_record_next(x, i, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree(&next, indices+1, len-1, true, ctx);
    }

    case Union: {
        const int64_t i = get_index_union(t, key, ctx);
        if (i < 0) {
            return xnd_error;
        }

        const uint8_t k = XND_UNION_TAG(x->ptr);
        if (i != k) {
            ndt_err_format(ctx, NDT_ValueError,
                "tag mismatch in union addressing: expected '%s', got '%s'",
                t->Union.tags[k], t->Union.tags[i]);
            return xnd_error;
        }

        const xnd_t next = xnd_union_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree(&next, indices+1, len-1, true, ctx);
    }

    case Array: {
        const int64_t shape = XND_ARRAY_SHAPE(x->ptr);
        const int64_t i = get_index(key, shape, ctx);
        if (i < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_array_next(x, i);
        return _xnd_subtree(&next, indices+1, len-1, true, ctx);
    }

    case Ref: {
        const xnd_t next = xnd_ref_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree(&next, indices, len, false, ctx);
    }

    case Constr: {
        const xnd_t next = xnd_constr_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree(&next, indices, len, false, ctx);
    }

    case Nominal: {
        const xnd_t next = xnd_nominal_next(x, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        return _xnd_subtree(&next, indices, len, false, ctx);
    }

    default:
        set_index_exception(indexable, ctx);
        return xnd_error;
    }
}

/*
 * Return a zero copy view of an xnd object.  If a dtype is indexable,
 * descend into the dtype.
 */
xnd_t
xnd_subtree(const xnd_t *x, const xnd_index_t indices[], int len, ndt_context_t *ctx)
{
    if (len < 0 || len > NDT_MAX_DIM) {
        ndt_err_format(ctx, NDT_IndexError, "too many indices");
        return xnd_error;
    }

    return _xnd_subtree(x, indices, len, false, ctx);
}

static xnd_t xnd_index(const xnd_t *x, const xnd_index_t indices[], int len, ndt_context_t *ctx);
static xnd_t xnd_slice(const xnd_t *x, const xnd_index_t indices[], int len, ndt_context_t *ctx);

static xnd_t
xnd_multikey(const xnd_t *x, const xnd_index_t indices[], int len, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    const xnd_index_t *key;

    assert(len >= 0);
    assert(ndt_is_concrete(t));
    assert(x->ptr != NULL);

    if (len > ndt_logical_ndim(t)) {
        ndt_err_format(ctx, NDT_IndexError, "too many indices");
        return xnd_error;
    }

    if (len == 0) {
        xnd_t next = *x;
        ndt_incref(next.type);
        return next;
    }

    key = &indices[0];

    switch (key->tag) {
    case Index:
        return xnd_index(x, indices, len, ctx);
    case Slice:
        return xnd_slice(x, indices, len, ctx);
    case FieldName:
        ndt_err_format(ctx, NDT_RuntimeError,
            "xnd_multikey: internal error: key must be index or slice");
        return xnd_error;
    }

    /* NOT REACHED: tags should be exhaustive */
    ndt_err_format(ctx, NDT_RuntimeError, "invalid index tag");
    return xnd_error;
}

/*
 * Return a view with a copy of the type.  Indexing into the dtype is
 * not permitted.
 */
static xnd_t
xnd_index(const xnd_t *x, const xnd_index_t indices[], int len, ndt_context_t *ctx)
{
    xnd_index_t xindices[NDT_MAX_DIM+1];
    const ndt_t *t = x->type;
    const xnd_index_t *key;

    assert(len > 0);
    assert(ndt_is_concrete(t));
    assert(x->ptr != NULL);

    /* Hidden element type, insert the stored index. */
    if (have_stored_index(t)) {
        xindices[0].tag = Index;
        xindices[0].Index = get_stored_index(t);
        for (int k = 0; k < len; k++) {
            xindices[k+1] = indices[k];
        }
        indices = xindices;
        len = len+1;
    }

    key = &indices[0];
    assert(key->tag == Index);

    switch (t->tag) {
    case FixedDim: {
        const int64_t i = get_index(key, t->FixedDim.shape, ctx);
        if (i < 0) {
            return xnd_error;
        }

        const xnd_t next = xnd_fixed_dim_next(x, i);
        return xnd_multikey(&next, indices+1, len-1, ctx);
    }

    case VarDim: case VarDimElem: {
        const ndt_t *u;

        if (ndt_is_optional(t)) {
            ndt_err_format(ctx, NDT_NotImplementedError,
                "optional dimensions are temporarily disabled");
            return xnd_error;
        }

        const int64_t i = get_index_var_elem(key, ctx);
        if (i == INT64_MIN) {
            return xnd_error;
        }

        const xnd_t next = xnd_var_dim_next(x, 0, 1, 0);
        const xnd_t tail = xnd_multikey(&next, indices+1, len-1, ctx);
        if (xnd_err_occurred(&tail)) {
            return xnd_error;
        }

        u = ndt_convert_to_var_elem(t, tail.type, i, ctx);
        ndt_decref(tail.type);
        if (u == NULL) {
            return xnd_error;
        }

        xnd_t ret = *x;
        ret.type = u;
        return ret;
    }

    default:
        ndt_err_format(ctx, NDT_IndexError, "type is not indexable");
        return xnd_error;
    }
}

static xnd_t
xnd_slice(const xnd_t *x, const xnd_index_t indices[], int len, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    const xnd_index_t *key;

    assert(len > 0);
    assert(ndt_is_concrete(t));
    assert(x->ptr != NULL);

    key = &indices[0];
    assert(key->tag == Slice);

    switch (t->tag) {
    case FixedDim: {
        int64_t start = key->Slice.start;
        int64_t stop = key->Slice.stop;
        int64_t step = key->Slice.step;
        int64_t shape;

        shape = xnd_slice_adjust_indices(t->FixedDim.shape, &start, &stop, step);

        const xnd_t next = xnd_fixed_dim_next(x, start);
        const xnd_t sliced = xnd_multikey(&next, indices+1, len-1, ctx);
        if (sliced.ptr == NULL) {
            return xnd_error;
        }

        xnd_t ret = *x;
        ret.type = ndt_fixed_dim(sliced.type, shape,
                                 t->Concrete.FixedDim.step * step,
                                 ctx);
        ndt_decref(sliced.type);
        if (ret.type == NULL) {
            return xnd_error;
        }
        ret.index = sliced.index;

        return ret;
    }

    case VarDim: {
        int64_t start = key->Slice.start;
        int64_t stop = key->Slice.stop;
        int64_t step = key->Slice.step;
        ndt_slice_t *slices;
        int32_t nslices;

        if (ndt_is_optional(t)) {
            ndt_err_format(ctx, NDT_NotImplementedError,
                "optional dimensions are temporarily disabled");
            return xnd_error;
        }

        xnd_t next = *x;
        next.type = t->VarDim.type;

        next = xnd_multikey(&next, indices+1, len-1, ctx);
        if (next.ptr == NULL) {
            return xnd_error;
        }

        slices = ndt_var_add_slice(&nslices, t, start, stop, step, ctx);
        if (slices == NULL) {
            ndt_decref(next.type);
            return xnd_error;
        }

        xnd_t ret = *x;
        ret.type = ndt_var_dim(next.type, t->Concrete.VarDim.offsets,
                               nslices, slices, false, ctx);
        ndt_decref(next.type);
        if (ret.type == NULL) {
            return xnd_error;
        }

        ret.index = next.index;

        return ret;
    }

    case VarDimElem: {
        int64_t i = t->VarDimElem.index;

        if (ndt_is_optional(t)) {
            ndt_err_format(ctx, NDT_NotImplementedError,
                "optional dimensions are temporarily disabled");
            return xnd_error;
        }

        const xnd_t next = xnd_var_dim_next(x, 0, 1, 0);
        const xnd_t tail = xnd_multikey(&next, indices, len, ctx);
        if (xnd_err_occurred(&tail)) {
            return xnd_error;
        }

        const ndt_t *u = ndt_convert_to_var_elem(t, tail.type, i, ctx);
        ndt_decref(tail.type);
        if (u == NULL) {
            return xnd_error;
        }

        xnd_t ret = *x;
        ret.type = u;
        return ret;
    }

    case Tuple: {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "slicing tuples is not supported");
        return xnd_error;
    }

    case Record: {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "slicing records is not supported");
        return xnd_error;
    }

    case Union: {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "slicing unions is not supported");
        return xnd_error;
    }

    default:
        ndt_err_format(ctx, NDT_IndexError, "type not sliceable");
        return xnd_error;
    }
}

/* Validate indices for mixed indexed/sliced var dimensions. */
static bool
validate_indices(const xnd_t *x, ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;

    assert(ndt_is_concrete(t));

    switch (t->tag) {
    case VarDim: {
        int64_t start, step, shape;

        shape = ndt_var_indices_non_empty(&start, &step, t, x->index, ctx);
        if (shape < 0) {
            return false;
        }

        for (int64_t i = 0; i < shape; i++) {
            const xnd_t next = xnd_var_dim_next(x, start, step, i);
            if (!validate_indices(&next, ctx)) {
                return false;
            }
        }

        return true;
    }

    case VarDimElem: {
        int64_t start, step, shape;

        shape = ndt_var_indices(&start, &step, t, x->index, ctx);
        if (shape < 0) {
            return false;
        }

        const int64_t k = adjust_index(t->VarDimElem.index, shape, ctx);
        if (k < 0) {
            return false;
        }

        const xnd_t next = xnd_var_dim_next(x, start, step, k);
        return validate_indices(&next, ctx);
    }

    default:
        return true;
    }
}

xnd_t
xnd_subscript(const xnd_t *x, const xnd_index_t indices[], int len,
              ndt_context_t *ctx)
{
    bool have_index = false;
    bool have_slice = false;

    if (len < 0 || len > NDT_MAX_DIM) {
        ndt_err_format(ctx, NDT_IndexError, "too many indices");
        return xnd_error;
    }

    for (int i = 0; i < len; i++) {
        if (indices[i].tag == Index) {
            have_index = true;
        }
        if (indices[i].tag == Slice) {
            have_slice = true;
        }
    }

    if (have_slice) {
        xnd_t res = xnd_multikey(x, indices, len, ctx);
        if (xnd_err_occurred(&res)) {
            return xnd_error;
        }

        if (have_index && !validate_indices(&res, ctx)) {
            ndt_decref(res.type);
            return xnd_error;
        }

        return res;
    }
    else {
        xnd_t res = xnd_subtree(x, indices, len, ctx);
        if (res.ptr == NULL) {
            return xnd_error;
        }

        ndt_incref(res.type);
        return res;
    }
}


/*****************************************************************************/
/*                                Unstable API                               */
/*****************************************************************************/

/* error return value */
const xnd_view_t xnd_view_error = {
  .flags = 0,
  .obj = NULL,
  .view = { .bitmap = {.data=NULL, .size=0, .next=NULL},
            .index = 0,
            .type = NULL,
            .ptr = NULL }
};

int
xnd_view_err_occurred(const xnd_view_t *x)
{
    return x->view.ptr == NULL;
}

void
xnd_view_clear(xnd_view_t *x)
{
    xnd_del_buffer(&x->view, x->flags);
    x->flags = 0;
    x->obj = NULL;
}

xnd_view_t
xnd_view_from_xnd(const void *obj, const xnd_t *x)
{
    xnd_view_t res;

    res.flags = 0;
    res.obj = obj;
    res.view = *x;

    return res;
}

xnd_view_t
xnd_view_subscript(const xnd_view_t *x, const xnd_index_t indices[], int len,
                   ndt_context_t *ctx)
{
    xnd_view_t res;

    res.flags = XND_OWN_TYPE;
    res.obj = x->obj;

    res.view = xnd_subscript(&x->view, indices, len, ctx);
    if (xnd_err_occurred(&res.view)) {
        return xnd_view_error;
    }

    return res;
}


/*****************************************************************************/
/*                                Float format                               */
/*****************************************************************************/

#define IEEE_LITTLE_ENDIAN 0
#define IEEE_BIG_ENDIAN    1
static int xnd_double_format = 0;
static int xnd_float_format = 0;

int
xnd_init_float(ndt_context_t *ctx)
{
    double x = 9006104071832581.0;
    float y = 16711938.0;

#ifndef _MSC_VER /* Suppress a warning, no need to check on Windows. */
    if (sizeof(double) != 8) {
        ndt_err_format(ctx, NDT_RuntimeError,
            "unsupported platform, need sizeof(double)==8");
        return -1;

    }

    if (sizeof(float) != 4) {
        ndt_err_format(ctx, NDT_RuntimeError,
            "unsupported platform, need sizeof(float)==4");
        return -1;
    }
#endif

    if (memcmp(&x, "\x43\x3f\xff\x01\x02\x03\x04\x05", 8) == 0) {
        xnd_double_format = IEEE_BIG_ENDIAN;
    }
    else if (memcmp(&x, "\x05\x04\x03\x02\x01\xff\x3f\x43", 8) == 0) {
        xnd_double_format = IEEE_LITTLE_ENDIAN;
    }
    else {
        ndt_err_format(ctx, NDT_RuntimeError,
            "unsupported platform, could not detect double endianness");
        return -1;
    }

    if (memcmp(&y, "\x4b\x7f\x01\x02", 4) == 0) {
        xnd_float_format = IEEE_BIG_ENDIAN;
    }
    else if (memcmp(&y, "\x02\x01\x7f\x4b", 4) == 0) {
        xnd_float_format = IEEE_LITTLE_ENDIAN;
    }
    else {
        ndt_err_format(ctx, NDT_RuntimeError,
            "unsupported platform, could not detect float endianness");
        return -1;
    }

    return 0;
}

bool
xnd_float_is_little_endian(void)
{
    return xnd_float_format==IEEE_LITTLE_ENDIAN;
}

bool
xnd_float_is_big_endian(void)
{
    return xnd_float_format==IEEE_BIG_ENDIAN;
}

bool
xnd_double_is_little_endian(void)
{
    return xnd_double_format==IEEE_LITTLE_ENDIAN;
}

bool
xnd_double_is_big_endian(void)
{
    return xnd_double_format==IEEE_BIG_ENDIAN;
}

static float
bfloat16_to_float(uint16_t b)
{
    float f = 0;
    uint16_t *p = (uint16_t *)((char *)&f);

    if (xnd_float_is_big_endian()) {
        p[0] = b;
    }
    else {
        p[1] = b;
    }

    return f;
}

/*
 * Unlike the corresponding Python conversion functions, Tensorflow does
 * not raise OverflowError.
 */
void
xnd_bfloat_pack(char *p, double x)
{
    float f = (float)x;
    uint16_t u16;

    u16 = xnd_round_to_bfloat16(f);
    PACK_SINGLE(p, u16, uint16_t, 0);
}

double
xnd_bfloat_unpack(char *p)
{
    uint16_t u16;

    UNPACK_SINGLE(u16, p, uint16_t, 0);
    return bfloat16_to_float(u16);
}

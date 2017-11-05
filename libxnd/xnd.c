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


#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include "ndtypes.h"
#include "xnd.h"


static int xnd_init(xnd_t x, const uint32_t flags, ndt_context_t *ctx);
static void xnd_clear(xnd_t xu, const uint32_t flags);


/*****************************************************************************/
/*                              Error handling                               */
/*****************************************************************************/

/* error return value */
const xnd_t xnd_error = {.index=0, .type=NULL, .ptr=NULL};

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
xnd_new(const ndt_t *t, uint32_t flags, ndt_context_t *ctx)
{
    xnd_t x;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError,
            "cannot create xnd container from abstract type");
        return NULL;
    }

    x.index = 0;
    x.type = t;

    x.ptr = ndt_aligned_calloc(t->data_align, t->data_size);
    if (x.ptr == NULL) {
        ndt_memory_error(ctx);
        return NULL;
    }

    if (xnd_init(x, flags, ctx) < 0) {
        ndt_aligned_free(x.ptr);
        return NULL;
    }

    return x.ptr;
}

/*
 * Initialize typed memory. If the XND_OWN_POINTERS flag is set, allocate
 * memory for all pointer subtypes and initialize that memory. Otherwise,
 * set pointers to NULL.
 *
 * Pointer subtypes include any type of the form "Pointer(t)".
 *
 * Never allocated are (sizes are not known):
 *   - "string" type (pointer to NUL-terminated UTF8 string)
 *   - data of the "bytes" type: {size: size_t, data: uint8_t *bytes}
 *
 * At all times the data pointers must be NULL or pointers to valid memory.
 */
static int
xnd_init(xnd_t x, uint32_t flags, ndt_context_t *ctx)
{
    const ndt_t *t = x.type;
    xnd_t next;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError,
            "cannot initialize concrete memory from abstract type");
        return -1;
    }

    switch (t->tag) {
    case FixedDim: {
        int64_t i;

        assert(x.index == 0);
        next.index = 0;
        next.type = t->FixedDim.type;

        for (i = 0; i < t->FixedDim.shape; i++) {
            next.ptr = x.ptr + i * t->Concrete.FixedDim.itemsize;
            if (xnd_init(next, flags, ctx) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case VarDim: {
        const int32_t noffsets = t->Concrete.VarDim.noffsets;
        int32_t start, stop;
        int64_t shape, i;

        if (x.index < 0 || x.index+1 >= noffsets) {
            ndt_err_format(ctx, NDT_RuntimeError,
                "xnd: offset index out of range: index=%" PRIi32 ", noffsets=%" PRIi32,
                x.index, noffsets);
            return -1;
        }

        start = t->Concrete.VarDim.offsets[x.index];
        stop = t->Concrete.VarDim.offsets[x.index+1];

        shape = stop - start;

        next.type = t->VarDim.type;
        next.ptr = x.ptr;

        for (i = 0; i < shape; i++) {
            next.index = start + i;

            if (xnd_init(next, flags, ctx) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case Tuple: {
        int64_t i;

        next.index = 0;

        for (i = 0; i < t->Tuple.shape; i++) {
            next.type = t->Tuple.types[i];
            next.ptr = x.ptr + t->Concrete.Tuple.offset[i];

            if (xnd_init(next, flags, ctx) < 0) {
                xnd_clear(next, flags);
                return -1;
            }
        }

        return 0;
    }

    case Record: {
        int64_t i;

        next.index = 0;

        for (i = 0; i < t->Record.shape; i++) {
            next.type = t->Record.types[i];
            next.ptr = x.ptr + t->Concrete.Record.offset[i];

            if (xnd_init(next, flags, ctx) < 0) {
                xnd_clear(next, flags);
                return -1;
            }
        }

        return 0;
    }

    /*
     * Pointer represents a pointer to an explicit type. If XND_OWN_POINTERS
     * is set, allocate memory for that type and set the pointer.
     */
    case Pointer: {
        if (flags & XND_OWN_POINTERS) {
            void *pointer = ndt_aligned_calloc(t->data_align, t->data_size);
            if (pointer == NULL) {
                ndt_err_format(ctx, NDT_MemoryError, "out of memory");
                return -1;
            }

            next.index = 0;
            next.type = t->Pointer.type;
            next.ptr = pointer;

            if (xnd_init(next, flags, ctx) < 0) {
                xnd_clear(next, flags);
                return -1;
            }

            XND_POINTER_DATA(x.ptr) = pointer;
        }

        return 0;
    }

    /* Constr is a named explicit type. */
    case Constr: {
        next.index = 0;
        next.type = t->Constr.type;
        next.ptr = x.ptr;

        if (xnd_init(next, flags, ctx) < 0) {
            xnd_clear(next, flags);
            return -1;
        }

        return 0;
    }

    /* Optional dtypes inside containers using validity bitmaps. */
    case OptionItem:
        return 0;

    /* The standalone option is represented as a tuple: (type, byte).
     * 'byte' is the flag that indicates whether an element is present
     * and is initialized to 0 (undefined, NA).
     */
    case Option:
        ndt_err_format(ctx, NDT_NotImplementedError,
                       "option semantics currently only defined for dimensions "
                       "and array members");
        return -1;

    /* Nominal represents an opaque piece of memory that just has a size
       and an alignment. If it represents a pointer, the pointer needs to
       be externally initialized and does not belong to the memory block. */
    case Nominal:
        return 0;

    /* Categorical is already initialized by calloc(). */
    case Categorical:
        return 0;

    /* Primitive types are already initialized by calloc(). */
    case Bool:
    case Int8: case Int16: case Int32: case Int64:
    case Uint8: case Uint16: case Uint32: case Uint64:
    case Float16: case Float32: case Float64:
    case Complex32: case Complex64: case Complex128:
    case FixedString: case FixedBytes:
    case Char: case String: case Bytes:
        return 0;

    case AnyKind: case SymbolicDim: case EllipsisDim: case Typevar:
    case ScalarKind: case SignedKind: case UnsignedKind: case FloatKind:
    case ComplexKind: case FixedStringKind: case FixedBytesKind: case Field:
    case Module:
        ndt_err_format(ctx, NDT_NotImplementedError, "modules not implemented");
        return -1;
    case Void: case Function:
        /* NOT REACHED: intercepted by ndt_is_abstract(). */
        ndt_err_format(ctx, NDT_RuntimeError, "unexpected abstract type");
        return -1;
   }

   /* NOT REACHED: tags should be exhaustive */
   ndt_err_format(ctx, NDT_RuntimeError, "invalid type tag");
   return -1;
}

/*
 * Create a type from 'datashape' and return a new master buffer for that
 * type. Any combination of flags that include XND_OWN_TYPE can be passed.
 */
xnd_master_t *
xnd_empty_from_string(const char *datashape, uint32_t flags, ndt_context_t *ctx)
{
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

    t = ndt_from_string(datashape, ctx);
    if (t == NULL) {
        ndt_free(x);
        return NULL;
    }

    ptr = xnd_new(t, flags, ctx);
    if (ptr == NULL) {
        ndt_del(t);
        ndt_free(x);
        return NULL;
    }

    x->flags = flags;
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
    xnd_master_t *x;
    char *ptr;

    if (flags & XND_OWN_TYPE) {
        ndt_err_format(ctx, NDT_InvalidArgumentError,
            "xnd_empty_from_type: XND_OWN_TYPE must not be set");
        return NULL;
    }

    x = ndt_alloc(1, sizeof *x);
    if (x == NULL) {
        return ndt_memory_error(ctx);
    }

    ptr = xnd_new(t, flags, ctx);
    if (ptr == NULL) {
        ndt_free(x);
        return NULL;
    }

    x->flags = flags;
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
xnd_clear_pointer(xnd_t *x, const uint32_t flags)
{
    assert(x->type->tag == Pointer);

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
xnd_clear(xnd_t x, const uint32_t flags)
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
        int64_t i;

        assert(x.index == 0);
        next.index = 0;
        next.type = t->FixedDim.type;

        for (i = 0; i < t->FixedDim.shape; i++) {
            next.ptr = x.ptr + i * t->Concrete.FixedDim.itemsize;
            xnd_clear(next, flags);
        }

        return;
    }

    case VarDim: {
        int32_t start, stop, shape, i;

        assert(0 <= x.index && x.index+1 < t->Concrete.VarDim.noffsets);

        start = t->Concrete.VarDim.offsets[x.index];
        stop = t->Concrete.VarDim.offsets[x.index+1];
        shape = stop - start;

        next.type = t->VarDim.type;
        next.ptr = x.ptr;

        for (i = 0; i < shape; i++) {
            next.index =  start + i;
            xnd_clear(next, flags);
        }

        return;
    }

    case Tuple: {
        int64_t i;

        next.index = 0;

        for (i = 0; i < t->Tuple.shape; i++) {
            next.type = t->Tuple.types[i];
            next.ptr = x.ptr + t->Concrete.Tuple.offset[i];
            xnd_clear(next, flags);
        }

        return;
    }

    case Record: {
        int64_t i;

        next.index = 0;

        for (i = 0; i < t->Record.shape; i++) {
            next.type = t->Record.types[i];
            next.ptr = x.ptr + t->Concrete.Record.offset[i];
            xnd_clear(next, flags);
        }

        return;
    }

    case Pointer: {
        next.index = 0;

        if (flags & XND_OWN_POINTERS) {
            next.type = t->Pointer.type;
            next.ptr = XND_POINTER_DATA(x.ptr);
            xnd_clear(next, flags);
            xnd_clear_pointer(&x, flags);
        }

        return;
    }

    case Constr: {
        next.index = 0;
        next.type = t->Constr.type;
        next.ptr = x.ptr;
        xnd_clear(next, flags);
        return;
    }

    case Nominal:
        /* The Nominal type is opaque: no deallocation strategy is known. */
        return;

    case Bool:
    case Int8: case Int16: case Int32: case Int64:
    case Uint8: case Uint16: case Uint32: case Uint64:
    case Float16: case Float32: case Float64:
    case Complex32: case Complex64: case Complex128:
    case FixedString: case FixedBytes:
        return;

    case String:
        xnd_clear_string(&x, flags);
        return;

    case Bytes:
        xnd_clear_bytes(&x, flags);
        return;

    case Categorical:
        /* Categorical values ar, flagse just indices into the categories. */
        return;

    case Char:
        /* Just a scalar. */
        return;

    case Option: case OptionItem:
        /* XXX Not implemented. */
        return;

    case Module:
        /* XXX Not implemented. */
        return;

    /* NOT REACHED: intercepted by ndt_is_abstract(). */
    case AnyKind: case SymbolicDim: case EllipsisDim: case Typevar:
    case ScalarKind: case SignedKind: case UnsignedKind: case FloatKind:
    case ComplexKind: case FixedStringKind: case FixedBytesKind: case Field:
    case Void: case Function:
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
    xnd_clear(x->master, x->flags);

    if (x->flags & XND_OWN_TYPE) {
        ndt_del((ndt_t *)x->master.type);
    }

    ndt_aligned_free(x->master.ptr);
    ndt_free(x);
}


/*****************************************************************************/
/*                 Subtrees (single elements are a special case)             */
/*****************************************************************************/

/* Return a typed subtree of a memory block */
xnd_t
xnd_subtree(xnd_t x, const int64_t *indices, int len, ndt_context_t *ctx)
{
    const ndt_t *t = x.type;
    xnd_t next;
    int64_t i;

    assert(ndt_is_concrete(t));

    if (len == 0) {
        if (ndt_is_optional(t)) {
            ndt_err_format(ctx, NDT_NotImplementedError,
                "options are temporarily disabled");
            return xnd_error;
        }
        return x;
    }

    i = indices[0];

    switch (t->tag) {
    case FixedDim: {
        if (i < 0 || i >= t->FixedDim.shape) {
            ndt_err_format(ctx, NDT_ValueError,
                "fixed dim index out of bounds");
            return xnd_error;
        }

        assert(x.index == 0);
        next.index = x.index;
        next.type = t->FixedDim.type;
        next.ptr = x.ptr + i * t->Concrete.FixedDim.stride;

        break;
    }

    case VarDim: {
        const int32_t noffsets = t->Concrete.VarDim.noffsets;
        int32_t start, stop;

        if (ndt_is_optional(t)) {
            ndt_err_format(ctx, NDT_NotImplementedError,
                "optional dimensions are temporarily disabled");
            return xnd_error;
        }

        if (x.index < 0 || x.index+1 >= noffsets) {
            ndt_err_format(ctx, NDT_RuntimeError,
                "var dim offset index out of bounds");
            return xnd_error;
        }

        start = t->Concrete.VarDim.offsets[x.index];
        stop = t->Concrete.VarDim.offsets[x.index+1];

        if (i < 0 || i >= stop) {
            ndt_err_format(ctx, NDT_ValueError, "var dim index out of bounds");
            return xnd_error;
        }

        next.index = start + i;
        next.type = t->VarDim.type;
        next.ptr = x.ptr;

        break;
    }

    case Tuple: {
        if (i < 0 || i >= t->Tuple.shape) {
            ndt_err_format(ctx, NDT_ValueError, "tuple index out of bounds");
            return xnd_error;
        }

        next.index = 0;
        next.type = t->Tuple.types[i];
        next.ptr += t->Concrete.Tuple.offset[i];

        break;
    }

    case Record: {
        if (i < 0 || i >= t->Record.shape) {
            ndt_err_format(ctx, NDT_ValueError, "record index out of bounds");
            return xnd_error;
        }

        next.type = t->Record.types[i];
        next.index = 0;
        next.ptr += t->Concrete.Record.offset[i];

        break;
    }

    default:
        ndt_err_format(ctx, NDT_ValueError, "type not indexable");
        return xnd_error;
    }

    return xnd_subtree(next, indices+1, len-1, ctx);
}

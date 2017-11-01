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
#include <assert.h>
#include "ndtypes.h"
#include "xnd.h"


/* error return value */
static xnd_t err = {NULL, 0, NULL};


/*****************************************************************************/
/*              Allocate and initialize empty typed memory                   */
/*****************************************************************************/

/* Allocate typed memory */
char *
xnd_new(const ndt_t *t, bool alloc_pointers, ndt_context_t *ctx)
{
    char *ptr;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError,
                       "cannot create xnd container from abstract type");
        return NULL;
    }

    ptr = ndt_aligned_calloc(t->data_align, t->data_size);
    if (ptr == NULL) {
        return ndt_memory_error(ctx);
    }

    if (xnd_init(ptr, t, alloc_pointers, ctx) < 0) {
        ndt_aligned_free(ptr);
        return NULL;
    }

    return ptr;
}

/*
 * Initialize typed memory. If 'alloc_pointers' is true, allocate memory
 * for all pointer subtypes and initialize that memory. Otherwise, set
 * pointers to NULL.
 *
 * Pointer subtypes include:
 *   - Any type of the form 'Pointer(t)'.
 *
 * Never allocated are (sizes are not known):
 *   - The 'String' type (pointer to NUL-terminated UTF8 string).
 *   - The 'Bytes' type ({size: size_t, data: char *bytes}).
 *
 *   At all times the data pointers must be NULL or pointers to valid memory.
 *   Once initialized, the pointers belong to the array.
 */
int
xnd_init(char *ptr, const ndt_t *t, bool alloc_pointers, ndt_context_t *ctx)
{
    char *item;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError,
                       "cannot initialize concrete memory from abstract type");
        return -1;
    }

    switch (t->tag) {
    case FixedDim: case VarDim: {
        const ndt_t *dims[NDT_MAX_DIM];
        const ndt_t *dtype;

        ndt_const_dims_dtype(dims, &dtype, t);
        return xnd_init(ptr, dtype, alloc_pointers, ctx);
    }

    case Tuple: {
        int64_t i;

        for (i = 0; i < t->Tuple.shape; i++) {
            item = ptr + t->Concrete.Tuple.offset[i];
            if (xnd_init(item, t->Tuple.types[i], alloc_pointers, ctx) < 0) {
                // xnd_clear(ptr, t);
                return -1;
            }
        }

        return 0;
    }

    case Record: {
        int64_t i;

        for (i = 0; i < t->Record.shape; i++) {
            item = ptr + t->Concrete.Record.offset[i];
            if (xnd_init(item, t->Record.types[i], alloc_pointers, ctx) < 0) {
                // xnd_clear(ptr, t);
                return -1;
            }
        }

        return 0;
    }

    /*
     * Pointer represents a pointer to an explicit type. If 'alloc_pointers'
     * is true, allocate memory for that type and set the pointer.  Otherwise,
     * if an external source sets the pointers later, they must a) have been
     * allocated by ndt_aligned_calloc() and b) they belong to the array and
     * will be traversed and then deallocated when the array is deallocated.
     *
     * If this is not desired, use the opaque 'Nominal' type instead.
     */
    case Pointer:
        if (alloc_pointers) {
            XND_POINTER_DATA(ptr) = ndt_aligned_calloc(t->data_align, t->data_size);
            if (XND_POINTER_DATA(ptr) == NULL) {
                ndt_err_format(ctx, NDT_MemoryError, "out of memory");
                return -1;
            }

            if (xnd_init(XND_POINTER_DATA(ptr), t->Pointer.type, alloc_pointers,
                ctx) < 0) {
                // xnd_clear(ptr, t);
                return -1;
            }
        }
        else {
            XND_POINTER_DATA(ptr) = NULL;
        }

        return 0;

    /* Constr is a named explicit type */
    case Constr:
        if (xnd_init(ptr, t->Constr.type, alloc_pointers, ctx) < 0) {
            // xnd_clear(ptr, t);
            return -1;
        }
        return 0;

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
       be externally initialized and does not belong to the array. */
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

/* Return the space required for a type.  Pointer types are allocated. */
xnd_t
xnd_empty(const char *datashape, ndt_context_t *ctx)
{
    xnd_t x;

    x.type = ndt_from_string(datashape, ctx);
    if (x.type == NULL) {
        return err;
    }

    x.index = 0;

    x.ptr = xnd_new(x.type, 1, ctx);
    if (x.ptr == NULL) {
        ndt_del((ndt_t *)x.type);
        return err;
    }

    return x;
}

/* Delete the space required for a type. Pointer types are deallocated. */
void
xnd_del(xnd_t x)
{
    // xnd_clear(x);
    ndt_del((ndt_t *)x.type);
    ndt_aligned_free(x.ptr);
}

/* Clear pointer types */
/* XXX void xnd_clear(char *ptr, const ndt_t *t) {} */


/*****************************************************************************/
/*                Subarrays (single elements are a special case)             */
/*****************************************************************************/

#if 0
/*
 * Set the validity bit of a subarray and of all dimensions that need to
 * be accessed to reach that subarray to 1.  Only useful for types that
 * contain optional dimensions or values.
 */
int
xnd_subarray_set_valid(xnd_t a, const int64_t *indices, int len,
                       ndt_context_t *ctx)
{
    const ndt_t *t = a.type;
    xnd_t next;
    int64_t shape;
    int i;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError, "array has abstract type");
        return -1;
    }

    if (len == 0) {
        switch (t->tag) {
        case OptionItem:
            assert(ndt_is_optional(t));
            XND_DATA_SET_VALID(&a);
            return 0;
        default:
            return 0;
        }
    }

    i = indices[0];

    switch (t->tag) {
    case Array:
        next.base = &a;
        next.type = t->Array.type;
        next.ptr = a.ptr + t->Concrete.Array.data[t->Concrete.Array.ndim_start];
        return xnd_subarray_set_valid(next, indices, len, ctx);
    case FixedDim:
        shape = t->FixedDim.shape;
        next.base = a.base;
        next.type = t->FixedDim.type;
        next.ptr = a.ptr + i * t->Concrete.FixedDim.stride;
        break;
    case VarDim:
        if (ndt_is_optional(t)) {
            XND_DATA_SET_VALID(&a);
        }
        shape = XND_VAR_SHAPE(&a);
        next.base = a.base;
        next.type = t->VarDim.type;
        next.ptr = XND_NEXT_DIM(&a) + t->Concrete.VarDim.suboffset + i * t->Concrete.VarDim.stride;
        break;
    case Tuple:
        shape = t->Tuple.shape;
        next.base = a.base;
        next.type = t->Tuple.types[i];
        next.ptr += t->Concrete.Tuple.offset[i];
        break;
    case Record:
        shape = t->Record.shape;
        next.base = a.base;
        next.type = t->Record.types[i];
        next.ptr += t->Concrete.Record.offset[i];
        break;

    default:
        ndt_err_format(ctx, NDT_ValueError, "type not indexable");
        return -1;
    }

    if (i < 0 || i >= shape) {
        ndt_err_format(ctx, NDT_ValueError, "index out of bounds");
        return -1;
    }

    return xnd_subarray_set_valid(next, indices+1, len-1, ctx);
}
#endif

/* Return a typed subarray */
xnd_t
xnd_subarray(const xnd_t x, const int64_t *indices, int len, ndt_context_t *ctx)
{
    const ndt_t *t = x.type;
    xnd_t next;
    int i;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError, "array has abstract type");
        return err;
    }

    if (len == 0) {
        if (ndt_is_optional(t)) {
            ndt_err_format(ctx, NDT_NotImplementedError,
                "options temporarily disabled");
            return err;
        }
        return x;
    }

    i = indices[0];

    switch (t->tag) {
    case FixedDim: {
        assert(x.index == 0);

        if (i < 0 || i >= t->FixedDim.shape) {
            ndt_err_format(ctx, NDT_ValueError,
                "fixed dim index out of bounds");
            return err;
        }

        next.type = t->FixedDim.type;
        next.index = 0;
        next.ptr = x.ptr + i * t->Concrete.FixedDim.stride;
        break;
    }

    case VarDim: {
        const int32_t noffsets = t->Concrete.VarDim.noffsets;
        int32_t start, stop;

        if (ndt_is_optional(t)) {
            ndt_err_format(ctx, NDT_NotImplementedError,
                "optional dimensions temporarily disabled");
            return err;
        }

        if (x.index+1 >= noffsets) {
            ndt_err_format(ctx, NDT_RuntimeError,
                "var dim offset index out of bounds");
            return err;
        }

        start = t->Concrete.VarDim.offsets[x.index];
        stop = t->Concrete.VarDim.offsets[x.index+1];

        if (i < 0 || i >= stop) {
            ndt_err_format(ctx, NDT_ValueError, "var dim index out of bounds");
            return err;
        }

        next.type = t->VarDim.type;
        next.index = start + i;
        next.ptr = x.ptr;
        break;
    }

    case Tuple: {
        if (i < 0 || i >= t->Tuple.shape) {
            ndt_err_format(ctx, NDT_ValueError, "tuple index out of bounds");
            return err;
        }

        next.type = t->Tuple.types[i];
        next.index = 0;
        next.ptr += t->Concrete.Tuple.offset[i];
        break;
    }

    case Record: {
        if (i < 0 || i >= t->Record.shape) {
            ndt_err_format(ctx, NDT_ValueError, "record index out of bounds");
            return err;
        }

        next.type = t->Record.types[i];
        next.index = 0;
        next.ptr += t->Concrete.Record.offset[i];
        break;
    }

    default:
        ndt_err_format(ctx, NDT_ValueError, "type not indexable");
        return err;
    }

    return xnd_subarray(next, indices+1, len-1, ctx);
}


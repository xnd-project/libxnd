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
static nd_array_t err = {NULL, NULL, NULL};


/*****************************************************************************/
/*              Allocate and initialize empty typed memory                   */
/*****************************************************************************/

/* Allocate typed memory */
char *
nd_new(const ndt_t *t, bool alloc_pointers, ndt_context_t *ctx)
{
    char *ptr;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError,
                       "cannot create concrete memory from abstract type");
        return NULL;
    }

    ptr = ndt_calloc(1, t->Concrete.size);
    if (ptr == NULL) {
        return ndt_memory_error(ctx);
    }

    if (nd_init(ptr, t, alloc_pointers, ctx) < 0) {
        ndt_free(ptr);
        return NULL;
    }

    return ptr;
}

/* Initialize the var dimensions represented by indey arrays. The index
   arrays use the smallest possible integer type. */
static int
init_var_data(char *ptr, enum ndt_dim dim_type, const int64_t *offsets,
              int nshapes, ndt_context_t *ctx)
{
    int i;

    /* Indices into the next dimension: cumulative sum of shapes */
    switch (dim_type) {
    case DimUint8: {
        uint8_t *p = (uint8_t *)ptr;

        for (i = 0; i < nshapes+1; i++) {
            p[i] = offsets[i];
        }
        return 0;
    }
    case DimUint16: {
        uint16_t *p = (uint16_t *)ptr;

        for (i = 0; i < nshapes+1; i++) {
            p[i] = offsets[i];
        }
        return 0;
    }
    case DimUint32: {
        uint32_t *p = (uint32_t *)ptr;

        for (i = 0; i < nshapes+1; i++) {
            p[i] = offsets[i];
        }
        return 0;
    }
    case DimInt32: {
        int32_t *p = (int32_t *)ptr;

        for (i = 0; i < nshapes+1; i++) {
            p[i] = offsets[i];
        }
        return 0;
    }
    case DimInt64: {
        int64_t *p = (int64_t *)ptr;

        for (i = 0; i < nshapes+1; i++) {
            p[i] = offsets[i];
        }
    }
    case DimNone:
        goto unknown_dimension_type;
    }

unknown_dimension_type:
    ndt_err_format(ctx, NDT_RuntimeError, "dimension type is not set");
    return -1;
}

/*
 * Initialize typed memory. If 'alloc_pointers' is true, allocate memory
 * for all pointer subtypes and initialize that memory. Otherwise, set
 * pointers to NULL.
 */
int
nd_init(char *ptr, const ndt_t *t, bool alloc_pointers, ndt_context_t *ctx)
{
    char *item;
    int ret;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError,
                       "cannot initialize concrete memory from abstract type");
        return -1;
    }

    switch (t->tag) {
    case Array: {
        const ndt_t *dims[NDT_MAX_DIM];
        const ndt_t *dtype;
        int i;

        ndt_const_dims_dtype(dims, &dtype, t->Array.type);

        for (i = 0; i < t->ndim; i++) {
            assert(ndt_is_concrete(dims[i]));

            /* possible bitmaps initialized by calloc() */
            switch (dims[i]->tag) {
            case FixedDim:
                break;
            case VarDim:
                ret = init_var_data(ptr + t->Concrete.Array.data[dims[i]->ndim],
                                    t->Concrete.Array.dim_type,
                                    dims[i]->Concrete.VarDim.offsets,
                                    dims[i]->Concrete.VarDim.nshapes,
                                    ctx);
                if (ret < 0) {
                    // nd_clear(ptr, t);
                    return -1;
                }
                break;
            default:
                abort(); /* NOT REACHED */
            }
        }

        return nd_init(ptr, dtype, alloc_pointers, ctx);
    }

    case Tuple: {
        int64_t i;

        for (i = 0; i < t->Tuple.shape; i++) {
            item = ptr + t->Concrete.Tuple.offset[i];
            if (nd_init(item, t->Tuple.types[i], alloc_pointers, ctx) < 0) {
                // nd_clear(ptr, t);
                return -1;
            }
        }

        return 0;
    }

    case Record: {
        int64_t i;

        for (i = 0; i < t->Record.shape; i++) {
            item = ptr + t->Concrete.Record.offset[i];
            if (nd_init(item, t->Record.types[i], alloc_pointers, ctx) < 0) {
                // nd_clear(ptr, t);
                return -1;
            }
        }

        return 0;
    }

    /*
     * Pointer represents a pointer to an explicit type. If 'alloc_pointers'
     * is true, allocate memory for that type and set the pointer.  Otherwise,
     * if an external source sets the pointers later, they must a) have been
     * allocated by ndt_alloc() and b) they belong to the array and will be
     * traversed and then deallocated when the array is deallocated.
     *
     * If this is not desired, use the opaque 'Nominal' type instead.
     */
    case Pointer:
        if (alloc_pointers) {
            ND_POINTER_DATA(ptr) = ndt_calloc(1, t->Concrete.size);
            if (ND_POINTER_DATA(ptr) == NULL) {
                ndt_err_format(ctx, NDT_MemoryError, "out of memory");
                return -1;
            }

            if (nd_init(ND_POINTER_DATA(ptr), t->Pointer.type, alloc_pointers,
                ctx) < 0) {
                // nd_clear(ptr, t);
                return -1;
            }
        }
        else {
            ND_POINTER_DATA(ptr) = NULL;
        }

        return 0;

    /* Constr is a named explicit type */
    case Constr:
        if (nd_init(ptr, t->Constr.type, alloc_pointers, ctx) < 0) {
            // nd_clear(ptr, t);
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

    case FixedDim: case VarDim:
        /* NOT REACHED: intercepted by Array. */
        ndt_err_format(ctx, NDT_RuntimeError, "unexpected fixed or var dim");
        return -1;

    case AnyKind: case SymbolicDim: case EllipsisDim: case Typevar:
    case ScalarKind: case SignedKind: case UnsignedKind: case FloatKind:
    case ComplexKind: case FixedStringKind: case FixedBytesKind: case Field:
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
nd_array_t
nd_empty(const char *datashape, ndt_context_t *ctx)
{
    nd_array_t a;

    a.type = ndt_from_string(datashape, ctx);
    if (a.type == NULL) {
        return err;
    }

    a.ptr = nd_new(a.type, 1, ctx);
    if (a.ptr == NULL) {
        ndt_del((ndt_t *)a.type);
        return err;
    }

    a.base = NULL;

    return a;
}

/* Delete the space required for a type. Pointer types are deallocated. */
void
nd_del(nd_array_t a)
{
    // nd_clear(a);
    ndt_del((ndt_t *)a.type);
    ndt_free(a.ptr);
}

/* Clear pointer types */
/* XXX void nd_clear(char *ptr, const ndt_t *t) {} */


/*****************************************************************************/
/*                Subarrays (single elements are a special case)             */
/*****************************************************************************/

/*
 * Set the validity bit of a subarray and of all dimensions that need to
 * be accessed to reach that subarray to 1.  Only useful for types that
 * contain optional dimensions or values.
 */
int
nd_subarray_set_valid(nd_array_t a, const int64_t *indices, int len,
                      ndt_context_t *ctx)
{
    const ndt_t *t = a.type;
    nd_array_t next;
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
            ND_DATA_SET_VALID(&a);
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
        return nd_subarray_set_valid(next, indices, len, ctx);
    case FixedDim:
        shape = t->FixedDim.shape;
        next.base = a.base;
        next.type = t->FixedDim.type;
        next.ptr = a.ptr + i * t->Concrete.FixedDim.stride;
        break;
    case VarDim:
        if (ndt_is_optional(t)) {
            ND_DATA_SET_VALID(&a);
        }
        shape = ND_VAR_SHAPE(&a);
        next.base = a.base;
        next.type = t->VarDim.type;
        next.ptr = ND_NEXT_DIM(&a) + t->Concrete.VarDim.suboffset + i * t->Concrete.VarDim.stride;
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

    return nd_subarray_set_valid(next, indices+1, len-1, ctx);
}

/* Return a typed subarray */
nd_array_t
nd_subarray(const nd_array_t a, const int64_t *indices, int len, ndt_context_t *ctx)
{
    const ndt_t *t = a.type;
    nd_array_t next;
    int64_t shape;
    int i;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError, "array has abstract type");
        return err;
    }

    if (len == 0) {
        if (ndt_is_optional(t)) {
            switch (t->tag) {
            case VarDim: case OptionItem:
                if (!ND_DATA_IS_VALID(&a)) {
                    next = a;
                    next.ptr = ND_MISSING;
                    return next;
                }
            default:
                return a;
            }
        }
        else {
            return a;
        }
    }

    i = indices[0];

    switch (t->tag) {
    case Array:
        next.base = &a;
        next.type = t->Array.type;
        next.ptr = a.ptr + t->Concrete.Array.data[t->Concrete.Array.ndim_start];
        return nd_subarray(next, indices, len, ctx);
    case FixedDim:
        shape = t->FixedDim.shape;
        next.base = a.base;
        next.type = t->FixedDim.type;
        next.ptr = a.ptr + i * t->Concrete.FixedDim.stride;
        break;
    case VarDim:
        if (ndt_is_optional(t) && !ND_DATA_IS_VALID(&a)) {
            goto missing_dimension_error;
        }
        shape = ND_VAR_SHAPE(&a);
        next.base = a.base;
        next.type = t->VarDim.type;
        next.ptr = ND_NEXT_DIM(&a) + t->Concrete.VarDim.suboffset + i * t->Concrete.VarDim.stride;
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
        return err;
    }

    if (i < 0 || i >= shape) {
        ndt_err_format(ctx, NDT_ValueError, "index out of bounds");
        return err;
    }

    return nd_subarray(next, indices+1, len-1, ctx);


missing_dimension_error:
    ndt_err_format(ctx, NDT_ValueError, "cannot index missing dimension");
    return err;
}

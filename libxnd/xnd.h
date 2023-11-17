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


#ifndef XND_H
#define XND_H


#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "ndtypes.h"

#ifdef __cplusplus
  #include <cstdint>
#else
  #include <stdint.h>
  #include <inttypes.h>
#endif


#ifdef _MSC_VER
  #if defined (XND_EXPORT)
    #define XND_API __declspec(dllexport)
  #elif defined(XND_IMPORT)
    #define XND_API __declspec(dllimport)
  #else
    #define XND_API
  #endif

  #include "malloc.h"
  #define ALLOCA(type, name, nmemb) type *name = _alloca(nmemb * sizeof(type))
#else
  #define XND_API

  #define ALLOCA(type, name, nmemb) type name[nmemb]
#endif


#if SIZE_MAX == UINT64_MAX
  #define XND_SSIZE_MAX INT64_MAX
#elif SIZE_MAX == UINT32_MAX
  #define XND_SSIZE_MAX INT32_MAX
#else
  #error "unsupported platform: need 32-bit or 64-bit size_t"
#endif


/*
 * Ownership flags: The library itself has no notion of how many exported
 * views a master buffer has. The Python bindings for example use Pythons's
 * reference counting to to keep track of exported memory blocks.
 */
#define XND_OWN_TYPE     0x00000001U /* type pointer */
#define XND_OWN_DATA     0x00000002U /* data pointer */
#define XND_OWN_STRINGS  0x00000004U /* embedded string pointers */
#define XND_OWN_BYTES    0x00000008U /* embedded bytes pointers */
#define XND_OWN_ARRAYS   0x00000010U /* embedded array pointers */
#define XND_OWN_POINTERS 0x00000020U /* embedded pointers */
#define XND_CUDA_MANAGED 0x00000040U /* cuda managed memory */

#define XND_OWN_ALL (XND_OWN_TYPE |    \
                     XND_OWN_DATA |    \
                     XND_OWN_STRINGS | \
                     XND_OWN_BYTES |   \
                     XND_OWN_ARRAYS |  \
                     XND_OWN_POINTERS)

#define XND_OWN_EMBEDDED (XND_OWN_DATA |    \
                          XND_OWN_STRINGS | \
                          XND_OWN_BYTES |   \
                          XND_OWN_ARRAYS |  \
                          XND_OWN_POINTERS)


/* Convenience macros to extract embedded values. */
#define XND_POINTER_DATA(ptr) (*((char **)ptr))
#define XND_STRING_DATA(ptr) ((*((const char **)ptr)) == NULL ? "" : (*((const char **)ptr)))
#define XND_BYTES_SIZE(ptr) (((ndt_bytes_t *)ptr)->size)
#define XND_BYTES_DATA(ptr) (((ndt_bytes_t *)ptr)->data)
#define XND_ARRAY_SHAPE(ptr) (((ndt_array_t *)ptr)->shape)
#define XND_ARRAY_DATA(ptr) (((ndt_array_t *)ptr)->data)
#define XND_UNION_TAG(ptr) (*((uint8_t *)ptr))


/* Bitmap tree. */
typedef struct xnd_bitmap xnd_bitmap_t;

struct xnd_bitmap {
    uint8_t *data;      /* bitmap */
    int64_t size;       /* number of subtree bitmaps in the "next" array */
    xnd_bitmap_t *next; /* array of bitmaps for subtrees */
};

/* Typed memory block, usually a view. */
typedef struct xnd {
    xnd_bitmap_t bitmap; /* bitmap tree */
    int64_t index;       /* linear index for var dims */
    const ndt_t *type;   /* type of the data */
    char *ptr;           /* data */
} xnd_t;

/* Master memory block. */
typedef struct xnd_master {
    uint32_t flags; /* ownership flags */
    xnd_t master;   /* typed memory */
} xnd_master_t;

/* Used in indexing and slicing. */
enum xnd_key { Index, FieldName, Slice };
typedef struct {
  enum xnd_key tag;
  union {
    int64_t Index;
    const char *FieldName;
    ndt_slice_t Slice;
  };
} xnd_index_t;


/* Unstable API: view with ownership tracking. */
typedef struct xnd_view {
    uint32_t flags;  /* flags that indicate resource ownership by the view */
    const void *obj; /* object that holds shared resources */
    xnd_t view;      /* typed memory */
} xnd_view_t;


/*****************************************************************************/
/*                         Create xnd memory blocks                          */
/*****************************************************************************/

XND_API xnd_master_t *xnd_empty_from_string(const char *s, uint32_t flags, ndt_context_t *ctx);
XND_API xnd_master_t *xnd_empty_from_type(const ndt_t *t, uint32_t flags, ndt_context_t *ctx);
XND_API void xnd_clear(xnd_t * const x, const uint32_t flags);
XND_API void xnd_del(xnd_master_t *x);

/* Create and delete pristine xnd_t buffers. */
XND_API xnd_master_t *xnd_from_xnd(xnd_t *src, uint32_t flags, ndt_context_t *ctx);
XND_API void xnd_del_buffer(xnd_t *x, uint32_t flags);


/*****************************************************************************/
/*                         Traverse xnd memory blocks                        */
/*****************************************************************************/

XND_API bool have_stored_index(const ndt_t *t);
XND_API int64_t get_stored_index(const ndt_t *t);
XND_API xnd_t apply_stored_index(const xnd_t *x, ndt_context_t *ctx);
XND_API xnd_t apply_stored_indices(const xnd_t *x, ndt_context_t *ctx);

XND_API xnd_t xnd_subtree_index(const xnd_t *x, const int64_t *indices, int len,
                                ndt_context_t *ctx);

XND_API xnd_t xnd_subtree(const xnd_t *x, const xnd_index_t indices[], int len,
                          ndt_context_t *ctx);

XND_API xnd_t xnd_subscript(const xnd_t *x, const xnd_index_t indices[], int len,
                            ndt_context_t *ctx);

XND_API xnd_t xnd_reshape(const xnd_t *x, int64_t shape[], int ndim, char order, ndt_context_t *ctx);

XND_API xnd_t *xnd_split(const xnd_t *x, int64_t *n, int max_outer, ndt_context_t *ctx);

XND_API int xnd_equal(const xnd_t *x, const xnd_t *y, ndt_context_t *ctx);
XND_API int xnd_strict_equal(const xnd_t *x, const xnd_t *y, ndt_context_t *ctx);
XND_API int xnd_identical(const xnd_t *x, const xnd_t *y, ndt_context_t *ctx);

XND_API int xnd_copy(xnd_t *y, const xnd_t *x, uint32_t flags, ndt_context_t *ctx);


/*****************************************************************************/
/*                               Bounds checking                             */
/*****************************************************************************/

XND_API int xnd_bounds_check(const ndt_t *t, const int64_t linear_index,
                             const int64_t bufsize, ndt_context_t *ctx);


/*****************************************************************************/
/*                                  Bitmaps                                  */
/*****************************************************************************/

XND_API int xnd_bitmap_init(xnd_bitmap_t *b, const ndt_t *t, ndt_context_t *ctx);
XND_API void xnd_bitmap_clear(xnd_bitmap_t *b);
XND_API xnd_bitmap_t xnd_bitmap_next(const xnd_t *x, int64_t i, ndt_context_t *ctx);
XND_API void xnd_set_valid(xnd_t *x);
XND_API void xnd_set_na(xnd_t *x);
XND_API int xnd_is_valid(const xnd_t *x);
XND_API int xnd_is_na(const xnd_t *x);
XND_API int xnd_bitmap_identical(const xnd_bitmap_t *xb, const xnd_bitmap_t *yb,
                                 const ndt_t *t, ndt_context_t *ctx);


/*****************************************************************************/
/*                               Error handling                              */
/*****************************************************************************/

XND_API extern const xnd_t xnd_error;
XND_API extern const xnd_bitmap_t xnd_bitmap_empty;

XND_API int xnd_err_occurred(const xnd_t *x);


/*****************************************************************************/
/*                                 Unstable API                              */
/*****************************************************************************/

XND_API extern const xnd_view_t xnd_view_error;

XND_API int xnd_view_err_occurred(const xnd_view_t *x);
XND_API void xnd_view_clear(xnd_view_t *x);
XND_API xnd_view_t xnd_view_from_xnd(const void *obj, const xnd_t *x);
XND_API xnd_view_t xnd_view_subscript(const xnd_view_t *x, const xnd_index_t indices[],
                                      int len, ndt_context_t *ctx);



/*****************************************************************************/
/*                                 Float format                              */
/*****************************************************************************/

XND_API int xnd_init_float(ndt_context_t *ctx);
XND_API bool xnd_float_is_little_endian(void);
XND_API bool xnd_float_is_big_endian(void);
XND_API bool xnd_double_is_little_endian(void);
XND_API bool xnd_double_is_big_endian(void);


/*****************************************************************************/
/*                                  BFloat16                                 */
/*****************************************************************************/

XND_API void xnd_bfloat_pack(char *p, double x);
XND_API double xnd_bfloat_unpack(char *p);


/*****************************************************************************/
/*                                   Cuda                                    */
/*****************************************************************************/

void *xnd_cuda_calloc_managed(uint16_t align, int64_t size, ndt_context_t *ctx);
void xnd_cuda_free(void *ptr);
int xnd_cuda_mem_prefetch_async(const void *ptr, int64_t count, int dev, ndt_context_t *ctx);
int xnd_cuda_device_synchronize(ndt_context_t *ctx);


/*****************************************************************************/
/*                           Static inline functions                         */
/*****************************************************************************/

/* Check index bounds and adjust negative indices. */
static inline int64_t
adjust_index(const int64_t i, const int64_t shape, ndt_context_t *ctx)
{
    const int64_t k = i < 0 ? i + shape : i;

    if (k < 0 || k >= shape || k > XND_SSIZE_MAX) {
        ndt_err_format(ctx, NDT_IndexError,
            "index with value %" PRIi64 " out of bounds", i);
        return -1;
    }

    return k;
}

/*
 * This looks inefficient, but both gcc and clang clean up unused xnd_t members.
 */
static inline int64_t
xnd_ndim(const xnd_t *x)
{
    return x->type->ndim;
}

static inline xnd_t
xnd_fixed_dim_next(const xnd_t *x, const int64_t i)
{
    const ndt_t *t = x->type;
    const ndt_t *u = t->FixedDim.type;
    const int64_t step = i * t->Concrete.FixedDim.step;
    xnd_t next;

    assert(t->tag == FixedDim);

    next.bitmap = x->bitmap;
    next.index = x->index + step;
    next.type = u;
    next.ptr = u->ndim==0 ? x->ptr + next.index * next.type->datasize : x->ptr;

    return next;
}

static inline int64_t
xnd_fixed_shape(const xnd_t *x)
{
    const ndt_t *t = x->type;
    assert(t->tag == FixedDim);
    return t->FixedDim.shape;
}

static inline int64_t
xnd_fixed_step(const xnd_t *x)
{
    const ndt_t *t = x->type;
    assert(t->tag == FixedDim);
    return t->Concrete.FixedDim.step;
}

static inline int64_t
xnd_fixed_shape_at(const xnd_t *x, const int i)
{
    const ndt_t *t = x->type;

    assert(0 <= i && i < t->ndim);
    assert(t->tag == FixedDim);

    for (int k = 0; k < i; k++) {
        t = t->FixedDim.type;
    }
    return t->FixedDim.shape;
}

static inline int64_t
xnd_fixed_stride(const xnd_t *x)
{
    const ndt_t *t = x->type;
    assert(t->tag == FixedDim);
    return t->Concrete.FixedDim.step * t->Concrete.FixedDim.itemsize;
}

static inline char *
xnd_fixed_apply_index(const xnd_t *x)
{
    assert(x->type->tag == FixedDim);
    return x->ptr + x->index * x->type->Concrete.FixedDim.itemsize;
}

static inline xnd_t
xnd_var_dim_next(const xnd_t *x, const int64_t start, const int64_t step,
                 const int64_t i)
{
    const ndt_t *t = x->type;
    const ndt_t *u = t->VarDim.type;
    xnd_t next;

    next.bitmap = x->bitmap;
    next.index = start + i * step;
    next.type = u;
    next.ptr = u->ndim==0 ? x->ptr + next.index * next.type->datasize : x->ptr;

    return next;
}

static inline xnd_t
xnd_tuple_next(const xnd_t *x, const int64_t i, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_next(x, i, ctx);
    if (ndt_err_occurred(ctx)) {
        return xnd_error;
    }

    next.index = 0;
    next.type = t->Tuple.types[i];
    next.ptr = x->ptr + t->Concrete.Tuple.offset[i];

    return next;
}

static inline xnd_t
xnd_record_next(const xnd_t *x, const int64_t i, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_next(x, i, ctx);
    if (ndt_err_occurred(ctx)) {
        return xnd_error;
    }

    next.index = 0;
    next.type = t->Record.types[i];
    next.ptr = x->ptr + t->Concrete.Record.offset[i];

    return next;
}

static inline xnd_t
xnd_union_next(const xnd_t *x, ndt_context_t *ctx)
{
    uint8_t i = XND_UNION_TAG(x->ptr);
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_next(x, i, ctx);
    if (ndt_err_occurred(ctx)) {
        return xnd_error;
    }

    next.index = 0;
    next.type = t->Union.types[i];
    next.ptr = x->ptr+1;

    return next;
}

static inline xnd_t
xnd_ref_next(const xnd_t *x, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_next(x, 0, ctx);
    if (ndt_err_occurred(ctx)) {
        return xnd_error;
    }

    next.index = 0;
    next.type = t->Ref.type;
    next.ptr = XND_POINTER_DATA(x->ptr);

    return next;
}

static inline xnd_t
xnd_constr_next(const xnd_t *x, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_next(x, 0, ctx);
    if (ndt_err_occurred(ctx)) {
        return xnd_error;
    }

    next.index = 0;
    next.type = t->Constr.type;
    next.ptr = x->ptr;

    return next;
}

static inline xnd_t
xnd_nominal_next(const xnd_t *x, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_next(x, 0, ctx);
    if (ndt_err_occurred(ctx)) {
        return xnd_error;
    }

    next.index = 0;
    next.type = t->Nominal.type;
    next.ptr = x->ptr;

    return next;
}

static inline xnd_t
xnd_array_next(const xnd_t *x, const int64_t i)
{
    const ndt_t *t = x->type;
    const ndt_t *u = t->Array.type;
    xnd_t next;

    assert(t->tag == Array);

    next.bitmap = x->bitmap;
    next.index = 0;
    next.type = u;
    next.ptr = XND_ARRAY_DATA(x->ptr) + i * next.type->datasize;

    return next;
}

#if NDT_SYS_BIG_ENDIAN == 1
  #define XND_REV_COND NDT_LITTLE_ENDIAN
#else
  #define XND_REV_COND NDT_BIG_ENDIAN
#endif

static inline void
memcpy_rev(char *dest, const char *src, size_t size)
{
    size_t i;

    for (i = 0; i < size; i++) {
        dest[i] = src[size-1-i];
    }
}

static inline void
bcopy_swap(char *dest, const char *src, size_t size, uint32_t flags)
{
    if (flags & XND_REV_COND) {
        memcpy_rev(dest, src, size);
    }
    else {
        memcpy(dest, src, size);
    }
}

static inline int
le(uint32_t flags)
{
#if NDT_SYS_BIG_ENDIAN == 1
    return flags & NDT_LITTLE_ENDIAN;
#else
    return !(flags & NDT_BIG_ENDIAN);
#endif
}


#define PACK_SINGLE(ptr, src, type, flags) \
    do {                                                      \
        type _x;                                              \
        _x = (type)src;                                       \
        bcopy_swap(ptr, (const char *)&_x, sizeof _x, flags); \
    } while (0)

#define UNPACK_SINGLE(dest, ptr, type, flags) \
    do {                                                \
        type _x;                                        \
        bcopy_swap((char *)&_x, ptr, sizeof _x, flags); \
        dest = _x;                                      \
    } while (0)

#define APPLY_STORED_INDICES_INT(x) \
    xnd_t _##x##tail;                              \
    if (have_stored_index(x->type)) {              \
        _##x##tail = apply_stored_indices(x, ctx); \
        if (xnd_err_occurred(&_##x##tail)) {       \
            return -1;                             \
        }                                          \
        x = &_##x##tail;                           \
    }

#define APPLY_STORED_INDICES_XND(x) \
    xnd_t _##x##tail;                              \
    if (have_stored_index(x->type)) {              \
        _##x##tail = apply_stored_indices(x, ctx); \
        if (xnd_err_occurred(&_##x##tail)) {       \
            return xnd_error;                      \
        }                                          \
        x = &_##x##tail;                           \
    }


#ifdef __cplusplus
} /* END extern "C" */
#endif


#endif /* XND_H */

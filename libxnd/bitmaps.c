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


const xnd_bitmap_t xnd_bitmap_empty = { .data = NULL, .size = 0, .next = NULL};


static int64_t
bitmap_size(int64_t nelem)
{
    return (nelem + 7) / 8;
}

static uint8_t *
bits_new(int64_t n, ndt_context_t *ctx)
{
    uint8_t *bits;

    bits = ndt_calloc(bitmap_size(n), 1);
    if (bits == NULL) {
        return ndt_memory_error(ctx);
    }

    return bits;
}

static xnd_bitmap_t *
bitmap_array_new(int64_t n, ndt_context_t *ctx)
{
    xnd_bitmap_t *b;

    b = ndt_calloc(n, sizeof *b);
    if (b == NULL) {
        return ndt_memory_error(ctx);
    }

    return b;
}

static int
bitmap_init(xnd_bitmap_t *b, const ndt_t *t, int64_t nitems, ndt_context_t *ctx)
{
    xnd_bitmap_t *next;
    int64_t shape, i, k;
    int64_t n;

    assert(ndt_is_concrete(t));
    assert(b->data == NULL);
    assert(b->size == 0);
    assert(b->next == NULL);

    if (ndt_is_optional(t)) {
         if (t->ndim > 0) {
             ndt_err_format(ctx, NDT_NotImplementedError,
                 "optional dimensions are not implemented");
             return -1;
         }

         b->data = bits_new(nitems, ctx);
         if (b->data == NULL) {
             return -1;
         }
    }

    if (!ndt_subtree_is_optional(t)) {
        return 0;
    }

    switch (t->tag) {
    case FixedDim: {
        shape = t->FixedDim.shape;
        return bitmap_init(b, t->FixedDim.type, nitems * shape, ctx);
    }

    case VarDim: {
        assert(nitems == 1);
        n = nitems;

        if (t->ndim == 1) {
            int32_t noffsets = t->Concrete.VarDim.offsets->n;
            n = t->Concrete.VarDim.offsets->v[noffsets-1];
        }

        return bitmap_init(b, t->VarDim.type, n, ctx);
    }

    case Tuple: {
        shape = t->Tuple.shape;

        n = nitems * shape;
        b->next = bitmap_array_new(n, ctx);
        if (b->next == NULL) {
            xnd_bitmap_clear(b);
            return -1;
        }
        b->size = n;

        for (i = 0; i < nitems; i++) {
            for (k = 0; k < shape; k++) {
                next = b->next + i*shape + k;
                if (bitmap_init(next, t->Tuple.types[k], 1, ctx) < 0) {
                    xnd_bitmap_clear(b);
                    return -1;
                }
            }
        }

        return 0;
    }

    case Record: {
        shape = t->Record.shape;

        n = nitems * shape;
        b->next = bitmap_array_new(n, ctx);
        if (b->next == NULL) {
            xnd_bitmap_clear(b);
            return -1;
        }
        b->size = n;

        for (i = 0; i < nitems; i++) {
            for (k = 0; k < shape; k++) {
                next = b->next + i*shape + k;
                if (bitmap_init(next, t->Record.types[k], 1, ctx) < 0) {
                    xnd_bitmap_clear(b);
                    return -1;
                }
            }
        }

        return 0;
    }

    case Union: {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "bitmaps are not implemented for union types");
        return -1;
    }

    case Ref: {
        b->next = bitmap_array_new(nitems, ctx);
        if (b->next == NULL) {
            xnd_bitmap_clear(b);
            return -1;
        }
        b->size = nitems;

        for (i = 0; i < nitems; i++) {
            next = b->next + i;
            if (bitmap_init(next, t->Ref.type, 1, ctx) < 0) {
                xnd_bitmap_clear(b);
                return -1;
            }
        }

        return 0;
    }

    case Constr: {
        b->next = bitmap_array_new(nitems, ctx);
        if (b->next == NULL) {
            xnd_bitmap_clear(b);
            return -1;
        }
        b->size = nitems;

        for (i = 0; i < nitems; i++) {
            next = b->next + i;
            if (bitmap_init(next, t->Constr.type, 1, ctx) < 0) {
                xnd_bitmap_clear(b);
                return -1;
            }
        }

        return 0;
    }

    case Nominal: {
        b->next = bitmap_array_new(nitems, ctx);
        if (b->next == NULL) {
            xnd_bitmap_clear(b);
            return -1;
        }
        b->size = nitems;

        for (i = 0; i < nitems; i++) {
            next = b->next + i;
            if (bitmap_init(next, t->Nominal.type, 1, ctx) < 0) {
                xnd_bitmap_clear(b);
                return -1;
            }
        }

        return 0;
    }

    default:
        return 0;
    }
}

int
xnd_bitmap_init(xnd_bitmap_t *b, const ndt_t *t, ndt_context_t *ctx)
{
    return bitmap_init(b, t, 1, ctx);
}

void
xnd_bitmap_clear(xnd_bitmap_t *b)
{
    int64_t i;

    ndt_free(b->data);
    b->data = NULL;

    if (b->next) {
        for (i = 0; i < b->size; i++) {
            xnd_bitmap_clear(b->next + i);
        }

        ndt_free(b->next);
        b->next = NULL;
    }
}

xnd_bitmap_t
xnd_bitmap_next(const xnd_t *x, int64_t i, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    xnd_bitmap_t next = {.data=NULL, .size=0, .next=NULL};
    int64_t shape;

    if (!ndt_subtree_is_optional(t)) {
         return next;
    }

    if (x->bitmap.next == NULL) {
        ndt_err_format(ctx, NDT_RuntimeError, "missing bitmap");
        return next;
    }

    switch (t->tag) {
    case Tuple:
        shape = t->Tuple.shape;
        break;
    case Record:
        shape = t->Record.shape;
        break;
    case Union:
        ndt_err_format(ctx, NDT_NotImplementedError,
            "bitmaps are not implemented for union types");
        return next;
    case Ref: case Constr: case Nominal:
        shape = 1;
        break;
    default:
        ndt_err_format(ctx, NDT_RuntimeError, "type has no subtree bitmaps");
        return next;
    }

    if (i < 0 || i >= shape) {
        ndt_err_format(ctx, NDT_ValueError, "invalid index");
        return next;
    }

    return x->bitmap.next[x->index * shape + i];
}

void
xnd_set_valid(xnd_t *x)
{
    const ndt_t *t = x->type;
    int64_t n = x->index;

    assert(ndt_is_optional(t));
    assert(0 <= n);

    x->bitmap.data[n / 8] |= ((uint8_t)1 << (n % 8));
}

void
xnd_set_na(xnd_t *x)
{
    const ndt_t *t = x->type;
    int64_t n = x->index;

    assert(ndt_is_optional(t));
    assert(0 <= n);

    x->bitmap.data[n / 8] &= ~((uint8_t)1 << (n % 8));
}

static int
_xnd_is_valid(const xnd_t *x)
{
    const ndt_t *t = x->type;
    int64_t n = x->index;

    assert(ndt_is_optional(t));
    assert(0 <= n);

    return x->bitmap.data[n / 8] & ((uint8_t)1 << (n % 8));
}

int
xnd_is_valid(const xnd_t *x)
{
    if (!ndt_is_optional(x->type)) {
        return 1;
    }

    return _xnd_is_valid(x);
}

int
xnd_is_na(const xnd_t *x)
{
    if (!ndt_is_optional(x->type)) {
        return 0;
    }

    return !_xnd_is_valid(x);
}

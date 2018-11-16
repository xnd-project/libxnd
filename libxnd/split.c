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
#include <string.h>
#include <inttypes.h>
#include "ndtypes.h"
#include "xnd.h"
#include "overflow.h"


static const xnd_index_t init_slice =
  { .tag = Slice,
    .Slice = {.start = -1, .stop = -1, .step = -1}};


static int64_t column(
  int64_t nrows, int64_t ncols,
  xnd_index_t *indices, int *nindices,
  int64_t row, int64_t col, int64_t r, int64_t q,
  int64_t m, const int64_t *ms, int len);


static void
free_slices(xnd_t *lst, int64_t len)
{
    for (int64_t i = 0; i < len; i++) {
        ndt_decref(lst[i].type);
    }

    ndt_free(lst);
}

static inline int64_t
start(int64_t i, int64_t r, int64_t q)
{
    return i < r ?  i*(q+1) : r+i*q;
}

static inline int64_t
stop(int64_t i, int64_t r, int64_t q)
{
    return i < r ? (i+1)*(q+1) : r+(i+1)*q;
}

static inline int64_t
step(int64_t i, int64_t r, int64_t q)
{
    return i < r ? q+1 : q;
}

static inline xnd_index_t
single_step_slice(int64_t i)
{
    xnd_index_t x;

    x.tag = Slice;
    x.Slice.start = i;
    x.Slice.stop = i+1;
    x.Slice.step = 1;

    return x;
}

static inline xnd_index_t
slice(int64_t i, int64_t r, int64_t q)
{
    xnd_index_t x;

    x.tag = Slice;
    x.Slice.start = start(i, r, q);
    x.Slice.stop = stop(i, r, q);
    x.Slice.step = 1; 

    return x;
}

static int64_t
prepend(int64_t nrows, int64_t ncols,
        xnd_index_t *indices, int *nindices,
        int64_t row, int64_t col, xnd_index_t s, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        assert(row+i < nrows && col < ncols);
        indices[(row+i)*ncols + col] = s;
        nindices[row+i]++;
    }
 
    return n;
}

static int64_t
last_column(int64_t nrows, int64_t ncols,
            xnd_index_t *indices, int *nindices,
            int64_t row, int64_t col, int64_t r, int64_t q, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        assert(row+i < nrows && col < ncols);
        indices[(row+i)*ncols + col] = slice(i, r, q);
        nindices[row+i]++;
    }

    return n;
}

static int64_t
schedule(int64_t nrows, int64_t ncols,
         xnd_index_t *indices, int *nindices,
         int64_t row, int64_t col, int64_t n, const int64_t *shape, int len)
{
    int64_t m;
    int64_t q;
    int64_t r;

    if (len == 0) {
        return 1;
    }

    m = shape[0];
    if (n <= m) {
        q = m / n; 
        r = m % n;
        return last_column(nrows, ncols, indices, nindices, row, col, r, q, n);
    }
    else {
        q = n / m;
        r = n % m;
        return column(nrows, ncols, indices, nindices, row, col, r, q, m, shape+1, len-1);
    }
}

static int64_t
column(int64_t nrows, int64_t ncols,
       xnd_index_t *indices, int *nindices,
       int64_t row, int64_t col, int64_t r, int64_t q, int64_t m,
       const int64_t *ms, int len)
{
    int64_t column_len = 0;
    int64_t n, subtree_len, block_len;
    xnd_index_t s;

    for (int64_t i = 0; i < m; i++) {
        n = step(i, r, q);
        s = single_step_slice(i);
        subtree_len = schedule(nrows, ncols, indices, nindices, row, col+1, n, ms, len);
        block_len = prepend(nrows, ncols, indices, nindices, row, col, s, subtree_len);
        row += block_len;
        column_len += block_len;
    }

    return column_len;
}

static int
get_shape(int64_t *shape, const ndt_t *t, int max_outer, ndt_context_t *ctx)
{
    int i;

    if (!ndt_is_ndarray(t)) {
        ndt_err_format(ctx, NDT_ValueError,
            "split function called on non-ndarray");
        return -1;
    }

    for (i = 0; i < max_outer && t->ndim > 0; i++, t=t->FixedDim.type) {
        shape[i] = t->FixedDim.shape;
        if (shape[i] <= 0) {
            ndt_err_format(ctx, NDT_ValueError,
                "split function called on invalid shape or shape with zeros");
            return -1;
        }
    }
    for (; t->ndim > 0; t=t->FixedDim.type) {
        if (t->FixedDim.shape <= 0) {
            ndt_err_format(ctx, NDT_ValueError,
                "split function called on invalid shape or shape with zeros");
            return -1;
        }
    }

    return i;
}

xnd_t *
xnd_split(const xnd_t *x, int64_t *nparts, int max_outer, ndt_context_t *ctx)
{
    bool overflow = false;
    int64_t shape[NDT_MAX_DIM];
    xnd_index_t *indices;
    int *nindices;
    xnd_t *result;
    int64_t nrows, nmemb;
    int ncols;

    if (*nparts < 1) {
        ndt_err_format(ctx, NDT_ValueError, "'n' parameter must be >= 1");
        return NULL;
    }
    nrows = *nparts;

    ncols = get_shape(shape, x->type, max_outer, ctx);
    if (ncols < 0) {
        return NULL;
    }

    nmemb = MULi64(nrows, ncols, &overflow);
    if (overflow) {
        ndt_err_format(ctx, NDT_ValueError, "'n' parameter is too large");
        return NULL;
    }

    indices = ndt_alloc(nmemb, sizeof *indices);
    if (indices == NULL) {
        return ndt_memory_error(ctx);
    }
    for (int64_t i = 0; i < nrows; i++) {
        for (int64_t k = 0; k < ncols; k++) {
            indices[i*ncols + k] = init_slice;
        }
    }

    nindices = ndt_alloc(nrows, sizeof *nindices);
    if (nindices == NULL) {
        ndt_free(indices);
        return ndt_memory_error(ctx);
    }
    for (int64_t i = 0; i < nrows; i++) {
        nindices[i] = 0;
    }

    nrows = schedule(nrows, ncols, indices, nindices, 0, 0, nrows, shape, ncols);

    result = ndt_alloc(nrows, sizeof *result);
    if (result == NULL) {
        ndt_free(nindices);
        ndt_free(indices);
        return ndt_memory_error(ctx);
    }

    for (int64_t i = 0; i < nrows; i++) {
        result[i] = xnd_multikey(x, indices+(i*ncols), nindices[i], ctx);
        if (ndt_err_occurred(ctx)) {
            ndt_free(nindices);
            ndt_free(indices);
            free_slices(result, i);
            return NULL;
        }
    }

    ndt_free(nindices);
    ndt_free(indices);
    *nparts = nrows;

    return result;
}

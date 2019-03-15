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
#include "contrib.h"
#include "overflow.h"


static bool
shape_equal(const ndt_ndarray_t *dest, const ndt_ndarray_t *src)
{
    if (dest->ndim != src->ndim) {
        return false;
    }

    for (int i = 0; i < src->ndim; i++) {
        if (dest->shape[i] != src->shape[i]) {
            return false;
        }
    }

    return true;
}

static int64_t
prod(const int64_t shape[], int N)
{
    bool overflow = false;
    int64_t p = 1;

    for (int64_t i = 0; i < N; i++) {
        p = MULi64(p, shape[i], &overflow);
        if (overflow) {
            return -1;
        }
    }

    return p;
}

static inline bool
zero_in_shape(const ndt_ndarray_t *x)
{
    for (int i = 0; i < x->ndim; i++) {
        if (x->shape[i] == 0) {
            return true;
        }
    }

    return false;
}

static void
init_contiguous_c_strides(ndt_ndarray_t *dest, const ndt_ndarray_t *src)
{
    int64_t q;
    int64_t i;

    if (src->ndim == 0 && dest->ndim == 0) {
        return;
    }

    q = 1;
    for (i = dest->ndim-1; i >= 0; i--) {
        dest->steps[i] = q;
        q *= dest->shape[i];
    }
}

static void
init_contiguous_f_strides(ndt_ndarray_t *dest, const ndt_ndarray_t *src)
{
    int64_t q;
    int64_t i;

    if (src->ndim == 0 && dest->ndim == 0) {
        return;
    }

    q = 1;
    for (i = 0; i < dest->ndim; i++) {
        dest->steps[i] = q;
        q *= dest->shape[i];
    }
}

xnd_t
xnd_reshape(const xnd_t *x, int64_t shape[], int ndim, char order,
            ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    ndt_ndarray_t src, dest;
    int64_t p, q;
    int ret;
    int use_fortran = 0;

    if (order == 'F') {
        use_fortran = 1;
    }
    else if (order == 'A') {
        use_fortran = ndt_is_f_contiguous(t);
    }
    else if (order != 'C') {
        ndt_err_format(ctx, NDT_ValueError, "'order' must be 'C', 'F' or 'A'");
        return xnd_error;
    }

    if (ndt_as_ndarray(&src, t, ctx) < 0) {
        return xnd_error;
    }

    dest.ndim = ndim;
    dest.itemsize = src.itemsize;
    for (int i = 0; i < ndim; i++) {
        dest.shape[i] = shape[i];
        dest.steps[i] = 0;
        dest.strides[i] = 0;
    }

    p = prod(src.shape, src.ndim);
    q = prod(dest.shape, dest.ndim);
    if (p < 0 || q < 0) {
        ndt_err_format(ctx, NDT_ValueError,
            "reshaped array has too many elements");
        return xnd_error;
    }
    if (p != q) {
        ndt_err_format(ctx, NDT_ValueError,
            "shapes do not have the same number of elements");
        return xnd_error;
    }

    if (shape_equal(&dest, &src)) {
        dest = src;
    }
    else if (zero_in_shape(&dest)) {
        ;
    }
    else if (!use_fortran && ndt_is_c_contiguous(t)) {
        init_contiguous_c_strides(&dest, &src);
    }
    else if (use_fortran && ndt_is_f_contiguous(t)) {
        init_contiguous_f_strides(&dest, &src);
    }
    else {
        ret = xnd_nocopy_reshape(dest.shape, dest.steps, dest.ndim,
                                 src.shape, src.steps, src.ndim, use_fortran);
        if (!ret) {
            ndt_err_format(ctx, NDT_ValueError, "inplace reshape not possible");
            return xnd_error;
        }
    }

    xnd_t res = *x;

    const ndt_t *u = ndt_copy(ndt_dtype(t), ctx);
    if (u == NULL) {
        return xnd_error;
    }

    for (int i = dest.ndim-1; i >= 0; i--) {
        const ndt_t *v = ndt_fixed_dim(u, dest.shape[i], dest.steps[i], ctx);
        ndt_decref(u);
        if (v == NULL) {
            return xnd_error;
        }
        u = v;
    }

    res.type = u;
    return res;
}

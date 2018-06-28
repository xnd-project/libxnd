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
#include <assert.h>
#include "ndtypes.h"
#include "xnd.h"
#include "contrib.h"


int
xnd_equal(const xnd_t * const x, const xnd_t * const y, ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;
    const ndt_t * const u = y->type;
    int n;

    assert(ndt_is_concrete(t) && ndt_is_concrete(u));

    if (t->tag != u->tag) {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "mixed type comparisons are not implemented");
        return -1;
    }

    if (xnd_is_na(x) || xnd_is_na(y)) {
        return 0;
    }

    switch (t->tag) {
    case FixedDim: {
        int64_t i;

        if (u->FixedDim.shape != t->FixedDim.shape) {
            return 0;
        }

        for (i = 0; i < t->FixedDim.shape; i++) {
            const xnd_t xnext = xnd_fixed_dim_next(x, i);
            const xnd_t ynext = xnd_fixed_dim_next(y, i);
            n = xnd_equal(&xnext, &ynext, ctx);
            if (n <= 0) return n;
        }

        return 1;
    }

    case VarDim: {
        int64_t xstart, xstep, xshape;
        int64_t ystart, ystep, yshape;
        int64_t i;

        xshape = ndt_var_indices(&xstart, &xstep, t, x->index, ctx);
        if (xshape < 0) {
            return -1;
        }

        yshape = ndt_var_indices(&ystart, &ystep, u, y->index, ctx);
        if (yshape < 0) {
            return -1;
        }

        if (yshape != xshape) {
            return 0;
        }

        for (i = 0; i < xshape; i++) {
            const xnd_t xnext = xnd_var_dim_next(x, xstart, xstep, i);
            const xnd_t ynext = xnd_var_dim_next(y, ystart, ystep, i);
            n = xnd_equal(&xnext, &ynext, ctx);
            if (n <= 0) return n;
        }

        return 1;
    }

    case Tuple: {
        if (u->Tuple.shape != t->Tuple.shape) {
            return 0;
        }

        for (int64_t i = 0; i < t->Tuple.shape; i++) {
            const xnd_t xnext = xnd_tuple_next(x, i, ctx);
            if (xnext.ptr == NULL) {
                return -1;
            }

            const xnd_t ynext = xnd_tuple_next(y, i, ctx);
            if (ynext.ptr == NULL) {
                return -1;
            }

            n = xnd_equal(&xnext, &ynext, ctx);
            if (n <= 0) return n;
        }

        return 1;
    }

    case Record: {
        if (u->Record.shape != t->Record.shape) {
            return 0;
        }

        for (int64_t i = 0; i < t->Record.shape; i++) {
            const xnd_t xnext = xnd_record_next(x, i, ctx);
            if (xnext.ptr == NULL) {
                return -1;
            }


            const xnd_t ynext = xnd_record_next(y, i, ctx);
            if (ynext.ptr == NULL) {
                return -1;
            }

            n = xnd_equal(&xnext, &ynext, ctx);
            if (n <= 0) return n;
        }

        return 1;
    }

    case Ref: {
        const xnd_t xnext = xnd_ref_next(x, ctx);
        if (xnext.ptr == NULL) {
            return -1;
        }

        const xnd_t ynext = xnd_ref_next(y, ctx);
        if (ynext.ptr == NULL) {
            return -1;
        }

        return xnd_equal(&xnext, &ynext, ctx);
    }

    case Constr: {
        const xnd_t xnext = xnd_constr_next(x, ctx);
        if (xnext.ptr == NULL) {
            return -1;
        }

        const xnd_t ynext = xnd_constr_next(y, ctx);
        if (ynext.ptr == NULL) {
            return -1;
        }

        return xnd_equal(&xnext, &ynext, ctx);
    }

    case Nominal: {
        const xnd_t xnext = xnd_nominal_next(x, ctx);
        if (xnext.ptr == NULL) {
            return -1;
        }

        const xnd_t ynext = xnd_nominal_next(y, ctx);
        if (ynext.ptr == NULL) {
            return -1;
        }

        return xnd_equal(&xnext, &ynext, ctx);
    }

    case Categorical: {
        int64_t i, k;

        if (!ndt_equal(t, u)) {
            return 0;
        }

        UNPACK_SINGLE(i, x->ptr, int64_t, t->flags);
        UNPACK_SINGLE(k, y->ptr, int64_t, u->flags);
        return i == k;
    }

    case Char: {
        ndt_err_format(ctx, NDT_NotImplementedError, "char not implemented");
        return -1;
    }

    case Bool: {
        bool a, b;
        UNPACK_SINGLE(a, x->ptr, bool, t->flags);
        UNPACK_SINGLE(b, y->ptr, bool, u->flags);
        return a == b;
    }

    case Int8: {
        int8_t a, b;
        UNPACK_SINGLE(a, x->ptr, int8_t, t->flags);
        UNPACK_SINGLE(b, y->ptr, int8_t, u->flags);
        return a == b;
    }

    case Int16: {
        int16_t a, b;
        UNPACK_SINGLE(a, x->ptr, int16_t, t->flags);
        UNPACK_SINGLE(b, y->ptr, int16_t, u->flags);
        return a == b;
    }

    case Int32: {
        int32_t a, b;
        UNPACK_SINGLE(a, x->ptr, int32_t, t->flags);
        UNPACK_SINGLE(b, y->ptr, int32_t, u->flags);
        return a == b;
    }

    case Int64: {
        int64_t a, b;
        UNPACK_SINGLE(a, x->ptr, int64_t, t->flags);
        UNPACK_SINGLE(b, y->ptr, int64_t, u->flags);
        return a == b;
    }

    case Uint8: {
        uint8_t a, b;
        UNPACK_SINGLE(a, x->ptr, uint8_t, t->flags);
        UNPACK_SINGLE(b, y->ptr, uint8_t, u->flags);
        return a == b;
    }

    case Uint16: {
        uint16_t a, b;
        UNPACK_SINGLE(a, x->ptr, uint16_t, t->flags);
        UNPACK_SINGLE(b, y->ptr, uint16_t, u->flags);
        return a == b;
    }

    case Uint32: {
        uint32_t a, b;
        UNPACK_SINGLE(a, x->ptr, uint32_t, t->flags);
        UNPACK_SINGLE(b, y->ptr, uint32_t, u->flags);
        return a == b;
    }

    case Uint64: {
        uint64_t a, b;
        UNPACK_SINGLE(a, x->ptr, uint64_t, t->flags);
        UNPACK_SINGLE(b, y->ptr, uint64_t, u->flags);
        return a == b;
    }

    case Float16: {
        double a = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        double b = xnd_float_unpack2((unsigned char *)y->ptr, le(u->flags));
        return a == b;
    }

    case Float32: {
        double a = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        double b = xnd_float_unpack4((unsigned char *)y->ptr, le(u->flags));
        return a == b;
    }

    case Float64: {
        double a = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        double b = xnd_float_unpack8((unsigned char *)y->ptr, le(u->flags));
        return a == b;
    }

    case Complex32: {
        double a_real, a_imag;
        double b_real, b_imag;

        a_real = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        a_imag = xnd_float_unpack2((unsigned char *)(x->ptr+2), le(t->flags));

        b_real = xnd_float_unpack2((unsigned char *)y->ptr, le(u->flags));
        b_imag = xnd_float_unpack2((unsigned char *)(y->ptr+2), le(u->flags));

        return a_real == b_real && a_imag == b_imag;
    }

    case Complex64: {
        double a_real, a_imag;
        double b_real, b_imag;

        a_real = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        a_imag = xnd_float_unpack4((unsigned char *)(x->ptr+4), le(t->flags));

        b_real = xnd_float_unpack4((unsigned char *)y->ptr, le(u->flags));
        b_imag = xnd_float_unpack4((unsigned char *)(y->ptr+4), le(u->flags));

        return a_real == b_real && a_imag == b_imag;
    }

    case Complex128: {
        double a_real, a_imag;
        double b_real, b_imag;

        a_real = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        a_imag = xnd_float_unpack8((unsigned char *)(x->ptr+8), le(t->flags));

        b_real = xnd_float_unpack8((unsigned char *)y->ptr, le(u->flags));
        b_imag = xnd_float_unpack8((unsigned char *)(y->ptr+8), le(u->flags));

        return a_real == b_real && a_imag == b_imag;
    }

    case FixedString: {
        if (t->FixedString.size != u->FixedString.size ||
            t->FixedString.encoding != u->FixedString.encoding ||
            t->datasize != u->datasize) {
            return 0;
        }
        return memcmp(x->ptr, y->ptr, t->datasize) == 0;
    }

    case FixedBytes: {
        if (t->FixedBytes.size != u->FixedBytes.size ||
            t->datasize != u->datasize) {
            return 0;
        }
        return memcmp(x->ptr, y->ptr, t->datasize) == 0;
    }

    case String: {
        const char *a = XND_POINTER_DATA(x->ptr);
        const char *b = XND_POINTER_DATA(y->ptr);
        return strcmp(a, b) == 0;
    }

    case Bytes: {
        char *a, *b;
        int64_t asize, bsize;

        asize = XND_BYTES_SIZE(x->ptr);
        bsize = XND_BYTES_SIZE(y->ptr);

        a = (char *)XND_BYTES_DATA(x->ptr);
        b = (char *)XND_BYTES_DATA(y->ptr);

        if (asize != bsize) {
            return 0;
        }

        return memcmp(a, b, asize) == 0;
    }

    /* NOT REACHED: xnd types must be concrete. */
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

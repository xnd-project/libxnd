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


/*****************************************************************************/
/*                      Equality with strict type checking                   */
/*****************************************************************************/

int
xnd_strict_equal(const xnd_t *x, const xnd_t *y, ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;
    const ndt_t * const u = y->type;
    int n;

    assert(ndt_is_concrete(t) && ndt_is_concrete(u));

    if (t->tag != u->tag) {
        return 0;
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
            n = xnd_strict_equal(&xnext, &ynext, ctx);
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
            n = xnd_strict_equal(&xnext, &ynext, ctx);
            if (n <= 0) return n;
        }

        return 1;
    }

    case VarDimElem: {
        const xnd_t xnext = apply_stored_index(x, ctx);
        if (xnd_err_occurred(&xnext)) {
            return -1;
        }

        const xnd_t ynext = apply_stored_index(y, ctx);
        if (xnd_err_occurred(&ynext)) {
            return -1;
        }

        return xnd_strict_equal(&xnext, &ynext, ctx);
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

            n = xnd_strict_equal(&xnext, &ynext, ctx);
            if (n <= 0) return n;
        }

        return 1;
    }

    case Record: {
        if (u->Record.shape != t->Record.shape) {
            return 0;
        }

        for (int64_t i = 0; i < t->Record.shape; i++) {
            if (strcmp(t->Record.names[i], u->Record.names[i]) != 0) {
                return 0;
            }

            const xnd_t xnext = xnd_record_next(x, i, ctx);
            if (xnext.ptr == NULL) {
                return -1;
            }

            const xnd_t ynext = xnd_record_next(y, i, ctx);
            if (ynext.ptr == NULL) {
                return -1;
            }

            n = xnd_strict_equal(&xnext, &ynext, ctx);
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

        return xnd_strict_equal(&xnext, &ynext, ctx);
    }

    case Constr: {
        if (strcmp(t->Constr.name, u->Constr.name) != 0) {
            return 0;
        }

        const xnd_t xnext = xnd_constr_next(x, ctx);
        if (xnext.ptr == NULL) {
            return -1;
        }

        const xnd_t ynext = xnd_constr_next(y, ctx);
        if (ynext.ptr == NULL) {
            return -1;
        }

        return xnd_strict_equal(&xnext, &ynext, ctx);
    }

    case Nominal: {
        if (strcmp(t->Nominal.name, u->Nominal.name) != 0) {
            return 0;
        }

        const xnd_t xnext = xnd_nominal_next(x, ctx);
        if (xnext.ptr == NULL) {
            return -1;
        }

        const xnd_t ynext = xnd_nominal_next(y, ctx);
        if (ynext.ptr == NULL) {
            return -1;
        }

        return xnd_strict_equal(&xnext, &ynext, ctx);
    }

    case Categorical: {
        int64_t i, k;

        if (!ndt_equal(t, u)) {
            return 0;
        }

        UNPACK_SINGLE(i, x->ptr, int64_t, t->flags);
        UNPACK_SINGLE(k, y->ptr, int64_t, u->flags);

        if (t->Categorical.types[i].tag == ValNA) {
            return 0;
        }

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

    case BFloat16: {
        double a = xnd_bfloat_unpack(x->ptr);
        double b = xnd_bfloat_unpack(y->ptr);
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

    case BComplex32: {
        double a_real, a_imag;
        double b_real, b_imag;

        a_real = xnd_bfloat_unpack(x->ptr);
        a_imag = xnd_bfloat_unpack(x->ptr+2);

        b_real = xnd_bfloat_unpack(y->ptr);
        b_imag = xnd_bfloat_unpack(y->ptr+2);

        return a_real == b_real && a_imag == b_imag;
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
        return memcmp(x->ptr, y->ptr, (size_t)t->datasize) == 0;
    }

    case FixedBytes: {
        if (t->FixedBytes.size != u->FixedBytes.size ||
            t->datasize != u->datasize) {
            return 0;
        }
        return memcmp(x->ptr, y->ptr, (size_t)t->datasize) == 0;
    }

    case String: {
        const char *a = XND_STRING_DATA(x->ptr);
        const char *b = XND_STRING_DATA(y->ptr);
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

        return memcmp(a, b, (size_t)asize) == 0;
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


/*****************************************************************************/
/*                           Equality with exact casts                       */
/*****************************************************************************/

/* Skip all ref chains. */
static int
equal_ref(const xnd_t *x, const xnd_t *y, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;
    const ndt_t *u = y->type;
    xnd_t xnext, ynext;

    while (t->tag == Ref) {
        xnext = xnd_ref_next(x, ctx);
        if (xnext.ptr == NULL) {
            return -1;
        }
        x = &xnext;
        t = x->type;
    }

    while (u->tag == Ref) {
        ynext = xnd_ref_next(y, ctx);
        if (ynext.ptr == NULL) {
            return -1;
        }
        y = &ynext;
        u = y->type;
    }

    return xnd_equal(x, y, ctx);
}

static int
equal_int64(const int64_t a, const xnd_t * const x)
{
    const ndt_t * const t = x->type;

    switch (t->tag) {
    case Int8: {
        int8_t b;
        UNPACK_SINGLE(b, x->ptr, int8_t, t->flags);
        return a == (int64_t)b;
    }

    case Int16: {
        int16_t b;
        UNPACK_SINGLE(b, x->ptr, int16_t, t->flags);
        return a == (int64_t)b;
    }

    case Int32: {
        int32_t b;
        UNPACK_SINGLE(b, x->ptr, int32_t, t->flags);
        return a == (int64_t)b;
    }

    case Int64: {
        int64_t b;
        UNPACK_SINGLE(b, x->ptr, int64_t, t->flags);
        return a == b;
    }

    case Uint8: {
        uint8_t b;
        UNPACK_SINGLE(b, x->ptr, uint8_t, t->flags);
        return a >= 0 ? (uint64_t)a == (uint64_t)b : 0;
    }

    case Uint16: {
        uint16_t b;
        UNPACK_SINGLE(b, x->ptr, uint16_t, t->flags);
        return a >= 0 ? (uint64_t)a == (uint64_t)b : 0;
    }

    case Uint32: {
        uint32_t b;
        UNPACK_SINGLE(b, x->ptr, uint32_t, t->flags);
        return a >= 0 ? (uint64_t)a == (uint64_t)b : 0;
    }

    case Uint64: {
        uint64_t b;
        UNPACK_SINGLE(b, x->ptr, uint64_t, t->flags);
        return a >= 0 ? (uint64_t)a == b : 0;
    }

    case BFloat16: {
        double b = xnd_bfloat_unpack(x->ptr);
        return llabs(a) <= 4503599627370496LL ? (double)a == b : 0;
    }

    case Float16: {
        double b = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        return llabs(a) <= 4503599627370496LL ? (double)a == b : 0;
    }

    case Float32: {
        double b = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        return llabs(a) <= 4503599627370496LL ? (double)a == b : 0;
    }

    case Float64: {
        double b = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        return llabs(a) <= 4503599627370496LL ? (double)a == b : 0;
    }

    case BComplex32: {
        double real = xnd_bfloat_unpack(x->ptr);
        double imag = xnd_bfloat_unpack(x->ptr+2);

        if (imag == 0.0 && llabs(a) <= 4503599627370496LL) {
            return (double)a == real;
        }

        return 0;
    }

    case Complex32: {
        double real = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        double imag = xnd_float_unpack2((unsigned char *)(x->ptr+2), le(t->flags));

        if (imag == 0.0 && llabs(a) <= 4503599627370496LL) {
            return (double)a == real;
        }

        return 0;
    }

    case Complex64: {
        double real = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        double imag = xnd_float_unpack4((unsigned char *)(x->ptr+4), le(t->flags));

        if (imag == 0.0 && llabs(a) <= 4503599627370496LL) {
            return (double)a == real;
        }

        return 0;
    }

    case Complex128: {
        double real = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        double imag = xnd_float_unpack8((unsigned char *)(x->ptr+8), le(t->flags));

        if (imag == 0.0 && llabs(a) <= 4503599627370496LL) {
            return (double)a == real;
        }

        return 0;
    }
    default:
        return 0;
    }
}

static int
equal_uint64(const uint64_t a, const xnd_t * const x)
{
    const ndt_t * const t = x->type;

    switch (t->tag) {
    case Int8: {
        int8_t b;
        UNPACK_SINGLE(b, x->ptr, int8_t, t->flags);
        return b >= 0 ? a == (uint64_t)b : 0;
    }

    case Int16: {
        int16_t b;
        UNPACK_SINGLE(b, x->ptr, int16_t, t->flags);
        return b >= 0 ? a == (uint64_t)b : 0;
    }

    case Int32: {
        int32_t b;
        UNPACK_SINGLE(b, x->ptr, int32_t, t->flags);
        return b >= 0 ? a == (uint64_t)b : 0;
    }

    case Int64: {
        int64_t b;
        UNPACK_SINGLE(b, x->ptr, int64_t, t->flags);
        return b >= 0 ? a == (uint64_t)b : 0;
    }

    case Uint8: {
        uint8_t b;
        UNPACK_SINGLE(b, x->ptr, uint8_t, t->flags);
        return a == (uint64_t)b;
    }

    case Uint16: {
        uint16_t b;
        UNPACK_SINGLE(b, x->ptr, uint16_t, t->flags);
        return a == (uint64_t)b;
    }

    case Uint32: {
        uint32_t b;
        UNPACK_SINGLE(b, x->ptr, uint32_t, t->flags);
        return a == (uint64_t)b;
    }

    case Uint64: {
        uint64_t b;
        UNPACK_SINGLE(b, x->ptr, uint64_t, t->flags);
        return a == b;
    }

    case BFloat16: {
        double b = xnd_bfloat_unpack(x->ptr);
        return a <= 4503599627370496ULL ? (double)a == b : 0;
    }

    case Float16: {
        double b = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        return a <= 4503599627370496ULL ? (double)a == b : 0;
    }

    case Float32: {
        double b = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        return a <= 4503599627370496ULL ? (double)a == b : 0;
    }

    case Float64: {
        double b = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        return a <= 4503599627370496ULL ? (double)a == b : 0;
    }

    case BComplex32: {
        double real = xnd_bfloat_unpack(x->ptr);
        double imag = xnd_bfloat_unpack(x->ptr+2);

        if (imag == 0.0 && a <= 4503599627370496ULL) {
            return (double)a == real;
        }

        return 0;
    }

    case Complex32: {
        double real = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        double imag = xnd_float_unpack2((unsigned char *)(x->ptr+2), le(t->flags));

        if (imag == 0.0 && a <= 4503599627370496ULL) {
            return (double)a == real;
        }

        return 0;
    }

    case Complex64: {
        double real = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        double imag = xnd_float_unpack4((unsigned char *)(x->ptr+4), le(t->flags));

        if (imag == 0.0 && a <= 4503599627370496ULL) {
            return (double)a == real;
        }

        return 0;
    }

    case Complex128: {
        double real = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        double imag = xnd_float_unpack8((unsigned char *)(x->ptr+8), le(t->flags));

        if (imag == 0.0 && a <= 4503599627370496ULL) {
            return (double)a == real;
        }

        return 0;
    }
    default:
        return 0;
    }
}

static int
equal_float64(const double a, const xnd_t * const x)
{
    const ndt_t * const t = x->type;

    switch (t->tag) {
    case Int8: {
        int8_t b;
        UNPACK_SINGLE(b, x->ptr, int8_t, t->flags);
        return a == (double)b;
    }

    case Int16: {
        int16_t b;
        UNPACK_SINGLE(b, x->ptr, int16_t, t->flags);
        return a == (double)b;
    }

    case Int32: {
        int32_t b;
        UNPACK_SINGLE(b, x->ptr, int32_t, t->flags);
        return a == (double)b;
    }

    case Int64: {
        int64_t b;
        UNPACK_SINGLE(b, x->ptr, int64_t, t->flags);
        return llabs(b) <= 4503599627370496LL ? a == (double)b : 0;
    }

    case Uint8: {
        uint8_t b;
        UNPACK_SINGLE(b, x->ptr, uint8_t, t->flags);
        return a == (double)b;
    }

    case Uint16: {
        uint16_t b;
        UNPACK_SINGLE(b, x->ptr, uint16_t, t->flags);
        return a == (double)b;
    }

    case Uint32: {
        uint32_t b;
        UNPACK_SINGLE(b, x->ptr, uint32_t, t->flags);
        return a == (double)b;
    }

    case Uint64: {
        uint64_t b;
        UNPACK_SINGLE(b, x->ptr, uint64_t, t->flags);
        return b <= 4503599627370496ULL ? a == (double)b : 0;
    }

    case BFloat16: {
        double b = xnd_bfloat_unpack(x->ptr);
        return a == b;
    }

    case Float16: {
        double b = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        return a == b;
    }

    case Float32: {
        double b = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        return a == b;
    }

    case Float64: {
        double b = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        return a == b;
    }

    case BComplex32: {
        double real, imag;

        real = xnd_bfloat_unpack(x->ptr);
        imag = xnd_bfloat_unpack(x->ptr+2);

        if (imag == 0.0) {
            return a == real;
        }

        return 0;
    }

    case Complex32: {
        double real, imag;

        real = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        imag = xnd_float_unpack2((unsigned char *)(x->ptr+2), le(t->flags));

        if (imag == 0.0) {
            return a == real;
        }

        return 0;
    }

    case Complex64: {
        double real, imag;

        real = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        imag = xnd_float_unpack4((unsigned char *)(x->ptr+4), le(t->flags));

        if (imag == 0.0) {
            return a == real;
        }

        return 0;
    }

    case Complex128: {
        double real, imag;

        real = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        imag = xnd_float_unpack8((unsigned char *)(x->ptr+8), le(t->flags));

        if (imag == 0.0) {
            return a == real;
        }

        return 0;
    }
    default:
        return 0;
    }
}

static int
equal_complex128(const double real, const double imag, const xnd_t * const x)
{
    const ndt_t * const t = x->type;

    switch (t->tag) {
    case Int8: {
        int8_t b;
        UNPACK_SINGLE(b, x->ptr, int8_t, t->flags);
        return imag == 0.0 ? real == (double)b : 0;
    }

    case Int16: {
        int16_t b;
        UNPACK_SINGLE(b, x->ptr, int16_t, t->flags);
        return imag == 0.0 ? real == (double)b : 0;
    }

    case Int32: {
        int32_t b;
        UNPACK_SINGLE(b, x->ptr, int32_t, t->flags);
        return imag == 0.0 ? real == (double)b : 0;
    }

    case Int64: {
        int64_t b;
        UNPACK_SINGLE(b, x->ptr, int64_t, t->flags);
        return imag == 0.0 && llabs(b) <= 4503599627370496LL ? real == (double)b : 0;
    }

    case Uint8: {
        uint8_t b;
        UNPACK_SINGLE(b, x->ptr, uint8_t, t->flags);
        return imag == 0.0 ? real == (double)b : 0;
    }

    case Uint16: {
        uint16_t b;
        UNPACK_SINGLE(b, x->ptr, uint16_t, t->flags);
        return imag == 0.0 ? real == (double)b : 0;
    }

    case Uint32: {
        uint32_t b;
        UNPACK_SINGLE(b, x->ptr, uint32_t, t->flags);
        return imag == 0.0 ? real == (double)b : 0;
    }

    case Uint64: {
        uint64_t b;
        UNPACK_SINGLE(b, x->ptr, uint64_t, t->flags);
        return b <= 4503599627370496ULL ? real == (double)b : 0;
    }

    case BFloat16: {
        double b = xnd_bfloat_unpack(x->ptr);
        return imag == 0.0 && real == b;
    }

    case Float16: {
        double b = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        return imag == 0.0 && real == b;
    }

    case Float32: {
        double b = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        return imag == 0.0 && real == b;
    }

    case Float64: {
        double b = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        return imag == 0.0 && real == b;
    }

    case BComplex32: {
        double a = xnd_bfloat_unpack(x->ptr);
        double b = xnd_bfloat_unpack(x->ptr+2);
        return real == a && imag == b;
    }

    case Complex32: {
        double a = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        double b = xnd_float_unpack2((unsigned char *)(x->ptr+2), le(t->flags));
        return real == a && imag == b;
    }

    case Complex64: {
        double a = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        double b = xnd_float_unpack4((unsigned char *)(x->ptr+4), le(t->flags));
        return real == a && imag == b;
    }

    case Complex128: {
        double a = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        double b = xnd_float_unpack8((unsigned char *)(x->ptr+8), le(t->flags));
        return real == a && imag == b;
    }
    default:
        return 0;
    }
}

int
xnd_equal(const xnd_t *x, const xnd_t *y, ndt_context_t *ctx)
{
    APPLY_STORED_INDICES_INT(x)
    APPLY_STORED_INDICES_INT(y)
    const ndt_t * const t = x->type;
    const ndt_t * const u = y->type;
    int n;

    if (xnd_is_na(x) || xnd_is_na(y)) {
        return 0;
    }

    if (t->tag == Ref || u->tag == Ref) {
        return equal_ref(x, y, ctx);
    }

    switch (t->tag) {
    case FixedDim: {
        int64_t i;

        if (u->tag != FixedDim || u->FixedDim.shape != t->FixedDim.shape) {
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

        if (u->tag != VarDim) {
            return 0;
        }

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
        if (u->tag != Tuple || u->Tuple.shape != t->Tuple.shape) {
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
        if (u->tag != Record || u->Record.shape != t->Record.shape) {
            return 0;
        }

        for (int64_t i = 0; i < t->Record.shape; i++) {
            if (strcmp(t->Record.names[i], u->Record.names[i]) != 0) {
                return 0;
            }

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

    case Constr: {
        if (u->tag != Constr || strcmp(u->Constr.name, t->Constr.name) != 0) {
            return 0;
        }

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
        if (u->tag != Nominal) {
            return 0;
        }

        if (strcmp(t->Nominal.name, u->Nominal.name) != 0) {
            return 0;
        }

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

        if (t->Categorical.types[i].tag == ValNA) {
            return 0;
        }

        return i == k;
    }

    case Char: {
        ndt_err_format(ctx, NDT_NotImplementedError, "char not implemented");
        return -1;
    }

    case Bool: {
        bool a, b;

        if (u->tag != Bool) {
            return 0;
        }

        UNPACK_SINGLE(a, x->ptr, bool, t->flags);
        UNPACK_SINGLE(b, y->ptr, bool, u->flags);
        return a == b;
    }

    case Int8: {
        int8_t a;
        UNPACK_SINGLE(a, x->ptr, int8_t, t->flags);
        return equal_int64(a, y);
    }

    case Int16: {
        int16_t a;
        UNPACK_SINGLE(a, x->ptr, int16_t, t->flags);
        return equal_int64(a, y);
    }

    case Int32: {
        int32_t a;
        UNPACK_SINGLE(a, x->ptr, int32_t, t->flags);
        return equal_int64(a, y);
    }

    case Int64: {
        int64_t a;
        UNPACK_SINGLE(a, x->ptr, int64_t, t->flags);
        return equal_int64(a, y);
    }

    case Uint8: {
        uint8_t a;
        UNPACK_SINGLE(a, x->ptr, uint8_t, t->flags);
        return equal_uint64(a, y);
    }

    case Uint16: {
        uint16_t a;
        UNPACK_SINGLE(a, x->ptr, uint16_t, t->flags);
        return equal_uint64(a, y);
    }

    case Uint32: {
        uint32_t a;
        UNPACK_SINGLE(a, x->ptr, uint32_t, t->flags);
        return equal_uint64(a, y);
    }

    case Uint64: {
        uint64_t a;
        UNPACK_SINGLE(a, x->ptr, uint64_t, t->flags);
        return equal_uint64(a, y);
    }

    case BFloat16: {
        double a = xnd_bfloat_unpack(x->ptr);
        return equal_float64(a, y);
    }

    case Float16: {
        double a = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        return equal_float64(a, y);
    }

    case Float32: {
        double a = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        return equal_float64(a, y);
    }

    case Float64: {
        double a = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        return equal_float64(a, y);
    }

    case BComplex32: {
        double real, imag;

        real = xnd_bfloat_unpack(x->ptr);
        imag = xnd_bfloat_unpack(x->ptr+2);

        if (imag == 0.0) {
            return equal_float64(real, y);
        }

        return equal_complex128(real, imag, y);
    }

    case Complex32: {
        double real, imag;

        real = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        imag = xnd_float_unpack2((unsigned char *)(x->ptr+2), le(t->flags));

        if (imag == 0.0) {
            return equal_float64(real, y);
        }

        return equal_complex128(real, imag, y);
    }

    case Complex64: {
        double real, imag;

        real = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        imag = xnd_float_unpack4((unsigned char *)(x->ptr+4), le(t->flags));

        if (imag == 0.0) {
            return equal_float64(real, y);
        }

        return equal_complex128(real, imag, y);
    }

    case Complex128: {
        double real, imag;

        real = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        imag = xnd_float_unpack8((unsigned char *)(x->ptr+8), le(t->flags));

        if (imag == 0.0) {
            return equal_float64(real, y);
        }

        return equal_complex128(real, imag, y);
    }

    case FixedString: {
        if (u->tag != FixedString ||
            u->FixedString.size != t->FixedString.size ||
            u->FixedString.encoding != t->FixedString.encoding ||
            u->datasize != t->datasize) {
            return 0;
        }
        return memcmp(x->ptr, y->ptr, (size_t)t->datasize) == 0;
    }

    case FixedBytes: {
        if (u->tag != FixedBytes ||
            u->FixedBytes.size != t->FixedBytes.size ||
            u->datasize != t->datasize) {
            return 0;
        }
        return memcmp(x->ptr, y->ptr, (size_t)t->datasize) == 0;
    }

    case String: {
        if (u->tag != String) {
            return 0;
        }

        const char *a = XND_STRING_DATA(x->ptr);
        const char *b = XND_STRING_DATA(y->ptr);
        return strcmp(a, b) == 0;
    }

    case Bytes: {
        char *a, *b;
        int64_t asize, bsize;

        if (u->tag != Bytes) {
            return 0;
        }

        asize = XND_BYTES_SIZE(x->ptr);
        bsize = XND_BYTES_SIZE(y->ptr);

        a = (char *)XND_BYTES_DATA(x->ptr);
        b = (char *)XND_BYTES_DATA(y->ptr);

        if (asize != bsize) {
            return 0;
        }

        return memcmp(a, b, (size_t)asize) == 0;
    }

    /* NOT REACHED: intercepted by apply_stored_indices(). */
    case VarDimElem:
    /* NOT REACHED: intercepted by equal_ref(). */
    case Ref:
        ndt_err_format(ctx, NDT_RuntimeError, "unexpected VarDimElem or Ref");
        return -1;

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

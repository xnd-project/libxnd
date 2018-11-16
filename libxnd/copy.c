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
/*                           Copying with exact casts                        */
/*****************************************************************************/

static int
type_error(ndt_context_t *ctx)
{
    ndt_err_format(ctx, NDT_TypeError,
        "incompatible types of source and destination in copy function");
    return -1;
}

static int
value_error(ndt_context_t *ctx)
{
    ndt_err_format(ctx, NDT_ValueError,
        "incompatible value for destination type in copy function");
    return -1;
}

/* Skip all ref chains. */
static int
copy_ref(xnd_t *y, const xnd_t *x, const uint32_t flags, ndt_context_t *ctx)
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

    return xnd_copy(y, x, flags, ctx);
}

static int
copy_int64(xnd_t * const x, const int64_t i64, ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;

    switch (t->tag) {
    case Int8: {
        if (i64 < INT8_MIN || i64 > INT8_MAX) {
            return value_error(ctx);
        }
        int8_t i8 = (int8_t)i64;
        PACK_SINGLE(x->ptr, i8, int8_t, t->flags);
        return 0;
    }

    case Int16: {
        if (i64 < INT16_MIN || i64 > INT16_MAX) {
            return value_error(ctx);
        }
        int16_t i16 = (int16_t)i64;
        PACK_SINGLE(x->ptr, i16, int16_t, t->flags);
        return 0;
    }

    case Int32: {
        if (i64 < INT32_MIN || i64 > INT32_MAX) {
            return value_error(ctx);
        }
        int32_t i32 = (int32_t)i64;
        PACK_SINGLE(x->ptr, i32, int32_t, t->flags);
        return 0;
    }

    case Int64: {
        PACK_SINGLE(x->ptr, i64, int64_t, t->flags);
        return 0;
    }

    case Uint8: {
        if (i64 < 0 || i64 > UINT8_MAX) {
            return value_error(ctx);
        }
        uint8_t u8 = (uint8_t)i64;
        PACK_SINGLE(x->ptr, u8, uint8_t, t->flags);
        return 0;
    }

    case Uint16: {
        if (i64 < 0 || i64 > UINT16_MAX) {
            return value_error(ctx);
        }
        uint16_t u16 = (uint16_t)i64;;
        PACK_SINGLE(x->ptr, u16, uint16_t, t->flags);
        return 0;
    }

    case Uint32: {
        if (i64 < 0 || i64 > UINT32_MAX) {
            return value_error(ctx);
        }
        uint32_t u32 = (uint32_t)i64;
        PACK_SINGLE(x->ptr, u32, uint32_t, t->flags);
        return 0;
    }

    case Uint64: {
        if (i64 < 0) {
            return value_error(ctx);
        }
        uint64_t u64 = (uint64_t)i64;
        PACK_SINGLE(x->ptr, u64, uint64_t, t->flags);
        return 0;
    }

    case Float16: {
        if (i64 < -4503599627370496LL || i64 > 4503599627370496LL) {
            return value_error(ctx);
        }
        double real = (double)i64;
        return xnd_float_pack2(real, (unsigned char *)x->ptr, le(t->flags), ctx);
    }

    case Float32: {
        if (i64 < -4503599627370496LL || i64 > 4503599627370496LL) {
            return value_error(ctx);
        }
        double real = (double)i64;
        return xnd_float_pack4(real, (unsigned char *)x->ptr, le(t->flags), ctx);
    }

    case Float64: {
        if (i64 < -4503599627370496LL || i64 > 4503599627370496LL) {
            return value_error(ctx);
        }
        double real = (double)i64;
        xnd_float_pack8(real, (unsigned char *)x->ptr, le(t->flags));
        return 0;
    }

    case Complex32: {
        if (i64 < -4503599627370496LL || i64 > 4503599627370496LL) {
            return value_error(ctx);
        }

        double real = (double)i64;
        double imag = 0.0;

        if (xnd_float_pack2(real, (unsigned char *)x->ptr, le(t->flags), ctx) < 0) {
            return -1;
        }

        return xnd_float_pack2(imag, (unsigned char *)(x->ptr+2), le(t->flags), ctx);
    }

    case Complex64: {
        if (i64 < -4503599627370496LL || i64 > 4503599627370496LL) {
            return value_error(ctx);
        }

        double real = (double)i64;
        double imag = 0.0;

        if (xnd_float_pack4(real, (unsigned char *)x->ptr, le(t->flags), ctx) < 0) {
            return -1;
        }

        return xnd_float_pack4(imag, (unsigned char *)(x->ptr+4), le(t->flags), ctx);
    }

    case Complex128: {
        if (i64 < -4503599627370496LL || i64 > 4503599627370496LL) {
            return value_error(ctx);
        }

        double real = (double)i64;
        double imag = 0.0;

        xnd_float_pack8(real, (unsigned char *)x->ptr, le(t->flags));
        xnd_float_pack8(imag, (unsigned char *)(x->ptr+8), le(t->flags));

        return 0;
    }

    default:
        return type_error(ctx);
    }
}

static int
copy_uint64(xnd_t * const x, const uint64_t u64, ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;

    switch (t->tag) {
    case Int8: {
        if (u64 > INT8_MAX) {
            return value_error(ctx);
        }
        int8_t i8 = (int8_t)u64;
        PACK_SINGLE(x->ptr, i8, int8_t, t->flags);
        return 0;
    }

    case Int16: {
        if (u64 > INT16_MAX) {
            return value_error(ctx);
        }
        int16_t i16 = (int16_t)u64;
        PACK_SINGLE(x->ptr, i16, int16_t, t->flags);
        return 0;
    }

    case Int32: {
        if (u64 > INT32_MAX) {
            return value_error(ctx);
        }
        int32_t i32 = (int32_t)u64;
        PACK_SINGLE(x->ptr, i32, int32_t, t->flags);
        return 0;
    }

    case Int64: {
        if (u64 > INT64_MAX) {
            return value_error(ctx);
        }
        int64_t i64 = (int64_t)u64;
        PACK_SINGLE(x->ptr, i64, int64_t, t->flags);
        return 0;
    }

    case Uint8: {
        if (u64 > UINT8_MAX) {
            return value_error(ctx);
        }
        uint8_t u8 = (uint8_t)u64;
        PACK_SINGLE(x->ptr, u8, uint8_t, t->flags);
        return 0;
    }

    case Uint16: {
        if (u64 > UINT16_MAX) {
            return value_error(ctx);
        }
        uint16_t u16 = (uint16_t)u64;
        PACK_SINGLE(x->ptr, u16, uint16_t, t->flags);
        return 0;
    }

    case Uint32: {
        if (u64 > UINT32_MAX) {
            return value_error(ctx);
        }
        uint32_t u32 = (uint32_t)u64;
        PACK_SINGLE(x->ptr, u32, uint32_t, t->flags);
        return 0;
    }

    case Uint64: {
        PACK_SINGLE(x->ptr, u64, uint64_t, t->flags);
        return 0;
    }

    case Float16: {
        if (u64 > 4503599627370496LL) {
            return value_error(ctx);
        }
        double real = (double)u64;
        return xnd_float_pack2(real, (unsigned char *)x->ptr, le(t->flags), ctx);
    }

    case Float32: {
        if (u64 > 4503599627370496LL) {
            return value_error(ctx);
        }
        double real = (double)u64;
        return xnd_float_pack4(real, (unsigned char *)x->ptr, le(t->flags), ctx);
    }

    case Float64: {
        if (u64 > 4503599627370496LL) {
            return value_error(ctx);
        }
        double real = (double)u64;
        xnd_float_pack8(real, (unsigned char *)x->ptr, le(t->flags));
        return 0;
    }

    case Complex32: {
        if (u64 > 4503599627370496LL) {
            return value_error(ctx);
        }

        double real = (double)u64;
        double imag = 0.0;

        if (xnd_float_pack2(real, (unsigned char *)x->ptr, le(t->flags), ctx) < 0) {
            return -1;
        }

        return xnd_float_pack2(imag, (unsigned char *)x->ptr+2, le(t->flags), ctx);
    }

    case Complex64: {
        if (u64 > 4503599627370496LL) {
            return value_error(ctx);
        }

        double real = (double)u64;
        double imag = 0.0;

        if (xnd_float_pack4(real, (unsigned char *)x->ptr, le(t->flags), ctx) < 0) {
            return -1;
        }

        return xnd_float_pack4(imag, (unsigned char *)(x->ptr+4), le(t->flags), ctx);
    }

    case Complex128: {
        if (u64 > 4503599627370496LL) {
            return value_error(ctx);
        }

        double real = (double)u64;
        double imag = 0.0;

        xnd_float_pack8(real, (unsigned char *)x->ptr, le(t->flags));
        xnd_float_pack8(imag, (unsigned char *)(x->ptr+8), le(t->flags));
        return 0;
    }

    default:
        return type_error(ctx);
    }
}

static inline bool
isint(double real)
{
    double intpart;
    double fracpart;

    fracpart = modf(real, &intpart);
    if (fracpart == 0.0) {
        return (!isinf(intpart) && !isnan(intpart));
    }

    return false;
}

static int
copy_float64(xnd_t * const x, const double real, ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;

    switch (t->tag) {
    case Int8: {
        if (!isint(real) || real < INT8_MIN || real > INT8_MAX) {
            return value_error(ctx);
        }
        int8_t i8 = (int8_t)real;
        PACK_SINGLE(x->ptr, i8, int8_t, t->flags);
        return 0;
    }

    case Int16: {
        if (!isint(real) || real < INT16_MIN || real > INT16_MAX) {
            return value_error(ctx);
        }
        int16_t i16 = (int16_t)real;
        PACK_SINGLE(x->ptr, i16, int16_t, t->flags);
        return 0;
    }

    case Int32: {
        if (!isint(real) || real < INT32_MIN || real > INT32_MAX) {
            return value_error(ctx);
        }
        int32_t i32 = (int32_t)real;
        PACK_SINGLE(x->ptr, i32, int32_t, t->flags);
        return 0;
    }

    case Int64: {
        if (!isint(real) || real < -4503599627370496LL || real > 4503599627370496LL) {
            return value_error(ctx);
        }
        int64_t i64 = (int64_t)real;
        PACK_SINGLE(x->ptr, i64, int64_t, t->flags);
        return 0;
    }

    case Uint8: {
        if (!isint(real) || real < 0 || real > UINT8_MAX) {
            return value_error(ctx);
        }
        uint8_t u8 = (uint8_t)real;
        PACK_SINGLE(x->ptr, u8, uint8_t, t->flags);
        return 0;
    }

    case Uint16: {
        if (!isint(real) || real < 0 || real > UINT16_MAX) {
            return value_error(ctx);
        }
        uint16_t u16 = (uint16_t)real;
        PACK_SINGLE(x->ptr, u16, uint16_t, t->flags);
        return 0;
    }

    case Uint32: {
        if (!isint(real) || real < 0 || real > UINT32_MAX) {
            return value_error(ctx);
        }
        uint32_t u32 = (uint32_t)real;
        PACK_SINGLE(x->ptr, u32, uint32_t, t->flags);
        return 0;
    }

    case Uint64: {
        if (!isint(real) || real < 0 || real > 4503599627370496ULL) {
            return value_error(ctx);
        }
        uint64_t u64 = (uint64_t)real;
        PACK_SINGLE(x->ptr, u64, uint64_t, t->flags);
        return 0;
    }

    case Float16: {
        return xnd_float_pack2(real, (unsigned char *)x->ptr, le(t->flags), ctx);
    }

    case Float32: {
        return xnd_float_pack4(real, (unsigned char *)x->ptr, le(t->flags), ctx);
    }

    case Float64: {
        xnd_float_pack8(real, (unsigned char *)x->ptr, le(t->flags));
        return 0;
    }

    case Complex32: {
        double imag = 0.0;

        if (xnd_float_pack2(real, (unsigned char *)x->ptr, le(t->flags), ctx) < 0) {
            return -1;
        }

        return xnd_float_pack2(imag, (unsigned char *)(x->ptr+2), le(t->flags), ctx);
    }

    case Complex64: {
        double imag = 0.0;

        if (xnd_float_pack4(real, (unsigned char *)x->ptr, le(t->flags), ctx) < 0) {
            return -1;
        }

        return xnd_float_pack4(imag, (unsigned char *)(x->ptr+4), le(t->flags), ctx);
    }

    case Complex128: {
        double imag = 0.0;

        xnd_float_pack8(real, (unsigned char *)x->ptr, le(t->flags));
        xnd_float_pack8(imag, (unsigned char *)(x->ptr+8), le(t->flags));

        return 0;
    }

    default:
        return type_error(ctx);
    }
}

static int
copy_complex128(xnd_t * const x, const double real, const double imag,
                ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;

    switch (t->tag) {
    case Int8: case Int16: case Int32: case Int64:
    case Uint8: case Uint16: case Uint32: case Uint64:
    case Float16: case Float32: case Float64: {
       if (imag == 0.0) {
           return copy_float64(x, real, ctx);
       }

       return type_error(ctx);
    }

    case Complex32: {
        if (xnd_float_pack2(real, (unsigned char *)x->ptr, le(t->flags), ctx) < 0) {
            return -1;
        }
        return xnd_float_pack2(imag, (unsigned char *)(x->ptr+2), le(t->flags), ctx);
    }

    case Complex64: {
        if (xnd_float_pack4(real, (unsigned char *)x->ptr, le(t->flags), ctx) < 0) {
            return -1;
        }
        return xnd_float_pack4(imag, (unsigned char *)(x->ptr+4), le(t->flags), ctx);
    }

    case Complex128: {
        xnd_float_pack8(real, (unsigned char *)x->ptr, le(t->flags));
        xnd_float_pack8(imag, (unsigned char *)(x->ptr+8), le(t->flags));
        return 0;
    }

    default:
       return type_error(ctx);
    }
}

int
xnd_copy(xnd_t *y, const xnd_t *x, uint32_t flags, ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;
    const ndt_t * const u = y->type;
    int n;

    if (xnd_is_na(x)) {
        if (!ndt_is_optional(u)) {
            ndt_err_format(ctx, NDT_TypeError,
                "cannot copy NA to destination with non-optional type");
            return -1;
        }

        xnd_set_na(y);
        return 0;
    }

    if (t->tag == Ref || u->tag == Ref) {
        return copy_ref(y, x, flags, ctx);
    }

    switch (t->tag) {
    case FixedDim: {
        int64_t i;

        if (u->tag != FixedDim || u->FixedDim.shape != t->FixedDim.shape) {
            return type_error(ctx);
        }

        for (i = 0; i < t->FixedDim.shape; i++) {
            const xnd_t xnext = xnd_fixed_dim_next(x, i);
            xnd_t ynext = xnd_fixed_dim_next(y, i);
            n = xnd_copy(&ynext, &xnext, flags, ctx);
            if (n < 0) return n;
        }

        return 0;
    }

    case VarDim: {
        int64_t xstart, xstep, xshape;
        int64_t ystart, ystep, yshape;
        int64_t i;

        if (u->tag != VarDim) {
            return type_error(ctx);
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
            return type_error(ctx);
        }

        for (i = 0; i < xshape; i++) {
            const xnd_t xnext = xnd_var_dim_next(x, xstart, xstep, i);
            xnd_t ynext = xnd_var_dim_next(y, ystart, ystep, i);
            n = xnd_copy(&ynext, &xnext, flags, ctx);
            if (n < 0) return n;
        }

        return 0;
    }

    case Tuple: {
        if (u->tag != Tuple || u->Tuple.shape != t->Tuple.shape) {
            return type_error(ctx);
        }

        for (int64_t i = 0; i < t->Tuple.shape; i++) {
            const xnd_t xnext = xnd_tuple_next(x, i, ctx);
            if (xnext.ptr == NULL) {
                return -1;
            }

            xnd_t ynext = xnd_tuple_next(y, i, ctx);
            if (ynext.ptr == NULL) {
                return -1;
            }

            n = xnd_copy(&ynext, &xnext, flags, ctx);
            if (n < 0) return n;
        }

        return 0;
    }

    case Record: {
        if (u->tag != Record || u->Record.shape != t->Record.shape) {
            return type_error(ctx);
        }

        for (int64_t i = 0; i < t->Record.shape; i++) {
            if (strcmp(t->Record.names[i], u->Record.names[i]) != 0) {
                return type_error(ctx);
            }

            const xnd_t xnext = xnd_record_next(x, i, ctx);
            if (xnext.ptr == NULL) {
                return -1;
            }

            xnd_t ynext = xnd_record_next(y, i, ctx);
            if (ynext.ptr == NULL) {
                return -1;
            }

            n = xnd_copy(&ynext, &xnext, flags, ctx);
            if (n < 0) return n;
        }

        return 0;
    }

    case Constr: {
        if (u->tag != Constr || strcmp(u->Constr.name, t->Constr.name) != 0) {
            return type_error(ctx);
        }

        const xnd_t xnext = xnd_constr_next(x, ctx);
        if (xnext.ptr == NULL) {
            return -1;
        }

        xnd_t ynext = xnd_constr_next(y, ctx);
        if (ynext.ptr == NULL) {
            return -1;
        }

        return xnd_copy(&ynext, &xnext, flags, ctx);
    }

    case Nominal: {
        if (u->tag != Nominal) {
            return type_error(ctx);
        }

        if (strcmp(t->Nominal.name, u->Nominal.name) != 0) {
            return type_error(ctx);
        }

        const xnd_t xnext = xnd_nominal_next(x, ctx);
        if (xnext.ptr == NULL) {
            return -1;
        }

        xnd_t ynext = xnd_nominal_next(y, ctx);
        if (ynext.ptr == NULL) {
            return -1;
        }

        return xnd_copy(&ynext, &xnext, flags, ctx);
    }

    case Categorical: {
        int64_t i64;

        if (!ndt_equal(t, u)) {
            return type_error(ctx);
        }

        UNPACK_SINGLE(i64, x->ptr, int64_t, t->flags);
        PACK_SINGLE(y->ptr, i64, int64_t, u->flags);
        return 0;
    }

    case Char: {
        ndt_err_format(ctx, NDT_NotImplementedError, "char not implemented");
        return -1;
    }

    case Bool: {
        bool b;

        if (u->tag != Bool) {
            return type_error(ctx);
        }

        UNPACK_SINGLE(b, x->ptr, bool, u->flags);
        PACK_SINGLE(y->ptr, b, bool, t->flags);
        return 0;
    }

    case Int8: {
        int8_t i8;
        UNPACK_SINGLE(i8, x->ptr, int8_t, t->flags);
        return copy_int64(y, i8, ctx);
    }

    case Int16: {
        int16_t i16;
        UNPACK_SINGLE(i16, x->ptr, int16_t, t->flags);
        return copy_int64(y, i16, ctx);
    }

    case Int32: {
        int32_t i32;
        UNPACK_SINGLE(i32, x->ptr, int32_t, t->flags);
        return copy_int64(y, i32, ctx);
    }

    case Int64: {
        int64_t i64;
        UNPACK_SINGLE(i64, x->ptr, int64_t, t->flags);
        return copy_int64(y, i64, ctx);
    }

    case Uint8: {
        uint8_t u8;
        UNPACK_SINGLE(u8, x->ptr, uint8_t, t->flags);
        return copy_uint64(y, u8, ctx);
    }

    case Uint16: {
        uint16_t u16;
        UNPACK_SINGLE(u16, x->ptr, uint16_t, t->flags);
        return copy_uint64(y, u16, ctx);
    }

    case Uint32: {
        uint32_t u32;
        UNPACK_SINGLE(u32, x->ptr, uint32_t, t->flags);
        return copy_uint64(y, u32, ctx);
    }

    case Uint64: {
        uint64_t u64;
        UNPACK_SINGLE(u64, x->ptr, uint64_t, t->flags);
        return copy_uint64(y, u64, ctx);
    }

    case Float16: {
        double real = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        return copy_float64(y, real, ctx);
    }

    case Float32: {
        double real = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        return copy_float64(y, real, ctx);
    }

    case Float64: {
        double real = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        return copy_float64(y, real, ctx);
    }

    case Complex32: {
        double real, imag;

        real = xnd_float_unpack2((unsigned char *)x->ptr, le(t->flags));
        imag = xnd_float_unpack2((unsigned char *)(x->ptr+2), le(t->flags));

        return copy_complex128(y, real, imag, ctx);
    }

    case Complex64: {
        double real, imag;

        real = xnd_float_unpack4((unsigned char *)x->ptr, le(t->flags));
        imag = xnd_float_unpack4((unsigned char *)(x->ptr+4), le(t->flags));

        return copy_complex128(y, real, imag, ctx);
    }

    case Complex128: {
        double real, imag;

        real = xnd_float_unpack8((unsigned char *)x->ptr, le(t->flags));
        imag = xnd_float_unpack8((unsigned char *)(x->ptr+8), le(t->flags));

        return copy_complex128(y, real, imag, ctx);
    }

    case FixedString: {
        if (u->tag != FixedString ||
            u->FixedString.size != t->FixedString.size ||
            u->FixedString.encoding != t->FixedString.encoding ||
            u->datasize != t->datasize) {
            return type_error(ctx);
        }
        memcpy(y->ptr, x->ptr, (size_t)t->datasize);
        return 0;
    }

    case FixedBytes: {
        if (u->tag != FixedBytes ||
            u->FixedBytes.size != t->FixedBytes.size ||
            u->datasize != t->datasize) {
            return type_error(ctx);
        }
        memcpy(y->ptr, x->ptr, (size_t)t->datasize);
        return 0;
    }

    case String: {
        char *s;

        if (u->tag != String) {
            return type_error(ctx);
        }

        s = ndt_strdup(XND_POINTER_DATA(x->ptr), ctx);
        if (s == NULL) {
            return -1;
        }

        if (XND_POINTER_DATA(y->ptr) != NULL) {
            if (!(flags & XND_OWN_EMBEDDED)) {
                ndt_err_format(ctx, NDT_RuntimeError,
                    "cannot free string pointer, xnd does not own it");
                ndt_free(s);
                return -1;
            }
            ndt_free(XND_POINTER_DATA(y->ptr));
        }

        XND_POINTER_DATA(y->ptr) = s;
        return 0;
    }

    case Bytes: {
        unsigned char *s;
        int64_t size;

        if (u->tag != Bytes) {
            return type_error(ctx);
        }

        size = XND_BYTES_SIZE(x->ptr);

        s = ndt_aligned_calloc(u->Bytes.target_align, size);
        if (s == NULL) {
            (void)ndt_memory_error(ctx);
            return -1;
        }
        memcpy(s, XND_BYTES_DATA(x->ptr), (size_t)size);

        if (XND_BYTES_DATA(y->ptr) != NULL) {
            if (!(flags & XND_OWN_EMBEDDED)) {
                ndt_err_format(ctx, NDT_RuntimeError,
                    "cannot free string pointer, xnd does not own it");
                ndt_aligned_free(s);
                return -1;
            }
            ndt_aligned_free(XND_BYTES_DATA(y->ptr));
        }

        XND_BYTES_SIZE(y->ptr) = size;
        XND_BYTES_DATA(y->ptr) = s;
        return 0;
    }

    /* NOT REACHED: intercepted by equal_ref(). */
    case Ref:

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

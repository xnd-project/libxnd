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
 *    tbufsizes list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    tbufsizes list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    tbufsizes software without specific prior written permission.
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
#include "ndtypes.h"
#include "xnd.h"
#include "overflow.h"

#ifndef _MSC_VER
#include "config.h"
#endif


typedef struct {
    int64_t index;
    const ndt_t *type;
    int64_t ptr;
} xnd_bounds_t;


static inline xnd_bounds_t
_fixed_dim_next(const xnd_bounds_t *x, const int64_t i, bool *overflow)
{
    const ndt_t *t = x->type;
    const ndt_t *u = t->FixedDim.type;
    const int64_t step = MULi64(i, t->Concrete.FixedDim.step, overflow);
    xnd_bounds_t next;

    next.index = ADDi64(x->index, step, overflow);
    next.type = u;
    next.ptr = x->ptr;

    if (u->ndim == 0) {
        int64_t tmp = MULi64(next.index, next.type->datasize, overflow);
        next.ptr = ADDi64(x->ptr, tmp, overflow);
    }

    return next;
}

static inline xnd_bounds_t
_var_dim_next(const xnd_bounds_t *x, const int64_t start, const int64_t step,
              const int64_t i, bool *overflow)
{
    const ndt_t *t = x->type;
    const ndt_t *u = t->VarDim.type;
    xnd_bounds_t next;
    int64_t tmp;

    tmp = MULi64(i, step, overflow);
    next.index = ADDi64(start, tmp, overflow);
    next.type = u;
    next.ptr = x->ptr;

    if (u->ndim == 0) {
        tmp = MULi64(next.index, next.type->datasize, overflow);
        next.ptr = ADDi64(x->ptr, tmp, overflow);
    }

    return next;
}

static inline xnd_bounds_t
_tuple_next(const xnd_bounds_t *x, const int64_t i, bool *overflow)
{
    const ndt_t *t = x->type;
    xnd_bounds_t next;

    next.index = 0;
    next.type = t->Tuple.types[i];
    next.ptr = ADDi64(x->ptr, t->Concrete.Tuple.offset[i], overflow);

    return next;
}

static inline xnd_bounds_t
_record_next(const xnd_bounds_t *x, const int64_t i, bool *overflow)
{
    const ndt_t *t = x->type;
    xnd_bounds_t next;

    next.index = 0;
    next.type = t->Record.types[i];
    next.ptr = ADDi64(x->ptr, t->Concrete.Record.offset[i], overflow);

    return next;
}

static inline xnd_bounds_t
_constr_next(const xnd_bounds_t *x)
{
    const ndt_t *t = x->type;
    xnd_bounds_t next;

    next.index = 0;
    next.type = t->Constr.type;
    next.ptr = x->ptr;

    return next;
}

static inline xnd_bounds_t
_nominal_next(const xnd_bounds_t *x)
{
    const ndt_t *t = x->type;
    xnd_bounds_t next;

    next.index = 0;
    next.type = t->Nominal.type;
    next.ptr = x->ptr;

    return next;
}


static int
_xnd_bounds_check(const xnd_bounds_t * const x, const int64_t bufsize, ndt_context_t *ctx)
{
    const ndt_t * const t = x->type;
    bool overflow = false;

    if (ndt_is_abstract(t)) {
        ndt_err_format(ctx, NDT_ValueError,
            "bounds checking requires a concrete type");
        return -1;
    }

    if (ndt_subtree_is_optional(t)) {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "bounds checking not implemented for optional types");
        return -1;
    }

    switch (t->tag) {
    case FixedDim: {
        if (t->FixedDim.shape > 0) {
            xnd_bounds_t next = _fixed_dim_next(x, 0, &overflow);
            if (_xnd_bounds_check(&next, bufsize, ctx) < 0) {
                return -1;
            }
        }

        if (t->FixedDim.shape > 1) {
            xnd_bounds_t next = _fixed_dim_next(x, t->FixedDim.shape-1, &overflow);
            if (_xnd_bounds_check(&next, bufsize, ctx) < 0) {
                return -1;
            }
        }

        if (overflow) {
            goto overflow_error;
        }

        return 0;
    }

    case VarDim: {
        int64_t start, step, shape;

        shape = ndt_var_indices(&start, &step, t, x->index, ctx);
        if (shape < 0) {
            return -1;
        }

        if (shape > 0) {
            xnd_bounds_t next = _var_dim_next(x, start, step, 0, &overflow);
            if (_xnd_bounds_check(&next, bufsize, ctx) < 0) {
                return -1;
            }
        }

        if (shape > 1) {
            xnd_bounds_t next = _var_dim_next(x, start, step, shape-1, &overflow);
            if (_xnd_bounds_check(&next, bufsize, ctx) < 0) {
                return -1;
            }
        }

        if (overflow) {
            goto overflow_error;
        }

        return 0;
    }

    case Tuple: {
        if (t->Tuple.shape > 0) {
            xnd_bounds_t next = _tuple_next(x, 0, &overflow);
            if (_xnd_bounds_check(&next, bufsize, ctx) < 0) {
                return -1;
            }
        }

        if (t->Tuple.shape > 1) {
            xnd_bounds_t next = _tuple_next(x, t->Tuple.shape-1, &overflow);
            if (_xnd_bounds_check(&next, bufsize, ctx) < 0) {
                return -1;
            }
        }

        if (overflow) {
            goto overflow_error;
        }

        return 0;
    }

    case Record: {
        if (t->Record.shape > 0) {
            xnd_bounds_t next = _record_next(x, 0, &overflow);
            if (_xnd_bounds_check(&next, bufsize, ctx) < 0) {
                return -1;
            }
        }

        if (t->Record.shape > 1) {
            xnd_bounds_t next = _record_next(x, t->Record.shape-1, &overflow);
            if (_xnd_bounds_check(&next, bufsize, ctx) < 0) {
                return -1;
            }
        }

        if (overflow) {
            goto overflow_error;
        }

        return 0;
    }

    case Union: {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "bounds checking union types is not implemented");
        return -1;
    }

    case Ref: {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "bounds checking ref types is not implemented");
        return -1;
    }

    case Constr: {
        xnd_bounds_t next = _constr_next(x);
        if (_xnd_bounds_check(&next, bufsize, ctx) < 0) {
            return -1;
        }

        return 0;
    }

    case Nominal: {
        xnd_bounds_t next = _nominal_next(x);
        if (_xnd_bounds_check(&next, bufsize, ctx) < 0) {
            return -1;
        }

        return 0;
    }

    case VarDimElem: {
        ndt_err_format(ctx, NDT_NotImplementedError,
           "cannot bounds check var elem dimension");
        return -1;
    }

    case Char: {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "char not implemented");
        return -1;
    }

    case String: case Bytes: {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "serialization for string and bytes not implemented");
        return -1;
    }

    case Categorical:
    case Bool:
    case Int8: case Int16: case Int32: case Int64:
    case Uint8: case Uint16: case Uint32: case Uint64:
    case BFloat16: case Float16: case Float32: case Float64:
    case BComplex32: case Complex32: case Complex64: case Complex128:
    case FixedString: case FixedBytes: {
        const int64_t min = x->ptr;
        const int64_t max = ADDi64(min, t->datasize, &overflow);

        if (overflow) {
            goto overflow_error;
        }

        if (min < 0 || max > bufsize) {
            ndt_err_format(ctx, NDT_ValueError, "bounds check failed");
            return -1;
        }

        return 0;
    }

    /* NOT REACHED: intercepted by ndt_is_abstract(). */
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

overflow_error:
    ndt_err_format(ctx, NDT_ValueError, "overflow in bounds check");
    return -1;
}

int
xnd_bounds_check(const ndt_t *t, const int64_t linear_index, const int64_t bufsize,
                 ndt_context_t *ctx)
{
    xnd_bounds_t x;

    x.index = linear_index;
    x.type = t;
    x.ptr = 0;

    return _xnd_bounds_check(&x, bufsize, ctx);
}

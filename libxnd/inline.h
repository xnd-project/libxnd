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


#ifndef INLINE_H
#define INLINE_H


#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "ndtypes.h"
#include "xnd.h"


/*****************************************************************************/
/*                          Internal inline functions                        */
/*****************************************************************************/

/* Counterparts of the functions in xnd.h. These versions ignore the bitmaps. */

static inline xnd_t
_fixed_dim_next(const xnd_t *x, const int64_t i)
{
    const ndt_t *t = x->type;
    const ndt_t *u = t->FixedDim.type;
    const int64_t step = i * t->Concrete.FixedDim.step;
    xnd_t next;

    next.bitmap = xnd_bitmap_empty;
    next.index = x->index + step;
    next.type = u;
    next.ptr = u->ndim==0 ? x->ptr + next.index * next.type->datasize : x->ptr;

    return next;
}

static inline xnd_t
_var_dim_next(const xnd_t *x, const int64_t start, const int64_t step,
              const int64_t i)
{
    const ndt_t *t = x->type;
    const ndt_t *u = t->VarDim.type;
    xnd_t next;

    next.bitmap = xnd_bitmap_empty;
    next.index = start + i * step;
    next.type = u;
    next.ptr = u->ndim==0 ? x->ptr + next.index * next.type->datasize : x->ptr;

    return next;
}

static inline xnd_t
_tuple_next(const xnd_t *x, const int64_t i)
{
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_empty;
    next.index = 0;
    next.type = t->Tuple.types[i];
    next.ptr = x->ptr + t->Concrete.Tuple.offset[i];

    return next;
}

static inline xnd_t
_record_next(const xnd_t *x, const int64_t i)
{
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_empty;
    next.index = 0;
    next.type = t->Record.types[i];
    next.ptr = x->ptr + t->Concrete.Record.offset[i];

    return next;
}

static inline xnd_t
_ref_next(const xnd_t *x)
{
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_empty;
    next.index = 0;
    next.type = t->Ref.type;
    next.ptr = XND_POINTER_DATA(x->ptr);

    return next;
}

static inline xnd_t
_constr_next(const xnd_t *x)
{
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_empty;
    next.index = 0;
    next.type = t->Constr.type;
    next.ptr = x->ptr;

    return next;
}

static inline xnd_t
_nominal_next(const xnd_t *x)
{
    const ndt_t *t = x->type;
    xnd_t next;

    next.bitmap = xnd_bitmap_empty;
    next.index = 0;
    next.type = t->Nominal.type;
    next.ptr = x->ptr;

    return next;
}


#endif /* INLINE_H */

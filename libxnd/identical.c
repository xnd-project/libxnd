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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "ndtypes.h"
#include "xnd.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/*****************************************************************************/
/*                      Structural identity                                  */
/*****************************************************************************/

static int
identical_with_pointers(const xnd_t *x, const xnd_t *y, ndt_context_t *ctx) {
  const ndt_t *const t = x->type;
  int n;
  int64_t i;

  switch (t->tag) {

  case Ref: {
    const xnd_t xnext = xnd_ref_next(x, ctx);
    if (xnext.ptr == NULL) {
      return -1;
    }
    const xnd_t ynext = xnd_ref_next(y, ctx);
    if (ynext.ptr == NULL) {
      return -1;
    }
    return xnd_identical(&xnext, &ynext, ctx);
  }

  case Bytes:
    if (XND_BYTES_SIZE(x->ptr) != XND_BYTES_SIZE(y->ptr)) {
      return 0;
    }
    return memcmp(XND_BYTES_DATA(x->ptr), XND_BYTES_DATA(y->ptr),
                  XND_BYTES_SIZE(x->ptr)) == 0;

  case String:
    return strcmp(XND_POINTER_DATA(x->ptr), XND_POINTER_DATA(y->ptr)) == 0;

  case FixedDim: {
    for (i = 0; i < t->FixedDim.shape; i++) {
      const xnd_t xnext = xnd_fixed_dim_next(x, i);
      const xnd_t ynext = xnd_fixed_dim_next(y, i);
      n = xnd_identical(&xnext, &ynext, ctx);
      if (n <= 0)
        return n;
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
    yshape = ndt_var_indices(&ystart, &ystep, t, y->index, ctx);
    if (yshape < 0) {
      return -1;
    }
    if (yshape != xshape) {
      return 0;
    }
    for (i = 0; i < xshape; i++) {
      const xnd_t xnext = xnd_var_dim_next(x, xstart, xstep, i);
      const xnd_t ynext = xnd_var_dim_next(y, ystart, ystep, i);
      n = xnd_identical(&xnext, &ynext, ctx);
      if (n <= 0)
        return n;
    }
    return 1;
  }

  case Tuple: {
    for (i = 0; i < t->Tuple.shape; i++) {
      const xnd_t xnext = xnd_tuple_next(x, i, ctx);
      if (xnext.ptr == NULL) {
        return -1;
      }
      const xnd_t ynext = xnd_tuple_next(y, i, ctx);
      if (ynext.ptr == NULL) {
        return -1;
      }
      n = xnd_identical(&xnext, &ynext, ctx);
      if (n <= 0)
        return n;
    }
    return 1;
  }

  case Record: {
    for (i = 0; i < t->Record.shape; i++) {
      const xnd_t xnext = xnd_record_next(x, i, ctx);
      if (xnext.ptr == NULL) {
        return -1;
      }
      const xnd_t ynext = xnd_record_next(y, i, ctx);
      if (ynext.ptr == NULL) {
        return -1;
      }
      n = xnd_identical(&xnext, &ynext, ctx);
      if (n <= 0)
        return n;
    }
    return 1;
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
    return xnd_identical(&xnext, &ynext, ctx);
  }

  default:
    /* NOT REACHED */
    ndt_err_format(ctx, NDT_NotImplementedError, "type tag");
    return -1;
  }

  /* NOT REACHED: tags should be exhaustive */
  ndt_err_format(ctx, NDT_RuntimeError, "invalid type tag");
  return -1;
}


int xnd_identical(const xnd_t *x, const xnd_t *y, ndt_context_t *ctx) {
  const ndt_t *const t = x->type;
  const ndt_t *const u = y->type;
  int n;
  assert(ndt_is_concrete(t) && ndt_is_concrete(u));
  // some quick tests
  if (x == y) {
    return 1;
  }
  if (t->datasize != u->datasize) {
    return 0;
  }
  // check for identical types
  n = ndt_equal(t, u);
  if (n <= 0) {
    return n;
  }
  // check for identical bitmaps
  n = xnd_bitmap_identical(&x->bitmap, &y->bitmap, t, ctx);
  if (n <= 0) {
    return n;
  }
  // check for empty objects
  if (t->datasize == 0) {
    return 1;
  }
  // check for identical index state
  if (x->index != y->index) {
    return 0;
  }
  // check pointer free instances
  if (ndt_is_pointer_free(t)) {
    return memcmp(x->ptr, y->ptr, t->datasize) == 0;
  }
  // xnd instance contains Ref, Bytes, or String items
  return identical_with_pointers(x, y, ctx);
}

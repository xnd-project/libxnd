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
//#include <inttypes.h>
//#include <stdint.h>
//#include <stdlib.h>
#include <string.h>

/*****************************************************************************/
/*                      Structural identity                                  */
/*****************************************************************************/

int xnd_identical(const xnd_t *x, const xnd_t *y, ndt_context_t *ctx) {

  const ndt_t *const t = x->type;
  const ndt_t *const u = y->type;
  int64_t sz;

  assert(ndt_is_concrete(t) && ndt_is_concrete(u));

  if (x == y) {
    return 1;
  }

  if (!ndt_equal(t, u)) {
    return 0;
  }

  if (!xnd_bitmap_identical(&x->bitmap, &y->bitmap, t, ctx)) {
    return 0;
  }

  sz = XND_BYTES_SIZE(x->ptr);
  if (sz != XND_BYTES_SIZE(y->ptr)) {
    return 0;
  }

  if (memcmp(XND_BYTES_DATA(x->ptr), XND_BYTES_DATA(y->ptr), sz)) {
    return 0;
  }

  return 1;
}

/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2017, plures
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
#include <assert.h>
#include "ndtypes.h"
#include "test.h"


int
test_fixed(void)
{
    ndt_context_t *ctx;
    nd_array_t a, b;
    uint16_t *p;
    int ret = 0;
    int i, j, k, l;
    int64_t indices[3];

    /* a1 = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]] */
    const char *type1 = "3 * 2 * 2 * uint16";
    uint16_t data1[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};


    ctx = ndt_context_new();
    if (ctx == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }


    /***** Type with fixed dimensions *****/
    a = nd_empty(type1, ctx);
    if (a.type == NULL) {
        goto error;
    }

    p = (uint16_t *)a.ptr;
    for (i = 0; i < ARRAY_SIZE(data1); i++) {
        p[i] = data1[i];
    }

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 2; j++) {
            for (k = 0; k < 2; k++) {
                indices[0] = i; indices[1] = j; indices[2] = k;
                l = i * 4 + j * 2 + k;
                b = nd_subarray(a, indices, 3, ctx);
                if (b.ptr == NULL) {
                    goto error;
                }
                assert(b.type->tag == Uint16);
                if (*(uint16_t *)b.ptr != data1[l]) {
                    ndt_err_format(ctx, NDT_RuntimeError, "unexpected value");
                    goto error;
                }
            }
        }
    }


out:
    nd_del(a);
    ndt_context_del(ctx);
    return ret;

error:
    ret = -1;
    ndt_err_fprint(stderr, ctx);
    goto out;
}



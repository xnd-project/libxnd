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
test_var(void)
{
    ndt_context_t *ctx;
    xnd_t a, b;
    uint16_t *p;
    int ret = 0;
    int i;

    /* a1 = [[[0, 1], [2, 3]], [[4, 5, 6], [7], [8]], [[9, 10]]] */
    const char *type1 = "3 * var(2,3,1) * var(2,2,3,1,1,2) * uint16";
    uint16_t data1[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int64_t indices1[11][3] =
      {{0,0,0},{0,0,1},
       {0,1,0},{0,1,1},
       {1,0,0},{1,0,1},{1,0,2},
       {1,1,0},
       {1,2,0},
       {2,0,0},{2,0,1}};

    /* a2 = [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [10, 11], [12, 13]], [[14, 15]], [[16, 17]]], [[[18, 19], [20, 21]]]] */
    const char *type2 = "3 * var(2,3,1) * var(2,2,3,1,1,2) * 2 * uint16";
    uint16_t data2[22] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    int64_t indices2[22][4] =
      {{0,0,0,0},{0,0,0,1},{0,0,1,0},{0,0,1,1},
       {0,1,0,0},{0,1,0,1},{0,1,1,0},{0,1,1,1},
       {1,0,0,0},{1,0,0,1},{1,0,1,0},{1,0,1,1},{1,0,2,0},{1,0,2,1},
       {1,1,0,0},{1,1,0,1},
       {1,2,0,0},{1,2,0,1},
       {2,0,0,0},{2,0,0,1},{2,0,1,0},{2,0,1,1}};

    /* a3 = [[[0, 1], [2, None]], [[4, 5, None], None, [7]], [[8, 9]]] */
    const char *type3 = "3 * var(2,3,1) * ?var(2,2,3,0,1,2) * ?uint16";
    uint16_t data3[10] = {0, 1, 2, 28712, 4, 5, 11211, 7, 8, 9};
    int64_t accessible_indices3[10][3] =
      {{0,0,0},{0,0,1},
       {0,1,0},
       {0,1,1}, // NA
       {1,0,0},{1,0,1},
       {1,0,2}, // NA
       {1,2,0},
       {2,0,0},{2,0,1}};
    int64_t defined_indices3[8][3] =
      {{0,0,0},{0,0,1},
       {0,1,0},
       {1,0,0},{1,0,1},
       {1,2,0},
       {2,0,0},{2,0,1}};
    // int64_t undefined_indices3[2][3] = {{0,1,1}, {1,0,2}};
    int64_t undefined_dimension3[2] = {1,1}; // NA
    int64_t invalid_element_access3[3] = {1,1,0}; // ValueError

    /* a2 = [None, [[0, 1], [2, 3]], [None, [None, 4]]] */
    const char *type4 = "3 * ?var(0,2,2) * ?var(2,2,0,2) * ?uint16";
    uint16_t data4[6] = {0, 1, 2, 3, 28712, 4};
    int64_t access4[12][3] =
      {{1,0,0},{1,0,1},
       {1,1,0},{1,1,1},
       {2,1,0}, // NA
       {2,1,1}};
    int64_t def4[5] = {0, 1, 2, 3, 5};

    int64_t undef4a[1] = {0};     // NA
    int64_t undef4b[2] = {2,0};   // NA
    int64_t undef4c[3] = {2,1,0}; // NA

    int64_t invalid4a[2] = {0,0};   // ValueError
    int64_t invalid4b[3] = {2,0,0}; // ValueError

    //
    // a = [[[0, 1], [2, 3]], [[4, 5, 6], [7], [8]], [[9, 10]]]
    // const char *type5_full = "3 * var(2,3,1) * var(2,2,3,1,1,2) * uint16";
    //
    // a[:, :, 1:3] = [[[1], [3]], [[5, 6], [], []], [[10]]]
    //
    const char *type5 = "3 * var(2,3,1) * var(1:1, 3:1, 5:2, 7:0, 8:0, 10:1, 11:0) * uint16";
    uint16_t data5[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int64_t indices5[5][3] =
      {{0,0,0},{0,1,0},
       {1,0,0},{1,0,1},
       {2,0,0}};
    int64_t def5[5] = {1, 3, 5, 6, 10};


    ctx = ndt_context_new();
    if (ctx == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    /***** Type with var dimensions *****/
    a = xnd_empty(type1, ctx);
    if (a.type == NULL) {
        goto error;
    }

    p = (uint16_t *)a.ptr;
    for (i = 0; i < 11; i++) {
        p[i] = data1[i];
    }

    for (i = 0; i < 11; i++) {
        b = xnd_subarray(a, indices1[i], 3, ctx);
        if (b.ptr == NULL) {
            goto error;
        }
        assert(b.type->tag == Uint16);
        if (*(uint16_t *)b.ptr != data1[i]) {
            ndt_err_format(ctx, NDT_RuntimeError, "unexpected value");
            goto error;
        }
    }
    xnd_del(a);


    /***** Type with mixed fixed and var dimensions *****/
    a = xnd_empty(type2, ctx);
    if (a.type == NULL) {
        goto error;
    }

    p = (uint16_t *)a.ptr;
    for (i = 0; i < 22; i++) {
        p[i] = data2[i];
    }

    for (i = 0; i < 22; i++) {
        b = xnd_subarray(a, indices2[i], 4, ctx);
        if (b.ptr == NULL) {
            goto error;
        }
        assert(b.type->tag == Uint16);
        if (*(uint16_t *)b.ptr != data2[i]) {
            ndt_err_format(ctx, NDT_RuntimeError, "unexpected value");
            goto error;
        }
    }
    xnd_del(a);


    /***** Type with optional values and dimensions *****/
    a = xnd_empty(type3, ctx);
    if (a.type == NULL) {
        goto error;
    }

    p = (uint16_t *)a.ptr;
    for (i = 0; i < 10; i++) {
        p[i] = data3[i];
    }

    /*** Set validity bits ***/
    for (i = 0; i < 8; i++) {
        if (xnd_subarray_set_valid(a, defined_indices3[i], 3, ctx) < 0) {
            goto error;
        }
    }

    /*** Defined values ***/
    for (i = 0; i < 3; i++) {
        b = xnd_subarray(a, accessible_indices3[i], 3, ctx);
        if (b.ptr == NULL) {
            goto error;
        }
        assert(b.type->tag == OptionItem);
        assert(b.type->OptionItem.type->tag == Uint16);
        if (*(uint16_t *)b.ptr != data3[i]) {
            ndt_err_format(ctx, NDT_RuntimeError, "unexpected value");
            goto error;
        }
    }

    for (i = 4; i < 6; i++) {
        b = xnd_subarray(a, accessible_indices3[i], 3, ctx);
        if (b.ptr == NULL) {
            goto error;
        }
        assert(b.type->tag == OptionItem);
        assert(b.type->OptionItem.type->tag == Uint16);
        if (*(uint16_t *)b.ptr != data3[i]) {
            ndt_err_format(ctx, NDT_RuntimeError, "unexpected value");
            goto error;
        }
    }

    for (i = 7; i < 10; i++) {
        b = xnd_subarray(a, accessible_indices3[i], 3, ctx);
        if (b.ptr == NULL) {
            goto error;
        }
        assert(b.type->tag == OptionItem);
        assert(b.type->OptionItem.type->tag == Uint16);
        if (*(uint16_t *)b.ptr != data3[i]) {
            ndt_err_format(ctx, NDT_RuntimeError, "unexpected value");
            goto error;
        }
    }

    /*** NA values ***/
    b = xnd_subarray(a, accessible_indices3[3], 3, ctx);
    if (b.ptr == NULL) {
        goto error;
    }
    assert(b.type->tag == OptionItem);
    assert(b.type->OptionItem.type->tag == Uint16);

    if (b.ptr != XND_MISSING) {
        ndt_err_format(ctx, NDT_RuntimeError, "expected missing value");
        goto error;
    }

    b = xnd_subarray(a, accessible_indices3[6], 3, ctx);
    if (b.ptr == NULL) {
        goto error;
    }
    assert(b.type->tag == OptionItem);
    assert(b.type->OptionItem.type->tag == Uint16);

    if (b.ptr != XND_MISSING) {
        ndt_err_format(ctx, NDT_RuntimeError, "expected missing value");
        goto error;
    }


    /*** NA dimension ***/
    b = xnd_subarray(a, undefined_dimension3, 2, ctx);
    if (b.ptr == NULL) {
        goto error;
    }

    assert(b.type->tag == VarDim);
    if (b.ptr != XND_MISSING) {
        ndt_err_format(ctx, NDT_RuntimeError, "expected missing dimension");
        goto error;
    }

    /*** Invalid element access in a missing dimension ***/
    b = xnd_subarray(a, invalid_element_access3, 3, ctx);
    if (b.ptr != NULL || ctx->err != NDT_ValueError) {
        goto error;
    }
    ndt_err_clear(ctx);

    xnd_del(a);


    /***** Type with fixed (optional) dimensions and optional values *****/
    a = xnd_empty(type4, ctx);
    if (a.type == NULL) {
        goto error;
    }

    p = (uint16_t *)a.ptr;
    for (i = 0; i < ARRAY_SIZE(data4); i++) {
        p[i] = data4[i];
    }

    /*** Set validity bits ***/
    for (i = 0; i < ARRAY_SIZE(def4); i++) {
        if (xnd_subarray_set_valid(a, access4[def4[i]], 3, ctx) < 0) {
            goto error;
        }
    }

    /*** Defined values ***/
    for (i = 0; i < ARRAY_SIZE(def4); i++) {
        b = xnd_subarray(a, access4[def4[i]], 3, ctx);
        if (b.ptr == NULL) {
            goto error;
        }
        assert(b.type->tag == OptionItem);
        assert(b.type->OptionItem.type->tag == Uint16);
        if (*(uint16_t *)b.ptr != data4[def4[i]]) {
            ndt_err_format(ctx, NDT_RuntimeError, "unexpected value %u %u",
                           *(uint16_t *)b.ptr, data4[def4[i]]);
            goto error;
        }
    }


    /*** NA values ***/
    b = xnd_subarray(a, access4[4], 3, ctx);
    if (b.ptr == NULL) {
        goto error;
    }
    assert(b.type->tag == OptionItem);
    assert(b.type->OptionItem.type->tag == Uint16);

    if (b.ptr != XND_MISSING) {
        ndt_err_format(ctx, NDT_RuntimeError, "expected missing value");
        goto error;
    }


    b = xnd_subarray(a, undef4a, ARRAY_SIZE(undef4a), ctx);
    if (b.ptr == NULL) {
        goto error;
    }
    assert(b.type->tag == VarDim);

    if (b.ptr != XND_MISSING) {
        ndt_err_format(ctx, NDT_RuntimeError, "expected missing value");
        goto error;
    }



    b = xnd_subarray(a, undef4b, ARRAY_SIZE(undef4b), ctx);
    if (b.ptr == NULL) {
        goto error;
    }
    assert(b.type->tag == VarDim);

    if (b.ptr != XND_MISSING) {
        ndt_err_format(ctx, NDT_RuntimeError, "expected missing value");
        goto error;
    }


    b = xnd_subarray(a, undef4c, ARRAY_SIZE(undef4c), ctx);
    if (b.ptr == NULL) {
        goto error;
    }
    assert(b.type->tag == OptionItem);

    if (b.ptr != XND_MISSING) {
        ndt_err_format(ctx, NDT_RuntimeError, "expected missing value");
        goto error;
    }


    /*** Invalid element access in a missing dimension ***/
    b = xnd_subarray(a, invalid4a, ARRAY_SIZE(invalid4a), ctx);
    if (b.ptr != NULL || ctx->err != NDT_ValueError) {
        goto error;
    }
    ndt_err_clear(ctx);

    b = xnd_subarray(a, invalid4b, ARRAY_SIZE(invalid4b), ctx);
    if (b.ptr != NULL || ctx->err != NDT_ValueError) {
        goto error;
    }
    ndt_err_clear(ctx);

    xnd_del(a);

    /***** Type with multidimensional slice *****/
    a = xnd_empty(type5, ctx);
    if (a.type == NULL) {
        goto error;
    }

    p = (uint16_t *)a.ptr;
    for (i = 0; i < ARRAY_SIZE(data5); i++) {
        p[i] = data5[i];
    }

    /*** Defined values ***/
    for (i = 0; i < ARRAY_SIZE(indices5); i++) {
        b = xnd_subarray(a, indices5[i], 3, ctx);
        if (b.ptr == NULL) {
            goto error;
        }
        assert(b.type->tag == Uint16);
        if (*(uint16_t *)b.ptr != data5[def5[i]]) {
            ndt_err_format(ctx, NDT_RuntimeError, "unexpected value %u %u",
                           *(uint16_t *)b.ptr, data5[def5[i]]);
            goto error;
        }
    }


    fprintf(stderr, "test_var (5 test cases)\n");


out:
    xnd_del(a);
    ndt_context_del(ctx);
    return ret;

error:
    ret = -1;
    ndt_err_fprint(stderr, ctx);
    goto out;
}



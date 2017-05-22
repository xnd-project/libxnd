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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "ndtypes.h"
#include "test.h"


static int
init_tests(void)
{
    ndt_context_t *ctx;

    ctx = ndt_context_new();
    if (ctx == NULL) {
        fprintf(stderr, "error: out of memory");
        return -1;
    }

    if (ndt_init(ctx) < 0) {
        ndt_err_fprint(stderr, ctx);
        ndt_context_del(ctx);
        return -1;
    }

    ndt_context_del(ctx);
    return 0;
}

static int (*tests[])(void) = {
  test_vararray,
  NULL
};

int
main(void)
{
    int (**f)(void);
    int success = 0;
    int fail = 0;

    if (init_tests() < 0) {
        return 1;
    }

    for (f = tests; *f != NULL; f++) {
        if ((*f)() < 0)
            fail++;
        else
            success++;
    }

    if (fail) {
        fprintf(stderr, "\nFAIL (failures=%d)\n", fail);
    }
    else {
        fprintf(stderr, "\n%d tests OK.\n", success);
    }

    ndt_finalize();
    return fail ? 1 : 0;
}

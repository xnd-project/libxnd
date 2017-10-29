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


#ifndef XND_H
#define XND_H


#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "ndtypes.h"


#ifdef _MSC_VER
  #if defined (EXPORT)
    #define XND_API __declspec(dllexport)
  #elif defined(IMPORT)
    #define XND_API __declspec(dllimport)
  #else
    #define XND_API
  #endif
#else
  #define XND_API
#endif


/* xnd.ptr: special value to indicate NA */
#define XND_MISSING ((void *)0x1)

/* convenience macros to extract embedded values */
#define XND_POINTER_DATA(ptr) (*((char **)ptr))
#define XND_BYTES_SIZE(ptr) (((ndt_bytes_t *)ptr)->size)
#define XND_BYTES_DATA(ptr) (((ndt_bytes_t *)ptr)->data)


/* typed memory */
typedef struct _xnd_t {
    const ndt_t *type;  /* type of the data */
    int64_t index;      /* linear index for var dims */
    char *ptr;          /* data */
} xnd_t;


/*****************************************************************************/
/*                               API functions                               */
/*****************************************************************************/

XND_API char *xnd_new(const ndt_t *t, bool alloc_pointers, ndt_context_t *ctx);
XND_API int xnd_init(char *ptr, const ndt_t *t, bool alloc_pointers, ndt_context_t *ctx);
XND_API void xnd_del(xnd_t a);
XND_API xnd_t xnd_empty(const char *datashape, ndt_context_t *ctx);
XND_API int xnd_subarray_set_valid(xnd_t a, const int64_t *indices, int len, ndt_context_t *ctx);
XND_API xnd_t xnd_subarray(const xnd_t a, const int64_t *indices, int len, ndt_context_t *ctx);


#endif /* XND_H */

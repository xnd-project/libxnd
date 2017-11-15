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


/*
 * Ownership flags: The library itself has no notion of how many exported
 * views a master buffer has. The Python bindings for example use Pythons's
 * reference counting to to keep track of exported memory blocks.
 */
#define XND_OWN_TYPE     0x00000001U /* type pointer */
#define XND_OWN_STRINGS  0x00000002U /* embedded string pointers */
#define XND_OWN_BYTES    0x00000004U /* embedded bytes pointers */
#define XND_OWN_POINTERS 0x00000008U /* embedded pointers */

#define XND_OWN_ALL (XND_OWN_TYPE|XND_OWN_STRINGS|XND_OWN_BYTES|XND_OWN_POINTERS)
#define XND_OWN_EMBEDDED (XND_OWN_STRINGS|XND_OWN_BYTES|XND_OWN_POINTERS)


/* Convenience macros to extract embedded values. */
#define XND_POINTER_DATA(ptr) (*((char **)ptr))
#define XND_BYTES_SIZE(ptr) (((ndt_bytes_t *)ptr)->size)
#define XND_BYTES_DATA(ptr) (((ndt_bytes_t *)ptr)->data)


/* Typed memory block, usually a view. */
typedef struct xnd {
    int64_t index;      /* linear index for var dims */
    const ndt_t *type;  /* type of the data */
    char *ptr;          /* data */
} xnd_t;

/* Master memory block. */
typedef struct xnd_master {
    uint32_t flags; /* ownership flags */
    xnd_t master;   /* typed memory */
} xnd_master_t;

extern const xnd_t xnd_error;


/*****************************************************************************/
/*                         Create xnd memory blocks                          */
/*****************************************************************************/

XND_API xnd_master_t *xnd_empty_from_string(const char *datashape, uint32_t flags, ndt_context_t *ctx);
XND_API xnd_master_t *xnd_empty_from_type(const ndt_t *t, uint32_t flags, ndt_context_t *ctx);
XND_API void xnd_del(xnd_master_t *x);


/*****************************************************************************/
/*                         Traverse xnd memory blocks                        */
/*****************************************************************************/

XND_API xnd_t xnd_subtree(xnd_t x, const int64_t *indices, int len, ndt_context_t *ctx);


#endif /* XND_H */

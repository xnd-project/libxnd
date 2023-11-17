/* BSD 3-Clause License
 *
 * Copyright (c) 2018, Quansight and Sameer Deshmukh
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
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


/* File containing internal declarations for Ruby XND wrapper.
 *
 * Author: Sameer Deshmukh (@v0dro)
*/

#ifndef RUBY_XND_INTERNAL_H
#define RUBY_XND_INTERNAL_H

//#define XND_DEBUG 1

#ifdef XND_DEBUG
#include <assert.h>
#endif

#include <float.h>
#include "ruby.h"
#include "ruby/encoding.h"
#include "ruby_ndtypes.h"
#include "ruby_xnd.h"
#include "util.h"
#include "float_pack_unpack.h"

extern VALUE mRubyXND_GCGuard;

/* typedefs */
typedef struct XndObject XndObject;
typedef struct MemoryBlockObject MemoryBlockObject;

#include "gc_guard.h"

/* macros */
#if SIZEOF_LONG == SIZEOF_VOIDP
# define PTR2NUM(x)   (LONG2NUM((long)(x)))
# define NUM2PTR(x)   ((void*)(NUM2ULONG(x)))
#elif SIZEOF_LONG_LONG == SIZEOF_VOIDP
# define PTR2NUM(x)   (LL2NUM((LONG_LONG)(x)))
# define NUM2PTR(x)   ((void*)(NUM2ULL(x)))
#else
# error ---->> ruby requires sizeof(void*) == sizeof(long) or sizeof(LONG_LONG) to be compiled. <<----
#endif

/* Convert C int 't' to Ruby 'true' or 'false'. */
#define INT2BOOL(t) (t ? Qtrue : Qfalse)

#ifdef WORDS_BIGENDIAN
#define IEEE_BIG_ENDIAN_P 1
#define IEEE_LITTLE_ENDIAN_P NULL
#else
#define IEEE_LITTLE_ENDIAN_P 1
#define IEEE_BIG_ENDIAN_P NULL
#endif

#endif  /* RUBY_XND_INTERNAL_H */

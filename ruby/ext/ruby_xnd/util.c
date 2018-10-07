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

#include "ruby.h"
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include "ndtypes.h"

void
raise_error(void)
{
  VALUE exeception = rb_errinfo();

  rb_set_errinfo(Qnil);
  rb_exc_raise(exeception);
}

void
set_error_info(VALUE err, const char * msg)
{
  rb_set_errinfo(rb_exc_new2(err, msg));
}

size_t
safe_downcast(int64_t size)
{
#if SIZE_MAX < INT64_MAX
    if (size > INT32_MAX) {
      rb_raise(rb_eSizeError,
                "sizes should never exceed INT32_MAX on 32-bit platforms.");
    }
#endif
    return (size_t)size;
}

bool
check_invariants(const ndt_t *t)
{
#if SIZE_MAX < INT64_MAX
    return safe_downcast(t->datasize) >= 0;
#else
    (void)t;
    return 1;
#endif
}

VALUE
array_new(int64_t size)
{
#if SIZE_MAX < INT64_MAX
    size_t n = safe_downcast(size);
    return n < 0 ? NULL : rb_ary_new2(n);
#else
    return rb_ary_new2(size);
#endif
}

VALUE
bytes_from_string_and_size(const char *str, int64_t size)
{
#if SIZE_MAX < INT64_MAX
    size_t n = safe_downcast(size);
    return n < 0 ? NULL : rb_str_new(str, n);
#else
    return rb_str_new(str, size);
#endif
}

long long
mod(long long a, long long b)
{
    long long r = a % b;
    return r < 0 ? r + b : r;
}

void
rb_range_unpack(VALUE range, long long *begin, long long *end, long long *step, size_t size)
{
  /* FIXME: As of 27 Aug. 2018 Ruby trunk implements step as a property of
     Range and XND will support it as and when it is available. Maybe for 
     now we can implement a #step iterator in a separate method.
  */
  *step = 1;
  VALUE rb_begin = rb_funcall(range, rb_intern("begin"), 0, NULL);
  VALUE rb_end = rb_funcall(range, rb_intern("end"), 0, NULL);
  int exclude_end = RTEST(rb_funcall(range, rb_intern("exclude_end?"), 0, NULL));

  if (RB_TYPE_P(rb_begin, T_FLOAT)) {
    double value = RFLOAT_VALUE(rb_begin);

    if (isinf(value)) {
      *begin = 0;  
    }
  }
  else {
    long long temp = NUM2LL(rb_begin);

    if (temp < 0) {            /* if negative index map to positive. */
      temp = mod(temp, (long long)size);
    }

    *begin = temp;
  }

  if (RB_TYPE_P(rb_end, T_FLOAT)) {
    double value = RFLOAT_VALUE(rb_end);

    if (isinf(value)) {
      *end = INT64_MAX;
      return;
    }
  }
  else {
    long long temp = NUM2LL(rb_end);

    if (temp < 0) {              /* if negative index map to ppositive. */
      temp = mod(temp, (long long)size);
    }

    *end = temp;
  }

  /* a[0..0] in Ruby returns the 0th index. 
     a[0...0] in Ruby returns empty array like a[0:0] in Python.
     libxnd does not include the last index by default.  
     */
  if (!exclude_end) {          
    *end += 1;
  }
}

int
ndt_exists(void)
{
  return RTEST(rb_const_get(rb_cObject, rb_intern("NDT")));
}

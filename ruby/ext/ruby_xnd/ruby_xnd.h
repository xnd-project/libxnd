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

/* File containing headers for Ruby XND wrapper. 
 *
 * Author: Sameer Deshmukh (@v0dro)
*/
#ifndef RUBY_XND_H
#define RUBY_XND_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ruby.h"
#include "ndtypes.h"
#include "xnd.h"

  size_t rb_xnd_hash_size(VALUE hash);
  int rb_xnd_get_complex_values(VALUE comp, double *real, double *imag);
  /* Return true if obj is of type XND. */
  int rb_xnd_check_type(VALUE obj);
  const xnd_t * rb_xnd_const_xnd(VALUE xnd);
  VALUE rb_xnd_empty_from_type(ndt_t *t);
  VALUE rb_xnd_from_xnd(xnd_t *x);
  
  typedef struct XndObject XndObject;

#ifdef __cplusplus
}
#endif

#endif  /* RUBY_XND_H */

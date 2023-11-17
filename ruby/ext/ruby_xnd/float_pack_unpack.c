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

/* Functions for packing and unpacking floats from char * arrays. 

   Author: Sameer Deshmukh (@v0dro)
*/

#include "ruby_xnd_internal.h"

/* Pack a 32-bit float into a contiguous unsigned char* buffer. 
   
   Reference:
   https://github.com/python/cpython/blob/master/Include/floatobject.h#L77

   @param num The number to be packed.
   @param ptr
   @param le Le is a boolean argument. True if you want the string in
   litte-endian format, false if you want it in big-endian format.
   
   @return 0 if success. Raise ruby error if failure.
*/
int
rb_xnd_pack_float32(double num, unsigned char* ptr, int le)
{
  float y = (float)num;
  int i, incr = 1;

  if (isinf(y) && !isinf(num)) {
    rb_raise(rb_eRangeError, "cannot fit value in 32-bit floating point number.");
  }

  unsigned char s[sizeof(float)];
  memcpy(s, &y, sizeof(float));

#ifdef XND_DEBUG
  if (!le) { // choose big-endian
    ptr += 3;
    incr = -1;
  }
#else
  if ((IEEE_LITTLE_ENDIAN_P && !le) || (IEEE_BIG_ENDIAN_P && le)) { // choose big-endian
    ptr += 3;
    incr = -1;
  }
#endif

  for (i = 0; i < sizeof(float); ++i) {
    *ptr = s[i];
    ptr += incr;
  }

  return 0;
}

/* Unpack a 32-bit float from a contiguos unsigned char* buffer. 
  
   @param ptr : 
   @param le Le is a boolean argument. True if you want the string in
   litte-endian format, false if you want it in big-endian format.

   @return unpacked number as a double.
*/
int
rb_xnd_unpack_float32(float *x, unsigned char* ptr, int le)
{
#ifdef XND_DEBUG
  if (!le) // big-endian
#else
    if ((IEEE_LITTLE_ENDIAN_P && !le) || (IEEE_BIG_ENDIAN_P && le))// big-endian
#endif
      {
        char buf[4];
        char *d = &buf[3];
        int i;

        for (i = 0; i < sizeof(float); ++i) {
          *d-- = *ptr++;
        }
        memcpy(x, buf, sizeof(float));
      }
    else {
      memcpy(x, ptr, sizeof(float));
    }

  return 0;
}

/* Pack 64 bit floating point number into an unsigned char array.

   @param num
   @param ptr
   @param le
 */
int
rb_xnd_pack_float64(double num, unsigned char *ptr, int le)
{
  int i, incr = 1;

  unsigned char s[sizeof(double)];
  memcpy(s, &num, sizeof(double));

#ifdef XND_DEBUG
  if (!le) { // choose big-endian
    ptr += 7;
    incr = -1;
  }
#else
  if ((IEEE_LITTLE_ENDIAN_P && !le) || (IEEE_BIG_ENDIAN_P && le)) { // choose big-endian
    ptr += 7;
    incr = -1;
  }
#endif

  for (i = 0; i < sizeof(double); ++i) {
    *ptr = s[i];
    ptr += incr;
  }

  return 0;
}

/* Unpack a 64-bit floating point number from an unsigned char array and return
   the resulting number as a type double.
 */
int
rb_xnd_unpack_float64(double *x, unsigned char *ptr, int le)
{
#ifdef XND_DEBUG
  if (!le) { // big-endian 
#else
    if ((IEEE_LITTLE_ENDIAN_P && !le) || (IEEE_BIG_ENDIAN_P && le)) { // big-endian
#endif
        char buf[sizeof(double)];
        char *d = &buf[sizeof(double)-1];
        int i;

        for (i = 0; i < sizeof(double); ++i) {
          *d-- = *ptr++;
        }
        memcpy(x, buf, sizeof(double));
      }
    else {
      memcpy(x, ptr, sizeof(double));
    }

  return 0;
}

#ifdef XND_DEBUG
/* Functions for testing packing/unpacking functions. 

   In order to avoid the need for injecting the dependency of a C testing framework,
   these are tests that work with the pack/unpack functions and are called in the
   Init_ function if XND_DEBUG is defined.
*/
void
test_pack_float32(void)
{
  double num = 16448.0;
  int i;
  unsigned char ptr[4];

  /* test big endian */
  unsigned char ans_bige[4] = {0x46, 0x80, 0x80, 0x00};
  rb_xnd_pack_float32(num, ptr, 0);
  for (i = 0; i < 4; i++) {
    assert(ans_bige[i] == ptr[i]);
  }

  /* test little endian */
  unsigned char ans_lite[4] = {0, 0X80, 0X80, 0X46};
  rb_xnd_pack_float32(num, ptr, 1);
  for (i = 0; i < 4; i++) {
    assert(ans_lite[i] == ptr[i]);
  }
}

void test_unpack_float32(void)
{
  float answer = 16448.0, result = 0.0;
  
  /* test big endian */
  unsigned char ptr_bige[4] = {0x46, 0x80, 0x80, 0x00};
  rb_xnd_unpack_float32(&result, ptr_bige, 0);
  assert(answer == result);
  
  /* test little endian */
  unsigned char ptr_lite[4] = {0, 0X80, 0X80, 0X46};
  rb_xnd_unpack_float32(&result, ptr_lite, 1);
  assert(answer == result);
}

void test_pack_float64(void)
{
  double num = 16448.0;
  int i;
  unsigned char ptr[8];

  /* test big endian. */
  unsigned char ans_bige[8] = {0x40, 0xD0, 0x10, 0, 0, 0, 0, 0};
  rb_xnd_pack_float64(num, ptr, 0);
  for (i = 0; i < 8; i++) {
    assert(ans_bige[i] == ptr[i]);
  }

  /* test little endian. */
  unsigned char ans_lite[8] = {0, 0, 0, 0, 0, 0X10, 0XD0, 0X40};
  rb_xnd_pack_float64(num, ptr, 1);
  for (i = 0; i < 8; i++) {
    assert(ans_lite[i] == ptr[i]);
  }

  double a = 1.0;
  unsigned char ans_lite_a[8] = {0, 0, 0, 0, 0, 0, 0xF0, 0x3F};
  rb_xnd_pack_float64(a, ptr, 1);
  for (i = 0; i < 8; i++) {
    assert(ans_lite_a[i] == ptr[i]);
  }
}

void test_unpack_float64(void)
{
  double answer = 16448.0, result = 0.0;

  /* test big-endian */
  unsigned char ptr_bige[8] = {0x40, 0xD0, 0x10, 0, 0, 0, 0, 0};
  rb_xnd_unpack_float64(&result, ptr_bige, 0);
  assert(answer == result);

  /* test little-endian */
  unsigned char ptr_lite[8] = {0, 0, 0, 0, 0, 0X10, 0XD0, 0X40};
  rb_xnd_unpack_float64(&result, ptr_lite, 1);
  assert(answer == result);

  double a = 1.0;
  unsigned char ans_lite_a[8] = {0, 0, 0, 0, 0, 0, 0xF0, 0x3F};
  rb_xnd_unpack_float64(&result, ans_lite_a, 1);
  assert(a == result);
}

void run_float_pack_unpack_tests(void)
{
  test_pack_float32();
  test_unpack_float32();
  test_pack_float64();
  test_unpack_float64();
}
#endif

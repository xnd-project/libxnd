/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Modified and adapted for gumath. */

#ifndef BFLOAT16_H
#define BFLOAT16_H


#include <stdint.h>
#include <math.h>


union FP32 {
    unsigned int u;
    float f;
};

// Converts a float point to bfloat16, with round-nearest-to-even as rounding
// method.
// TODO: There is a slightly faster implementation (8% faster on CPU)
// than this (documented in cl/175987786), that is exponentially harder to
// understand and document. Switch to the faster version when converting to
// BF16 becomes compute-bound.
static inline uint16_t
xnd_round_to_bfloat16(float v)
{
    uint32_t input;
    union FP32 f;
    f.f = v;
    input = f.u;
    uint16_t output;

    if (isnan(v)) {
      // If the value is a NaN, squash it to a qNaN with msb of fraction set,
      // this makes sure after truncation we don't end up with an inf.
      //
      // qNaN magic: All exponent bits set + most significant bit of fraction
      // set.
      output = 0x7fc0;
    } else {
      // Fast rounding algorithm that rounds a half value to nearest even. This
      // reduces expected error when we convert a large number of floats. Here
      // is how it works:
      //
      // Definitions:
      // To convert a float 32 to bfloat16, a float 32 can be viewed as 32 bits
      // with the following tags:
      //
      // Sign |  Exp (8 bits) | Frac (23 bits)
      //  S     EEEEEEEE         FFFFFFLRTTTTTTTTTTTTTTT
      //
      //  S: Sign bit.
      //  E: Exponent bits.
      //  F: First 6 bits of fraction.
      //  L: Least significant bit of resulting bfloat16 if we truncate away the
      //  rest of the float32. This is also the 7th bit of fraction
      //  R: Rounding bit, 8th bit of fraction.
      //  T: Sticky bits, rest of fraction, 15 bits.
      //
      // To round half to nearest even, there are 3 cases where we want to round
      // down (simply truncate the result of the bits away, which consists of
      // rounding bit and sticky bits) and two cases where we want to round up
      // (truncate then add one to the result).
      //
      // The fast converting algorithm simply adds lsb (L) to 0x7fff (15 bits of
      // 1s) as the rounding bias, adds the rounding bias to the input, then
      // truncates the last 16 bits away.
      //
      // To understand how it works, we can analyze this algorithm case by case:
      //
      // 1. L = 0, R = 0:
      //   Expect: round down, this is less than half value.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input may create any carry, depending on
      //   whether there is any value set to 1 in T bits.
      //   - R may be set to 1 if there is a carry.
      //   - L remains 0.
      //   - Note that this case also handles Inf and -Inf, where all fraction
      //   bits, including L, R and Ts are all 0. The output remains Inf after
      //   this algorithm.
      //
      // 2. L = 1, R = 0:
      //   Expect: round down, this is less than half value.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 1 = 0x8000
      //   - Adding rounding bias to input doesn't change sticky bits but
      //   adds 1 to rounding bit.
      //   - L remains 1.
      //
      // 3. L = 0, R = 1, all of T are 0:
      //   Expect: round down, this is exactly at half, the result is already
      //   even (L=0).
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input sets all sticky bits to 1, but
      //   doesn't create a carry.
      //   - R remains 1.
      //   - L remains 0.
      //
      // 4. L = 1, R = 1:
      //   Expect: round up, this is exactly at half, the result needs to be
      //   round to the next even number.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 1 = 0x8000
      //   - Adding rounding bias to input doesn't change sticky bits, but
      //   creates a carry from rounding bit.
      //   - The carry sets L to 0, creates another carry bit and propagate
      //   forward to F bits.
      //   - If all the F bits are 1, a carry then propagates to the exponent
      //   bits, which then creates the minimum value with the next exponent
      //   value. Note that we won't have the case where exponents are all 1,
      //   since that's either a NaN (handled in the other if condition) or inf
      //   (handled in case 1).
      //
      // 5. L = 0, R = 1, any of T is 1:
      //   Expect: round up, this is greater than half.
      //
      //   Algorithm:
      //   - Rounding bias: 0x7fff + 0 = 0x7fff
      //   - Adding rounding bias to input creates a carry from sticky bits,
      //   sets rounding bit to 0, then create another carry.
      //   - The second carry sets L to 1.
      //
      // Examples:
      //
      //  Exact half value that is already even:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0     1000000000000000
      //
      //     This falls into case 3. We truncate the rest of 16 bits and no
      //     carry is created into F and L:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0
      //
      //  Exact half value, round to next even number:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 0 1     1000000000000000
      //
      //     This falls into case 4. We create a carry from R and T,
      //     which then propagates into L and F:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 0      0 0 0 0 0 1 0
      //
      //
      //  Max denormal value round to min normal value:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     0 0 0 0 0 0 0 0      1 1 1 1 1 1 1     1111111111111111
      //
      //     This falls into case 4. We create a carry from R and T,
      //     propagate into L and F, which then propagates into exponent
      //     bits:
      //
      //    Output:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     0 0 0 0 0 0 0 1      0 0 0 0 0 0 0
      //
      //  Max normal value round to Inf:
      //    Input:
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit) | Frac (last 16 bit)
      //     S     E E E E E E E E      F F F F F F L     RTTTTTTTTTTTTTTT
      //     0     1 1 1 1 1 1 1 0      1 1 1 1 1 1 1     1111111111111111
      //
      //     This falls into case 4. We create a carry from R and T,
      //     propagate into L and F, which then propagates into exponent
      //     bits:
      //
      //    Sign |  Exp (8 bit)     | Frac (first 7 bit)
      //     S     E E E E E E E E      F F F F F F L
      //     0     1 1 1 1 1 1 1 1      0 0 0 0 0 0 0
      //
      //
      // Least significant bit of resulting bfloat.
      uint32_t lsb = (input >> 16) & 1;
      uint32_t rounding_bias = 0x7fff + lsb;
      input += rounding_bias;
      output = (uint16_t)(input >> 16);
    }
    return output;
}


#endif  // BFLOAT16_H

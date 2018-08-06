/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2017-2018, plures
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


#include <stdint.h>


/*****************************************************************************/
/*                       Arithmetic with overflow checking                   */
/*****************************************************************************/

#if defined(__GNUC__) && __GNUC__ >= 5 && !defined(__INTEL_COMPILER)
static inline int64_t
ADDi64(int64_t a, int64_t b, bool *overflow)
{
    int64_t c;
    *overflow |= __builtin_add_overflow(a, b, &c);
    return c;
}

static inline int64_t
SUBi64(int64_t a, int64_t b, bool *overflow)
{
    int64_t c;
    *overflow |= __builtin_sub_overflow(a, b, &c);
    return c;
}

static inline int64_t
MULi64(int64_t a, int64_t b, bool *overflow)
{
    int64_t c;
    *overflow |= __builtin_mul_overflow(a, b, &c);
    return c;
}

static inline size_t
MULi64_size(int64_t a, int64_t b, bool *overflow)
{
    int64_t c;
    *overflow |= __builtin_mul_overflow(a, b, &c);
#if SIZE_MAX < INT64_MAX
    *overflow |= (c > INT32_MAX);
#endif
    return (size_t)c;
}

static inline int64_t
ABSi64(int64_t a, bool *overflow)
{
    if (a == INT64_MIN) {
        *overflow = 1;
        return INT64_MIN;
    }
    return a >= 0 ? a : -a;
}

static inline uint16_t
ADDu16(uint16_t a, uint16_t b, bool *overflow)
{
    uint16_t c;
    *overflow |= __builtin_add_overflow(a, b, &c);
    return c;
}
#else
static inline int64_t
ADDi64(int64_t a, int64_t b, bool *overflow)
{
    int64_t c = (uint64_t)a + (uint64_t)b;
    *overflow |= ((a < 0 && b < 0 && c >= 0) || (a >= 0 && b >= 0 && c < 0));
    return c;
}

static inline int64_t
SUBi64(int64_t a, int64_t b, bool *overflow)
{
    int64_t c = (uint64_t)a - (uint64_t)b;
    *overflow |= ((a < 0 && b >= 0 && c >= 0) || (a >= 0 && b < 0 && c < 0));
    return c;
}

static inline int64_t
MULi64(int64_t a, int64_t b, bool *overflow)
{
    int64_t c = (uint64_t)a * (uint64_t)b;
    *overflow |= ((b < 0 && a == INT64_MIN) || (b != 0 && a != c / b));
    return c;
}

static inline size_t
MULi64_size(int64_t a, int64_t b, bool *overflow)
{
    int64_t c = (uint64_t)a * (uint64_t)b;
    *overflow |= ((b < 0 && a == INT64_MIN) || (b != 0 && a != c / b));
#if SIZE_MAX < INT64_MAX
    *overflow |= (c > INT32_MAX);
#endif
    return (size_t)c;
}

static inline int64_t
ABSi64(int64_t a, bool *overflow)
{
    if (a == INT64_MIN) {
        *overflow = 1;
        return INT64_MIN;
    }
    return a >= 0 ? a : -a;
}

static inline uint16_t
ADDu16(uint16_t a, uint16_t b, bool *overflow)
{
    uint16_t c = a + b;
    *overflow |= (c < a);
    return c;
}
#endif /* OVERFLOW_H */

#ifndef CONTRIB_H
#define CONTRIB_H


#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>


/* PSF copyright: Written by Jim Hugunin and Chris Chase. */
static inline int64_t
xnd_slice_adjust_indices(int64_t length,
                         int64_t *start, int64_t *stop, int64_t step)
{
    /* this is harder to get right than you might think */

    assert(step != 0);
    assert(step >= -INT64_MAX);

    if (*start < 0) {
        *start += length;
        if (*start < 0) {
            *start = (step < 0) ? -1 : 0;
        }
    }
    else if (*start >= length) {
        *start = (step < 0) ? length - 1 : length;
    }

    if (*stop < 0) {
        *stop += length;
        if (*stop < 0) {
            *stop = (step < 0) ? -1 : 0;
        }
    }
    else if (*stop >= length) {
        *stop = (step < 0) ? length - 1 : length;
    }

    if (step < 0) {
        if (*stop < *start) {
            return (*start - *stop - 1) / (-step) + 1;
        }
    }
    else {
        if (*start < *stop) {
            return (*stop - *start - 1) / step + 1;
        }
    }

    return 0;
}

/* PSF copyright: Original by Eli Stevens and Mark Dickinson. */
static inline int
xnd_float_pack2(double x, unsigned char *p, int le, ndt_context_t *ctx)
{
    unsigned char sign;
    int e;
    double f;
    unsigned short bits;
    int incr = 1;

    if (x == 0.0) {
        sign = (copysign(1.0, x) == -1.0);
        e = 0;
        bits = 0;
    }
    else if (isinf(x)) {
        sign = (x < 0.0);
        e = 0x1f;
        bits = 0;
    }
    else if (isnan(x)) {
        /* There are 2046 distinct half-precision NaNs (1022 signaling and
           1024 quiet), but there are only two quiet NaNs that don't arise by
           quieting a signaling NaN; we get those by setting the topmost bit
           of the fraction field and clearing all other fraction bits. We
           choose the one with the appropriate sign. */
        sign = (copysign(1.0, x) == -1.0);
        e = 0x1f;
        bits = 512;
    }
    else {
        sign = (x < 0.0);
        if (sign) {
            x = -x;
        }

        f = frexp(x, &e);
        if (f < 0.5 || f >= 1.0) {
            ndt_err_format(ctx, NDT_RuntimeError,
                           "frexp() result out of range");
            return -1;
        }

        /* Normalize f to be in the range [1.0, 2.0) */
        f *= 2.0;
        e--;

        if (e >= 16) {
            goto overflow;
        }
        else if (e < -25) {
            /* |x| < 2**-25. Underflow to zero. */
            f = 0.0;
            e = 0;
        }
        else if (e < -14) {
            /* |x| < 2**-14. Gradual underflow */
            f = ldexp(f, 14 + e);
            e = 0;
        }
        else /* if (!(e == 0 && f == 0.0)) */ {
            e += 15;
            f -= 1.0; /* Get rid of leading 1 */
        }

        f *= 1024.0; /* 2**10 */
        /* Round to even */
        bits = (unsigned short)f; /* Note the truncation */
        assert(bits < 1024);
        assert(e < 31);
        if ((f - bits > 0.5) || ((f - bits == 0.5) && (bits % 2 == 1))) {
            ++bits;
            if (bits == 1024) {
                /* The carry propagated out of a string of 10 1 bits. */
                bits = 0;
                ++e;
                if (e == 31)
                    goto overflow;
            }
        }
    }

    bits |= (e << 10) | (sign << 15);

    /* Write out result. */
    if (le) {
        p += 1;
        incr = -1;
    }

    /* First byte */
    *p = (unsigned char)((bits >> 8) & 0xFF);
    p += incr;

    /* Second byte */
    *p = (unsigned char)(bits & 0xFF);

    return 0;

overflow:
    ndt_err_format(ctx, NDT_ValueError,
        "float too large to pack with float16 type");
    return -1;
}

/* PSF copyright: Original by Tim Peters. */
static inline int
xnd_float_pack4(double x, unsigned char *p, int le, ndt_context_t *ctx)
{
    float y = (float)x;
    int i, incr = 1;

    if (isinf(y) && !isinf(x)) {
        goto overflow;
    }

    unsigned char s[sizeof(float)];
    memcpy(s, &y, sizeof(float));

    if ((xnd_float_is_little_endian() && !le) ||
        (xnd_float_is_big_endian() && le)) {
        p += 3;
        incr = -1;
    }

    for (i = 0; i < 4; i++) {
        *p = s[i];
        p += incr;
    }

    return 0;

overflow:
    ndt_err_format(ctx, NDT_ValueError,
        "float too large to pack with float32 type");
    return -1;
}

/* PSF copyright: Original by Tim Peters. */
static inline void
xnd_float_pack8(double x, unsigned char *p, int le)
{
    const unsigned char *s = (unsigned char*)&x;
    int i, incr = 1;

    if ((xnd_double_is_little_endian() && !le) ||
        (xnd_double_is_big_endian() && le)) {
        p += 7;
        incr = -1;
    }

    for (i = 0; i < 8; i++) {
        *p = *s++;
        p += incr;
    }
}

/* PSF copyright: Original by Eli Stevens and Mark Dickinson. */
static inline double
xnd_float_unpack2(const unsigned char *p, int le)
{
    unsigned char sign;
    int e;
    unsigned int f;
    double x;
    int incr = 1;

    if (le) {
        p += 1;
        incr = -1;
    }

    /* First byte */
    sign = (*p >> 7) & 1;
    e = (*p & 0x7C) >> 2;
    f = (*p & 0x03) << 8;
    p += incr;

    /* Second byte */
    f |= *p;

    if (e == 0x1f) {
        if (f == 0) {
            /* Infinity */
            return sign ? -INFINITY : INFINITY;
        }
        else {
            /* NaN */
            return sign ? -NAN : NAN;
        }
    }

    x = (double)f / 1024.0;

    if (e == 0) {
        e = -14;
    }
    else {
        x += 1.0;
        e -= 15;
    }
    x = ldexp(x, e);

    if (sign)
        x = -x;

    return x;
}

/* PSF copyright: Original by Tim Peters. */
static inline double
xnd_float_unpack4(const unsigned char *p, int le)
{
    float x;

    if ((xnd_float_is_little_endian() && !le) ||
        (xnd_float_is_big_endian() && le)) {
        char buf[4];
        char *d = &buf[3];
        int i;

        for (i = 0; i < 4; i++) {
            *d-- = *p++;
        }
        memcpy(&x, buf, 4);
    }
    else {
        memcpy(&x, p, 4);
    }

    return x;
}

/* PSF copyright: Original by Tim Peters. */
static inline double
xnd_float_unpack8(const unsigned char *p, int le)
{
    double x;

    if ((xnd_double_is_little_endian() && !le) ||
        (xnd_double_is_big_endian() && le)) {
        char buf[8];
        char *d = &buf[7];
        int i;

        for (i = 0; i < 8; i++) {
            *d-- = *p++;
        }
        memcpy(&x, buf, 8);
    }
    else {
        memcpy(&x, p, 8);
    }

    return x;
}


#endif /* CONTRIB_H */

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

    if ((xnd_double_is_little_endian() && !le) ||
        (xnd_double_is_big_endian() && le)) {
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

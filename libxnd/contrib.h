#ifndef CONTRIB_H
#define CONTRIB_H


#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>


/* PSF copyright: Written by Jim Hugunin and Chris Chase. */
int64_t
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


#endif /* CONTRIB_H */

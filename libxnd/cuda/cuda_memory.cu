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
#include "ndtypes.h"


/*****************************************************************************/
/*                  Expose some cuda memory functions                        */
/*****************************************************************************/

extern "C" void *
xnd_cuda_calloc_managed(uint16_t align, int64_t size, ndt_context_t *ctx)
{
    char *ptr;
    cudaError_t err;

    if (size < 0 || (uint64_t)size > SIZE_MAX) {
        ndt_err_format(ctx, NDT_ValueError,
            "cudaMallocManaged: invalid size");
        return NULL;
    }

    err = cudaMallocManaged(&ptr, (size_t)size);
    if (err != cudaSuccess) {
        ndt_err_format(ctx, NDT_MemoryError,
            "cudaMallocManaged: allocation failed");
        return NULL;
    }

    if (((uintptr_t)ptr) % align != 0) {
        ndt_err_format(ctx, NDT_ValueError,
            "cudaMallocManaged: alignment requirement too large");
        cudaFree(ptr);
        return NULL;
    }

    memset(ptr, '\0', (size_t)size);

    return ptr;
}

extern "C" void
xnd_cuda_free(void *ptr)
{
    cudaError_t err;

    err = cudaFree(ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree: unexpected failure\n");
    }
}

extern "C" int
xnd_cuda_mem_prefetch_async(const void *ptr, int64_t count, int dev,
                            ndt_context_t *ctx)
{
    cudaError_t err;

    if (count < 0 || (uint64_t)count > SIZE_MAX) {
        ndt_err_format(ctx, NDT_ValueError,
            "cudaMemPrefetchAsync: invalid count");
        return -1;
    }

    err = cudaMemPrefetchAsync(&ptr, (size_t)count, dev);
    if (err != cudaSuccess) {
        ndt_err_format(ctx, NDT_MemoryError,
            "cudaMemPrefetchAsync: prefetching failed");
        return -1;
    }

    return 0;
}

extern "C" int
xnd_cuda_device_synchronize(ndt_context_t *ctx)
{
    cudaError_t err;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        ndt_err_format(ctx, NDT_RuntimeError,
            "cuda device synchronization failed");
        return -1;
    }

    return 0;
}

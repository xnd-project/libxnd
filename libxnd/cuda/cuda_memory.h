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


#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include <stdint.h>
#include "ndtypes.h"


/*****************************************************************************/
/*                  Expose some cuda memory functions                        */
/*****************************************************************************/

#ifdef __cpplusplus
extern "C" void *xnd_cuda_calloc_managed(uint16_t align, int64_t size, ndt_context_t *ctx);
extern "C" void xnd_cuda_free(void *ptr);
extern "C" int xnd_cuda_mem_prefetch_async(const void *ptr, int64_t count, int dev, ndt_context_t *ctx);
extern "C" int xnd_cuda_device_synchronize(ndt_context_t *ctx);
#else
void *xnd_cuda_calloc_managed(uint16_t align, int64_t size, ndt_context_t *ctx);
void xnd_cuda_free(void *ptr);
int xnd_cuda_mem_prefetch_async(const void *ptr, int64_t count, int dev, ndt_context_t *ctx);
int xnd_cuda_device_synchronize(ndt_context_t *ctx);
#endif


#endif /* CUDA_MEMORY_H */

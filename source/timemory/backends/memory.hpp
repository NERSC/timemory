// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "timemory/backends/device.hpp"
#include "timemory/macros/os.hpp"

#if !defined(TIMEMORY_WINDOWS)
#    define RESTRICT(TYPE) TYPE __restrict__
#else
#    define RESTRICT(TYPE) TYPE
#endif

#if defined(__INTEL_COMPILER)
#    define ASSUME_ALIGNED_MEMORY(ARRAY, WIDTH) __assume_aligned(ARRAY, WIDTH)
#elif defined(__xlC__)
#    define ASSUME_ALIGNED_MEMORY(ARRAY, WIDTH) __alignx(WIDTH, ARRAY)
#else
#    define ASSUME_ALIGNED_MEMORY(ARRAY, WIDTH)
#endif

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>

// provide allocation functions
#if defined(TIMEMORY_LINUX) || defined(TIMEMORY_WINDOWS)
#    include <malloc.h>
#elif defined(TIMEMORY_MACOS)
#    include <malloc/malloc.h>
#endif

namespace tim
{
namespace memory
{
//--------------------------------------------------------------------------------------//
//  aligned allocation, alignment should be specified in bits
//
namespace hidden
{
template <typename Tp, typename DeviceT>
Tp*
allocate_aligned(std::size_t size, std::size_t alignment,
                 std::enable_if_t<std::is_same<DeviceT, device::cpu>::value, int> = 0)
{
#if defined(_ISOC11_SOURCE)
    return static_cast<Tp*>(aligned_alloc(alignment, size * sizeof(Tp)));
#elif defined(TIMEMORY_MACOS) ||                                                         \
    (defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L))
    void* ptr = nullptr;
    auto  ret = posix_memalign(&ptr, alignment, size * sizeof(Tp));
    (void) ret;
    return static_cast<Tp*>(ptr);
#elif defined(TIMEMORY_WINDOWS)
    return static_cast<Tp*>(_aligned_malloc(size * sizeof(Tp), alignment));
#else
    return new Tp[size];
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename DeviceT>
Tp*
allocate_aligned(std::size_t size, std::size_t,
                 std::enable_if_t<std::is_same<DeviceT, device::gpu>::value, int> = 0)
{
    return cuda::malloc<Tp>(size);
}

//--------------------------------------------------------------------------------------//
//  free aligned array, should be used in conjunction with data allocated with above
//
template <typename Tp, typename DeviceT>
void
free_aligned(Tp* ptr,
             std::enable_if_t<std::is_same<DeviceT, device::cpu>::value, int> = 0)
{
#if defined(_ISOC11_SOURCE) || defined(TIMEMORY_MACOS) || (_POSIX_C_SOURCE >= 200112L)
    free(static_cast<void*>(ptr));
#elif defined(TIMEMORY_WINDOWS)
    _aligned_free(ptr);
#else
    delete[] ptr;
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename DeviceT>
void
free_aligned(Tp* ptr,
             std::enable_if_t<std::is_same<DeviceT, device::gpu>::value, int> = 0)
{
    cuda::free(ptr);
}

}  // namespace hidden

//--------------------------------------------------------------------------------------//
//  aligned allocation, alignment should be specified in bits
//
template <typename Tp, typename DeviceT = device::cpu>
Tp*
allocate_aligned(std::size_t size, std::size_t alignment)
{
    return hidden::allocate_aligned<Tp, DeviceT>(size, alignment);
}

//--------------------------------------------------------------------------------------//
//  free aligned array, should be used in conjunction with data allocated with above
//
template <typename Tp, typename DeviceT = device::cpu>
void
free_aligned(Tp* ptr)
{
    hidden::free_aligned<Tp, DeviceT>(ptr);
}

}  // namespace memory
}  // namespace tim

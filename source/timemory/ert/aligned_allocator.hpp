// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#if defined(_WIN32) || defined(_WIN64)
#    if !defined(_WINDOWS)
#        define _WINDOWS
#    endif
#elif defined(__APPLE__) || defined(__MACH__)
#    if !defined(_MACOS)
#        define _MACOS
#    endif
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#    if !defined(_LINUX)
#        define _LINUX
#    endif
#endif

#if !defined(_WINDOWS)
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
#if defined(_LINUX) || defined(_WINDOWS)
#    include <malloc.h>
#elif defined(_MACOS)
#    include <malloc/malloc.h>
#endif

#include "timemory/backends/device.hpp"
#include "timemory/bits/settings.hpp"

namespace tim
{
namespace ert
{
namespace details
{
// variadic template function for eliminating unused-* warnings
template <typename... _Args>
void
consume_parameters(_Args&&...)
{}
}  // namespace details

template <bool B, typename T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

//--------------------------------------------------------------------------------------//
//  aligned allocation, alignment should be specified in bits
//
namespace hidden
{
template <typename _Tp, typename _Device,
          enable_if_t<std::is_same<_Device, device::cpu>::value> = 0>
_Tp*
allocate_aligned(std::size_t size, std::size_t alignment)
{
#if defined(__INTEL_COMPILER)
    return static_cast<_Tp*>(_mm_malloc(size * sizeof(_Tp), alignment));
#elif defined(_ISOC11_SOURCE)
    return static_cast<_Tp*>(aligned_alloc(alignment, size * sizeof(_Tp)));
#elif defined(_MACOS) || (_POSIX_C_SOURCE >= 200112L)
    void* ptr = nullptr;
    auto  ret = posix_memalign(&ptr, alignment, size * sizeof(_Tp));
    details::consume_parameters(ret);
    return static_cast<_Tp*>(ptr);
#elif defined(_WINDOWS)
    return static_cast<_Tp*>(_aligned_malloc(size * sizeof(_Tp), alignment));
#else
    return new _Tp[size];
#endif
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Device,
          enable_if_t<std::is_same<_Device, device::gpu>::value> = 0>
_Tp*
allocate_aligned(std::size_t size, std::size_t)
{
    return cuda::malloc<_Tp>(size);
}

//--------------------------------------------------------------------------------------//
//  free aligned array, should be used in conjunction with data allocated with above
//
template <typename _Tp, typename _Device,
          enable_if_t<std::is_same<_Device, device::cpu>::value> = 0>
void
free_aligned(_Tp* ptr)
{
#if defined(__INTEL_COMPILER)
    _mm_free(static_cast<void*>(ptr));
#elif defined(_ISOC11_SOURCE) || defined(_MACOS) || (_POSIX_C_SOURCE >= 200112L) ||      \
    defined(_WINDOWS)
    free(static_cast<void*>(ptr));
#else
    delete[] ptr;
#endif
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Device,
          enable_if_t<std::is_same<_Device, device::gpu>::value> = 0>
void
free_aligned(_Tp* ptr)
{
    cuda::free(ptr);
}

}  // namespace hidden

//--------------------------------------------------------------------------------------//
//  aligned allocation, alignment should be specified in bits
//
template <typename _Tp, typename _Device = device::cpu>
_Tp*
allocate_aligned(std::size_t size, std::size_t alignment)
{
    return hidden::allocate_aligned<_Tp, _Device>(size, alignment);
}

//--------------------------------------------------------------------------------------//
//  free aligned array, should be used in conjunction with data allocated with above
//
template <typename _Tp, typename _Device = device::cpu>
void
free_aligned(_Tp* ptr)
{
    hidden::free_aligned<_Tp, _Device>(ptr);
}

//--------------------------------------------------------------------------------------//
//  allocator that can be used with STL containers
//
//  default template parameter _Align_v uses (8 * sizeof(_Tp)) because alignment should
//  be specified in bits and sizeof(_Tp) return bytes so for float (sizeof == 4), this
//  would be 32-bit alignment and for double (sizeof == 8), this would be 64-bit alignment
//
template <typename _Tp, std::size_t _Align_v = 8 * sizeof(_Tp)>
class aligned_allocator
{
public:
    // The following will be the same for virtually all allocators.
    using value_type      = _Tp;
    using pointer         = _Tp*;
    using reference       = _Tp&;
    using const_pointer   = const _Tp*;
    using const_reference = const _Tp&;
    using size_type       = std::size_t;
    using difference_type = ptrdiff_t;

public:
    // constructors and destructors
    aligned_allocator() {}
    aligned_allocator(const aligned_allocator&) {}
    aligned_allocator(aligned_allocator&&) {}
    template <typename U>
    aligned_allocator(const aligned_allocator<U, _Align_v>&)
    {}
    ~aligned_allocator() {}

public:
    // operators
    aligned_allocator& operator=(const aligned_allocator&) = delete;
    aligned_allocator& operator==(aligned_allocator&&)     = delete;
    bool operator!=(const aligned_allocator& other) const { return !(*this == other); }
    bool operator==(const aligned_allocator&) const { return true; }

public:
    _Tp*       address(_Tp& r) const { return &r; }
    const _Tp* address(const _Tp& s) const { return &s; }

    std::size_t max_size() const
    {
        // avoid signed/unsigned warnings independent of size_t definition
        return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(_Tp);
    }

    // The following must be the same for all allocators.
    template <typename U>
    struct rebind
    {
        typedef aligned_allocator<U, _Align_v> other;
    };

    void construct(_Tp* const p, const _Tp& t) const
    {
        void* const pv = static_cast<void*>(p);
        new(pv) _Tp(t);
    }

    template <typename... _Args>
    void construct(_Tp* const p, _Args&&... args) const
    {
        ::new((void*) p) _Tp(std::forward<_Args>(args)...);
    }

    void destroy(_Tp* const p) const { p->~_Tp(); }

    _Tp* allocate(const std::size_t n) const
    {
        if(n == 0)
            return nullptr;

        // integer overflow check that throws std::length_error in case of overflow
        if(n > max_size())
        {
            throw std::length_error(
                "aligned_allocator<_Tp>::allocate() - Integer overflow.");
        }

        // Mallocator wraps malloc().
        void* const ptr = allocate_aligned<_Tp, device::cpu>(n, _Align_v);

        // throw std::bad_alloc in the case of memory allocation failure.
        if(ptr == nullptr)
        {
            std::cerr << "Allocation of type " << typeid(_Tp).name() << " of size " << n
                      << " and alignment " << _Align_v << " failed. ptr = " << ptr
                      << std::endl;
            throw std::bad_alloc();
        }

        return static_cast<_Tp*>(ptr);
    }

    void deallocate(_Tp* const ptr, const std::size_t) const
    {
        free_aligned<_Tp, device::cpu>(ptr);
    }

    // same for all allocators that ignore hints.
    template <typename U>
    _Tp* allocate(const std::size_t n, const U* /* const hint */) const
    {
        return allocate(n);
    }

    // returns the alignment in bytes
    std::size_t get_alignment() { return _Align_v / 8; }
};

}  // namespace ert
}  // namespace tim

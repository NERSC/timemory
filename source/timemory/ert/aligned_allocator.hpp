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

/** \file timemory/ert/aligned_allocator.hpp
 * \headerfile timemory/ert/aligned_allocator.hpp "timemory/ert/aligned_allocator.hpp"
 * Provides an aligned allocator
 *
 */

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
#include "timemory/settings/declaration.hpp"

namespace tim
{
namespace ert
{
namespace details
{
// variadic template function for eliminating unused-* warnings
template <typename... ArgsT>
void
consume_parameters(ArgsT&&...)
{}
}  // namespace details

template <bool B, typename T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

//--------------------------------------------------------------------------------------//
//  aligned allocation, alignment should be specified in bits
//
namespace hidden
{
template <typename Tp, typename DeviceT,
          enable_if_t<std::is_same<DeviceT, device::cpu>::value> = 0>
Tp*
allocate_aligned(std::size_t size, std::size_t alignment)
{
#if defined(_ISOC11_SOURCE)
    return static_cast<Tp*>(aligned_alloc(alignment, size * sizeof(Tp)));
#elif defined(_MACOS) || (_POSIX_C_SOURCE >= 200112L)
    void* ptr = nullptr;
    auto  ret = posix_memalign(&ptr, alignment, size * sizeof(Tp));
    details::consume_parameters(ret);
    return static_cast<Tp*>(ptr);
#elif defined(_WINDOWS)
    return static_cast<Tp*>(_aligned_malloc(size * sizeof(Tp), alignment));
#else
    return new Tp[size];
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename DeviceT,
          enable_if_t<std::is_same<DeviceT, device::gpu>::value> = 0>
Tp*
allocate_aligned(std::size_t size, std::size_t)
{
    return cuda::malloc<Tp>(size);
}

//--------------------------------------------------------------------------------------//
//  free aligned array, should be used in conjunction with data allocated with above
//
template <typename Tp, typename DeviceT,
          enable_if_t<std::is_same<DeviceT, device::cpu>::value> = 0>
void
free_aligned(Tp* ptr)
{
#if defined(_ISOC11_SOURCE) || defined(_MACOS) || (_POSIX_C_SOURCE >= 200112L) ||        \
    defined(_WINDOWS)
    free(static_cast<void*>(ptr));
#else
    delete[] ptr;
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename DeviceT,
          enable_if_t<std::is_same<DeviceT, device::gpu>::value> = 0>
void
free_aligned(Tp* ptr)
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

//--------------------------------------------------------------------------------------//
//  allocator that can be used with STL containers
//
//  default template parameter _Align_v uses (8 * sizeof(Tp)) because alignment should
//  be specified in bits and sizeof(Tp) return bytes so for float (sizeof == 4), this
//  would be 32-bit alignment and for double (sizeof == 8), this would be 64-bit alignment
//
template <typename Tp, std::size_t _Align_v = 8 * sizeof(Tp)>
class aligned_allocator
{
public:
    // The following will be the same for virtually all allocators.
    using value_type      = Tp;
    using pointer         = Tp*;
    using reference       = Tp&;
    using const_pointer   = const Tp*;
    using const_reference = const Tp&;
    using size_type       = std::size_t;
    using difference_type = ptrdiff_t;

public:
    // constructors and destructors
    aligned_allocator() {}
    aligned_allocator(const aligned_allocator&) {}
    aligned_allocator(aligned_allocator&&) noexcept {}
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
    Tp*       address(Tp& r) const { return &r; }
    const Tp* address(const Tp& s) const { return &s; }

    std::size_t max_size() const
    {
        // avoid signed/unsigned warnings independent of size_t definition
        return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(Tp);
    }

    // The following must be the same for all allocators.
    template <typename U>
    struct rebind
    {
        typedef aligned_allocator<U, _Align_v> other;
    };

    void construct(Tp* const p, const Tp& t) const
    {
        void* const pv = static_cast<void*>(p);
        new(pv) Tp(t);
    }

    template <typename... ArgsT>
    void construct(Tp* const p, ArgsT&&... args) const
    {
        ::new((void*) p) Tp(std::forward<ArgsT>(args)...);
    }

    void destroy(Tp* const p) const { p->~Tp(); }

    Tp* allocate(const std::size_t n) const
    {
        if(n == 0)
            return nullptr;

        // integer overflow check that throws std::length_error in case of overflow
        if(n > max_size())
        {
            throw std::length_error(
                "aligned_allocator<Tp>::allocate() - Integer overflow.");
        }

        // Mallocator wraps malloc().
        void* const ptr = allocate_aligned<Tp, device::cpu>(n, _Align_v);

        // throw std::bad_alloc in the case of memory allocation failure.
        if(ptr == nullptr)
        {
            std::cerr << "Allocation of type " << typeid(Tp).name() << " of size " << n
                      << " and alignment " << _Align_v << " failed. ptr = " << ptr
                      << std::endl;
            throw std::bad_alloc();
        }

        return static_cast<Tp*>(ptr);
    }

    void deallocate(Tp* const ptr, const std::size_t) const
    {
        free_aligned<Tp, device::cpu>(ptr);
    }

    // same for all allocators that ignore hints.
    template <typename U>
    Tp* allocate(const std::size_t n, const U* /* const hint */) const
    {
        return allocate(n);
    }

    // returns the alignment in bytes
    std::size_t get_alignment() { return _Align_v / 8; }
};

}  // namespace ert
}  // namespace tim

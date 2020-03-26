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

/**
 * \file timemory/manager/macros.hpp
 * \brief Include the macros for manager
 */

#pragma once

//======================================================================================//
//
// Define macros for manager
//
//======================================================================================//
//
#if !defined(__library_ctor__)
#    if !defined(_WINDOWS)
#        define __library_ctor__ __attribute__((constructor))
#    else
#        define __library_ctor__ static
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(__library_dtor__)
#    if !defined(_WINDOWS)
#        define __library_dtor__ __attribute__((destructor))
#    else
#        define __library_dtor__
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_MANAGER_SOURCE)
//
#    define TIMEMORY_MANAGER_LINKAGE(...) __VA_ARGS__
//
#elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_MANAGER_EXTERN)
//
#    define TIMEMORY_MANAGER_LINKAGE(...) extern __VA_ARGS__
//
#else
//
#    define TIMEMORY_MANAGER_LINKAGE(...) inline __VA_ARGS__
//
#endif
//
//--------------------------------------------------------------------------------------//
//
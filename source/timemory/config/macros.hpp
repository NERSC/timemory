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
 * \file timemory/config/macros.hpp
 * \brief Include the macros for config
 */

#pragma once

#include "timemory/dll.hpp"

//======================================================================================//
//
//                              Define macros for config
//
//======================================================================================//
//
#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_CONFIG_EXTERN)
#    define TIMEMORY_USE_CONFIG_EXTERN
#endif
//
#if defined(TIMEMORY_CONFIG_SOURCE)
#    define TIMEMORY_CONFIG_LINKAGE(...) __VA_ARGS__
#elif defined(TIMEMORY_USE_CONFIG_EXTERN)
#    define TIMEMORY_CONFIG_LINKAGE(...) extern __VA_ARGS__
#else
#    define TIMEMORY_CONFIG_LINKAGE(...) inline __VA_ARGS__
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_INIT)
#    define TIMEMORY_INIT(...) ::tim::timemory_init(__VA_ARGS__)
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_FINALIZE)
#    define TIMEMORY_FINALIZE() ::tim::timemory_finalize()
#endif
//
//--------------------------------------------------------------------------------------//

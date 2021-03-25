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

#include "timemory/macros/os.hpp"

//======================================================================================//
//
//      Compiler
//
//======================================================================================//

//  clang compiler
#if defined(__clang__) && !defined(_TIMEMORY_CLANG)
#    define _TIMEMORY_CLANG 1
#endif

//--------------------------------------------------------------------------------------//

//  nvcc compiler
#if defined(__NVCC__) && !defined(_TIMEMORY_NVCC)
#    define _TIMEMORY_NVCC 1
#endif

//--------------------------------------------------------------------------------------//

//  nvcc compiler
#if defined(__CUDACC__) && !defined(_TIMEMORY_CUDACC)
#    define _TIMEMORY_CUDACC 1
#endif

//--------------------------------------------------------------------------------------//

//  assume openmp target is enabled
// #if defined(_TIMEMORY_CUDACC) && defined(_OPENMP) && !defined(_TIMEMORY_OPENMP_TARGET)
// #    define _TIMEMORY_OPENMP_TARGET 1
// #endif

//--------------------------------------------------------------------------------------//

//  Intel compiler
#if defined(__INTEL_COMPILER) && !defined(_TIMEMORY_INTEL)
#    define _TIMEMORY_INTEL 1
#    if __INTEL_COMPILER < 1500
#        warning "Intel compilers < 1500 have been known to have compiler errors"
#    endif
#endif

//--------------------------------------------------------------------------------------//

// GNU compiler
#if defined(__GNUC__) && !defined(_TIMEMORY_CLANG) && !defined(_TIMEMORY_GNU)
#    if(__GNUC__ <= 4 && __GNUC_MINOR__ < 9)
#        warning "GCC compilers < 4.9 have been known to have compiler errors"
#        define _TIMEMORY_GNU 1
#    elif(__GNUC__ >= 4 && __GNUC_MINOR__ >= 9) || __GNUC__ >= 5
#        define _TIMEMORY_GNU
#    endif
#endif

//--------------------------------------------------------------------------------------//

//  MSVC compiler
#if defined(_MSC_VER) && _MSC_VER > 0 && !defined(_TIMEMORY_MSVC)
#    define _TIMEMORY_MSVC 1
#endif

//======================================================================================//
//
//      Demangling
//
//======================================================================================//

#if(defined(_TIMEMORY_GNU) || defined(_TIMEMORY_CLANG) || defined(_TIMEMORY_INTEL) ||    \
    defined(_TIMEMORY_NVCC)) &&                                                          \
    defined(TIMEMORY_UNIX)
#    if !defined(TIMEMORY_ENABLE_DEMANGLE)
#        define TIMEMORY_ENABLE_DEMANGLE 1
#    endif
#endif

//======================================================================================//
//
//      WINDOWS WARNINGS
//
//======================================================================================//

#if defined(_TIMEMORY_MSVC) && !defined(TIMEMORY_MSVC_WARNINGS)

#    pragma warning(disable : 4003)   // not enough actual params
#    pragma warning(disable : 4061)   // enum in switch of enum not explicitly handled
#    pragma warning(disable : 4068)   // unknown pragma
#    pragma warning(disable : 4127)   // conditional expr is constant
#    pragma warning(disable : 4129)   // unrecognized char escape
#    pragma warning(disable : 4146)   // unsigned
#    pragma warning(disable : 4217)   // locally defined symbol
#    pragma warning(disable : 4244)   // possible loss of data
#    pragma warning(disable : 4251)   // needs to have dll-interface to be used
#    pragma warning(disable : 4242)   // possible loss of data (assignment)
#    pragma warning(disable : 4244)   // conversion from 'double' to 'LONGLONG'
#    pragma warning(disable : 4245)   // signed/unsigned mismatch (init conversion)
#    pragma warning(disable : 4267)   // possible loss of data (implicit conversion)
#    pragma warning(disable : 4305)   // truncation from 'double' to 'float'
#    pragma warning(disable : 4355)   // this used in base member init list
#    pragma warning(disable : 4365)   // signed/unsigned mismatch
#    pragma warning(disable : 4464)   // relative include path contains '..'
#    pragma warning(disable : 4522)   // multiple assignment operators specified
#    pragma warning(disable : 4625)   // copy ctor implicitly defined as deleted
#    pragma warning(disable : 4626)   // copy assign implicitly defined as deleted
#    pragma warning(disable : 4661)   // no suitable definition for template inst
#    pragma warning(disable : 4700)   // uninitialized local variable used
#    pragma warning(disable : 4710)   // function not inlined
#    pragma warning(disable : 4711)   // function selected for auto inline expansion
#    pragma warning(disable : 4786)   // ID truncated to '255' char in debug info
#    pragma warning(disable : 4820)   // bytes padded after data member
#    pragma warning(disable : 4834)   // discarding return value with 'nodiscard'
#    pragma warning(disable : 4996)   // function may be unsafe
#    pragma warning(disable : 5219)   // implicit conversion; possible loss of data
#    pragma warning(disable : 5026)   // move ctor implicitly defined as deleted
#    pragma warning(disable : 5027)   // move assign implicitly defined as deleted
#    pragma warning(disable : 26495)  // Always initialize member variable

#endif

//======================================================================================//
//
//      DISABLE WINDOWS MIN/MAX macros
//
//======================================================================================//

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)

#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif

#endif

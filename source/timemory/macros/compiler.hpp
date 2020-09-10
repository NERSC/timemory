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
#if defined(__clang__)
#    define _TIMEMORY_CLANG
#endif

//--------------------------------------------------------------------------------------//

//  nvcc compiler
#if defined(__NVCC__)
#    define _TIMEMORY_NVCC
#endif

//--------------------------------------------------------------------------------------//

//  Intel compiler
#if defined(__INTEL_COMPILER)
#    define _TIMEMORY_INTEL
#    if __INTEL_COMPILER < 1500
#        warning "Intel compilers < 1500 have been known to have compiler errors"
#    endif
#endif

//--------------------------------------------------------------------------------------//

// GNU compiler
#if defined(__GNUC__) && !defined(_TIMEMORY_CLANG)
#    if(__GNUC__ <= 4 && __GNUC_MINOR__ < 9)
#        warning "GCC compilers < 4.9 have been known to have compiler errors"
#    elif(__GNUC__ >= 4 && __GNUC_MINOR__ >= 9) || __GNUC__ >= 5
#        define _TIMEMORY_GNU
#    endif
#endif

//======================================================================================//
//
//      Demangling
//
//======================================================================================//

#if(defined(_TIMEMORY_GNU) || defined(_TIMEMORY_CLANG) || defined(_TIMEMORY_INTEL) ||    \
    defined(_TIMEMORY_NVCC)) &&                                                          \
    defined(_UNIX)
#    if !defined(TIMEMORY_ENABLE_DEMANGLE)
#        define TIMEMORY_ENABLE_DEMANGLE 1
#    endif
#endif

//======================================================================================//
//
//      WINDOWS WARNINGS
//
//======================================================================================//

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)

#    pragma warning(disable : 4003)   // not enough actual params
#    pragma warning(disable : 4068)   // unknown pragma
#    pragma warning(disable : 4129)   // unrecognized char escape
#    pragma warning(disable : 4146)   // unsigned
#    pragma warning(disable : 4217)   // locally defined symbol
#    pragma warning(disable : 4244)   // possible loss of data
#    pragma warning(disable : 4251)   // needs to have dll-interface to be used
#    pragma warning(disable : 4267)   // possible loss of data
#    pragma warning(disable : 4305)   // truncation from 'double' to 'float'
#    pragma warning(disable : 4522)   // multiple assignment operators specified
#    pragma warning(disable : 4661)   // no suitable definition for template inst
#    pragma warning(disable : 4700)   // uninitialized local variable used
#    pragma warning(disable : 4786)   // ID truncated to '255' char in debug info
#    pragma warning(disable : 4996)   // function may be unsafe
#    pragma warning(disable : 26495)  // Always initialize member variable (cereal issue)

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

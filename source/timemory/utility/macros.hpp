//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.
//

/** \file macros.hpp
 * \headerfile macros.hpp "timemory/utility/macros.hpp"
 * Useful macros for:
 *   - Operating system
 *   - Language
 *   - Compiler
 *   - Windows-specific macros
 */

#pragma once

#include "timemory/dll.hpp"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>

//======================================================================================//
//
//      Operating System
//
//======================================================================================//

// machine bits
#if defined(__x86_64__)
#    if !defined(_64BIT)
#        define _64BIT
#    endif
#else
#    if !defined(_32BIT)
#        define _32BIT
#    endif
#endif

//--------------------------------------------------------------------------------------//
// base operating system

#if defined(_WIN32) || defined(_WIN64)
#    if !defined(_WINDOWS)
#        define _WINDOWS
#    endif

//--------------------------------------------------------------------------------------//

#elif defined(__APPLE__) || defined(__MACH__)
#    if !defined(_MACOS)
#        define _MACOS
#    endif
#    if !defined(_UNIX)
#        define _UNIX
#    endif

//--------------------------------------------------------------------------------------//

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#    if !defined(_LINUX)
#        define _LINUX
#    endif
#    if !defined(_UNIX)
#        define _UNIX
#    endif

//--------------------------------------------------------------------------------------//

#elif defined(__unix__) || defined(__unix) || defined(unix)
#    if !defined(_UNIX)
#        define _UNIX
#    endif
#endif

//======================================================================================//
//
//      LANGUAGE
//
//======================================================================================//

// Define C++14
#ifndef CXX14
#    if __cplusplus > 201103L  // C++14
#        define CXX14
#    endif
#endif

//--------------------------------------------------------------------------------------//

// Define C++17
#ifndef CXX17
#    if __cplusplus > 201402L  // C++17
#        define CXX17
#    endif
#endif

//--------------------------------------------------------------------------------------//

#if !defined(CXX14)
#    if !defined(_WINDOWS)
#        error "timemory requires __cplusplus > 201103L (C++14)"
#    endif
#endif

//--------------------------------------------------------------------------------------//

#if !defined(IF_CONSTEXPR)
#    if defined(CXX17)
#        define IF_CONSTEXPR(...) if constexpr(__VA_ARGS__)
#    else
#        define IF_CONSTEXPR(...) if(__VA_ARGS__)
#    endif
#endif

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

#if defined(_WINDOWS)
#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
#    include <Windows.h>
#endif

//======================================================================================//
//
//      Quick way to create a globally accessible setting
//
//======================================================================================//

#if !defined(CREATE_STATIC_VARIABLE_ACCESSOR)
#    define CREATE_STATIC_VARIABLE_ACCESSOR(TYPE, FUNC_NAME, VARIABLE)                   \
        static TYPE& FUNC_NAME()                                                         \
        {                                                                                \
            static TYPE _instance = Type::VARIABLE;                                      \
            return _instance;                                                            \
        }
#endif

//--------------------------------------------------------------------------------------//

#if !defined(CREATE_STATIC_FUNCTION_ACCESSOR)
#    define CREATE_STATIC_FUNCTION_ACCESSOR(TYPE, FUNC_NAME, VARIABLE)                   \
        static TYPE& FUNC_NAME()                                                         \
        {                                                                                \
            static TYPE _instance = Type::VARIABLE();                                    \
            return _instance;                                                            \
        }
#endif

//======================================================================================//
//
//      DEBUG
//
//======================================================================================//

#if !defined(PRINT_HERE)
#    define PRINT_HERE(fmt, ...)                                                         \
        fprintf(stderr, "> [%s@'%s':%i] " fmt "...\n", __FUNCTION__, __FILE__, __LINE__, \
                __VA_ARGS__)
#endif

#if !defined(DEBUG_PRINT_HERE)
#    if defined(DEBUG)
#        define DEBUG_PRINT_HERE(fmt, ...)                                               \
            if(::tim::settings::debug())                                                 \
            {                                                                            \
                fprintf(stderr, "> [%s@'%s':%i] " fmt "...\n", __FUNCTION__, __FILE__,   \
                        __LINE__, __VA_ARGS__);                                          \
            }
#    else
#        define DEBUG_PRINT_HERE(fmt, ...)
#    endif
#endif

#if !defined(VERBOSE_PRINT_HERE)
#    define VERBOSE_PRINT_HERE(VERBOSE_LEVEL, fmt, ...)                                  \
        if(::tim::settings::verbose() >= VERBOSE_LEVEL)                                  \
        {                                                                                \
            fprintf(stderr, "> [%s@'%s':%i] " fmt "...\n", __FUNCTION__, __FILE__,       \
                    __LINE__, __VA_ARGS__);                                              \
        }
#endif

#if !defined(PRETTY_PRINT_HERE)
#    if defined(_TIMEMORY_GNU) || defined(_TIMEMORY_CLANG)
#        define PRETTY_PRINT_HERE(fmt, ...)                                              \
            fprintf(stderr, "> [%s@'%s':%i] " fmt "...\n", __PRETTY_FUNCTION__,          \
                    __FILE__, __LINE__, __VA_ARGS__)
#    else
#        define PRETTY_PRINT_HERE(fmt, ...)                                              \
            fprintf(stderr, "> [%s@'%s':%i] " fmt "...\n", __FUNCTION__, __FILE__,       \
                    __LINE__, __VA_ARGS__)
#    endif
#endif

#if defined(DEBUG)

template <typename... Args>
inline void
__LOG(std::string file, int line, const char* msg, Args&&... args)
{
    if(file.find("/") != std::string::npos)
        file = file.substr(file.find_last_of("/"));
    fprintf(stderr, "[Log @ %s:%i]> ", file.c_str(), line);
    fprintf(stderr, msg, std::forward<Args>(args)...);
    fprintf(stderr, "\n");
}

//--------------------------------------------------------------------------------------//

inline void
__LOG(std::string file, int line, const char* msg)
{
    if(file.find("/") != std::string::npos)
        file = file.substr(file.find_last_of("/"));
    fprintf(stderr, "[Log @ %s:%i]> %s\n", file.c_str(), line, msg);
}

//--------------------------------------------------------------------------------------//
// auto insert the file and line
#    define _LOG(...) __LOG(__FILE__, __LINE__, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

template <typename... Args>
inline void
_DBG(const char* msg, Args&&... args)
{
    fprintf(stderr, msg, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//

inline void
_DBG(const char* msg)
{
    fprintf(stderr, "%s", msg);
}

#else
#    define _LOG(...)                                                                    \
        {}
#    define _DBG(...)                                                                    \
        {}
#endif

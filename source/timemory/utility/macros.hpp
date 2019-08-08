//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
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

// Define C++11
#ifndef CXX11
#    if __cplusplus > 199711L  // C++11
#        define CXX11
#    endif
#endif

//--------------------------------------------------------------------------------------//

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

#if(defined(_TIMEMORY_GNU) || defined(_TIMEMORY_CLANG) || defined(_TIMEMORY_INTEL)) &&   \
    defined(_UNIX)
#    if !defined(_TIMEMORY_ENABLE_DEMANGLE)
#        define _TIMEMORY_ENABLE_DEMANGLE 1
#    endif
#endif

//======================================================================================//
//
//      GLOBAL LINKING
//
//======================================================================================//

// Define macros for WIN32 for importing/exporting external symbols to DLLs
#if defined(_WINDOWS) && !defined(_TIMEMORY_ARCHIVE)
#    if defined(_TIMEMORY_DLL)
#        define tim_api __declspec(dllexport)
#        define tim_api_static static __declspec(dllexport)
#    else
#        define tim_api __declspec(dllimport)
#        define tim_api_static static __declspec(dllimport)
#    endif
#else
#    define tim_api
#    define tim_api_static static
#endif

//======================================================================================//
//
//      WINDOWS WARNINGS
//
//======================================================================================//

#if defined(_WINDOWS)
#    pragma warning(disable : 4786)  // ID truncated to '255' char in debug info
#    pragma warning(disable : 4068)  // unknown pragma
#    pragma warning(disable : 4003)  // not enough actual params
#    pragma warning(disable : 4244)  // possible loss of data
#    pragma warning(disable : 4146)  // unsigned
#    pragma warning(disable : 4129)  // unrecognized char escape
#    pragma warning(disable : 4996)  // function may be unsafe
#    pragma warning(disable : 4267)  // possible loss of data
#    pragma warning(disable : 4700)  // uninitialized local variable used
#    pragma warning(disable : 4217)  // locally defined symbol
#    pragma warning(disable : 4251)  // needs to have dll-interface to be used

#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
#endif

//======================================================================================//
//
//      EXTERN TEMPLATE DECLARE AND INSTANTIATE
//
//======================================================================================//

#define TIMEMORY_DECLARE_EXTERN_TUPLE(...)                                               \
    extern template class tim::auto_tuple<__VA_ARGS__>;                                  \
    extern template class tim::component_tuple<__VA_ARGS__>;                             \
    extern template class tim::auto_list<__VA_ARGS__>;                                   \
    extern template class tim::component_list<__VA_ARGS__>;

#define TIMEMORY_INSTANTIATE_EXTERN_TUPLE(...)                                           \
    template class tim::auto_tuple<__VA_ARGS__>;                                         \
    template class tim::component_tuple<__VA_ARGS__>;                                    \
    template class tim::auto_list<__VA_ARGS__>;                                          \
    template class tim::component_list<__VA_ARGS__>;

#define TIMEMORY_EXTERN_STORAGE_TYPE(OBJ_TYPE) tim::storage<tim::component::OBJ_TYPE>

#define TIMEMORY_DECLARE_EXTERN_STORAGE(OBJ_TYPE)                                        \
    extern template tim::details::storage_singleton_t<TIMEMORY_EXTERN_STORAGE_TYPE(      \
        OBJ_TYPE)>&                                                                      \
    tim::get_storage_singleton<TIMEMORY_EXTERN_STORAGE_TYPE(OBJ_TYPE)>();

#define TIMEMORY_INSTANTIATE_EXTERN_STORAGE(OBJ_TYPE)                                    \
    template tim::details::storage_singleton_t<TIMEMORY_EXTERN_STORAGE_TYPE(OBJ_TYPE)>&  \
    tim::get_storage_singleton<TIMEMORY_EXTERN_STORAGE_TYPE(OBJ_TYPE)>();

/*
// Accept any number of args >= N, but expand to just the Nth one.
// Here, N == 6.
#define TIMEMORY_GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

// Define some macros to help us create overrides based on the
// arity of a for-each-style macro.
#define TIMEMORY_FE_0(_call, ...)
#define TIMEMORY_FE_1(_call, x) _call(x);
#define TIMEMORY_FE_2(_call, x, ...) _call(x) TIMEMORY_FE_1(_call, __VA_ARGS__)
#define TIMEMORY_FE_3(_call, x, ...) _call(x) TIMEMORY_FE_2(_call, __VA_ARGS__)
#define TIMEMORY_FE_4(_call, x, ...) _call(x) TIMEMORY_FE_3(_call, __VA_ARGS__)
#define TIMEMORY_FE_5(_call, x, ...) _call(x) TIMEMORY_FE_4(_call, __VA_ARGS__)
#define TIMEMORY_FE_6(_call, x, ...) _call(x) TIMEMORY_FE_5(_call, __VA_ARGS__)
#define TIMEMORY_FE_7(_call, x, ...) _call(x) TIMEMORY_FE_6(_call, __VA_ARGS__)
#define TIMEMORY_FE_8(_call, x, ...) _call(x) TIMEMORY_FE_7(_call, __VA_ARGS__)
#define TIMEMORY_FE_9(_call, x, ...) _call(x) TIMEMORY_FE_8(_call, __VA_ARGS__)
#define TIMEMORY_FE_10(_call, x, ...) _call(x) TIMEMORY_FE_9(_call, __VA_ARGS__)
*/
/*
 * Provide a for-each construct for variadic macros. Supports up
 * to 9 args.
 *
 * Example usage1:
 *     #define FWD_DECLARE_CLASS(cls) class cls;
 *     CALL_MACRO_X_FOR_EACH(FWD_DECLARE_CLASS, Foo, Bar)
 *
 * Example usage 2:
 *     #define START_NS(ns) namespace ns {
 *     #define END_NS(ns) }
 *     #define MY_NAMESPACES System, Net, Http
 *     CALL_MACRO_X_FOR_EACH(START_NS, MY_NAMESPACES)
 *     typedef foo int;
 *     CALL_MACRO_X_FOR_EACH(END_NS, MY_NAMESPACES)
 */
/*
#define CALL_MACRO_X_FOR_EACH(x, ...)                                                    \
    TIMEMORY_GET_NTH_ARG("ignored", ##__VA_ARGS__, TIMEMORY_FE_9, TIMEMORY_FE_8,         \
                         TIMEMORY_FE_7, TIMEMORY_FE_6, TIMEMORY_FE_5, TIMEMORY_FE_4,     \
                         TIMEMORY_FE_3, TIMEMORY_FE_2, TIMEMORY_FE_1, TIMEMORY_FE_0)     \
    (x, ##__VA_ARGS__)
*/

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
//      FLOATING POINT EXCEPTIONS
//
//======================================================================================//

#if !defined(_WINDOWS)
#    define init_priority(N) __attribute__((init_priority(N)))
#    define init_construct(N) __attribute__((constructor(N)))
#    define __c_ctor__ __attribute__((constructor))
#    define __c_dtor__ __attribute__((destructor))
#else
#    define init_priority(N)
#    define init_construct(N)
#    define __c_ctor__
#    define __c_dtor__
#endif

//======================================================================================//
//
//      DEBUG
//
//======================================================================================//

#if !defined(PRINT_HERE)
#    define PRINT_HERE(extra)                                                            \
        printf("> [%s@'%s':%i] %s...\n", __FUNCTION__, __FILE__, __LINE__, extra)
#endif

#if !defined(DEBUG_PRINT_HERE)
#    if defined(DEBUG)
#        define DEBUG_PRINT_HERE(extra)                                                  \
            printf("> [%s@'%s':%i] %s...\n", __FUNCTION__, __FILE__, __LINE__, extra)
#    else
#        define DEBUG_PRINT_HERE(extra)
#    endif
#endif

#if !defined(PRETTY_PRINT_HERE)
#    if defined(_TIMEMORY_GNU) || defined(_TIMEMORY_CLANG)
#        define PRETTY_PRINT_HERE(extra)                                                 \
            printf("> [%s@'%s':%i] %s...\n", __PRETTY_FUNCTION__, __FILE__, __LINE__,    \
                   extra)
#    else
#        define PRETTY_PRINT_HERE(extra)                                                 \
            printf("> [%s@'%s':%i] %s...\n", __FUNCTION__, __FILE__, __LINE__, extra)
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
        {                                                                                \
        }
#    define _DBG(...)                                                                    \
        {                                                                                \
        }
#endif

//======================================================================================//
//
//      GOOGLE PERF-TOOLS
//
//======================================================================================//

#if !defined(TIMEMORY_USE_GPERF)
#    if defined(TIMEMORY_USE_GPERF_HEAP_PROFILER)
#        include <gperftools/heap-profiler.h>
#    endif
#    if defined(TIMEMORY_USE_GPERF_CPU_PROFILER)
#        include <gperftools/profiler.h>
#    endif
#elif defined(TIMEMORY_USE_GPERF)
#    include <gperftools/heap-profiler.h>
#    include <gperftools/profiler.h>
#endif

namespace gperf
{
namespace cpu
{
#if defined(TIMEMORY_USE_GPERF_CPU_PROFILER)
//--------------------------------------------------------------------------------------//
inline int
profiler_start(const std::string& name)
{
    return ProfilerStart(name.c_str());
}
//--------------------------------------------------------------------------------------//
inline void
profiler_stop()
{
    ProfilerStop();
}
//--------------------------------------------------------------------------------------//
#else
//--------------------------------------------------------------------------------------//
inline int
profiler_start(const std::string&)
{
    return 0;
}
//--------------------------------------------------------------------------------------//
inline void
profiler_stop()
{
}
//--------------------------------------------------------------------------------------//
#endif

}  // namespace cpu

namespace heap
{
#if defined(TIMEMORY_USE_GPERF_HEAP_PROFILER)
//--------------------------------------------------------------------------------------//
inline int
profiler_start(const std::string& name)
{
    HeapProfilerStart(name.c_str());
    return 0;
}
//--------------------------------------------------------------------------------------//
inline void
profiler_stop()
{
    HeapProfilerStop();
}
//--------------------------------------------------------------------------------------//
#else
//--------------------------------------------------------------------------------------//
inline int
profiler_start(const std::string&)
{
    return 0;
}
//--------------------------------------------------------------------------------------//
inline void
profiler_stop()
{
}
//--------------------------------------------------------------------------------------//
#endif

}  // namespace heap

//--------------------------------------------------------------------------------------//

inline void
profiler_start(const std::string& name)
{
    int ret = cpu::profiler_start(name) + heap::profiler_start(name);
    if(ret != 0)
        std::cerr << "Profiler failed to start for \"" << name << "\"..." << std::endl;
}

//--------------------------------------------------------------------------------------//

inline void
profiler_stop()
{
    cpu::profiler_stop();
    heap::profiler_stop();
}

//--------------------------------------------------------------------------------------//
}  // namespace gperf

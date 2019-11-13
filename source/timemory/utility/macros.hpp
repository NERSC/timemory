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
#    else
#        if defined(_TIMEMORY_LINK_LIBRARY)
#            define tim_api __declspec(dllimport)
#        else
#            define tim_api
#        endif
#    endif
#else
#    define tim_api
#endif

//======================================================================================//
//
//      WINDOWS WARNINGS
//
//======================================================================================//

#if defined(_WINDOWS)
#    pragma warning(disable : 4786)   // ID truncated to '255' char in debug info
#    pragma warning(disable : 4068)   // unknown pragma
#    pragma warning(disable : 4003)   // not enough actual params
#    pragma warning(disable : 4244)   // possible loss of data
#    pragma warning(disable : 4146)   // unsigned
#    pragma warning(disable : 4129)   // unrecognized char escape
#    pragma warning(disable : 4996)   // function may be unsafe
#    pragma warning(disable : 4267)   // possible loss of data
#    pragma warning(disable : 4700)   // uninitialized local variable used
#    pragma warning(disable : 4217)   // locally defined symbol
#    pragma warning(disable : 4251)   // needs to have dll-interface to be used
#    pragma warning(disable : 4522)   // multiple assignment operators specified
#    pragma warning(disable : 26495)  // Always initialize member variable (cereal issue)

#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
#endif

//======================================================================================//
//
//      EXTERN TEMPLATE DECLARE AND INSTANTIATE
//
//======================================================================================//

#if !defined(_WINDOWS)
#    define _EXTERN_NAME_COMBINE(X, Y) X##Y
#    define _EXTERN_TUPLE_ALIAS(Y) _EXTERN_NAME_COMBINE(extern_tuple_, Y)
#    define _EXTERN_LIST_ALIAS(Y) _EXTERN_NAME_COMBINE(extern_list_, Y)

//--------------------------------------------------------------------------------------//
//      extern declaration
//
#    define TIMEMORY_DECLARE_EXTERN_TUPLE(_ALIAS, ...)                                   \
        extern template class tim::auto_tuple<__VA_ARGS__>;                              \
        extern template class tim::component_tuple<__VA_ARGS__>;                         \
        using _EXTERN_TUPLE_ALIAS(_ALIAS) = tim::component_tuple<__VA_ARGS__>;

#    define TIMEMORY_DECLARE_EXTERN_LIST(_ALIAS, ...)                                    \
        extern template class tim::auto_list<__VA_ARGS__>;                               \
        extern template class tim::component_list<__VA_ARGS__>;                          \
        using _EXTERN_LIST_ALIAS(_ALIAS) = tim::component_list<__VA_ARGS__>;

#    define TIMEMORY_DECLARE_EXTERN_HYBRID(_ALIAS)                                       \
        extern template class tim::auto_hybrid<_EXTERN_TUPLE_ALIAS(_ALIAS),              \
                                               _EXTERN_LIST_ALIAS(_ALIAS)>;              \
        extern template class tim::component_hybrid<_EXTERN_TUPLE_ALIAS(_ALIAS),         \
                                                    _EXTERN_LIST_ALIAS(_ALIAS)>;

//--------------------------------------------------------------------------------------//
//      extern instantiation
//
#    define TIMEMORY_INSTANTIATE_EXTERN_TUPLE(_ALIAS, ...)                               \
        template class tim::auto_tuple<__VA_ARGS__>;                                     \
        template class tim::component_tuple<__VA_ARGS__>;                                \
        using _EXTERN_TUPLE_ALIAS(_ALIAS) = tim::component_tuple<__VA_ARGS__>;

#    define TIMEMORY_INSTANTIATE_EXTERN_LIST(_ALIAS, ...)                                \
        template class tim::auto_list<__VA_ARGS__>;                                      \
        template class tim::component_list<__VA_ARGS__>;                                 \
        using _EXTERN_LIST_ALIAS(_ALIAS) = tim::component_list<__VA_ARGS__>;

#    define TIMEMORY_INSTANTIATE_EXTERN_HYBRID(_ALIAS)                                   \
        template class tim::auto_hybrid<_EXTERN_TUPLE_ALIAS(_ALIAS),                     \
                                        _EXTERN_LIST_ALIAS(_ALIAS)>;                     \
        template class tim::component_hybrid<_EXTERN_TUPLE_ALIAS(_ALIAS),                \
                                             _EXTERN_LIST_ALIAS(_ALIAS)>;

//--------------------------------------------------------------------------------------//
//      extern storage singleton
//
#    define TIMEMORY_DECLARE_EXTERN_INIT(TYPE)                                           \
        template <>                                                                      \
        details::storage_singleton_t<storage<component::TYPE>>&                          \
        get_storage_singleton<storage<component::TYPE>>();                               \
        template <>                                                                      \
        details::storage_singleton_t<storage<component::TYPE>>&                          \
        get_noninit_storage_singleton<storage<component::TYPE>>();                       \
        extern template class storage<component::TYPE>;

#    define TIMEMORY_INSTANTIATE_EXTERN_INIT(TYPE)                                       \
        template <>                                                                      \
        details::storage_singleton_t<storage<component::TYPE>>&                          \
        get_storage_singleton<storage<component::TYPE>>()                                \
        {                                                                                \
            using _storage_t           = storage<component::TYPE>;                       \
            using _single_t            = details::storage_singleton_t<_storage_t>;       \
            static _single_t _instance = _single_t::instance();                          \
            return _instance;                                                            \
        }                                                                                \
        template <>                                                                      \
        details::storage_singleton_t<storage<component::TYPE>>&                          \
        get_noninit_storage_singleton<storage<component::TYPE>>()                        \
        {                                                                                \
            using _storage_t           = storage<component::TYPE>;                       \
            using _single_t            = details::storage_singleton_t<_storage_t>;       \
            static _single_t _instance = _single_t::instance_ptr();                      \
            return _instance;                                                            \
        }                                                                                \
        template class storage<component::TYPE>;

#else

#    define _EXTERN_NAME_COMBINE(X, Y) X##Y
#    define _EXTERN_TUPLE_ALIAS(Y) _EXTERN_NAME_COMBINE(extern_tuple_, Y)
#    define _EXTERN_LIST_ALIAS(Y) _EXTERN_NAME_COMBINE(extern_list_, Y)

//--------------------------------------------------------------------------------------//
//      extern declaration
//
#    define TIMEMORY_DECLARE_EXTERN_TUPLE(...)
#    define TIMEMORY_DECLARE_EXTERN_LIST(...)
#    define TIMEMORY_DECLARE_EXTERN_HYBRID(...)

//--------------------------------------------------------------------------------------//
//      extern instantiation
//
#    define TIMEMORY_INSTANTIATE_EXTERN_TUPLE(...)
#    define TIMEMORY_INSTANTIATE_EXTERN_LIST(...)
#    define TIMEMORY_INSTANTIATE_EXTERN_HYBRID(...)

//--------------------------------------------------------------------------------------//
//      extern storage
//
#    define TIMEMORY_EXTERN_INIT_TYPE(...)
#    define TIMEMORY_DECLARE_EXTERN_INIT(...)
#    define TIMEMORY_INSTANTIATE_EXTERN_INIT(...)

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
        {}
#    define _DBG(...)                                                                    \
        {}
#endif

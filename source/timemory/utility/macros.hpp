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

#pragma once

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>

#if defined(TIMEMORY_CORE_SOURCE)
#    define TIMEMORY_UTILITY_SOURCE
#elif defined(TIMEMORY_USE_CORE_EXTERN)
#    define TIMEMORY_USE_UTILITY_EXTERN
#endif
//
#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_UTILITY_EXTERN)
#    define TIMEMORY_USE_UTILITY_EXTERN
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
        (fprintf(stderr, "[pid=%i][tid=%i][%s:%i@'%s']> " fmt "...\n",                   \
                 (int) ::tim::process::get_id(), (int) ::tim::threading::get_id(),       \
                 __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__),                         \
         fflush(stderr))
#endif

#if !defined(DEBUG_PRINT_HERE)
#    if defined(DEBUG)
#        define DEBUG_PRINT_HERE(fmt, ...)                                               \
            if(::tim::settings::debug())                                                 \
            {                                                                            \
                fprintf(stderr, "[pid=%i][tid=%i][%s:%i@'%s']> " fmt "...\n",            \
                        (int) ::tim::process::get_id(),                                  \
                        (int) ::tim::threading::get_id(), __FILE__, __LINE__,            \
                        __FUNCTION__, __VA_ARGS__);                                      \
                fflush(stderr);                                                          \
            }
#    else
#        define DEBUG_PRINT_HERE(fmt, ...)
#    endif
#endif

#if !defined(VERBOSE_PRINT_HERE)
#    define VERBOSE_PRINT_HERE(VERBOSE_LEVEL, fmt, ...)                                  \
        if(::tim::settings::verbose() >= VERBOSE_LEVEL)                                  \
        {                                                                                \
            fprintf(stderr, "[pid=%i][tid=%i][%s:%i@'%s']> " fmt "...\n",                \
                    (int) ::tim::process::get_id(), (int) ::tim::threading::get_id(),    \
                    __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__);                      \
            fflush(stderr);                                                              \
        }
#endif

#if !defined(CONDITIONAL_PRINT_HERE)
#    define CONDITIONAL_PRINT_HERE(CONDITION, fmt, ...)                                  \
        if(CONDITION)                                                                    \
        {                                                                                \
            fprintf(stderr, "[pid=%i][tid=%i][%s:%i@'%s']> " fmt "...\n",                \
                    (int) ::tim::process::get_id(), (int) ::tim::threading::get_id(),    \
                    __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__);                      \
            fflush(stderr);                                                              \
        }
#endif

#if !defined(PRETTY_PRINT_HERE)
#    if defined(_TIMEMORY_GNU) || defined(_TIMEMORY_CLANG)
#        define PRETTY_PRINT_HERE(fmt, ...)                                              \
            (fprintf(stderr, "[pid=%i][tid=%i][%s:%i@'%s']> " fmt "...\n",               \
                     (int) ::tim::process::get_id(), (int) ::tim::threading::get_id(),   \
                     __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__),                     \
             fflush(stderr))
#    else
#        define PRETTY_PRINT_HERE(fmt, ...)                                              \
            (fprintf(stderr, "[pid=%i][tid=%i][%s:%i@'%s']> " fmt "...\n",               \
                     (int) ::tim::process::get_id(), (int) ::tim::threading::get_id(),   \
                     __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__),                     \
             fflush(stderr))
#    endif
#endif

#if defined(DEBUG)

template <typename... Args>
inline void
__LOG(std::string file, int line, const char* msg, Args&&... args)
{
    if(file.find('/') != std::string::npos)
        file = file.substr(file.find_last_of('/'));
    fprintf(stderr, "[Log @ %s:%i]> ", file.c_str(), line);
    fprintf(stderr, msg, std::forward<Args>(args)...);
    fprintf(stderr, "\n");
}

//--------------------------------------------------------------------------------------//

inline void
__LOG(std::string file, int line, const char* msg)
{
    if(file.find('/') != std::string::npos)
        file = file.substr(file.find_last_of('/'));
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

//======================================================================================//
//
// Define macros for utility
//
//======================================================================================//
//
#if defined(TIMEMORY_UTILITY_SOURCE)
#    define TIMEMORY_UTILITY_LINKAGE(...) __VA_ARGS__
#elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_UTILITY_EXTERN)
#    define TIMEMORY_UTILITY_LINKAGE(...) __VA_ARGS__
#else
#    define TIMEMORY_UTILITY_LINKAGE(...) inline __VA_ARGS__
#endif

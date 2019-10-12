//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

/** \file ctimemory.h
 * \headerfile ctimemory.h "timemory/ctimemory.h"
 * This header file provides the C interface to TiMemory and generally just
 * redirects to functions implemented in ctimemory.cpp
 *
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(DISABLE_TIMEMORY)

#    define TIMEMORY_SETTINGS_INIT                                                       \
        {                                                                                \
        }
#    define TIMEMORY_INIT(...)
#    define TIMEMORY_BLANK_AUTO_TIMER(...) NULL
#    define TIMEMORY_BASIC_AUTO_TIMER(...) NULL
#    define TIMEMORY_AUTO_TIMER(...) NULL
#    define TIMEMORY_BASIC_AUTO_TUPLE(...) NULL
#    define TIMEMORY_BLANK_AUTO_TUPLE(...) NULL
#    define TIMEMORY_AUTO_TUPLE(...) NULL
#    define FREE_TIMEMORY_AUTO_TIMER(...)
#    define FREE_TIMEMORY_AUTO_TUPLE(...)

#else  // !defined(DISABLE_TIMEMORY)

//======================================================================================//
//
//      Operating System
//
//======================================================================================//

// machine bits
#    if defined(__x86_64__)
#        if !defined(_64BIT)
#            define _64BIT
#        endif
#    else
#        if !defined(_32BIT)
#            define _32BIT
#        endif
#    endif

//--------------------------------------------------------------------------------------//
// base operating system

#    if defined(_WIN32) || defined(_WIN64)
#        if !defined(_WINDOWS)
#            define _WINDOWS
#        endif
//--------------------------------------------------------------------------------------//

#    elif defined(__APPLE__) || defined(__MACH__)
#        if !defined(_MACOS)
#            define _MACOS
#        endif
#        if !defined(_UNIX)
#            define _UNIX
#        endif
//--------------------------------------------------------------------------------------//

#    elif defined(__linux__) || defined(__linux) || defined(linux) ||                    \
        defined(__gnu_linux__)
#        if !defined(_LINUX)
#            define _LINUX
#        endif
#        if !defined(_UNIX)
#            define _UNIX
#        endif
//--------------------------------------------------------------------------------------//

#    elif defined(__unix__) || defined(__unix) || defined(unix) || defined(_)
#        if !defined(_UNIX)
#            define _UNIX
#        endif
#    endif

//======================================================================================//
//
//      Windows DLL settings
//
//======================================================================================//

// Define macros for WIN32 for importing/exporting external symbols to DLLs
#    if !defined(tim_api) && !defined(tim_api_static)
#        if defined(_WINDOWS) && !defined(_TIMEMORY_ARCHIVE)
#            if defined(_TIMEMORY_DLL)
#                define tim_api __declspec(dllexport)
#                define tim_api_static static __declspec(dllexport)
#            else
#                define tim_api __declspec(dllimport)
#                define tim_api_static static __declspec(dllimport)
#            endif
#        else
#            define tim_api
#            define tim_api_static static
#        endif
#    endif

//======================================================================================//
//
//      C component enum
//
//======================================================================================//

#    include "timemory/bits/ctimemory.h"

//======================================================================================//
//
//      C struct for settings
//
//======================================================================================//

#    if defined(__cplusplus)
extern "C"
{
#    endif

    typedef struct
    {
        int enabled;
        int auto_output;
        int file_output;
        int text_output;
        int json_output;
        int cout_output;
        int precision;
        int width;
        int scientific;
        // skipping remainder
    } timemory_settings;

#    if defined(__cplusplus)
}
#    endif

//======================================================================================//
//
//      C function declaration
//
//======================================================================================//
#    if !defined(TIMEMORY_EXTERN_C)
#        if defined(__cplusplus)
#            define TIMEMORY_EXTERN_C "C"
#        else
#            define TIMEMORY_EXTERN_C
#        endif
#    endif

extern TIMEMORY_EXTERN_C tim_api void
                         c_timemory_init(int argc, char** argv, timemory_settings);
extern TIMEMORY_EXTERN_C tim_api int
                         c_timemory_enabled(void);
extern TIMEMORY_EXTERN_C tim_api void*
                         c_timemory_create_auto_timer(const char*);
extern TIMEMORY_EXTERN_C tim_api void
                         c_timemory_delete_auto_timer(void*);
extern TIMEMORY_EXTERN_C tim_api void*
                         c_timemory_create_auto_tuple(const char*, ...);
extern TIMEMORY_EXTERN_C tim_api void
                         c_timemory_delete_auto_tuple(void*);
extern TIMEMORY_EXTERN_C tim_api const char*
                         c_timemory_string_combine(const char*, const char*);
extern TIMEMORY_EXTERN_C tim_api const char*
                         c_timemory_auto_str(const char*, const char*, const char*, int);

//======================================================================================//
//
//      C timemory macros
//
//======================================================================================//

// only define for C
#    if !defined(__cplusplus)

#        if !defined(__FUNCTION__) && defined(__func__)
#            define __FUNCTION__ __func__
#        endif

#        if defined(TIMEMORY_PRETTY_FUNCTION) && !defined(_WINDOWS)
#            define __TIMEMORY_FUNCTION__ __PRETTY_FUNCTION__
#        else
#            define __TIMEMORY_FUNCTION__ __FUNCTION__
#        endif

// stringify some macro -- uses TIMEMORY_STRINGIFY2 which does the actual
//   "stringify-ing" after the macro has been substituted by it's result
#        if !defined(TIMEMORY_STRINGIZE)
#            define TIMEMORY_STRINGIZE(X) TIMEMORY_STRINGIZE2(X)
#        endif

// actual stringifying
#        if !defined(TIMEMORY_STRINGIZE2)
#            define TIMEMORY_STRINGIZE2(X) #            X
#        endif

// stringify the __LINE__ macro
#        if !defined(TIMEMORY_LINE_STRING)
#            define TIMEMORY_LINE_STRING TIMEMORY_STRINGIZE(__LINE__)
#        endif

//--------------------------------------------------------------------------------------//
//
#        if !defined(TIMEMORY_AUTO_LABEL)
#            define TIMEMORY_AUTO_LABEL(c_str)                                           \
                c_timemory_auto_str(__TIMEMORY_FUNCTION__, c_str, __FILE__, __LINE__)
#        endif

//--------------------------------------------------------------------------------------//
#        define TIMEMORY_SETTINGS_INIT { 1, -1, -1, -1, -1, -1, -1, -1, -1 };
#        define TIMEMORY_INIT(argc, argv, settings) c_timemory_init(argc, argv, settings)

//--------------------------------------------------------------------------------------//

#        if !defined(TIMEMORY_SPRINTF)
#            define TIMEMORY_SPRINTF(VAR, LEN, FMT, ...)                                 \
                char VAR[LEN];                                                           \
                sprintf(VAR, FMT, __VA_ARGS__);
#        endif

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BLANK_AUTO_TIMER(c_str)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_BLANK_AUTO_TIMER("label");
 *          ...
 *          FREE_TIMEMORY_AUTO_TIMER(timer);
 *      }
 */
#        define TIMEMORY_BLANK_AUTO_TIMER(c_str) c_timemory_create_auto_timer(c_str)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_AUTO_TIMER(c_str)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_BASIC_AUTO_TIMER("");
 *          ...
 *          FREE_TIMEMORY_AUTO_TIMER(timer);
 *      }
 */
#        define TIMEMORY_BASIC_AUTO_TIMER(c_str)                                         \
            c_timemory_create_auto_timer(                                                \
                c_timemory_string_combine(__TIMEMORY_FUNCTION__, c_str))

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_AUTO_TIMER(str)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_AUTO_TIMER("");
 *          ...
 *          FREE_TIMEMORY_AUTO_TIMER(timer);
 *      }
 *
 */
#        define TIMEMORY_AUTO_TIMER(c_str)                                               \
            c_timemory_create_auto_timer(                                                \
                c_timemory_auto_str(__TIMEMORY_FUNCTION__, c_str, __FILE__, __LINE__))

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_MARKER(c_str, ...)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_BASIC_MARKER("", WALL_CLOCK, CPU_CLOCK);
 *          ...
 *          FREE_TIMEMORY_MARKER(timer);
 *      }
 *
 */
#        define TIMEMORY_BASIC_MARKER(c_str, ...)                                        \
            c_timemory_create_auto_tuple(                                                \
                c_timemory_string_combine(__TIMEMORY_FUNCTION__, c_str), __VA_ARGS__,    \
                TIMEMORY_COMPONENTS_END)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_BLANK_MARKER(c_str, ...)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_BLANK_MARKER("id", WALL_CLOCK, CPU_CLOCK);
 *          ...
 *          FREE_TIMEMORY_MARKER(timer);
 *      }
 *
 */
#        define TIMEMORY_BLANK_MARKER(c_str, ...)                                        \
            c_timemory_create_auto_tuple(c_str, __VA_ARGS__, TIMEMORY_COMPONENTS_END)

//--------------------------------------------------------------------------------------//
/*! \def TIMEMORY_MARKER(c_str, ...)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_MARKER("", WALL_CLOCK, SYS_CLOCK, CPU_CLOCK);
 *          ...
 *          FREE_TIMEMORY_MARKER(timer);
 *      }
 *
 */
#        define TIMEMORY_MARKER(c_str, ...)                                              \
            c_timemory_create_auto_tuple(                                                \
                c_timemory_auto_str(__TIMEMORY_FUNCTION__, c_str, __FILE__, __LINE__),   \
                __VA_ARGS__, TIMEMORY_COMPONENTS_END)

//--------------------------------------------------------------------------------------//
/*! \def FREE_TIMEMORY_AUTO_TIMER(ctimer)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_AUTO_TIMER("");
 *          ...
 *          FREE_TIMEMORY_AUTO_TIMER(timer);
 *      }
 */
#        define FREE_TIMEMORY_AUTO_TIMER(ctimer)                                         \
            c_timemory_delete_auto_timer((void*) ctimer)

//--------------------------------------------------------------------------------------//
/*! \def FREE_TIMEMORY_MARKER(ctimer)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_MARKER("", WALL_CLOCK);
 *          ...
 *          FREE_TIMEMORY_MARKER(timer);
 *      }
 */
#        define FREE_TIMEMORY_MARKER(ctimer) c_timemory_delete_auto_tuple((void*) ctimer)

//--------------------------------------------------------------------------------------//

#    endif  // !defined(__cplusplus)

#endif  // !defined(DISABLE_TIMEMORY)

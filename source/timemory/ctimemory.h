//  MIT License
//  
//  Copyright (c) 2018, The Regents of the University of California, 
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//  
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#ifndef ctimemory_h_
#define ctimemory_h_

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

//============================================================================//
//
//      Operating System
//
//============================================================================//

// machine bits
#if defined(__x86_64__)
#   if !defined(_64BIT)
#       define _64BIT
#   endif
#else
#   if !defined(_32BIT)
#       define _32BIT
#   endif
#endif

//----------------------------------------------------------------------------//
// base operating system

#if defined(_WIN32) || defined(_WIN64)
#   if !defined(_WINDOWS)
#       define _WINDOWS
#   endif
//----------------------------------------------------------------------------//

#elif defined(__APPLE__) || defined(__MACH__)
#   if !defined(_MACOS)
#       define _MACOS
#   endif
#   if !defined(_UNIX)
#       define _UNIX
#   endif
//----------------------------------------------------------------------------//

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#   if !defined(_LINUX)
#       define _LINUX
#   endif
#   if !defined(_UNIX)
#       define _UNIX
#   endif
//----------------------------------------------------------------------------//

#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(_)
#   if !defined(_UNIX)
#       define _UNIX
#   endif
#endif

//============================================================================//
//
//      Windows DLL settings
//
//============================================================================//

// Define macros for WIN32 for importing/exporting external symbols to DLLs
#if defined(_WINDOWS) && !defined(_TIMEMORY_ARCHIVE)
#   if defined(_TIMEMORY_DLL)
#       define tim_api __declspec(dllexport)
#       define tim_api_static static __declspec(dllexport)
#   else
#       define tim_api __declspec(dllimport)
#       define tim_api_static static __declspec(dllimport)
#   endif
#else
#   define tim_api
#   define tim_api_static static
#endif

//============================================================================//
//
//      C function declaration
//
//============================================================================//


tim_api void*       c_timemory_create_auto_timer    (const char*, int);
tim_api void        c_timemory_report               (const char*);
tim_api void        c_timemory_print                (void);
tim_api void        c_timemory_delete_auto_timer    (void*);
tim_api const char* c_timemory_string_combine       (const char*, const char*);
tim_api const char* c_timemory_auto_timer_str       (const char*, const char*,
                                                     const char*, int);
tim_api void        c_timemory_record_memory        (int);

//============================================================================//
//
//      C timemory macros
//
//============================================================================//

#if !defined(__FUNCTION__) && defined(__func__)
#   define __FUNCTION__ __func__
#endif

#if defined(TIMEMORY_PRETTY_FUNCTION) && !defined(_WINDOWS)
#   define __TIMEMORY_FUNCTION__ __PRETTY_FUNCTION__
#else
#   define __TIMEMORY_FUNCTION__ __FUNCTION__
#endif

// stringify some macro -- uses TIMEMORY_STRINGIFY2 which does the actual
//   "stringify-ing" after the macro has been substituted by it's result
#if !defined(TIMEMORY_STRINGIZE)
#   define TIMEMORY_STRINGIZE(X) TIMEMORY_STRINGIZE2(X)
#endif

// actual stringifying
#if !defined(TIMEMORY_STRINGIZE2)
#   define TIMEMORY_STRINGIZE2(X) #X
#endif

// stringify the __LINE__ macro
#if !defined(TIMEMORY_LINE_STRING)
#   define TIMEMORY_LINE_STRING TIMEMORY_STRINGIZE(__LINE__)
#endif

//----------------------------------------------------------------------------//
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
#if !defined(TIMEMORY_BASIC_AUTO_TIMER)
#   define TIMEMORY_BASIC_AUTO_TIMER(c_str) \
    c_timemory_create_auto_timer(c_timemory_string_combine(__TIMEMORY_FUNCTION__, c_str), __LINE__)
#endif

//----------------------------------------------------------------------------//
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
#if !defined(TIMEMORY_AUTO_TIMER)
#   define TIMEMORY_AUTO_TIMER(c_str) \
    c_timemory_create_auto_timer(c_timemory_auto_timer_str(__TIMEMORY_FUNCTION__, c_str, __FILE__, __LINE__), __LINE__)
#endif

//----------------------------------------------------------------------------//
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
#if !defined(FREE_TIMEMORY_AUTO_TIMER)
#   define FREE_TIMEMORY_AUTO_TIMER(ctimer) \
    c_timemory_delete_auto_timer((void*) ctimer);
#endif

//----------------------------------------------------------------------------//

#if !defined(TIMEMORY_PRINT)
#   define TIMEMORY_PRINT() c_timemory_print()
#endif

//----------------------------------------------------------------------------//

#if !defined(TIMEMORY_REPORT)
#   define TIMEMORY_REPORT(fname) c_timemory_report(fname)
#endif

//----------------------------------------------------------------------------//

#if !defined(TIMEMORY_RECORD_MEMORY)
#   define TIMEMORY_RECORD_MEMORY(code) c_timemory_record_memory(code)
#endif

//----------------------------------------------------------------------------//

#endif


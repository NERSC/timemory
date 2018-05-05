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

//----------------------------------------------------------------------------//

void*       c_timemory_create_auto_timer    (const char*, int);
void        c_timemory_report               (const char*);
void        c_timemory_delete_auto_timer    (void*);
const char* c_timemory_string_combine       (const char*, const char*);
const char* c_timemory_auto_timer_str       (const char*, const char*,
                                             const char*, int);

//----------------------------------------------------------------------------//

#if !defined(__FUNCTION__) && defined(__func__)
#   define __FUNCTION__ __func__
#endif

// stringify some macro -- uses TIMEMORY_C_STRINGIFY2 which does the actual
//   "stringify-ing" after the macro has been substituted by it's result
#if !defined(TIMEMORY_C_STRINGIZE)
#   define TIMEMORY_C_STRINGIZE(X) TIMEMORY_C_STRINGIZE2(X)
#endif

// actual stringifying
#if !defined(TIMEMORY_C_STRINGIZE2)
#   define TIMEMORY_C_STRINGIZE2(X) #X
#endif

// stringify the __LINE__ macro
#if !defined(TIMEMORY_C_LINE_STRING)
#   define TIMEMORY_C_LINE_STRING TIMEMORY_C_STRINGIZE(__LINE__)
#endif

//----------------------------------------------------------------------------//
/*! \def TIMEMORY_BASIC_C_AUTO_TIMER(c_str)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_BASIC_C_AUTO_TIMER("");
 *          ...
 *          TIMEMORY_C_AUTO_TIMER_DELETE(timer);
 *      }
*/
#if !defined(TIMEMORY_BASIC_C_AUTO_TIMER)
#   define TIMEMORY_BASIC_C_AUTO_TIMER(c_str) \
    c_timemory_create_auto_timer(c_timemory_string_combine(__FUNCTION__, c_str), __LINE__)
#endif

//----------------------------------------------------------------------------//
/*! \def TIMEMORY_C_AUTO_TIMER(str)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_C_AUTO_TIMER("");
 *          ...
 *          FREE_TIMEMORY_C_AUTO_TIMER(timer);
 *      }
 *
 */
#if !defined(TIMEMORY_C_AUTO_TIMER)
#   define TIMEMORY_C_AUTO_TIMER(c_str) \
    c_timemory_create_auto_timer(c_timemory_auto_timer_str(__FUNCTION__, c_str, __FILE__, __LINE__), __LINE__)
#endif

//----------------------------------------------------------------------------//
/*! \def FREE_TIMEMORY_C_AUTO_TIMER(ctimer)
 *
 * Usage:
 *
 *      void some_func()
 *      {
 *          void* timer = new TIMEMORY_C_AUTO_TIMER("");
 *          ...
 *          FREE_TIMEMORY_C_AUTO_TIMER(timer);
 *      }
*/
#if !defined(FREE_TIMEMORY_C_AUTO_TIMER)
#   define FREE_TIMEMORY_C_AUTO_TIMER(ctimer) \
    c_timemory_delete_auto_timer((void*) ctimer);
#endif

//----------------------------------------------------------------------------//

#if !defined(TIMEMORY_C_REPORT)
#   define TIMEMORY_C_REPORT(fname) c_timemory_report(fname)
#endif
//----------------------------------------------------------------------------//

#endif


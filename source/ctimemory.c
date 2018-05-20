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

#include "timemory/ctimemory.h"
#include "assert.h"

#if !defined(pfunc)
#   if defined(DEBUG)
#       define pfunc printf("TiMemory -- calling %s@\"%s\":%i...\n", __FUNCTION__, __FILE__, __LINE__)
#   else
#       define pfunc
#   endif
#endif

//============================================================================//
// declaration of C++ defined functions (timemory/auto_timer.hpp)

extern void         cxx_timemory_initialization     (void);
extern void         cxx_timemory_finalization       (void);
extern int          cxx_timemory_enabled            (void);
extern void         cxx_timemory_report             (const char*);
extern void         cxx_timemory_print              (void);
extern void*        cxx_timemory_create_auto_timer  (const char*, int,
                                                     const char*, int);
extern void*        cxx_timemory_delete_auto_timer  (void*);
extern const char*  cxx_timemory_string_combine     (const char*, const char*);
extern const char*  cxx_timemory_auto_timer_str     (const char*, const char*,
                                                     const char*, int);
extern void         cxx_timemory_record_memory      (int);

//============================================================================//

void* c_timemory_create_auto_timer(const char* tag, int lineno)
{
    return (cxx_timemory_enabled())
            ? cxx_timemory_create_auto_timer(tag, lineno, "_c_", false)
            : NULL;
}

//============================================================================//

void c_timemory_delete_auto_timer(void* ctimer)
{
    ctimer = cxx_timemory_delete_auto_timer(ctimer);
    assert(ctimer == NULL);
}

//============================================================================//

void c_timemory_report(const char* fname)
{
    cxx_timemory_report(fname);
}

//============================================================================//

void c_timemory_print(void)
{
    cxx_timemory_print();
}

//============================================================================//

const char* c_timemory_string_combine(const char* _a, const char* _b)
{
    return cxx_timemory_string_combine(_a, _b);
}

//============================================================================//

const char* c_timemory_auto_timer_str(const char* _a, const char* _b,
                                      const char* _c, int _d)
{
    return cxx_timemory_auto_timer_str(_a, _b, _c, _d);
}

//============================================================================//

void c_timemory_record_memory(int _record_memory)
{
    cxx_timemory_record_memory(_record_memory);
}

//============================================================================//


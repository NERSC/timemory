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

#include "timemory/ctimemory.h"
#include "assert.h"
#include <stdarg.h>
#include <stdlib.h>

#if !defined(TIMEMORY_EXTERN_C)
#    if defined(__cplusplus)
#        define TIMEMORY_EXTERN_C "C"
#    else
#        define TIMEMORY_EXTERN_C
#    endif
#endif

//======================================================================================//
// declaration of C++ defined functions (timemory/auto_timer.hpp)

extern void
cxx_timemory_init(int, char**, timemory_settings);
extern int
cxx_timemory_enabled(void);
extern void*
cxx_timemory_create_auto_timer(const char*, int);
extern void*
cxx_timemory_create_auto_tuple(const char*, int, int, const int*);
extern void*
cxx_timemory_delete_auto_timer(void*);
extern void*
cxx_timemory_delete_auto_tuple(void*);
extern const char*
cxx_timemory_string_combine(const char*, const char*);
extern const char*
cxx_timemory_auto_timer_str(const char*, const char*, const char*, int);

//======================================================================================//
#if defined(__cplusplus)
extern "C"
{
#endif
    //======================================================================================//

    tim_api void c_timemory_init(int argc, char** argv, timemory_settings _settings)
    {
        cxx_timemory_init(argc, argv, _settings);
    }

    //======================================================================================//

    tim_api int c_timemory_enabled(void) { return cxx_timemory_enabled(); }

    //======================================================================================//

    tim_api void* c_timemory_create_auto_timer(const char* tag, int lineno)
    {
        return (cxx_timemory_enabled()) ? cxx_timemory_create_auto_timer(tag, lineno)
                                        : NULL;
    }

    //======================================================================================//

    tim_api void* c_timemory_create_auto_tuple(const char* tag, int lineno, ...)
    {
        if(!cxx_timemory_enabled())
            return NULL;

        const int max_size       = (int) TIMEMORY_COMPONENTS_END;
        int       num_components = 0;
        int*      components     = (int*) malloc(max_size * sizeof(int));
        if(!components)
            return NULL;
        va_list args;
        va_start(args, lineno);
        for(int i = 0; i < max_size; ++i)
        {
            int comp = va_arg(args, int);
            if(comp >= TIMEMORY_COMPONENTS_END)
                break;
            ++num_components;
            components[i] = comp;
        }
        va_end(args);

        void* ptr = NULL;
        if(num_components > 0)
            ptr = cxx_timemory_create_auto_tuple(tag, lineno, num_components, components);
        free(components);

        return ptr;
    }

    //======================================================================================//

    tim_api void c_timemory_delete_auto_timer(void* ctimer)
    {
        ctimer = cxx_timemory_delete_auto_timer(ctimer);
        assert(ctimer == NULL);
    }

    //======================================================================================//

    tim_api void c_timemory_delete_auto_tuple(void* ctuple)
    {
        ctuple = cxx_timemory_delete_auto_tuple(ctuple);
        assert(ctuple == NULL);
    }

    //======================================================================================//

    tim_api const char* c_timemory_string_combine(const char* _a, const char* _b)
    {
        return cxx_timemory_string_combine(_a, _b);
    }

    //======================================================================================//

    tim_api const char* c_timemory_auto_str(const char* _a, const char* _b,
                                            const char* _c, int _d)
    {
        return cxx_timemory_auto_timer_str(_a, _b, _c, _d);
    }

//======================================================================================//
#if defined(__cplusplus)
}
#endif
//======================================================================================//

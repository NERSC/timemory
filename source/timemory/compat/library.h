// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

/** \file timemory_c.h
 * \headerfile timemory_c.h "timemory/timemory_c.h"
 * This header file provides the C interface to TiMemory
 *
 */

#pragma once

#if defined(__cplusplus)
#    include <cstdint>
#    include <cstdio>
#    include <cstdlib>
#else
#    include <stdbool.h>
#    include <stddef.h>
#    include <stdint.h>
#    include <stdio.h>
#    include <stdlib.h>
#    include <string.h>
#endif

#include "timemory/compat/macros.h"
#include "timemory/enum.h"

//======================================================================================//
//
//      C struct for settings
//
//======================================================================================//

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

//======================================================================================//

#if defined(__cplusplus)
extern "C"
{
#endif  // if defined(__cplusplus)

    extern void        c_timemory_init(int argc, char** argv, timemory_settings);
    extern int         c_timemory_enabled(void);
    extern void*       c_timemory_create_auto_timer(const char*);
    extern void        c_timemory_delete_auto_timer(void*);
    extern void*       c_timemory_create_auto_tuple(const char*, ...);
    extern void        c_timemory_delete_auto_tuple(void*);
    extern const char* c_timemory_blank_label(const char*);
    extern const char* c_timemory_basic_label(const char*, const char*);
    extern const char* c_timemory_label(const char*, const char*, int, const char*);

    extern uint64_t timemory_get_unique_id(void);
    extern void timemory_create_record(const char* name, uint64_t* id, int n, int* ct);
    extern void timemory_delete_record(uint64_t nid);
    extern void timemory_init_library(int argc, char** argv);
    extern void timemory_finalize_library(void);

    extern void timemory_set_default(const char* components);
    extern void timemory_push_components(const char* components);
    extern void timemory_push_components_enum(int args, ...);
    extern void timemory_pop_components(void);

    extern void timemory_begin_record(const char* name, uint64_t* id);
    extern void timemory_begin_record_enum(const char* name, uint64_t*, ...);
    extern void timemory_begin_record_types(const char* name, uint64_t*, const char*);

    extern uint64_t timemory_get_begin_record(const char* name);
    extern uint64_t timemory_get_begin_record_enum(const char* name, ...);
    extern uint64_t timemory_get_begin_record_types(const char* name, const char* ctypes);

    extern void timemory_end_record(uint64_t id);

    typedef void (*timemory_create_func_t)(const char*, uint64_t*, int, int*);
    typedef void (*timemory_delete_func_t)(uint64_t);

    extern tim_api timemory_create_func_t timemory_create_function;
    extern tim_api timemory_delete_func_t timemory_delete_function;

#if defined(__cplusplus)
}
#endif  // if defined(__cplusplus)

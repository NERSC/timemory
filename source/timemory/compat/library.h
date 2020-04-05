// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
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
/*
// c library
#pragma weak c_timemory_init
#pragma weak c_timemory_finalize
#pragma weak c_timemory_enabled
#pragma weak c_timemory_create_auto_timer
#pragma weak c_timemory_delete_auto_timer
#pragma weak c_timemory_create_auto_tuple
#pragma weak c_timemory_delete_auto_tuple
#pragma weak c_timemory_blank_label
#pragma weak c_timemory_basic_label
#pragma weak c_timemory_label
// cxx library
#pragma weak cxx_timemory_init
#pragma weak cxx_timemory_enabled
#pragma weak cxx_timemory_create_auto_timer
#pragma weak cxx_timemory_create_auto_tuple
#pragma weak cxx_timemory_delete_auto_timer
#pragma weak cxx_timemory_delete_auto_tuple
#pragma weak cxx_timemory_label
*/
// function pointers
#pragma weak timemory_create_function
#pragma weak timemory_create_function
// generic library
#pragma weak timemory_get_unique_id
#pragma weak timemory_create_record
#pragma weak timemory_delete_record
#pragma weak timemory_init_library
#pragma weak timemory_finalize_library
#pragma weak timemory_pause
#pragma weak timemory_resume
#pragma weak timemory_set_default
#pragma weak timemory_push_components
#pragma weak timemory_push_components_enum
#pragma weak timemory_pop_components
#pragma weak timemory_begin_record
#pragma weak timemory_begin_record_types
#pragma weak timemory_begin_record_enum
#pragma weak timemory_get_begin_record
#pragma weak timemory_get_begin_record_types
#pragma weak timemory_get_begin_record_enum
#pragma weak timemory_end_record
#pragma weak timemory_push_region
#pragma weak timemory_pop_region
#pragma weak timemory_register_trace
#pragma weak timemory_deregister_trace
#pragma weak timemory_dyninst_init
#pragma weak timemory_dyninst_finalize
#pragma weak timemory_init_trace
#pragma weak timemory_fini_trace
#pragma weak timemory_mpi_init_stub
#pragma weak timemory_get_rank

//======================================================================================//

#if !defined(TIMEMORY_DECL)
#    define TIMEMORY_DECL extern tim_dll
#endif

//======================================================================================//

#if defined(__cplusplus)
extern "C"
{
#endif  // if defined(__cplusplus)

    typedef void (*timemory_create_func_t)(const char*, uint64_t*, int, int*);
    typedef void (*timemory_delete_func_t)(uint64_t);

    TIMEMORY_DECL timemory_create_func_t timemory_create_function;
    TIMEMORY_DECL timemory_delete_func_t timemory_delete_function;

    TIMEMORY_DECL void        c_timemory_init(int argc, char** argv, timemory_settings);
    TIMEMORY_DECL void        c_timemory_finalize(void);
    TIMEMORY_DECL int         c_timemory_enabled(void);
    TIMEMORY_DECL void*       c_timemory_create_auto_timer(const char*);
    TIMEMORY_DECL void        c_timemory_delete_auto_timer(void*);
    TIMEMORY_DECL void*       c_timemory_create_auto_tuple(const char*, ...);
    TIMEMORY_DECL void        c_timemory_delete_auto_tuple(void*);
    TIMEMORY_DECL const char* c_timemory_blank_label(const char*);
    TIMEMORY_DECL const char* c_timemory_basic_label(const char*, const char*);
    TIMEMORY_DECL const char* c_timemory_label(const char*, const char*, int,
                                               const char*);

    TIMEMORY_DECL void  cxx_timemory_init(int, char**, timemory_settings);
    TIMEMORY_DECL int   cxx_timemory_enabled(void);
    TIMEMORY_DECL void* cxx_timemory_create_auto_timer(const char*);
    TIMEMORY_DECL void* cxx_timemory_create_auto_tuple(const char*, int, const int*);
    TIMEMORY_DECL void* cxx_timemory_delete_auto_timer(void*);
    TIMEMORY_DECL void* cxx_timemory_delete_auto_tuple(void*);
    TIMEMORY_DECL const char* cxx_timemory_label(int, int, const char*, const char*,
                                                 const char*);

    TIMEMORY_DECL uint64_t timemory_get_unique_id(void);
    TIMEMORY_DECL void     timemory_create_record(const char* name, uint64_t* id, int n,
                                                  int* ct);
    TIMEMORY_DECL void     timemory_delete_record(uint64_t nid);
    TIMEMORY_DECL void     timemory_init_library(int argc, char** argv);
    TIMEMORY_DECL void     timemory_finalize_library(void);

    TIMEMORY_DECL void timemory_pause(void);
    TIMEMORY_DECL void timemory_resume(void);

    TIMEMORY_DECL void timemory_set_default(const char* components);
    TIMEMORY_DECL void timemory_push_components(const char* components);
    TIMEMORY_DECL void timemory_push_components_enum(int args, ...);
    TIMEMORY_DECL void timemory_pop_components(void);

    TIMEMORY_DECL void timemory_begin_record(const char* name, uint64_t* id);
    TIMEMORY_DECL void timemory_begin_record_enum(const char* name, uint64_t*, ...);
    TIMEMORY_DECL void timemory_begin_record_types(const char* name, uint64_t*,
                                                   const char*);

    TIMEMORY_DECL uint64_t timemory_get_begin_record(const char* name);
    TIMEMORY_DECL uint64_t timemory_get_begin_record_enum(const char* name, ...);
    TIMEMORY_DECL uint64_t timemory_get_begin_record_types(const char* name,
                                                           const char* ctypes);

    TIMEMORY_DECL void timemory_end_record(uint64_t id);

    TIMEMORY_DECL void timemory_push_region(const char* name);
    TIMEMORY_DECL void timemory_pop_region(const char* name);

    TIMEMORY_DECL int64_t timemory_register_trace(const char* name);
    TIMEMORY_DECL void    timemory_deregister_trace(const char* name);
    TIMEMORY_DECL void    timemory_dyninst_init(void);
    TIMEMORY_DECL void    timemory_dyninst_finalize(void);
    TIMEMORY_DECL void    timemory_init_trace(uint64_t id);
    TIMEMORY_DECL void    timemory_fini_trace(void);
    TIMEMORY_DECL void    timemory_mpi_init_stub(int rank);
    TIMEMORY_DECL int     timemory_get_rank(void);

#if defined(__cplusplus)
}
#endif  // if defined(__cplusplus)

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

#if !defined(TIMEMORY_DECL)
#    define TIMEMORY_DECL extern tim_dll
#endif

#if !defined(TIMEMORY_CDECL)
#    define TIMEMORY_CDECL extern tim_cdll
#endif

#if defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_GOTCHA)
#    define TIMEMORY_MPI_GOTCHA
#endif

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

    typedef void (*timemory_create_func_t)(const char*, uint64_t*, int, int*);
    typedef void (*timemory_delete_func_t)(uint64_t);

    TIMEMORY_DECL timemory_create_func_t timemory_create_function
                                         TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL timemory_delete_func_t timemory_delete_function
                                         TIMEMORY_VISIBILITY("default");

    TIMEMORY_CDECL void c_timemory_init(int argc, char** argv, timemory_settings)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_CDECL void  c_timemory_finalize(void) TIMEMORY_VISIBILITY("default");
    TIMEMORY_CDECL int   c_timemory_enabled(void) TIMEMORY_VISIBILITY("default");
    TIMEMORY_CDECL void* c_timemory_create_auto_timer(const char*)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_CDECL void c_timemory_delete_auto_timer(void*)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_CDECL void* c_timemory_create_auto_tuple(const char*, ...)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_CDECL void c_timemory_delete_auto_tuple(void*)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_CDECL const char* c_timemory_blank_label(const char*)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_CDECL const char* c_timemory_basic_label(const char*, const char*)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_CDECL const char* c_timemory_label(const char*, const char*, int,
                                                const char*)
        TIMEMORY_VISIBILITY("default");

    TIMEMORY_DECL void cxx_timemory_init(int, char**, timemory_settings)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL int   cxx_timemory_enabled(void) TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void* cxx_timemory_create_auto_timer(const char*)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void* cxx_timemory_create_auto_tuple(const char*, int, const int*)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void* cxx_timemory_delete_auto_timer(void*)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void* cxx_timemory_delete_auto_tuple(void*)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL const char* cxx_timemory_label(int, int, const char*, const char*,
                                                 const char*)
        TIMEMORY_VISIBILITY("default");

    TIMEMORY_DECL uint64_t timemory_get_unique_id(void) TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void     timemory_create_record(const char* name, uint64_t* id, int n,
                                                  int* ct) TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void     timemory_delete_record(uint64_t nid)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_init_library(int argc, char** argv)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_finalize_library(void) TIMEMORY_VISIBILITY("default");

    TIMEMORY_DECL void timemory_pause(void) TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_resume(void) TIMEMORY_VISIBILITY("default");

    TIMEMORY_DECL void timemory_set_default(const char* components)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_push_components(const char* components)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_push_components_enum(int args, ...)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_pop_components(void) TIMEMORY_VISIBILITY("default");

    TIMEMORY_DECL void timemory_begin_record(const char* name, uint64_t* id)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_begin_record_enum(const char* name, uint64_t*, ...)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_begin_record_types(const char* name, uint64_t*,
                                                   const char*)
        TIMEMORY_VISIBILITY("default");

    TIMEMORY_DECL uint64_t timemory_get_begin_record(const char* name)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL uint64_t timemory_get_begin_record_enum(const char* name, ...)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL uint64_t timemory_get_begin_record_types(const char* name,
                                                           const char* ctypes)
        TIMEMORY_VISIBILITY("default");

    TIMEMORY_DECL void timemory_end_record(uint64_t id) TIMEMORY_VISIBILITY("default");

    TIMEMORY_DECL void timemory_push_region(const char* name)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_pop_region(const char* name)
        TIMEMORY_VISIBILITY("default");

    TIMEMORY_DECL void timemory_push_trace(const char* name)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_pop_trace(const char* name)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_trace_init(const char*, bool, const char*)
        TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_trace_finalize(void) TIMEMORY_VISIBILITY("default");
    TIMEMORY_DECL void timemory_trace_set_env(const char*, const char*)
        TIMEMORY_VISIBILITY("default");

#if defined(TIMEMORY_MPI_GOTCHA)
    TIMEMORY_DECL void timemory_trace_set_mpi(bool use) TIMEMORY_VISIBILITY("default");
#endif

#if defined(__cplusplus)
}
#endif  // if defined(__cplusplus)

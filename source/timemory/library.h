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

#pragma once

#if defined(__cplusplus)
#    include <cstdint>
#    include <cstdio>
#    include <cstdlib>
#else
#    include <stdint.h>
#    include <stdio.h>
#    include <stdlib.h>
#endif

// enumeration
#include "timemory/bits/ctimemory.h"

//--------------------------------------------------------------------------------------//
//
#if defined(__cplusplus)
//
//--------------------------------------------------------------------------------------//

// for generating names
#    include "timemory/variadic/macros.hpp"

extern "C"
{
//--------------------------------------------------------------------------------------//
//
#endif  // if defined(__cplusplus)

    extern uint64_t timemory_get_unique_id();
    extern void timemory_create_record(const char* name, uint64_t* id, int n, int* ct);
    extern void timemory_delete_record(uint64_t nid);
    extern void timemory_init_library(int argc, char** argv);
    extern void timemory_finalize_library();
    extern void timemory_set_default(const char* components);
    extern void timemory_push_components(const char* components);
    extern void timemory_pop_components();
    extern void timemory_begin_record(const char* name, uint64_t* id);
    extern void timemory_begin_record_types(const char* name, uint64_t*, const char*);
    extern uint64_t timemory_get_begin_record(const char* name);
    extern uint64_t timemory_get_begin_record_types(const char* name, const char* ctypes);
    extern void     timemory_end_record(uint64_t id);

    typedef void (*timemory_create_func_t)(const char*, uint64_t*, int, int*);
    typedef void (*timemory_delete_func_t)(uint64_t);

    extern tim_api timemory_create_func_t timemory_create_function;
    extern tim_api timemory_delete_func_t timemory_delete_function;

//
//--------------------------------------------------------------------------------------//
//
#if defined(__cplusplus)

}  // extern "C"

//--------------------------------------------------------------------------------------//

struct timemory_scoped_record
{
    timemory_scoped_record(const char* name)
    : m_nid(timemory_get_begin_record(name))
    {
    }

    timemory_scoped_record(const char* name, const char* components)
    : m_nid(timemory_get_begin_record_types(name, components))
    {
    }

    ~timemory_scoped_record() { timemory_end_record(m_nid); }

private:
    uint64_t m_nid = std::numeric_limits<uint64_t>::max();
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline _Tp&
timemory_tl_static(const _Tp& _initial = {})
{
    static thread_local _Tp _instance = _initial;
    return _instance;
}

//--------------------------------------------------------------------------------------//

#endif  // if defined(__cplusplus)

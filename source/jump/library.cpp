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

#include "timemory/backends/process.hpp"
#include "timemory/environment.hpp"

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

//--------------------------------------------------------------------------------------//
//
//      timemory stubs
//
//--------------------------------------------------------------------------------------//

#define DLSYM_JUMP_FUNCTION(VARNAME, HANDLE, FUNCNAME)                                   \
    if(HANDLE)                                                                           \
    {                                                                                    \
        *(void**) (&VARNAME) = dlsym(HANDLE, FUNCNAME);                                  \
        if(VARNAME == nullptr)                                                           \
        {                                                                                \
            fprintf(stderr, "[timemory-jump@%s][pid=%i]> %s\n", FUNCNAME,                \
                    tim::process::get_id(), dlerror());                                  \
        }                                                                                \
    }                                                                                    \
    else                                                                                 \
    {                                                                                    \
        VARNAME = nullptr;                                                               \
    }

//--------------------------------------------------------------------------------------//

class jump
{
public:
    void (*timemory_push_components_jump)(const char*);
    void (*timemory_pop_components_jump)(void);

    void (*timemory_push_region_jump)(const char*);
    void (*timemory_pop_region_jump)(const char*);

    void (*timemory_add_hash_id_jump)(uint64_t, const char*);
    void (*timemory_push_trace_jump)(const char*);
    void (*timemory_pop_trace_jump)(const char*);
    void (*timemory_push_trace_hash_jump)(uint64_t);
    void (*timemory_pop_trace_hash_jump)(uint64_t);
    void (*timemory_trace_init_jump)(const char*, bool, const char*);
    void (*timemory_trace_finalize_jump)(void);
    void (*timemory_trace_set_env_jump)(const char*, const char*);
    void (*timemory_trace_set_mpi_jump)(bool, bool);

    jump(std::string&& libpath)
    {
        PRINT_HERE("%s", "");
        auto libhandle = dlopen(libpath.c_str(), RTLD_LAZY);

        if(!libhandle)
            fprintf(stderr, "%s\n", dlerror());

        dlerror(); /* Clear any existing error */

        /* Initialize all pointers */
        DLSYM_JUMP_FUNCTION(timemory_push_components_jump, libhandle,
                            "timemory_push_components")
        DLSYM_JUMP_FUNCTION(timemory_pop_components_jump, libhandle,
                            "timemory_pop_components")

        /*
        *(void**) (&timemory_push_components_jump) =
            dlsym(libhandle, "timemory_push_components");

        if(!timemory_push_components_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_push_components_jump = nullptr;
        }

        *(void**) (&timemory_pop_components_jump) =
            dlsym(libhandle, "timemory_pop_components");
        */

        if(!timemory_pop_components_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_pop_components_jump = nullptr;
        }

        *(void**) (&timemory_push_region_jump) = dlsym(libhandle, "timemory_push_region");

        if(!timemory_push_region_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_push_region_jump = nullptr;
        }

        *(void**) (&timemory_pop_region_jump) = dlsym(libhandle, "timemory_pop_region");

        if(!timemory_pop_region_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_pop_region_jump = nullptr;
        }

        *(void**) (&timemory_add_hash_id_jump) = dlsym(libhandle, "timemory_add_hash_id");

        if(!timemory_add_hash_id_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_add_hash_id_jump = nullptr;
        }

        *(void**) (&timemory_push_trace_jump) = dlsym(libhandle, "timemory_push_trace");

        if(!timemory_push_trace_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_push_trace_jump = nullptr;
        }

        *(void**) (&timemory_pop_trace_jump) = dlsym(libhandle, "timemory_pop_trace");

        if(!timemory_pop_trace_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_pop_trace_jump = nullptr;
        }

        *(void**) (&timemory_push_trace_hash_jump) =
            dlsym(libhandle, "timemory_push_trace_hash");

        if(!timemory_push_trace_hash_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_push_trace_hash_jump = nullptr;
        }

        *(void**) (&timemory_pop_trace_hash_jump) =
            dlsym(libhandle, "timemory_pop_trace_hash");

        if(!timemory_pop_trace_hash_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_pop_trace_hash_jump = nullptr;
        }

        *(void**) (&timemory_trace_init_jump) = dlsym(libhandle, "timemory_trace_init");

        if(!timemory_trace_init_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_trace_init_jump = nullptr;
        }

        *(void**) (&timemory_trace_finalize_jump) =
            dlsym(libhandle, "timemory_trace_finalize");

        if(!timemory_trace_finalize_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_trace_finalize_jump = nullptr;
        }

        *(void**) (&timemory_trace_set_env_jump) =
            dlsym(libhandle, "timemory_trace_set_env");

        if(!timemory_trace_set_env_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_trace_set_env_jump = nullptr;
        }

        *(void**) (&timemory_trace_set_mpi_jump) =
            dlsym(libhandle, "timemory_trace_set_mpi");

        if(!timemory_trace_set_mpi_jump)
        {
            fprintf(stderr, "%s\n", dlerror());
            timemory_trace_set_mpi_jump = nullptr;
        }

        dlclose(libhandle);
    }
};

//--------------------------------------------------------------------------------------//

std::unique_ptr<jump>&
get_jump()
{
    PRINT_HERE("%s", "");
    static std::unique_ptr<jump> obj = std::make_unique<jump>(
        tim::get_env<std::string>("TIMEMORY_JUMP_LIBRARY", "libtimemory.so"));
    return obj;
}

//--------------------------------------------------------------------------------------//
//
//      timemory symbols
//
//--------------------------------------------------------------------------------------//
extern "C"
{
    void timemory_push_components(const char* name)
    {
        (*get_jump()->timemory_push_components_jump)(name);
    }

    void timemory_pop_components(void) { (*get_jump()->timemory_pop_components_jump)(); }

    void timemory_push_region(const char* name)
    {
        (*get_jump()->timemory_push_region_jump)(name);
    }

    void timemory_pop_region(const char* name)
    {
        (*get_jump()->timemory_pop_region_jump)(name);
    }

    void timemory_add_hash_id(uint64_t hash, const char* name)
    {
        (*get_jump()->timemory_add_hash_id_jump)(hash, name);
    }

    void timemory_push_trace(const char* name)
    {
        (*get_jump()->timemory_push_trace_jump)(name);
    }

    void timemory_pop_trace(const char* name)
    {
        (*get_jump()->timemory_pop_trace_jump)(name);
    }

    void timemory_push_trace_hash(uint64_t hash)
    {
        (*get_jump()->timemory_push_trace_hash_jump)(hash);
    }

    void timemory_pop_trace_hash(uint64_t hash)
    {
        (*get_jump()->timemory_pop_trace_hash_jump)(hash);
    }

    void timemory_trace_init(const char* a, bool b, const char* c)
    {
        (*get_jump()->timemory_trace_init_jump)(a, b, c);
    }

    void timemory_trace_finalize(void) { (*get_jump()->timemory_trace_finalize_jump)(); }

    void timemory_trace_set_env(const char* a, const char* b)
    {
        (*get_jump()->timemory_trace_set_env_jump)(a, b);
    }

    void timemory_trace_set_mpi(bool a, bool b)
    {
        (*get_jump()->timemory_trace_set_mpi_jump)(a, b);
    }
}

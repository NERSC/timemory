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

#include "timemory/library.h"
#include "timemory/timemory.hpp"
//
#include "timemory/components/gotcha/mpip.hpp"

#include <memory>
#include <set>
#include <unordered_map>

using namespace tim::component;

using api_t         = tim::api::native_tag;
using mpi_toolset_t = tim::component_list<user_mpip_bundle>;
using mpip_handle_t = mpip_handle<mpi_toolset_t, api_t>;
uint64_t global_id  = 0;

extern "C"
{
    void timemory_mpip_library_ctor() {}

    uint64_t init_timemory_mpip_tools()
    {
        // provide environment variable for enabling/disabling
        if(tim::get_env<bool>("ENABLE_TIMEMORY_MPIP", true))
        {
            configure_mpip<mpi_toolset_t, api_t>();
            user_mpip_bundle::global_init(nullptr);
            return activate_mpip<mpi_toolset_t, api_t>();
        }
        else
        {
            return 0;
        }
    }

    uint64_t stop_timemory_mpip_tools(uint64_t id)
    {
        return deactivate_mpip<mpi_toolset_t, api_t>(id);
    }

    void register_timemory_mpip() { global_id = init_timemory_mpip_tools(); }
    void deregister_timemory_mpip() { global_id = stop_timemory_mpip_tools(global_id); }

    // Below are for FORTRAN codes
    void     timemory_mpip_library_ctor_() {}
    uint64_t init_timemory_mpip_tools_() { return init_timemory_mpip_tools(); }
    uint64_t stop_timemory_mpip_tools_(uint64_t id)
    {
        return stop_timemory_mpip_tools(id);
    }
    void register_timemory_mpip_() { register_timemory_mpip(); }
    void deregister_timemory_mpip_() { deregister_timemory_mpip(); }

}  // extern "C"

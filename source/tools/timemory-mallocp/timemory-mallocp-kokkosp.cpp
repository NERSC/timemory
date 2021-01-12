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

#if !defined(TIMEMORY_SOURCE)
#    define TIMEMORY_SOURCE
#endif

#include "timemory/api/kokkosp.hpp"
#include "timemory/timemory.hpp"
#include "timemory/tools/timemory-mallocp.h"

//--------------------------------------------------------------------------------------//

namespace
{
static const std::string kokkos_banner =
    "#---------------------------------------------------------------------------#";

//--------------------------------------------------------------------------------------//

bool
configure_environment()
{
    tim::set_env("KOKKOS_PROFILE_LIBRARY", "libtimemory-mallocp-kokkosp.so", 0);
    return true;
}

static auto env_configured = (configure_environment(), true);

}  // namespace

//--------------------------------------------------------------------------------------//

extern "C"
{
    void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                              const uint32_t devInfoCount, void* deviceInfo)
    {
        tim::consume_parameters(devInfoCount, deviceInfo);

        tim::set_env("TIMEMORY_TIME_OUTPUT", "ON", 0);
        tim::set_env("TIMEMORY_COUT_OUTPUT", "OFF", 0);
        tim::set_env("TIMEMORY_ADD_SECONDARY", "OFF", 0);

        printf("%s\n", kokkos_banner.c_str());
        printf("# KokkosP: timemory Connector (sequence is %d, version: %llu)\n", loadSeq,
               (unsigned long long) interfaceVer);
        printf("%s\n\n", kokkos_banner.c_str());

        if(tim::settings::output_path() == "timemory-output")
        {
            tim::timemory_init("mallocp-kokkosp");
        }
        assert(env_configured);

        timemory_register_mallocp();
    }

    void kokkosp_finalize_library()
    {
        printf("\n%s\n", kokkos_banner.c_str());
        printf("KokkosP: Finalization of timemory Connector. Complete.\n");
        printf("%s\n\n", kokkos_banner.c_str());

        timemory_deregister_mallocp();

        tim::timemory_finalize();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_for(const char* name, uint32_t, uint64_t*)
    {
        timemory_push_mallocp(name);
    }

    void kokkosp_end_parallel_for(uint64_t kernid) { timemory_pop_mallocp(""); }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_reduce(const char* name, uint32_t, uint64_t*)
    {
        timemory_push_mallocp(name);
    }

    void kokkosp_end_parallel_reduce(uint64_t kernid) { timemory_pop_mallocp(""); }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_scan(const char* name, uint32_t, uint64_t*)
    {
        timemory_push_mallocp(name);
    }

    void kokkosp_end_parallel_scan(uint64_t) { timemory_pop_mallocp(""); }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_fence(const char* name, uint32_t, uint64_t*)
    {
        timemory_push_mallocp(name);
    }

    void kokkosp_end_fence(uint64_t) { timemory_pop_mallocp(""); }

    //----------------------------------------------------------------------------------//

    void kokkosp_push_profile_region(const char* name) { timemory_push_mallocp(name); }

    void kokkosp_pop_profile_region() { timemory_pop_mallocp(""); }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_deep_copy(SpaceHandle, const char* dst_name, const void*,
                                 SpaceHandle, const char* src_name, const void*, uint64_t)
    {
        timemory_push_mallocp(src_name);
        timemory_push_mallocp(dst_name);
    }

    void kokkosp_end_deep_copy()
    {
        timemory_pop_mallocp("");
        timemory_pop_mallocp("");
    }

    //----------------------------------------------------------------------------------//
}

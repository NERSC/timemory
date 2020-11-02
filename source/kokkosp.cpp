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

namespace kokkosp = tim::kokkosp;

//--------------------------------------------------------------------------------------//

namespace tim
{
template <>
inline auto
invoke_preinit<kokkosp::memory_tracker>(long)
{
    kokkosp::memory_tracker::label()       = "kokkos_memory";
    kokkosp::memory_tracker::description() = "Kokkos Memory tracker";
}
}  // namespace tim

//--------------------------------------------------------------------------------------//

namespace
{
static std::string kokkos_banner =
    "#---------------------------------------------------------------------------#";

//--------------------------------------------------------------------------------------//

bool
configure_environment()
{
    tim::set_env("KOKKOS_PROFILE_LIBRARY", "libtimemory.so", 0);
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
            // timemory_init is expecting some args so generate some
            std::array<char*, 1> cstr = { { strdup("kokkosp") } };
            tim::timemory_init(1, cstr.data());
            free(cstr[0]);
        }
        assert(env_configured);

        // search unique and fallback environment variables
        kokkosp::kokkos_bundle::global_init();

        // add at least one
        if(kokkosp::kokkos_bundle::bundle_size() == 0)
            kokkosp::kokkos_bundle::configure<tim::component::wall_clock>();
    }

    void kokkosp_finalize_library()
    {
        printf("\n%s\n", kokkos_banner.c_str());
        printf("KokkosP: Finalization of timemory Connector. Complete.\n");
        printf("%s\n\n", kokkos_banner.c_str());

        kokkosp::cleanup();

        tim::timemory_finalize();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname = TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
        *kernid    = kokkosp::get_unique_id();
        kokkosp::create_profiler<kokkosp::kokkos_bundle>(pname, *kernid);
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(*kernid);
    }

    void kokkosp_end_parallel_for(uint64_t kernid)
    {
        kokkosp::stop_profiler<kokkosp::kokkos_bundle>(kernid);
        kokkosp::destroy_profiler<kokkosp::kokkos_bundle>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname = TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
        *kernid    = kokkosp::get_unique_id();
        kokkosp::create_profiler<kokkosp::kokkos_bundle>(pname, *kernid);
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(*kernid);
    }

    void kokkosp_end_parallel_reduce(uint64_t kernid)
    {
        kokkosp::stop_profiler<kokkosp::kokkos_bundle>(kernid);
        kokkosp::destroy_profiler<kokkosp::kokkos_bundle>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname = TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
        *kernid    = kokkosp::get_unique_id();
        kokkosp::create_profiler<kokkosp::kokkos_bundle>(pname, *kernid);
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(*kernid);
    }

    void kokkosp_end_parallel_scan(uint64_t kernid)
    {
        kokkosp::stop_profiler<kokkosp::kokkos_bundle>(kernid);
        kokkosp::destroy_profiler<kokkosp::kokkos_bundle>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_push_profile_region(const char* name)
    {
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().push_back(
            kokkosp::profiler_t<kokkosp::kokkos_bundle>(name));
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().back().start();
    }

    void kokkosp_pop_profile_region()
    {
        if(kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().empty())
            return;
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().back().stop();
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_create_profile_section(const char* name, uint32_t* secid)
    {
        *secid = kokkosp::get_unique_id();
        auto pname =
            TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "section", *secid), name);
        kokkosp::create_profiler<kokkosp::kokkos_bundle>(pname, *secid);
    }

    void kokkosp_destroy_profile_section(uint32_t secid)
    {
        kokkosp::destroy_profiler<kokkosp::kokkos_bundle>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_start_profile_section(uint32_t secid)
    {
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(secid);
    }

    void kokkosp_stop_profile_section(uint32_t secid)
    {
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_allocate_data(const SpaceHandle space, const char* label,
                               const void* const ptr, const uint64_t size)
    {
        if(!tim::settings::enabled())
            return;
        auto itr = kokkosp::get_profiler_memory_map<kokkosp::kokkos_bundle>().insert(
            { ptr, kokkosp::profiler_t<kokkosp::kokkos_bundle>(
                       TIMEMORY_JOIN('/', "kokkos/allocate", space.name, label)) });
        if(itr.second)
        {
            itr.first->second.audit(space, label, ptr, size);
            itr.first->second.store(std::plus<int64_t>{}, size);
            itr.first->second.start();
        }
    }

    void kokkosp_deallocate_data(const SpaceHandle space, const char* label,
                                 const void* const ptr, const uint64_t size)
    {
        auto itr = kokkosp::get_profiler_memory_map<kokkosp::kokkos_bundle>().find(ptr);
        if(itr != kokkosp::get_profiler_memory_map<kokkosp::kokkos_bundle>().end())
        {
            itr->second.stop();
            itr->second.store(std::minus<int64_t>{}, 0);
            itr->second.audit(space, label, ptr, size);
            kokkosp::get_profiler_memory_map<kokkosp::kokkos_bundle>().erase(itr);
        }
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_deep_copy(SpaceHandle dst_handle, const char* dst_name,
                                 const void* dst_ptr, SpaceHandle src_handle,
                                 const char* src_name, const void* src_ptr, uint64_t size)
    {
        if(!tim::settings::enabled())
            return;
        auto name = TIMEMORY_JOIN('/', "kokkos/deep_copy",
                                  TIMEMORY_JOIN('=', dst_handle.name, dst_name),
                                  TIMEMORY_JOIN('=', src_handle.name, src_name));
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().emplace_back(
            kokkosp::profiler_t<kokkosp::kokkos_bundle>(name));
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().back().audit(
            dst_handle, dst_name, dst_ptr, src_handle, src_name, src_ptr, size);
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().back().store(
            std::plus<int64_t>{}, size);
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().back().start();
    }

    void kokkosp_end_deep_copy()
    {
        if(kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().empty())
            return;
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().back().stop();
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().back().store(
            std::minus<int64_t>{}, 0);
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().pop_back();
    }

    //----------------------------------------------------------------------------------//
}

//--------------------------------------------------------------------------------------//

TIMEMORY_INITIALIZE_STORAGE(kokkosp::memory_tracker)

//--------------------------------------------------------------------------------------//

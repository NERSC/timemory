
#include <cassert>
#include <cstdlib>
#include <execinfo.h>
#include <iostream>
#include <string>
#include <vector>

#include "timemory/runtime/configure.hpp"
#include "timemory/timemory.hpp"

#include "kp_timemory.hpp"

#if __cplusplus > 201402L  // C++17
#    define if_constexpr if constexpr
#else
#    define if_constexpr if
#endif

static std::string spacer =
    "#---------------------------------------------------------------------------#";

// this just differentiates Kokkos from other user_bundles
struct KokkosProfiler;
using KokkosUserBundle = tim::component::user_bundle<0, KokkosProfiler>;

// memory trackers
struct KokkosMemory
{};
using KokkosMemoryTracker = data_tracker<int64_t, KokkosMemory>;

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, KokkosMemoryTracker, std::true_type);
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, KokkosMemoryTracker, std::true_type);

// set up the configuration of tools
using profile_entry_t =
    tim::component_hybrid<tim::component_tuple<KokkosUserBundle>,
                          tim::component_list<cpu_util, KokkosMemoryTracker>>;

// various data structurs used
using section_entry_t = std::tuple<std::string, profile_entry_t>;
using profile_stack_t = std::vector<profile_entry_t>;
using profile_map_t   = std::unordered_map<uint64_t, profile_entry_t>;
using section_map_t   = std::unordered_map<uint64_t, section_entry_t>;
using pointer_map_t   = std::map<const void* const, profile_entry_t>;

//--------------------------------------------------------------------------------------//

static uint64_t
get_unique_id()
{
    static thread_local uint64_t _instance = 0;
    return _instance++;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
Tp&
get_tl_static()
{
    static thread_local Tp _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

static profile_map_t&
get_profile_map()
{
    return get_tl_static<profile_map_t>();
}

//--------------------------------------------------------------------------------------//

static pointer_map_t&
get_pointer_map()
{
    return get_tl_static<pointer_map_t>();
}

//--------------------------------------------------------------------------------------//

static profile_stack_t&
get_profile_stack()
{
    return get_tl_static<profile_stack_t>();
}

//--------------------------------------------------------------------------------------//

static void
create_profiler(const std::string& pname, uint64_t kernid)
{
    get_profile_map().insert(std::make_pair(kernid, profile_entry_t(pname, true)));
}

//--------------------------------------------------------------------------------------//

static void
destroy_profiler(uint64_t kernid)
{
    if(get_profile_map().find(kernid) != get_profile_map().end())
        get_profile_map().erase(kernid);
}

//--------------------------------------------------------------------------------------//

static void
start_profiler(uint64_t kernid)
{
    if(get_profile_map().find(kernid) != get_profile_map().end())
        get_profile_map().at(kernid).start();
}

//--------------------------------------------------------------------------------------//

static void
stop_profiler(uint64_t kernid)
{
    if(get_profile_map().find(kernid) != get_profile_map().end())
        get_profile_map().at(kernid).stop();
}

//======================================================================================//
//
//      Kokkos symbols
//
//======================================================================================//

extern "C" void
kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                     const uint32_t devInfoCount, void* deviceInfo)
{
    printf("%s\n", spacer.c_str());
    printf("# KokkosP: timemory Connector (sequence is %d, version: %llu)\n", loadSeq,
           (unsigned long long) interfaceVer);
    printf("%s\n\n", spacer.c_str());

    KokkosMemoryTracker::label()       = "kokkos_memory";
    KokkosMemoryTracker::description() = "Kokkos Memory tracker";

    // if using roofline, we want to suppress time_output which
    // would result in the second pass (required by roofline) to end
    // up in a different directory
    bool use_roofline            = tim::get_env<bool>("KOKKOS_ROOFLINE", false);
    auto papi_events             = tim::get_env<std::string>("PAPI_EVENTS", "");
    tim::settings::papi_events() = papi_events;

    // timemory_init is expecting some args so generate some
    auto  dir  = TIMEMORY_JOIN("_", loadSeq, interfaceVer, devInfoCount);
    char* cstr = strdup(dir.c_str());
    tim::timemory_init(1, &cstr);
    free(cstr);

    std::string default_components =
        (use_roofline) ? "gpu_roofline_flops, cpu_roofline" : "wall_clock, peak_rss";

    if(!tim::settings::papi_events().empty() && !use_roofline)
        default_components += ", papi_vector";

    // check environment variables "KOKKOS_TIMEMORY_COMPONENTS" and
    // "KOKKOS_PROFILE_COMPONENTS"
    tim::env::configure<KokkosUserBundle>(
        "KOKKOS_TIMEMORY_COMPONENTS",
        tim::get_env("KOKKOS_PROFILE_COMPONENTS", default_components));
}

extern "C" void
kokkosp_finalize_library()
{
    printf("\n%s\n", spacer.c_str());
    printf("KokkosP: Finalization of timemory Connector. Complete.\n");
    printf("%s\n\n", spacer.c_str());

    for(auto& itr : get_profile_map())
        itr.second.stop();
    get_profile_map().clear();

    tim::timemory_finalize();
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid)
{
    auto pname = TIMEMORY_JOIN("/", "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
    *kernid    = get_unique_id();
    create_profiler(pname, *kernid);
    start_profiler(*kernid);
}

extern "C" void
kokkosp_end_parallel_for(uint64_t kernid)
{
    stop_profiler(kernid);
    destroy_profiler(kernid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid)
{
    auto pname = TIMEMORY_JOIN("/", "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
    *kernid    = get_unique_id();
    create_profiler(pname, *kernid);
    start_profiler(*kernid);
}

extern "C" void
kokkosp_end_parallel_reduce(uint64_t kernid)
{
    stop_profiler(kernid);
    destroy_profiler(kernid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid)
{
    auto pname = TIMEMORY_JOIN("/", "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
    *kernid    = get_unique_id();
    create_profiler(pname, *kernid);
    start_profiler(*kernid);
}

extern "C" void
kokkosp_end_parallel_scan(uint64_t kernid)
{
    stop_profiler(kernid);
    destroy_profiler(kernid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_push_profile_region(const char* name)
{
    get_profile_stack().push_back(profile_entry_t(name, true));
    get_profile_stack().back().start();
}

extern "C" void
kokkosp_pop_profile_region()
{
    if(get_profile_stack().empty())
        return;
    get_profile_stack().back().stop();
    get_profile_stack().pop_back();
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_create_profile_section(const char* name, uint32_t* secid)
{
    *secid     = get_unique_id();
    auto pname = TIMEMORY_JOIN("/", "kokkos", TIMEMORY_JOIN("", "section", secid), name);
    create_profiler(pname, *secid);
}

extern "C" void
kokkosp_destroy_profile_section(uint32_t secid)
{
    destroy_profiler(secid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_start_profile_section(uint32_t secid)
{
    start_profiler(secid);
}

extern "C" void
kokkosp_stop_profile_section(uint32_t secid)
{
    start_profiler(secid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_allocate_data(const SpaceHandle space, const char* label, const void* const ptr,
                      const uint64_t size)
{
    get_pointer_map().insert({ ptr, profile_entry_t(TIMEMORY_JOIN("/", "kokkos/allocate",
                                                                  space.name, label)) });
    get_pointer_map()[ptr].get_list().init<KokkosMemoryTracker>();
    get_pointer_map()[ptr].start();
    get_pointer_map()[ptr].store(std::plus<int64_t>{}, size);
}

extern "C" void
kokkosp_deallocate_data(const SpaceHandle, const char*, const void* const ptr,
                        const uint64_t)
{
    auto itr = get_pointer_map().find(ptr);
    if(itr != get_pointer_map().end())
    {
        itr->second.stop();
        get_pointer_map().erase(itr);
    }
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_begin_deep_copy(SpaceHandle dst_handle, const char* dst_name, const void*,
                        SpaceHandle src_handle, const char* src_name, const void*,
                        uint64_t size)
{
    auto name = TIMEMORY_JOIN("/", "kokkos/deep_copy",
                              TIMEMORY_JOIN('=', dst_handle.name, dst_name),
                              TIMEMORY_JOIN('=', src_handle.name, src_name));
    get_profile_stack().push_back(profile_entry_t(name, true));
    get_profile_stack().back().get_list().init<KokkosMemoryTracker>();
    get_profile_stack().back().start();
    get_profile_stack().back().store(std::plus<int64_t>{}, size);
}

extern "C" void
kokkosp_end_deep_copy()
{
    if(get_profile_stack().empty())
        return;
    get_profile_stack().back().stop();
    get_profile_stack().pop_back();
}

//--------------------------------------------------------------------------------------//


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
    bool use_roofline = tim::get_env<bool>("KOKKOS_ROOFLINE", false);
    // store this for later
    std::string folder = tim::settings::output_path();

    auto papi_events              = tim::get_env<std::string>("PAPI_EVENTS", "");
    tim::settings::time_output()  = false;  // output in sub-dir with time
    tim::settings::papi_events()  = papi_events;
    tim::settings::auto_output()  = true;   // print when destructing
    tim::settings::cout_output()  = true;   // print to stdout
    tim::settings::text_output()  = true;   // print text files
    tim::settings::json_output()  = true;   // print to json
    tim::settings::banner()       = true;   // suppress banner
    tim::settings::mpi_finalize() = false;  // don't finalize MPI during timemory_finalize

    // timemory_init is expecting some args so generate some
    std::stringstream ss;
    ss << loadSeq << "_" << interfaceVer << "_" << devInfoCount;
    auto cstr = const_cast<char*>(ss.str().c_str());
    tim::timemory_init(1, &cstr, "", "");
    // over-ride the output path set by timemory_init to the
    // original setting
    tim::settings::output_path() = folder;

    // the environment variable to configure components
    std::string env_var = "KOKKOS_TIMEMORY_COMPONENTS";
    // if roofline is enabled, provide nothing by default
    // if roofline is not enabled, profile wall-clock by default
    std::string components = (use_roofline) ? "" : "wall_clock;peak_rss";
    // query the environment
    auto env_result = tim::get_env(env_var, components);
    std::transform(env_result.begin(), env_result.end(), env_result.begin(),
                   [](unsigned char c) -> unsigned char { return std::tolower(c); });
    // if a roofline component is not set in the environment, then add both the
    // cpu and gpu roofline
    if(use_roofline && env_result.find("roofline") == std::string::npos)
        env_result = TIMEMORY_JOIN(";", env_result, "gpu_roofline", "cpu_roofline");
    // configure the bundle to use these components
    tim::configure<KokkosUserBundle>(tim::enumerate_components(tim::delimit(env_result)));

#if defined(TIMEMORY_USE_GOTCHA)
    //
    //  This is not really a general tool, especially not the GOTCHA that
    //  intercepts the rand and srand. It is more of a demonstration
    //  of how to use the gotcha interface
    //
    auto gotcha_lvl = tim::get_env("KOKKOS_GOTCHA_MODE", 0);
    if(gotcha_lvl == 1 || gotcha_lvl > 2)
    {
        // when explicitly configured here, the gotcha wrappers are immediately generated
        TIMEMORY_C_GOTCHA(rand_gotcha_t, 0, srand);
        TIMEMORY_C_GOTCHA(rand_gotcha_t, 1, rand);
    }
    if(gotcha_lvl >= 2)
    {
        // for malloc/free specifically, we make sure the default activation
        // of the gotcha is off. Wrapping malloc/free has the potential to include
        // a limited number of malloc/free calls within the timemory library itself
        misc_gotcha_t::get_default_ready() = false;
        // when the initializer is overloaded, the gotcha is fully scoped
        // via reference counting. When no components containing this gotcha
        // is alive, the gotcha is disabled and all function calls use the original
        // wrappee
        misc_gotcha_t::get_initializer() = []() {
            TIMEMORY_C_GOTCHA(misc_gotcha_t, 0, malloc);
            TIMEMORY_C_GOTCHA(misc_gotcha_t, 1, free);
        };
    }
#endif
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

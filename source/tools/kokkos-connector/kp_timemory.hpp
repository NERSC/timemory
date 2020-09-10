
#pragma once

#include "timemory/timemory.hpp"

#include <cstdint>

//--------------------------------------------------------------------------------------//

using namespace tim::component;

using empty_t = tim::component_tuple<>;

//--------------------------------------------------------------------------------------//

struct SpaceHandle
{
    char name[64];
};

struct KokkosPDeviceInfo
{
    uint32_t deviceID;
};

enum Space
{
    SPACE_HOST,
    SPACE_CUDA
};

enum
{
    NSPACES = 2
};

inline Space
get_space(SpaceHandle const& handle)
{
    switch(handle.name[0])
    {
        case 'H': return SPACE_HOST;
        case 'C': return SPACE_CUDA;
    }
    abort();
    return SPACE_HOST;
}

inline const char*
get_space_name(int space)
{
    switch(space)
    {
        case SPACE_HOST: return "HOST";
        case SPACE_CUDA: return "CUDA";
    }
    abort();
    return nullptr;
}

//--------------------------------------------------------------------------------------//

// memory trackers
struct KokkosMemory
{};
using KokkosMemoryTracker = data_tracker<int64_t, KokkosMemory>;

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, KokkosMemoryTracker, std::true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, KokkosMemoryTracker, std::true_type)

using memory_entry_t =
    tim::component_tuple<wall_clock, user_global_bundle, KokkosMemoryTracker>;
using memory_map_t   = std::map<const void* const, memory_entry_t>;
using memory_stack_t = std::vector<memory_entry_t>;

//--------------------------------------------------------------------------------------//

inline uint64_t
get_unique_id()
{
    static thread_local uint64_t _instance = 0;
    return _instance++;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline Tp&
get_tl_static()
{
    static thread_local Tp _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

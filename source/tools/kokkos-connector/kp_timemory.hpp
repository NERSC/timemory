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

#pragma once

#if defined(FORCE_HIDDEN_VISIBILITY) && !_MSC_VER
#    define TIMEMORY_INTERNAL __attribute__((visibility("internal")))
#    define TIMEMORY_EXTERNAL __attribute__((visibility("default")))
#    define TIMEMORY_VISIBILITY(...)
#endif

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


#include "kp_timemory.hpp"

#include "timemory/operations/types/storage_initializer.hpp"
#include "timemory/timemory.hpp"

#include <cstdint>

//--------------------------------------------------------------------------------------//

namespace tim
{
template <>
inline auto
invoke_preinit<KokkosMemoryTracker>(long)
{
    KokkosMemoryTracker::label()       = "kokkos_memory";
    KokkosMemoryTracker::description() = "Kokkos Memory tracker";
}
}  // namespace tim

//--------------------------------------------------------------------------------------//

inline memory_map_t&
get_memory_map()
{
    return get_tl_static<memory_map_t>();
}

//--------------------------------------------------------------------------------------//

inline memory_stack_t&
get_memory_stack()
{
    return get_tl_static<memory_stack_t>();
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_allocate_data(const SpaceHandle space, const char* label, const void* const ptr,
                      const uint64_t size)
{
    if(!tim::settings::enabled())
        return;
    auto itr = get_memory_map().insert(
        { ptr,
          memory_entry_t(TIMEMORY_JOIN('/', "kokkos/allocate", space.name, label)) });
    if(itr.second)
    {
        itr.first->second.audit(space, label, ptr, size);
        itr.first->second.store(std::plus<int64_t>{}, size);
        itr.first->second.start();
    }
}

extern "C" void
kokkosp_deallocate_data(const SpaceHandle space, const char* label, const void* const ptr,
                        const uint64_t size)
{
    auto itr = get_memory_map().find(ptr);
    if(itr != get_memory_map().end())
    {
        itr->second.stop();
        itr->second.store(std::minus<int64_t>{}, 0);
        itr->second.audit(space, label, ptr, size);
        get_memory_map().erase(itr);
    }
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_begin_deep_copy(SpaceHandle dst_handle, const char* dst_name, const void* dst_ptr,
                        SpaceHandle src_handle, const char* src_name, const void* src_ptr,
                        uint64_t size)
{
    if(!tim::settings::enabled())
        return;
    auto name = TIMEMORY_JOIN('/', "kokkos/deep_copy",
                              TIMEMORY_JOIN('=', dst_handle.name, dst_name),
                              TIMEMORY_JOIN('=', src_handle.name, src_name));
    get_memory_stack().emplace_back(memory_entry_t(name, true));
    get_memory_stack().back().audit(dst_handle, dst_name, dst_ptr, src_handle, src_name,
                                    src_ptr, size);
    get_memory_stack().back().store(std::plus<int64_t>{}, size);
    get_memory_stack().back().start();
}

extern "C" void
kokkosp_end_deep_copy()
{
    if(get_memory_stack().empty())
        return;
    get_memory_stack().back().stop();
    get_memory_stack().back().store(std::minus<int64_t>{}, 0);
    get_memory_stack().pop_back();
}

//--------------------------------------------------------------------------------------//

TIMEMORY_STORAGE_INITIALIZER(KokkosMemoryTracker, kokkos_memory_tracker)

//--------------------------------------------------------------------------------------//

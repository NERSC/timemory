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

#include "timemory/defines.h"

#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_CORE_EXTERN)
#    define TIMEMORY_USE_CORE_EXTERN
#endif

#include "timemory/macros/os.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <thread>

#if defined(_LINUX)
#    include <fstream>
#    include <pthread.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

#if defined(_MACOS)
#    include <sys/sysctl.h>
#endif

#if defined(_WINDOWS)
#    include <processthreadsapi.h>
#endif

namespace tim
{
namespace threading
{
//
//--------------------------------------------------------------------------------------//
//
using id_t            = std::thread::id;
using native_handle_t = std::thread::native_handle_type;
//
//--------------------------------------------------------------------------------------//
//
static inline int64_t
get_id()
{
    static std::atomic<int64_t> _global_counter(0);
    static thread_local int64_t _this_id = _global_counter++;
    return _this_id;
}
//
//--------------------------------------------------------------------------------------//
//
static inline id_t
get_tid()
{
    return std::this_thread::get_id();
}
//
//--------------------------------------------------------------------------------------//
//
static inline id_t
get_master_tid()
{
    static id_t _instance = get_tid();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
static inline bool
is_master_thread()
{
    return (get_tid() == get_master_tid());
}
//
//--------------------------------------------------------------------------------------//
//
inline uint32_t
get_sys_tid()
{
#if defined(_LINUX)
    return syscall(SYS_gettid);
#elif defined(_WINDOWS)
    return GetCurrentThreadId();
#else
    return static_cast<uint32_t>(get_id());
#endif
}
//
//--------------------------------------------------------------------------------------//
//
struct affinity
{
    using functor_t          = std::function<int64_t(int64_t)>;
    using cpu_affinity_map_t = std::map<int64_t, int64_t>;

    enum MODE
    {
        COMPACT  = 0,
        SCATTER  = 1,
        SPREAD   = 2,
        EXPLICIT = 3
    };

    static auto    hw_concurrency() { return std::thread::hardware_concurrency(); }
    static int64_t hw_physicalcpu();
    static cpu_affinity_map_t& get_affinity_map();
    static MODE&               get_mode();
    static functor_t&          get_algorithm();
    static int64_t             set();
    static int64_t             set(native_handle_t athread);
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace threading
}  // namespace tim

#if !defined(TIMEMORY_CORE_SOURCE) && !defined(TIMEMORY_USE_CORE_EXTERN)
#    include "timemory/backends/threading.cpp"
#endif

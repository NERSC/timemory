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

/** \file backends/threading.hpp
 * \headerfile backends/threading.hpp "timemory/backends/threading.hpp"
 * Defines threading backend functions
 *
 */

#pragma once

#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <thread>

#if defined(_LINUX)
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
    using functor_t = std::function<int64_t(int64_t)>;

    static auto hw_concurrency() { return std::thread::hardware_concurrency(); }

    static auto hw_physicalcpu()
    {
        static int64_t _value = []() -> int64_t {
#if defined(_MACOS)
            int    count;
            size_t count_len = sizeof(count);
            sysctlbyname("hw.physicalcpu", &count, &count_len, nullptr, 0);
            return static_cast<int64_t>(count);
#elif defined(_LINUX)
            std::ifstream ifs("/proc/cpuinfo");
            if(ifs)
            {
                std::set<int64_t> core_ids;
                std::string       line;
                while(true)
                {
                    getline(ifs, line);
                    if(!ifs.good())
                        break;
                    if(line.find("core id") != std::string::npos)
                    {
                        auto cid = from_string<int64_t>(delimit(line, " :,;").back());
                        if(cid >= 0)
                            core_ids.insert(cid);
                    }
                }
                return core_ids.size();
            }
            return hw_concurrency();
#else
            return hw_concurrency();
#endif
        }();
        return _value;
    }

    using cpu_affinity_map_t = std::map<int64_t, int64_t>;

    static auto& get_affinity_map()
    {
        static cpu_affinity_map_t _instance;
        return _instance;
    }

    enum MODE
    {
        COMPACT  = 0,
        SCATTER  = 1,
        SPREAD   = 2,
        EXPLICIT = 3
    };

    static MODE& get_mode()
    {
        static MODE _instance = COMPACT;
        return _instance;
    }

    static functor_t& get_algorithm()
    {
        static functor_t _instance = [](int64_t tid) {
            //
            //  assigns the cpu affinity in a compact sequence
            //
            static functor_t _compact_instance = [](int64_t _tid) -> int64_t {
                static std::atomic<int64_t> _counter(0);
                static thread_local int64_t _this_count = _counter++;
                auto                        proc_itr    = get_affinity_map().find(_tid);
                if(proc_itr == get_affinity_map().end())
                    get_affinity_map()[_tid] = _this_count;
                return get_affinity_map()[_tid];
            };
            //
            //  assigns the cpu affinity in a scattered sequence
            //
            static functor_t _scatter_instance = [](int64_t _tid) -> int64_t {
                static std::atomic<int64_t> _counter(0);
                static thread_local int64_t _this_count = _counter++;
                auto _val     = (_this_count * hw_physicalcpu()) % hw_concurrency();
                auto proc_itr = get_affinity_map().find(_tid);
                if(proc_itr == get_affinity_map().end())
                    get_affinity_map()[_tid] = _val;
                return get_affinity_map()[_tid];
            };
            //
            //  assigns the cpu affinity explicitly
            //
            static functor_t _explicit_instance = [](int64_t _tid) -> int64_t {
                auto proc_itr = get_affinity_map().find(_tid);
                if(proc_itr != get_affinity_map().end())
                    return proc_itr->second;
                return -1;
            };
            //
            //  checks the configured mode and applies the appropriate algorithm
            //
            switch(get_mode())
            {
                case COMPACT: return _compact_instance(tid);
                case SCATTER:
                case SPREAD: return _scatter_instance(tid);
                case EXPLICIT: return _explicit_instance(tid);
            };
            //
            // default to compact algorithm
            //
            return _compact_instance(tid);
        };
        //
        return _instance;
    }

    static int64_t set()
    {
#if defined(_LINUX)
        auto proc_id = get_algorithm()(get_id());
        if(proc_id >= 0)
        {
            cpu_set_t cpuset;
            pthread_t thread = pthread_self();
            CPU_ZERO(&cpuset);
            CPU_SET(proc_id, &cpuset);
            pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        }
        return proc_id;
#else
        return -1;
#endif
    }

    static int64_t set(native_handle_t athread)
    {
#if defined(_LINUX)
        auto      proc_id = get_algorithm()(get_id());
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(proc_id, &cpuset);
        pthread_setaffinity_np(athread, sizeof(cpu_set_t), &cpuset);
        return proc_id;
#else
        consume_parameters(athread);
        return -1;
#endif
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace threading
}  // namespace tim

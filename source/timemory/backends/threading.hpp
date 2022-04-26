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

#include "timemory/defines.h"
#include "timemory/macros/os.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <utility>

#if defined(TIMEMORY_UNIX)
#    include <pthread.h>
#endif

#if defined(TIMEMORY_LINUX)
#    include <fstream>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

#if defined(TIMEMORY_MACOS)
#    include <sys/sysctl.h>
#endif

#if defined(TIMEMORY_WINDOWS)
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
namespace internal
{
inline std::vector<int64_t>&
get_available_ids()
{
    static auto _v = []() {
        auto _tmp = std::vector<int64_t>{};
        _tmp.reserve(TIMEMORY_MAX_THREADS);
        return _tmp;
    }();
    return _v;
}
//
/// add thread ids to this set to avoid them being recycled
/// when the thread is destroyed
inline std::set<int64_t>&
get_reserved_ids()
{
    static auto _v = std::set<int64_t>{ 0 };
    return _v;
}
//
struct recycle_ids
{
    operator bool() const { return value; }

#if defined(TIMEMORY_FORCE_UNIQUE_THREAD_IDS) && TIMEMORY_FORCE_UNIQUE_THREAD_IDS > 0
    // ignore assignments
    recycle_ids& operator=(bool) { return *this; }
#else
    // allow assignments
    recycle_ids& operator=(bool _v)
    {
        value = _v;
        return *this;
    }
#endif

private:
    bool value = false;
};
//
}  // namespace internal
//
//--------------------------------------------------------------------------------------//
//
inline internal::recycle_ids&
recycle_ids()
{
    static auto _v = internal::recycle_ids{};
    return _v;
}
//
inline int64_t
get_id()
{
    static std::atomic<int64_t> _global_counter{ 0 };
    static thread_local auto    _this_id = []() {
        int64_t _id = -1;
        if(recycle_ids() && _global_counter >= TIMEMORY_MAX_THREADS)
        {
            auto_lock_t _lk{ type_mutex<internal::recycle_ids>() };
            auto&       _avail = internal::get_available_ids();
            if(!_avail.empty())
            {
                // always grab from front
                _id = _avail.at(0);
                for(size_t i = 1; i < _avail.size(); ++i)
                    _avail[i - 1] = _avail[i];
                _avail.pop_back();
            }
        }

        if(_id < 0)
            _id = _global_counter++;

        return std::make_pair(_id, scope::destructor{ [_id]() {
                                  auto_lock_t _lk{ type_mutex<internal::recycle_ids>() };
                                  if(internal::get_reserved_ids().count(_id) == 0)
                                      internal::get_available_ids().emplace_back(_id);
                              } });
    }();
    return _this_id.first;
}
//
inline auto
add_reserved_id(int64_t _v = get_id())
{
    auto_lock_t _lk{ type_mutex<internal::recycle_ids>() };
    if(_v > 0)
        internal::get_reserved_ids().emplace(_v);
    return internal::get_reserved_ids();
}
//
inline auto
erase_reserved_id(int64_t _v = get_id())
{
    auto_lock_t _lk{ type_mutex<internal::recycle_ids>() };
    if(_v > 0)
        internal::get_reserved_ids().erase(_v);
    return internal::get_reserved_ids();
}
//
//--------------------------------------------------------------------------------------//
//
inline id_t
get_tid()
{
    return std::this_thread::get_id();
}
//
//--------------------------------------------------------------------------------------//
//
inline id_t
get_main_tid()
{
    static id_t _instance = get_tid();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
is_master_thread()
{
    return (get_tid() == get_main_tid());
}
//
//--------------------------------------------------------------------------------------//
//
inline uint32_t
get_sys_tid()
{
#if defined(TIMEMORY_LINUX)
    return syscall(SYS_gettid);
#elif defined(TIMEMORY_WINDOWS)
    return GetCurrentThreadId();
#else
    return static_cast<uint32_t>(get_id());
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline void
set_thread_name(const char* _name)
{
#if defined(TIMEMORY_UNIX)
    auto _length_error = [_name]() {
        fprintf(stderr,
                "[threading::set_thread_name] the length of '%s' + null-terminator (%i) "
                "exceeds the max allowed limit (usually 16)\n",
                _name, (int) (strlen(_name) + 1));
    };

    constexpr size_t _size = 16;
    size_t           _n    = std::min<size_t>(_size - 1, strlen(_name));
    char             _buff[_size];
    memset(_buff, '\0', _size * sizeof(char));
    memcpy(_buff, _name, _n * sizeof(char));
#endif

#if defined(TIMEMORY_LINUX)
    auto _err = pthread_setname_np(pthread_self(), _buff);
    if(_err == ERANGE)
        _length_error();
#elif defined(TIMEMORY_MACOS)
    auto _err = pthread_setname_np(_buff);
    if(_err == ERANGE)
        _length_error();
#elif defined(TIMEMORY_WINDOWS)
    auto     _n     = strlen(_name);
    wchar_t* _wname = new wchar_t[_n + 1];
    for(size_t i = 0; i < _n; ++i)
        _wname[i] = _name[i];
    _wname[_n] = '\0';
    SetThreadDescription(GetCurrentThread(), _wname);
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline std::string
get_thread_name()
{
#if defined(TIMEMORY_UNIX)
    constexpr size_t _buff_len = 32;
    char             _buff[_buff_len];
    memset(_buff, '\0', _buff_len * sizeof(char));
    auto _err = pthread_getname_np(pthread_self(), _buff, _buff_len);
    if(_err == ERANGE)
    {
        fprintf(stderr,
                "[threading::get_thread_name] buffer for pthread_getname_np was not "
                "large enough: %zu\n",
                _buff_len);
    }
    return std::string{ _buff };
#elif defined(TIMEMORY_WINDOWS)
    wchar_t*    data  = nullptr;
    std::string _name = {};
    HRESULT     hr    = GetThreadDescription(GetCurrentThread(), &data);
    if(SUCCEEDED(hr))
    {
        constexpr size_t _buff_len = 64;
        char             _buff[_buff_len];
        _name.resize(_buff_len);
        for(size_t i = 0; i < _buff_len; ++i)
        {
            _name[i] = data[i];
            if(data[i] == '\0')
                break;
        }
        LocalFree(data);
        _name.shrink_to_fit();
    }
    return _name;
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
#if defined(TIMEMORY_MACOS)
            int    count;
            size_t count_len = sizeof(count);
            sysctlbyname("hw.physicalcpu", &count, &count_len, nullptr, 0);
            return static_cast<int64_t>(count);
#elif defined(TIMEMORY_LINUX)
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
#if defined(TIMEMORY_LINUX)
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
#if defined(TIMEMORY_LINUX)
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

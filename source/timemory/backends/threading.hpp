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

#include <cstdint>
#include <thread>

#if defined(_LINUX)
#    include <pthread.h>
#endif

namespace tim
{
namespace threading
{
using native_handle_t = std::thread::native_handle_type;

inline int64_t
get_id()
{
    static std::atomic<int64_t> _global_counter(0);
    static thread_local int64_t _this_id = _global_counter++;
    return _this_id;
}

struct affinity
{
    using functor_t = std::function<int64_t(int64_t)>;

    static functor_t& get_algorithm()
    {
        static functor_t _instance = [](int64_t tid) {
            consume_parameters(tid);
            static std::atomic<int64_t> _counter(0);
            return _counter++;
        };
        return _instance;
    }

    static int64_t set()
    {
#if defined(_LINUX)
        auto      proc_id = get_algorithm()(get_id());
        cpu_set_t cpuset;
        pthread_t thread = pthread_self();
        CPU_ZERO(&cpuset);
        CPU_SET(proc_id, &cpuset);
        pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
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
}  // namespace threading
}  // namespace tim

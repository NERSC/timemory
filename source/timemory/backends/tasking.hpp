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

#include "timemory/api.hpp"
#include "timemory/utility/types.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

#if defined(TIMEMORY_USE_PTL)
#    include <PTL/TaskGroup.hh>
#    include <PTL/ThreadPool.hh>
#endif

namespace tim
{
namespace tasking
{
//
using mutex_t   = std::mutex;
using lock_t    = std::unique_lock<mutex_t>;
using condvar_t = std::condition_variable;
//
namespace internal
{
template <typename ApiT = TIMEMORY_API>
mutex_t&
get_mutex()
{
    static mutex_t _v{};
    return _v;
}
//
#if defined(TIMEMORY_USE_PTL)
//
template <typename ApiT = TIMEMORY_API>
PTL::ThreadPool&
get_thread_pool()
{
    static auto _v = PTL::ThreadPool{ get_env<uint64_t>("TIMEMORY_NUM_THREADS", 1) };
    return _v;
}
//
template <typename ApiT = TIMEMORY_API>
PTL::TaskGroup<void>&
get_task_group()
{
    static auto _v = PTL::TaskGroup<void>{ &get_thread_pool<ApiT>() };
    return _v;
}
#else
//
template <typename ApiT = TIMEMORY_API>
std::atomic<bool>&
get_completed()
{
    static std::atomic<bool> _v{ false };
    return _v;
}
//
template <typename ApiT = TIMEMORY_API>
condvar_t&
get_condvar()
{
    static condvar_t _v{};
    return _v;
}
//
#endif
}  // namespace internal
//
#if defined(TIMEMORY_USE_PTL)
//
template <typename ApiT = TIMEMORY_API, typename FuncT>
auto
submit(FuncT&& _func)
{
    lock_t _lk{ internal::get_mutex() };
    internal::get_task_group<ApiT>().exec(std::forward<FuncT>(_func));
    return []() { internal::get_task_group<ApiT>().join(); };
}
//
template <typename ApiT = TIMEMORY_API>
void
shutdown()
{
    internal::get_task_group<ApiT>().join();
    internal::get_thread_pool().set_verbose(-1);
    internal::get_thread_pool().destroy_threadpool();
}
//
#else
//
template <typename ApiT = TIMEMORY_API>
void
shutdown()
{
    fprintf(stderr, "[%s][%s:%i]\n", __FUNCTION__, __FILE__, __LINE__);
    auto& _completed = internal::get_completed();
    auto& _cv        = internal::get_condvar();
    auto& _mutex     = internal::get_mutex();
    _completed.store(true);
    lock_t _lk{ _mutex };
    _cv.notify_one();
}
//
template <typename ApiT = TIMEMORY_API, typename FuncT>
auto
submit(FuncT&& _func)
{
    using task_queue_t =
        std::vector<std::pair<std::function<void()>, std::atomic<bool>*>>;
    static auto&        _completed = internal::get_completed();
    static auto&        _cv        = internal::get_condvar();
    static auto&        _mutex     = internal::get_mutex();
    static task_queue_t _queue{};
    static auto         _run = []() {
        while(!_completed)
        {
            lock_t _lk{ _mutex };
            _cv.wait(_lk, []() -> bool { return _completed; });
            if(!_lk.owns_lock())
                _lk.lock();
            task_queue_t _local_queue{};
            std::swap(_local_queue, _queue);
            _lk.unlock();

            for(auto& itr : _local_queue)
            {
                itr.first();
                itr.second->store(true);
            }
        }
    };
    static std::thread _thread{ _run };
    static auto        _dtor = (_thread.detach(), scope::destructor{ []() {
                             _completed.store(true);
                             _cv.notify_one();
                         } });

    auto   _wait = new std::atomic<bool>{ false };
    lock_t _lk{ _mutex };
    _queue.emplace_back(std::forward<FuncT>(_func), _wait);
    _cv.notify_one();

    return [_wait]() {
        while(!_wait->load())
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
        }
        delete _wait;
    };
}
//
#endif
}  // namespace tasking
}  // namespace tim

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

/** \file timemory/ert/barrier.hpp
 * \headerfile timemory/ert/barrier.hpp "timemory/ert/barrier.hpp"
 * Provides multi-threading barriers
 *
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace tim
{
namespace ert
{
using std::size_t;

//--------------------------------------------------------------------------------------//
//  creates a multithreading barrier
//
class thread_barrier
{
public:
    using size_type = int64_t;
    using mutex_t   = std::mutex;
    using condvar_t = std::condition_variable;
    using atomic_t  = std::atomic<size_type>;
    using lock_t    = std::unique_lock<mutex_t>;

public:
    explicit thread_barrier(size_t nthreads)
    : m_master(std::this_thread::get_id())
    , m_num_threads(nthreads)
    , m_notify(0)
    , m_future(m_promise.get_future().share())
    {}

    thread_barrier(const thread_barrier&) = delete;
    thread_barrier(thread_barrier&&)      = delete;

    thread_barrier& operator=(const thread_barrier&) = delete;
    thread_barrier& operator=(thread_barrier&&) = delete;

    size_type size() const { return m_num_threads; }

    // call from worker thread -- spin wait (fast)
    void spin_wait()
    {
        if(is_master())
        {
#if defined(TIMEMORY_INTERNAL_TESTING)
            TIMEMORY_EXCEPTION("master thread calling worker wait function\n");
#else
            return;
#endif
        }

        {
            lock_t lk(m_mutex);
            ++m_counter;
            ++m_waiting;
        }

        while(m_counter < m_num_threads)
        {
            while(spin_lock.test_and_set(std::memory_order_acquire))  // acquire lock
                ;                                                     // spin
            spin_lock.clear(std::memory_order_release);
        }

        {
            lock_t lk(m_mutex);
            --m_waiting;
            if(m_waiting == 0)
                m_counter = 0;  // reset barrier
        }
    }

    // call from worker thread -- condition variable wait (slower)
    void cv_wait()
    {
        if(is_master())
        {
#if defined(TIMEMORY_INTERNAL_TESTING)
            TIMEMORY_EXCEPTION("master thread calling worker wait function\n");
#else
            return;
#endif
        }

        lock_t lk(m_mutex);
        ++m_counter;
        ++m_waiting;
        m_cv.wait(lk, [&] { return m_counter >= m_num_threads; });
        m_cv.notify_one();
        --m_waiting;
        if(m_waiting == 0)
            m_counter = 0;  // reset barrier
    }

    // workers increment an atomic until and wait on future until
    // master sets the promise once the
    void notify_wait()
    {
        if(is_master())
        {
            lock_t lk(m_mutex);
            while(m_notify.load() < m_num_threads)
                m_cv.wait(lk);
            m_promise.set_value();
            while(m_notify.load() > 0)
            {
            }
            std::promise<void>       _ptmp;
            std::shared_future<void> _ftmp = _ptmp.get_future().share();
            std::swap(m_promise, _ptmp);
            std::swap(m_future, _ftmp);
        }
        else
        {
            {
                lock_t lk(m_mutex);
                ++m_notify;
                m_cv.notify_one();
            }
            m_future.wait();
            --m_notify;
        }
    }

    // check if this is the thread the created barrier
    bool is_master() const { return std::this_thread::get_id() == m_master; }

private:
    // the constructing thread will be set to master
    std::thread::id  m_master      = std::this_thread::get_id();
    size_type        m_num_threads = 0;  // number of threads that will wait on barrier
    size_type        m_waiting     = 0;  // number of threads waiting on lock
    size_type        m_counter     = 0;  // number of threads that have entered wait func
    std::atomic_flag spin_lock     = ATOMIC_FLAG_INIT;  // for spin lock
    mutex_t          m_mutex;
    condvar_t        m_cv;
    std::atomic<size_type>   m_notify;
    std::promise<void>       m_promise;
    std::shared_future<void> m_future;
};

}  // namespace ert
}  // namespace tim

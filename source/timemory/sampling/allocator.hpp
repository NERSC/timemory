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

#include "timemory/backends/threading.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/storage/ring_buffer.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <exception>
#include <future>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <semaphore.h>
#include <sstream>
#include <thread>
#include <vector>

#define TIMEMORY_SEMAPHORE_CHECK(VAL)                                                    \
    if(VAL != 0)                                                                         \
    {                                                                                    \
        perror(#VAL);                                                                    \
        throw std::runtime_error(#VAL);                                                  \
    }

#define TIMEMORY_SEMAPHORE_CHECK_MSG(VAL, ...)                                           \
    if(VAL != 0)                                                                         \
    {                                                                                    \
        perror(__VA_ARGS__);                                                             \
        throw std::runtime_error(__VA_ARGS__);                                           \
    }

#define TIMEMORY_SEMAPHORE_HANDLE_EINTR(FUNC, EC, ...)                                   \
    EC = FUNC(__VA_ARGS__);                                                              \
    if(EC != 0)                                                                          \
        EC = errno;                                                                      \
    while(EC == EINTR)                                                                   \
    {                                                                                    \
        EC = FUNC(__VA_ARGS__);                                                          \
        if(EC != 0)                                                                      \
            EC = errno;                                                                  \
    }

#define TIMEMORY_SEMAPHORE_TRYWAIT(SEM, EC)                                              \
    TIMEMORY_SEMAPHORE_HANDLE_EINTR(sem_trywait, EC, &SEM)

namespace tim
{
namespace sampling
{
using semaphore_t = sem_t;

enum class sigmask_scope : short
{
    thread  = 0,
    process = 1
};

inline sigset_t
block_signals(const std::set<int>& _signals, sigmask_scope _scope)
{
    sigset_t _old;
    sigset_t _new;

    sigemptyset(&_new);
    for(auto itr : _signals)
        sigaddset(&_new, itr);

    auto _err = (_scope == sigmask_scope::thread)
                    ? pthread_sigmask(SIG_BLOCK, &_new, &_old)
                    : sigprocmask(SIG_BLOCK, &_new, &_old);

    if(_err != 0)
    {
        std::string _msg =
            (_scope == sigmask_scope::thread) ? "pthread_sigmask" : "sigprocmask";
        perror(_msg.c_str());
        throw std::runtime_error(_msg);
    }

    return _old;
}

inline sigset_t
unblock_signals(const std::set<int>& _signals, sigmask_scope _scope)
{
    sigset_t _old;
    sigset_t _new;

    sigemptyset(&_new);
    for(auto itr : _signals)
        sigaddset(&_new, itr);

    auto _err = (_scope == sigmask_scope::thread)
                    ? pthread_sigmask(SIG_UNBLOCK, &_new, &_old)
                    : sigprocmask(SIG_UNBLOCK, &_new, &_old);

    if(_err != 0)
    {
        std::string _msg =
            (_scope == sigmask_scope::thread) ? "pthread_sigmask" : "sigprocmask";
        perror(_msg.c_str());
        throw std::runtime_error(_msg);
    }

    return _old;
}

/// \struct tim::sampling::allocator
/// \tparam Tp The tim::sampling::sampler template type
///
/// \brief This structure handles allocating new memory outside of the signal handler
/// for \ref tim::sampling::sampler when it is dynamic. The sampler signals when
/// it's buffer is full and the thread owned by this class wakes up, swaps out
/// the full buffer for an empty one and then moves the data into it's memory
template <typename Tp>
struct allocator
{
    using this_type   = allocator<Tp>;
    using data_type   = typename Tp::bundle_type;
    using buffer_type = data_storage::ring_buffer<data_type>;
    using lock_type   = std::unique_lock<std::mutex>;

    allocator(Tp* _obj);
    ~allocator();

    allocator(const allocator&)     = delete;
    allocator(allocator&&) noexcept = default;
    allocator& operator=(const allocator&) = delete;
    allocator& operator=(allocator&&) noexcept = default;

    const auto& get_data() const;
    bool        is_alive() const { return m_alive; }
    void        join();
    void        restart(Tp* _obj);
    void        block_signal(int _v);
    void        emplace(buffer_type&&);

    static void execute(allocator*, Tp*);

private:
    /// this function makes sure that allocator isn't interrupted by signals
    void block_pending_signals();

    void update_data();

    std::atomic<bool>        m_alive{ false };
    int                      m_verbose = 0;
    int64_t                  m_tid     = threading::get_id();
    semaphore_t              m_sem;
    std::mutex               m_signal_mutex;
    std::mutex               m_buffer_mutex;
    std::promise<void>       m_promise                 = {};
    std::vector<data_type>   m_data                    = {};
    std::vector<buffer_type> m_buffers                 = {};
    std::set<int>            m_block_signals_pending   = { SIGALRM, SIGVTALRM, SIGPROF };
    std::set<int>            m_block_signals_completed = {};
    std::exception_ptr       m_thread_exception        = {};
    std::thread              m_thread;  // construct thread after data to prevent UB
};

template <typename Tp>
allocator<Tp>::allocator(Tp* _obj)
: m_thread{ &execute, this, _obj }
{
    m_promise.get_future().wait();
}

template <typename Tp>
allocator<Tp>::~allocator()
{
    if(m_thread.joinable())
        m_thread.join();

    sem_destroy(&m_sem);
}

template <typename Tp>
void
allocator<Tp>::join()
{
    if(m_thread.joinable())
        m_thread.join();
    if(m_thread_exception)
        std::rethrow_exception(m_thread_exception);
}

template <typename Tp>
void
allocator<Tp>::restart(Tp* _obj)
{
    if(!_obj)
        return;
    _obj->m_exit();
    if(m_thread.joinable())
        m_thread.join();
    m_promise = std::promise<void>{};
    m_alive.store(false);
    m_thread = std::thread{ &execute, this, _obj };
    m_promise.get_future().wait();
}

template <typename Tp>
void
allocator<Tp>::block_signal(int _v)
{
    lock_type _lk{ m_signal_mutex, std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();

    if(m_block_signals_completed.count(_v) == 0)
    {
        m_block_signals_pending.emplace(_v);
    }
}

template <typename Tp>
void
allocator<Tp>::emplace(buffer_type&& _v)
{
    lock_type _lk{ m_buffer_mutex };
    m_buffers.emplace_back(std::move(_v));
}

template <typename Tp>
void
allocator<Tp>::block_pending_signals()
{
    lock_type _lk{ m_signal_mutex, std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();

    auto& _pending = m_block_signals_pending;
    if(!_pending.empty())
    {
        auto& _completed = m_block_signals_completed;
        for(const auto& itr : _pending)
        {
            sampling::block_signals({ itr }, sigmask_scope::thread);
            _completed.emplace(itr);
        }
        _pending.clear();
    }
}

template <typename Tp>
void
allocator<Tp>::update_data()
{
    // swap out buffer array for a temporary to avoid
    // holding lock while processing
    auto _buffers = std::vector<buffer_type>{};
    {
        lock_type _lk{ m_buffer_mutex, std::defer_lock };
        if(!_lk.owns_lock())
            _lk.lock();
        std::swap(m_buffers, _buffers);
    }
    size_t _n = 0;
    for(auto& itr : _buffers)
        _n += itr.count();
    m_data.reserve(m_data.size() + _n);
    for(auto& itr : _buffers)
    {
        while(!itr.is_empty())
        {
            auto _v = data_type{};
            itr.read(&_v);
            m_data.emplace_back(std::move(_v));
        }
        itr.destroy();
    }
}

template <typename Tp>
const auto&
allocator<Tp>::get_data() const
{
    const_cast<this_type*>(this)->update_data();
    return m_data;
}

template <typename Tp>
void
allocator<Tp>::execute(allocator* _alloc, Tp* _obj)
{
    threading::set_thread_name(
        std::string{ "samp.alloc." + std::to_string(_alloc->m_tid) }.c_str());

    bool   _completed  = false;
    size_t _swap_count = 0;
    sem_t& _sem        = _alloc->m_sem;
    TIMEMORY_SEMAPHORE_CHECK(sem_init(&_sem, 0, 0));

    _alloc->m_alive.store(true);
    try
    {
        _obj->set_notify([&]() { TIMEMORY_SEMAPHORE_CHECK(sem_post(&_sem)); });

        _obj->set_exit([&_sem, &_completed]() {
            _completed = true;
            int _val   = 0;
            do
            {
                std::this_thread::sleep_for(std::chrono::microseconds{ 10 });
                TIMEMORY_SEMAPHORE_CHECK(sem_getvalue(&_sem, &_val));
            } while(_val > 0);
            TIMEMORY_SEMAPHORE_CHECK(sem_post(&_sem));
        });

        _alloc->m_promise.set_value();

        while(!_completed)
        {
            _alloc->block_pending_signals();

            auto _buff = buffer_type{};

            TIMEMORY_SEMAPHORE_CHECK(sem_wait(&_sem));
            if(!_obj->m_filled.is_empty())
            {
                std::swap(_buff, _obj->m_filled);
                ++_swap_count;
                if(!_buff.is_empty())
                    _alloc->emplace(std::move(_buff));
            }
        }
    } catch(...)
    {
        // Set the exception pointer in case of an exception
        _alloc->m_thread_exception = std::current_exception();
    }

    _obj->set_notify([]() {});
    _obj->set_exit([]() {});
    _alloc->m_alive.store(false);

    if(_alloc->m_verbose > 0)
    {
        std::cerr << "[" << _alloc->m_tid << "] Sampler allocator performed "
                  << _swap_count << " swaps and has " << _alloc->m_data.size()
                  << " entries\n";
    }
}

template <>
struct allocator<void>
{
    template <typename Tp>
    explicit allocator(Tp*)
    {}

    ~allocator()                    = default;
    allocator(const allocator&)     = delete;
    allocator(allocator&&) noexcept = default;
    allocator& operator=(const allocator&) = delete;
    allocator& operator=(allocator&&) noexcept = default;

    static constexpr bool is_alive() { return false; }
    static constexpr void block_signal(int) {}

    template <typename Tp>
    static constexpr void emplace(Tp&&)
    {}

    template <typename Tp>
    static constexpr void restart(Tp*)
    {}
};

}  // namespace sampling
}  // namespace tim

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

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/defines.h"
#include "timemory/log/logger.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/signals/signal_mask.hpp"
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
    {                                                                                    \
        if(VAL != 0)                                                                     \
        {                                                                                \
            perror(#VAL);                                                                \
            throw std::runtime_error(#VAL);                                              \
        }                                                                                \
    }

#define TIMEMORY_SEMAPHORE_CHECK_MSG(VAL, ...)                                           \
    {                                                                                    \
        if(VAL != 0)                                                                     \
        {                                                                                \
            perror(__VA_ARGS__);                                                         \
            throw std::runtime_error(__VA_ARGS__);                                       \
        }                                                                                \
    }

#define TIMEMORY_SEMAPHORE_HANDLE_EINTR(FUNC, EC, ...)                                   \
    {                                                                                    \
        EC = FUNC(__VA_ARGS__);                                                          \
        if(EC != 0)                                                                      \
            EC = errno;                                                                  \
        while(EC == EINTR)                                                               \
        {                                                                                \
            EC = FUNC(__VA_ARGS__);                                                      \
            if(EC != 0)                                                                  \
                EC = errno;                                                              \
        }                                                                                \
    }

#define TIMEMORY_SEMAPHORE_TRYWAIT(SEM, EC)                                              \
    TIMEMORY_SEMAPHORE_HANDLE_EINTR(sem_trywait, EC, &SEM)

namespace tim
{
namespace sampling
{
using semaphore_t = sem_t;

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

    auto&       get_data();
    const auto& get_data() const;
    bool        is_alive() const { return m_alive; }
    void        join();
    void        start() { m_start.set_value(); }
    void        restart(Tp* _obj);
    void        block_signal(int _v);
    void        emplace(buffer_type&&);
    void        set_verbose(int _v) { m_verbose = _v; }

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
    std::promise<void>       m_ready                   = {};
    std::promise<void>       m_start                   = {};
    std::vector<data_type>   m_data                    = {};
    std::vector<buffer_type> m_buffers                 = {};
    std::set<int>            m_block_signals_pending   = { SIGALRM, SIGVTALRM, SIGPROF };
    std::set<int>            m_block_signals_completed = {};
    std::exception_ptr       m_thread_exception        = {};
    std::function<void()>    m_exit                    = []() {};
    std::thread              m_thread;  // construct thread after data to prevent UB
};

template <typename Tp>
allocator<Tp>::allocator(Tp* _obj)
: m_thread{ &execute, this, _obj }
{
    TIMEMORY_SEMAPHORE_CHECK(sem_init(&m_sem, 0, 0));
    m_alive.store(true);
    m_ready.get_future().wait_for(std::chrono::milliseconds{ 100 });
}

template <typename Tp>
allocator<Tp>::~allocator()
{
    join();
    if(sem_destroy(&m_sem) != 0)
        TIMEMORY_PRINTF(stderr, "failed to destroy semaphore in sampling allocator");
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
    m_start.set_value();
    m_exit();
    join();
    if(!m_alive)
    {
        m_start = std::promise<void>{};
        m_ready = std::promise<void>{};
        m_alive.store(true);
        m_thread = std::thread{ &execute, this, _obj };
        m_ready.get_future().wait_for(std::chrono::milliseconds{ 100 });
    }
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
            signals::block_signals({ itr }, signals::sigmask_scope::thread);
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
auto&
allocator<Tp>::get_data()
{
    update_data();
    return m_data;
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
    threading::offset_this_id(true);
    threading::set_thread_name(
        std::string{ "samp.alloc." + std::to_string(_alloc->m_tid) }.c_str());

    using buffer_vec_t = std::vector<buffer_type>;

    bool          _completed         = false;
    bool          _clean_exit        = true;
    size_t        _swap_count        = 0;
    const int64_t _local_buffer_size = 10;
    auto          _local_buffer      = buffer_vec_t{};
    auto          _pending           = std::atomic<int64_t>{ 0 };
    auto          _wait              = std::atomic<bool>{ false };
    semaphore_t*  _sem               = &_alloc->m_sem;

    _local_buffer.reserve(_local_buffer_size);

    auto _swap_buffer = [&](buffer_vec_t& _new_buffer) {
        if(_local_buffer.empty() || _pending.load() == 0)
            return;
        _wait.store(true);
        std::swap(_local_buffer, _new_buffer);
        auto _n = _new_buffer.size();
        _wait.store(false);
        _pending -= _n;
        for(auto& itr : _new_buffer)
        {
            if(!itr.is_empty())
            {
                _alloc->emplace(std::move(itr));
                ++_swap_count;
            }
        }
    };
    auto _notify = [&](semaphore_t* _sem_v) {
        if(_alloc && _sem_v)
        {
            TIMEMORY_SEMAPHORE_CHECK(sem_post(_sem_v));
        }
    };
    auto _exit = [&]() {
        _completed = true;
        TIMEMORY_SEMAPHORE_CHECK(sem_post(_sem));
    };

    _alloc->m_exit = _exit;

    try
    {
        _obj->set_notify([&]() { _notify(_sem); });

        _obj->set_move([&](buffer_type&& _buffer) {
            ++_pending;
            if(_pending.load() >= _local_buffer_size)
                _notify(_sem);
            while(_wait.load() || _pending.load() >= _local_buffer_size)
                std::this_thread::yield();
            _local_buffer.emplace_back(std::move(_buffer));
            _notify(_sem);
        });

        _obj->set_exit(_exit);

        _alloc->m_ready.set_value();

        // wait for start post
        _alloc->m_start.get_future().wait_for(std::chrono::milliseconds{ 100 });
        // block the pending signals
        _alloc->block_pending_signals();

        while(!_completed)
        {
            auto _new_buffer = buffer_vec_t{};
            _new_buffer.reserve(_local_buffer_size);

            TIMEMORY_SEMAPHORE_CHECK(sem_wait(_sem));

            _swap_buffer(_new_buffer);
        }
    } catch(...)
    {
        _clean_exit = false;
        // Set the exception pointer in case of an exception
        _alloc->m_thread_exception = std::current_exception();
    }

    if(_clean_exit)
    {
        while(_pending.load() > 0)
        {
            auto _new_buffer = buffer_vec_t{};
            _new_buffer.reserve(_pending.load());
            _swap_buffer(_new_buffer);
        }

        auto _new_buffer = buffer_vec_t{};
        _new_buffer.reserve(1);

        _swap_buffer(_new_buffer);

        _alloc->update_data();

        if(_alloc->m_verbose >= 1)
        {
            log::stream(std::cerr, log::color::info())
                << "[" << TIMEMORY_PROJECT_NAME << "][pid=" << process::get_id()
                << "][tid=" << _alloc->m_tid << "] Sampler allocator performed "
                << _swap_count << " swaps and has " << _alloc->m_data.size()
                << " entries\n";
        }
    }
    _sem = nullptr;

    // disable communication with allocator
    _obj->set_notify([]() {});
    _obj->set_move([](buffer_type&&) {});
    _obj->set_exit([]() {});
    _alloc->m_exit = []() {};

    _alloc->m_alive.store(false);
}

template <>
struct allocator<void>
{
    template <typename Tp>
    explicit allocator(Tp*)
    {}

    allocator()                     = default;
    ~allocator()                    = default;
    allocator(const allocator&)     = delete;
    allocator(allocator&&) noexcept = default;
    allocator& operator=(const allocator&) = delete;
    allocator& operator=(allocator&&) noexcept = default;

    static constexpr bool is_alive() { return false; }
    static constexpr void block_signal(int) {}
    static constexpr void join() {}

    template <typename Tp>
    static constexpr void emplace(Tp&&)
    {}

    template <typename Tp>
    static constexpr void restart(Tp*)
    {}

    void set_verbose(int) {}
};

}  // namespace sampling
}  // namespace tim

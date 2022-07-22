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

#ifndef TIMEMORY_SAMPLING_SAMPLER_CPP_
#define TIMEMORY_SAMPLING_SAMPLER_CPP_

#include "timemory/utility/types.hpp"

#include <initializer_list>
#include <limits>

#if !defined(TIMEMORY_SAMPLING_SAMPLER_HPP_)
#    include "timemory/sampling/sampler.hpp"
#endif

#include "timemory/backends/threading.hpp"
#include "timemory/components/base.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/operations/types/sample.hpp"
#include "timemory/sampling/allocator.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/macros.hpp"

// C++ includes
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// C includes
#include <cassert>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(TIMEMORY_USE_LIBEXPLAIN)
#    include <libexplain/execvp.h>
#endif

#if defined(TIMEMORY_UNIX)
#    include <unistd.h>
#endif

#if !defined(TIMEMORY_SAMPLER_DEPTH_DEFAULT)
#    define TIMEMORY_SAMPLER_DEPTH_DEFAULT 64
#endif

#if !defined(TIMEMORY_SAMPLER_OFFSET_DEFAULT)
#    define TIMEMORY_SAMPLER_OFFSET_DEFAULT 3
#endif

#if !defined(TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT)
#    if defined(TIMEMORY_USE_LIBUNWIND)
#        define TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT true
#    else
#        define TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT false
#    endif
#endif

namespace tim
{
namespace sampling
{
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
auto
sampler<CompT<Types...>, N, SigIds...>::get_latest_samples()
{
    std::vector<bundle_type*> _last{};
    auto_lock_t               lk(type_mutex<this_type>());
    _last.reserve(get_persistent_data().m_instances.size());
    for(auto& itr : get_persistent_data().m_instances)
        _last.emplace_back(itr->get_last());
    return _last;
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
sampler<CompT<Types...>, N, SigIds...>::sampler(std::string _label, signal_set_t _good,
                                                signal_set_t _bad)
: m_good{ std::move(_good) }
, m_bad{ std::move(_bad) }
, m_alloc{ this }
, m_label{ std::move(_label) }
{
    _init_sampler(threading::get_id());
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
sampler<CompT<Types...>, N, SigIds...>::sampler(std::string _label, int64_t _tid,
                                                signal_set_t _good, signal_set_t _bad)
: m_good{ std::move(_good) }
, m_bad{ std::move(_bad) }
, m_alloc{ this }
, m_label{ std::move(_label) }
{
    _init_sampler(_tid);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
sampler<CompT<Types...>, N, SigIds...>::sampler(std::string _label, int64_t _tid,
                                                signal_set_t _good, int _verbose,
                                                signal_set_t _bad)
: m_verbose{ _verbose }
, m_good{ std::move(_good) }
, m_bad{ std::move(_bad) }
, m_alloc{ this }
, m_label{ std::move(_label) }
{
    _init_sampler(_tid);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
sampler<CompT<Types...>, N, SigIds...>::~sampler()
{
    m_exit();
    auto_lock_t _lk{ type_mutex<this_type>(), std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();
    auto _erase_samplers = [](auto& _samplers, this_type* _ptr) {
        auto itr = std::find(_samplers.begin(), _samplers.end(), _ptr);
        if(itr != _samplers.end())
        {
            _samplers.erase(itr);
            return true;
        }
        return false;
    };
    if(!_erase_samplers(get_samplers(threading::get_id()), this))
    {
        for(auto& itr : get_persistent_data().m_thread_instances)
            _erase_samplers(itr.second, this);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename... Args, typename Tp, enable_if_t<Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::sample(Args&&... _args)
{
    m_last = &(m_data.at((m_idx++) % N));
    if(m_backtrace)
    {
        // e.g. _depth == 4 and _offset == 3 means get last 4 of 7 backtrace entries
        constexpr auto _depth  = trait::backtrace_depth<this_type>::value;
        constexpr auto _offset = trait::backtrace_offset<this_type>::value;
        IF_CONSTEXPR(trait::backtrace_use_libunwind<this_type>::value)
        {
            m_last->sample(get_unw_backtrace<_depth, _offset>(),
                           std::forward<Args>(_args)...);
        }
        else m_last->sample(get_native_backtrace<_depth, _offset>(),
                            std::forward<Args>(_args)...);
    }
    else
    {
        m_last->sample(std::forward<Args>(_args)...);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename... Args, typename Tp, enable_if_t<!Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::sample(Args&&... _args)
{
    assert(m_buffer_size > 0);
    if(m_data.size() == m_buffer_size || m_data.capacity() < m_buffer_size)
    {
        m_notify();
        m_wait();
    }

    assert(m_data.capacity() >= m_buffer_size);
    m_data.emplace_back(bundle_type{});
    m_last = &m_data.back();
    if(m_backtrace)
    {
        // e.g. _depth == 4 and _offset == 3 means get last 4 of 7 backtrace entries
        constexpr auto _depth  = trait::backtrace_depth<this_type>::value;
        constexpr auto _offset = trait::backtrace_offset<this_type>::value;
        IF_CONSTEXPR(trait::backtrace_use_libunwind<this_type>::value)
        {
            m_last->sample(get_unw_backtrace<_depth, _offset>(),
                           std::forward<Args>(_args)...);
        }
        else m_last->sample(get_native_backtrace<_depth, _offset>(),
                            std::forward<Args>(_args)...);
    }
    else
    {
        m_last->sample(std::forward<Args>(_args)...);
    }
}
//
//--------------------------------------------------------------------------------------//
// one or more signals specified in template parameters
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::start()
{
    TIMEMORY_CONDITIONAL_PRINT_HERE(m_verbose >= 2, "starting (index: %zu)", m_idx);
    auto cnt = tracker_type::start();
    base_type::set_started();
    for(auto& itr : m_data)
        itr.start();
    if(cnt.second == 0)
        configure({ SigIds... });
}
//
//--------------------------------------------------------------------------------------//
// one or more signals specified in template parameters
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::stop()
{
    TIMEMORY_CONDITIONAL_PRINT_HERE(m_verbose >= 2, "stopping (index: %zu)", m_idx);
    auto cnt = tracker_type::stop();
    base_type::set_stopped();
    for(auto& itr : m_data)
        itr.stop();
    if(cnt.second == 0)
        stop({});
}
//
//--------------------------------------------------------------------------------------//
// no signals specified in template parameters
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<!Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::start()
{
    TIMEMORY_CONDITIONAL_PRINT_HERE(m_verbose >= 2, "starting (index: %zu)", m_idx);
    auto cnt = tracker_type::start();
    if(cnt.second == 0 && !m_alloc.is_alive())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(m_verbose >= 2,
                                        "restarting allocator (index: %zu)", m_idx);
        m_alloc.restart(this);
    }
    base_type::set_started();
    for(auto& itr : m_data)
        itr.start();
}
//
//--------------------------------------------------------------------------------------//
// no signals specified in template parameters
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<!Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::stop()
{
    TIMEMORY_CONDITIONAL_PRINT_HERE(m_verbose >= 2, "stopping (index: %zu)", m_idx);
    auto cnt = tracker_type::stop();
    base_type::set_stopped();
    for(auto& itr : m_data)
        itr.stop();
    if(cnt.second == 0)
        stop({});
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::stop(std::set<int> _signals)
{
    if(_signals.empty())
    {
        // if specified by set_signals(...)
        for(auto itr : m_timer_data.m_signals)
            _signals.emplace(itr);
        // if specified by template parameters
        TIMEMORY_FOLD_EXPRESSION(_signals.emplace(SigIds));
    }

    TIMEMORY_CONDITIONAL_PRINT_HERE(m_verbose >= 2, "stopping %zu signals (index: %zu)",
                                    _signals.size(), m_idx);

    for(auto itr : _signals)
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(m_verbose >= 2, "stopping signal %i (index: %zu)",
                                        itr, m_idx);
        auto _itimer = get_itimer(itr);
        if(_itimer < 0)
        {
            if(itr < SIGRTMIN || itr > SIGRTMAX)
            {
                TIMEMORY_EXCEPTION(TIMEMORY_JOIN(
                    " ", "Error! Alarm cannot be set for signal", itr,
                    "because the signal does not map to a known itimer "
                    "value or is not a real timer >= SIGRTMIN and <= SIGRTMAX"));
            }
            auto& _timer = m_timer_data.m_timer[itr];
            timer_delete(_timer);
        }
        else
        {
            auto&       _original_it = m_timer_data.m_original_itimerval[itr];
            itimerval_t _curr;
            check_itimer(getitimer(_itimer, &_curr));
            // stop the alarm
            if(_curr.it_interval.tv_usec > 0 || _curr.it_interval.tv_sec > 0)
                check_itimer(setitimer(_itimer, &_original_it, &_curr));

            m_timer_data.m_active[itr] = false;
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<Tp::value>>
typename sampler<CompT<Types...>, N, SigIds...>::bundle_type&
sampler<CompT<Types...>, N, SigIds...>::get(size_t idx)
{
    return m_data.at(idx % N);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<Tp::value>>
const typename sampler<CompT<Types...>, N, SigIds...>::bundle_type&
sampler<CompT<Types...>, N, SigIds...>::get(size_t idx) const
{
    return m_data.at(idx % N);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<!Tp::value>>
typename sampler<CompT<Types...>, N, SigIds...>::bundle_type&
sampler<CompT<Types...>, N, SigIds...>::get(size_t idx)
{
    return m_data.at(idx);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<!Tp::value>>
const typename sampler<CompT<Types...>, N, SigIds...>::bundle_type&
sampler<CompT<Types...>, N, SigIds...>::get(size_t idx) const
{
    return m_data.at(idx);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::execute(int signum)
{
    // prevent re-entry from different signals
    static thread_local semaphore_t _sem = []() {
        semaphore_t _v{};
        TIMEMORY_CONDITIONAL_PRINT_HERE(settings::debug(), "initializing %s",
                                        "semaphore");
        int _err = 0;
        TIMEMORY_SEMAPHORE_HANDLE_EINTR(sem_init, _err, &_v, 0, 1)
        TIMEMORY_SEMAPHORE_CHECK_MSG(_err, "sem_init(&_v, 0, 1)")
        TIMEMORY_CONDITIONAL_PRINT_HERE(settings::debug(), "%s initialized", "semaphore");
        return _v;
    }();

    IF_CONSTEXPR(trait::prevent_reentry<this_type>::value)
    {
        int _err = 0;
        TIMEMORY_SEMAPHORE_TRYWAIT(_sem, _err);

        if(_err == EAGAIN)
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(
                settings::debug(), "Ignoring signal %i (raised while sampling)", signum);
            return;
        }
        else if(_err != 0)
        {
            std::stringstream _msg{};
            _msg << "sem_trywait(&_sem) returned error code: " << _err;
            perror(_msg.str().c_str());
            throw std::runtime_error(_msg.str().c_str());
        }
    }

    for(auto& itr : get_samplers(threading::get_id()))
    {
        if(!itr)
            continue;

        TIMEMORY_CONDITIONAL_PRINT_HERE(
            itr->m_verbose >= 4, "sampling signal %i (index: %zu)", signum, itr->m_idx);

        IF_CONSTEXPR(trait::check_signals<this_type>::value)
        {
            if(itr->is_good(signum))
            {
                itr->sample(signum);
            }
            else if(itr->is_bad(signum))
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE(
                    itr->m_verbose >= 0,
                    "sampler instance received unexpected signal %i (index: %zu)", signum,
                    itr->m_idx);
            }
        }
        else { itr->sample(signum); }
    }

    IF_CONSTEXPR(trait::prevent_reentry<this_type>::value)
    {
        TIMEMORY_SEMAPHORE_CHECK(sem_post(&_sem));
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::execute(int signum, siginfo_t* _info, void* _data)
{
    // prevent re-entry from different signals
    static thread_local semaphore_t _sem = []() {
        semaphore_t _v{};
        TIMEMORY_CONDITIONAL_PRINT_HERE(settings::debug(), "initializing %s",
                                        "semaphore");
        int _err = 0;
        TIMEMORY_SEMAPHORE_HANDLE_EINTR(sem_init, _err, &_v, 0, 1)
        TIMEMORY_SEMAPHORE_CHECK_MSG(_err, "sem_init(&_v, 0, 1)")
        TIMEMORY_CONDITIONAL_PRINT_HERE(settings::debug(), "%s initialized", "semaphore");
        return _v;
    }();
    static thread_local auto _sem_dtor = scope::destructor{ []() {
        TIMEMORY_CONDITIONAL_PRINT_HERE(settings::debug(), "destroying %s", "semaphore");
        int                  _err      = 0;
        TIMEMORY_SEMAPHORE_HANDLE_EINTR(sem_destroy, _err, &_sem)
        TIMEMORY_SEMAPHORE_CHECK_MSG(_err, "sem_destroy(&_sem)")
        TIMEMORY_CONDITIONAL_PRINT_HERE(settings::debug(), "%s destroyed", "semaphore");
    } };

    IF_CONSTEXPR(trait::prevent_reentry<this_type>::value)
    {
        int _err = 0;
        TIMEMORY_SEMAPHORE_TRYWAIT(_sem, _err);

        if(_err == EAGAIN)
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(
                settings::debug(), "Ignoring signal %i (raised while sampling)", signum);
            return;
        }
        else if(_err != 0)
        {
            std::stringstream _msg{};
            _msg << "sem_trywait(&_sem) returned error code: " << _err;
            perror(_msg.str().c_str());
            throw std::runtime_error(_msg.str().c_str());
        }
    }

    for(auto& itr : get_samplers(threading::get_id()))
    {
        if(!itr)
            continue;

        TIMEMORY_CONDITIONAL_PRINT_HERE(
            itr->m_verbose >= 4, "sampling signal %i (index: %zu)", signum, itr->m_idx);

        IF_CONSTEXPR(trait::check_signals<this_type>::value)
        {
            if(itr->is_good(signum))
            {
                itr->sample(signum, _info, _data);
            }
            else if(itr->is_bad(signum))
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE(
                    itr->m_verbose >= 0,
                    "sampler instance received unexpected signal %i (index: %zu)", signum,
                    itr->m_idx);
            }
        }
        else { itr->sample(signum, _info, _data); }
    }

    IF_CONSTEXPR(trait::prevent_reentry<this_type>::value)
    {
        TIMEMORY_SEMAPHORE_CHECK(sem_post(&_sem));
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::configure(std::set<int> _signals, int _verbose,
                                                  bool _unblock)
{
    _verbose = std::max<int>(_verbose, m_verbose);

    TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose >= 3, "configuring sampler (index: %zu)",
                                    m_idx);

    size_t wait_count = 0;
    {
        auto_lock_t _lk{ type_mutex<this_type>() };
        for(auto& itr : get_samplers(threading::get_id()))
            wait_count += itr->count();
    }

    if(wait_count == 0)
    {
        if(_verbose >= 1)
        {
            fprintf(stderr,
                    "[sampler::configure]> No existing sampler has been configured to "
                    "sample at a specific signal or fail at a specific signal. itimer "
                    "for will not be set. Sampler will only wait for target pid to "
                    "exit\n");
        }
        _signals.clear();
    }
    else
    {
        // if specified by set_signals(...)
        for(auto itr : m_timer_data.m_signals)
            _signals.emplace(itr);
        // if specified by template parameters
        TIMEMORY_FOLD_EXPRESSION(_signals.emplace(SigIds));
    }

    if(!_signals.empty())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose >= 3,
                                        "configuring %zu signal handlers (index: %zu)",
                                        _signals.size(), m_idx);
        auto& _custom_sa = m_timer_data.m_custom_sigaction;

        memset(&_custom_sa, 0, sizeof(_custom_sa));

        if(m_timer_data.m_flags & (1 << SA_SIGINFO))
            _custom_sa.sa_sigaction = &this_type::execute;
        else
            _custom_sa.sa_handler = &this_type::execute;
        _custom_sa.sa_flags = m_timer_data.m_flags;

        // block signals on thread while configuring
        sampling::block_signals(_signals, sigmask_scope::thread);

        // start the interval timer
        for(const auto& itr : _signals)
        {
            // already active
            if(m_timer_data.m_active[itr])
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE(
                    _verbose >= 3, "handler for signal %i is already active (index: %zu)",
                    itr, m_idx);
                continue;
            }
            TIMEMORY_CONDITIONAL_PRINT_HERE(
                _verbose >= 3, "configuring handler for signal %i (index: %zu)", itr,
                m_idx);

            // get the associated itimer type
            auto _itimer = get_itimer(itr);
            if(_itimer < 0 && (itr < SIGRTMIN || itr > SIGRTMAX))
            {
                TIMEMORY_EXCEPTION(TIMEMORY_JOIN(
                    " ", "Error! Alarm cannot be set for signal", itr,
                    "because the signal does not map to a known itimer "
                    "value or is not a real timer >= SIGRTMIN and <= SIGRTMAX"));
            }

            // configure the sigaction
            if(sigaction(itr, &_custom_sa, &m_timer_data.m_original_sigaction) == 0)
            {
                m_timer_data.m_signals.insert(itr);
            }
            else
            {
                TIMEMORY_EXCEPTION(TIMEMORY_JOIN(
                    " ", "Error! sigaction could not be set for signal", itr));
            }

            // ensure itimerval exists
            if(m_timer_data.m_custom_itimerval.count(itr) == 0)
            {
                set_delay(m_timer_data.m_delay, { itr });
                set_frequency(m_timer_data.m_freq, { itr });
            }

            if(_itimer < 0)
            {
                auto& _timer                  = m_timer_data.m_timer[itr];
                auto& _sigevt                 = m_timer_data.m_sigevent[itr];
                _sigevt.sigev_notify          = SIGEV_SIGNAL;
                _sigevt.sigev_signo           = itr;
                _sigevt.sigev_value.sival_ptr = &_timer;

                auto& _orig = m_timer_data.m_original_itimerspec[itr];
                auto& _curr = m_timer_data.m_custom_itimerspec[itr];

                _curr              = get_itimerspec(m_timer_data.m_custom_itimerval[itr]);
                auto _timer_create = [&]() {
                    return timer_create(CLOCK_REALTIME, &_sigevt, &_timer);
                };

                int _ret = 0;
                while((_ret = _timer_create()) == EAGAIN)
                {
                }

                switch(_ret)
                {
                    case EINVAL:
                        TIMEMORY_EXCEPTION("timer_create failed! Received invalid data");
                        break;
                    case ENOMEM:
                        TIMEMORY_EXCEPTION(
                            "timer_create failed! Could not allocate memory");
                        break;
                    case ENOTSUP:
                        TIMEMORY_EXCEPTION("timer_create failed! Kernel does not support "
                                           "creating a timer "
                                           "against this clock id: CLOCK_REALTIME ("
                                           << CLOCK_REALTIME << ")");
                        break;
                    case EPERM:
                        TIMEMORY_EXCEPTION("timer_create failed! Caller did not have the "
                                           "CAP_WAKE_ALARM capability");
                        break;
                }
                _ret = timer_settime(_timer, 0, &_curr, &_orig);
                switch(_ret)
                {
                    case EFAULT:
                        TIMEMORY_EXCEPTION(
                            "timer_settime failed! Received invalid pointer");
                        break;
                    case EINVAL:
                        TIMEMORY_EXCEPTION("timer_settime failed! timer id is invalid");
                        break;
                }

                m_timer_data.m_original_itimerval[itr] = get_itimerval(_orig);
            }
            else
            {
                // start the alarm (throws if fails)
                check_itimer(setitimer(_itimer,
                                       &(m_timer_data.m_custom_itimerval.at(itr)),
                                       &(m_timer_data.m_original_itimerval[itr])),
                             true);
            }

            m_timer_data.m_active[itr] = true;
        }
    }

    TIMEMORY_CONDITIONAL_PRINT_HERE(
        _verbose >= 3, "signal handler configuration complete (index: %zu)", m_idx);

    // unblock signals on thread after configuring
    if(!_signals.empty() && _unblock)
        sampling::unblock_signals(_signals, sigmask_scope::thread);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::reset(std::set<int> _signals, int _verbose,
                                              bool _unblock)
{
    _verbose = std::max<int>(_verbose, m_verbose);

    TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose >= 3, "resetting sampler (index: %zu)",
                                    m_idx);

    // if specified by set_signals(...)
    for(auto itr : m_timer_data.m_signals)
        _signals.emplace(itr);
    // if specified by template parameters
    TIMEMORY_FOLD_EXPRESSION(_signals.emplace(SigIds));

    if(!_signals.empty())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose >= 3,
                                        "Resetting %zu signal handlers (index: %zu)",
                                        _signals.size(), m_idx);

        // block signals on thread while resetting
        sampling::block_signals(_signals, sigmask_scope::thread);

        // start the interval timer
        for(const auto& itr : _signals)
        {
            // already active
            if(!m_timer_data.m_active[itr])
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE(
                    _verbose >= 3, "handler for signal %i is already reset (index: %zu)",
                    itr, m_idx);
                continue;
            }

            TIMEMORY_CONDITIONAL_PRINT_HERE(
                _verbose >= 3, "resetting handler for signal %i (index: %zu)", itr,
                m_idx);

            // get the associated itimer type
            auto _itimer = get_itimer(itr);
            if(_itimer < 0 && (itr < SIGRTMIN || itr > SIGRTMAX))
            {
                TIMEMORY_EXCEPTION(
                    TIMEMORY_JOIN(" ", "Error! Alarm cannot be reset for signal", itr,
                                  "because the signal does not map to a known itimer "
                                  "value\n"));
            }

            itimerval_t _itimer_zero;
            _itimer_zero.it_interval.tv_sec  = 0;
            _itimer_zero.it_interval.tv_usec = 0;
            _itimer_zero.it_value.tv_sec     = 0;
            _itimer_zero.it_value.tv_usec    = 0;

            if(_itimer < 0)
            {
                auto& _timer = m_timer_data.m_timer[itr];
                timer_delete(_timer);
            }
            else
            {
                // reset the alarm (throws if fails)
                check_itimer(setitimer(_itimer, &_itimer_zero, nullptr), true);
            }

            m_timer_data.m_active[itr] = false;
        }
    }

    TIMEMORY_CONDITIONAL_PRINT_HERE(
        _verbose >= 3, "signal handler configuration complete (index: %zu)", m_idx);

    // unblock signals on thread after configuring
    if(!_signals.empty() && _unblock)
        sampling::unblock_signals(_signals, sigmask_scope::thread);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::ignore(std::set<int> _signals)
{
    if(_signals.empty())
    {
        // if specified by set_signals(...)
        for(auto itr : m_timer_data.m_signals)
            _signals.emplace(itr);
        // if specified by template parameters
        TIMEMORY_FOLD_EXPRESSION(_signals.emplace(SigIds));
    }

    sampling::block_signals(_signals, sigmask_scope::process);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Func>
int
sampler<CompT<Types...>, N, SigIds...>::wait(const pid_t wait_pid, int _verbose,
                                             bool _debug, Func&& _callback)
{
    _verbose = std::max<int>(_verbose, m_verbose);
    if(_debug)
        _verbose = 100;

    if(_verbose >= 4)
        fprintf(stderr, "[%i]> waiting for pid %i...\n", process::get_id(), wait_pid);

    //----------------------------------------------------------------------------------//
    //
    auto print_info = [=](pid_t _pid, int _status, int _errv, int _retv) {
        if(_verbose >= 4)
        {
            fprintf(stderr, "[%i]> return code: %i, error value: %i, status: %i\n", _pid,
                    _retv, _errv, _status);
            fflush(stderr);
        }
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto diagnose_status = [=](pid_t _pid, int status) {
        if(_verbose >= 4)
            fprintf(stderr, "[%i]> diagnosing status %i...\n", _pid, status);

        if(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS)
        {
            if(_verbose >= 4 || (_debug && _verbose >= 2))
            {
                fprintf(stderr, "[%i]> program terminated normally with exit code: %i\n",
                        _pid, WEXITSTATUS(status));
            }
            // normal terminatation
            return 0;
        }

        int ret = WEXITSTATUS(status);
        if(WIFSTOPPED(status))
        {
            int sig = WSTOPSIG(status);
            // stopped with signal 'sig'
            if(_verbose >= 5)
            {
                fprintf(stderr, "[%i]> program stopped with signal %i. Exit code: %i\n",
                        _pid, sig, ret);
            }
        }
        else if(WCOREDUMP(status))
        {
            if(_verbose >= 5)
            {
                fprintf(stderr,
                        "[%i]> program terminated and produced a core dump. Exit "
                        "code: %i\n",
                        _pid, ret);
            }
        }
        else if(WIFSIGNALED(status))
        {
            ret = WTERMSIG(status);
            if(_verbose >= 5)
            {
                fprintf(stderr,
                        "[%i]> program terminated because it received a signal "
                        "(%i) that was not handled. Exit code: %i\n",
                        _pid, WTERMSIG(status), ret);
            }
        }
        else if(WIFEXITED(status) && WEXITSTATUS(status))
        {
            if(ret == 127 && (_verbose >= 5))
            {
                fprintf(stderr, "[%i]> execv failed\n", _pid);
            }
            else if(_verbose >= 5)
            {
                fprintf(stderr,
                        "[%i]> program terminated with a non-zero status. Exit "
                        "code: %i\n",
                        _pid, ret);
            }
        }
        else
        {
            if(_verbose >= 5)
                fprintf(stderr, "[%i]> program terminated abnormally.\n", _pid);
            ret = EXIT_FAILURE;
        }

        return ret;
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto waitpid_eintr = [&](pid_t _pid, int& status) {
        pid_t pid    = 0;
        int   errval = 0;
        int   retval = 0;

        while((pid = waitpid(WAIT_ANY, &status, 0)) == -1)
        {
            errval = errno;
            if(errval == EINTR)
                continue;
            if(errno != errval)
                perror("Unexpected error in waitpid_eitr");
            retval = diagnose_status(pid, status);
            print_info(pid, status, errval, retval);
            break;
        }

        if(errval == ECHILD)
        {
            do
            {
                retval = kill(_pid, 0);
                errval = errno;
                // retval = diagnose_status(_pid, status);
                // print_info(_pid, status, errval, retval);
                if(errval == ESRCH || retval == -1)
                    break;
                std::this_thread::sleep_for(std::chrono::microseconds(
                    static_cast<int64_t>(get_frequency(units::usec))));
            } while(true);
        }

        return errval;
    };
    //
    //----------------------------------------------------------------------------------//

    auto _signals = m_timer_data.m_signals;
    int  status   = 0;
    int  errval   = 0;

    // do not wait on self to exit so execute callback until
    if(_signals.empty() && wait_pid == process::get_id())
    {
        do
        {
            std::this_thread::sleep_for(std::chrono::microseconds(
                static_cast<int64_t>(get_frequency(units::usec))));
        } while(_callback(wait_pid, status, errval));
        return diagnose_status(wait_pid, status);
    }

    // loop while the errno is not EINTR (interrupt) and status designates
    // it was stopped because of signal
    int retval = 0;
    do
    {
        status = 0;
        errval = waitpid_eintr(wait_pid, status);
        print_info(wait_pid, status, errval, retval);
    } while((errval == EINTR &&
             _signals.count(retval = diagnose_status(wait_pid, status)) != 0) &&
            (_callback(wait_pid, status, errval)));

    print_info(wait_pid, status, errval, retval);

    return diagnose_status(wait_pid, status);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::set_delay(double fdelay, std::set<int> _signals,
                                                  bool _verbose)
{
    m_timer_data.m_delay = std::max<double>(fdelay, m_timer_data.m_delay);
    if(_signals.empty())
        _signals = m_timer_data.m_signals;
    for(const auto& itr : _signals)
    {
        sampling::set_delay(m_timer_data.m_custom_itimerval[itr], fdelay,
                            std::string{ "[" } + std::to_string(itr) + "]", _verbose);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::set_frequency(double        ffreq,
                                                      std::set<int> _signals,
                                                      bool          _verbose)
{
    m_timer_data.m_freq = ffreq;
    if(_signals.empty())
        _signals = m_timer_data.m_signals;
    for(const auto& itr : _signals)
    {
        sampling::set_frequency(m_timer_data.m_custom_itimerval[itr], ffreq,
                                std::string{ "[" } + std::to_string(itr) + "]", _verbose);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
double
sampler<CompT<Types...>, N, SigIds...>::get_delay(int64_t units, int signum) const
{
    auto _signals = m_timer_data.m_signals;
    for(auto itr : std::initializer_list<int>{ SigIds... })
        _signals.emplace(itr);

    if(_signals.empty())
        return 0.0;

    if(signum >= 0)
        return sampling::get_delay(m_timer_data.m_custom_itimerval.at(signum), units);

    double _val = std::numeric_limits<double>::max();
    for(const auto& itr : _signals)
    {
        _val = std::min<double>(
            _val, sampling::get_delay(m_timer_data.m_custom_itimerval.at(itr), units));
    }
    return _val;
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
double
sampler<CompT<Types...>, N, SigIds...>::get_period(int64_t units, int signum) const
{
    auto _signals = m_timer_data.m_signals;
    for(auto itr : std::initializer_list<int>{ SigIds... })
        _signals.emplace(itr);

    if(_signals.empty())
        return 0.0;

    if(signum >= 0)
        return sampling::get_period(m_timer_data.m_custom_itimerval.at(signum), units);

    double _val = std::numeric_limits<double>::max();
    for(const auto& itr : _signals)
    {
        _val = std::min<double>(
            _val, sampling::get_period(m_timer_data.m_custom_itimerval.at(itr), units));
    }
    return _val;
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
double
sampler<CompT<Types...>, N, SigIds...>::get_frequency(int64_t units, int signum) const
{
    auto _signals = m_timer_data.m_signals;
    for(auto itr : std::initializer_list<int>{ SigIds... })
        _signals.emplace(itr);

    if(_signals.empty())
        return 0.0;

    return 1.0 / get_period(units, signum);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
int
sampler<CompT<Types...>, N, SigIds...>::get_itimer(int _signal)
{
    int _itimer = -1;
    switch(_signal)
    {
        case SIGALRM: _itimer = ITIMER_REAL; break;
        case SIGVTALRM: _itimer = ITIMER_VIRTUAL; break;
        case SIGPROF: _itimer = ITIMER_PROF; break;
    }
    return _itimer;
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
bool
sampler<CompT<Types...>, N, SigIds...>::check_itimer(int _stat, bool _throw_exception)
{
    if(_stat == EFAULT)
    {
        auto msg =
            TIMEMORY_JOIN(" ", "Warning! setitimer returned EFAULT.",
                          "Either the new itimerval or the old itimerval was invalid");
        if(_throw_exception)
        {
            TIMEMORY_EXCEPTION(msg)
        }
        else
        {
            std::cerr << msg << '\n';
        }
    }
    else if(_stat == EINVAL)
    {
        auto msg = TIMEMORY_JOIN(" ", "Warning! setitimer returned EINVAL.",
                                 "Either the timer was not one of: ['ITIMER_REAL', "
                                 "'ITIMER_VIRTUAL', "
                                 "'ITIMER_PROF'] or the old itimerval was invalid");
        if(_throw_exception)
        {
            TIMEMORY_EXCEPTION(msg)
        }
        else
        {
            std::cerr << msg << '\n';
        }
    }
    return (_stat != EFAULT && _stat != EINVAL);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::_init_sampler(int64_t _tid)
{
    TIMEMORY_FOLD_EXPRESSION(m_good.insert(SigIds));
    m_data.fill(bundle_type{ m_label });
    m_last = &m_data.front();
    get_samplers(_tid).emplace_back(this);
    if(settings::debug())
        m_verbose += 16;
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<!Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::_init_sampler(int64_t _tid)
{
    TIMEMORY_FOLD_EXPRESSION(m_good.insert(SigIds));
    m_notify();
    m_wait();
    get_samplers(_tid).emplace_back(this);
    if(settings::debug())
        m_verbose += 16;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace sampling
}  // namespace tim

#endif

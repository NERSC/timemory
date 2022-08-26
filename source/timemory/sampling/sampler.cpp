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

#if !defined(TIMEMORY_SAMPLING_SAMPLER_HPP_)
#    include "timemory/sampling/sampler.hpp"
#endif

#include "timemory/backends/threading.hpp"
#include "timemory/components/base.hpp"
#include "timemory/log/color.hpp"
#include "timemory/log/macros.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/operations/types/sample.hpp"
#include "timemory/sampling/allocator.hpp"
#include "timemory/sampling/signals.hpp"
#include "timemory/sampling/timer.hpp"
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
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
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
template <template <typename...> class CompT, size_t N, typename... Types>
auto
sampler<CompT<Types...>, N>::get_latest_samples()
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
template <template <typename...> class CompT, size_t N, typename... Types>
sampler<CompT<Types...>, N>::sampler(std::string _label, int64_t _tid, int _verbose)
: m_verbose{ _verbose }
, m_alloc{ this }
, m_label{ std::move(_label) }
{
    _init_sampler(_tid);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
sampler<CompT<Types...>, N>::~sampler()
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
    m_alloc.join();
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename... Args, typename Tp, enable_if_t<Tp::value>>
void
sampler<CompT<Types...>, N>::sample(Args&&... _args)
{
    if(m_count > 0 && m_count % N == (N - 1))
    {
        bool _completed = false;
        m_notify(&_completed);
        while(!_completed)
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::microseconds{ 1 });
        }
    }
    m_last = &(get_data().at((m_count++) % N));
    IF_CONSTEXPR(trait::provide_backtrace<this_type>::value)
    {
        // e.g. _depth == 4 and _offset == 3 means get last 4 of 7 backtrace entries
        constexpr auto _depth  = trait::backtrace_depth<this_type>::value;
        constexpr auto _offset = trait::backtrace_offset<this_type>::value;
        IF_CONSTEXPR(trait::backtrace_use_libunwind<this_type>::value)
        {
            m_last->template invoke<operation::set_data>(
                get_unw_backtrace<_depth, _offset>());
            m_last->sample(get_unw_backtrace<_depth, _offset>(),
                           std::forward<Args>(_args)...);
        }
        else
        {
            m_last->template invoke<operation::set_data>(
                get_native_backtrace<_depth, _offset>());
            m_last->sample(get_native_backtrace<_depth, _offset>(),
                           std::forward<Args>(_args)...);
        }
    }
    else { m_last->sample(std::forward<Args>(_args)...); }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename... Args, typename Tp, enable_if_t<!Tp::value>>
void
sampler<CompT<Types...>, N>::sample(Args&&... _args)
{
    assert(m_buffer_size > 0);
    if(!m_buffer.is_initialized())
    {
        m_buffer = buffer_t{ m_buffer_size, true };
    }
    else if(m_buffer.is_full())
    {
        m_move(std::move(m_buffer));
        m_buffer = buffer_t{ m_buffer_size, true };
    }
    m_last = m_buffer.request();
    if(!m_last)
        return;
    ++m_count;
    IF_CONSTEXPR(trait::provide_backtrace<this_type>::value)
    {
        // e.g. _depth == 4 and _offset == 3 means get last 4 of 7 backtrace entries
        constexpr auto _depth  = trait::backtrace_depth<this_type>::value;
        constexpr auto _offset = trait::backtrace_offset<this_type>::value;
        IF_CONSTEXPR(trait::backtrace_use_libunwind<this_type>::value)
        {
            m_last->template invoke<operation::set_data>(
                get_unw_backtrace_raw<_depth, _offset>());
            m_last->template invoke<operation::set_data>(
                get_unw_backtrace<_depth, _offset>());
            m_last->sample(get_unw_backtrace<_depth, _offset>(),
                           std::forward<Args>(_args)...);
        }
        else
        {
            m_last->template invoke<operation::set_data>(
                get_native_backtrace<_depth, _offset>());
            m_last->sample(get_native_backtrace<_depth, _offset>(),
                           std::forward<Args>(_args)...);
        }
    }
    else { m_last->sample(std::forward<Args>(_args)...); }
}
//
//--------------------------------------------------------------------------------------//
// static allocator
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename Tp, enable_if_t<Tp::value>>
void
sampler<CompT<Types...>, N>::start()
{
    if(!base_type::get_is_running())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(m_verbose >= 2, "starting (index: %zu)", m_idx);
        tracker_type::start();
        base_type::set_started();
        for(auto& itr : get_data())
            itr.start();
        for(auto& itr : m_timers)
            itr->start();
    }
}
//
//--------------------------------------------------------------------------------------//
// static allocator
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename Tp, enable_if_t<Tp::value>>
void
sampler<CompT<Types...>, N>::stop()
{
    if(base_type::get_is_running())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(m_verbose >= 2, "stopping (index: %zu)", m_idx);
        tracker_type::stop();
        base_type::set_stopped();
        for(auto& itr : m_timers)
            itr->stop();
        for(auto& itr : get_data())
            itr.stop();
    }
}
//
//--------------------------------------------------------------------------------------//
// dynamic allocator
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename Tp, enable_if_t<!Tp::value>>
void
sampler<CompT<Types...>, N>::start()
{
    if(!base_type::get_is_running())
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
        // notify the allocator thread so that it can wake and block the signals
        // before we start the timers
        m_alloc.start();
        for(auto& itr : m_timers)
            itr->start();
    }
}
//
//--------------------------------------------------------------------------------------//
// dynamic allocator
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename Tp, enable_if_t<!Tp::value>>
void
sampler<CompT<Types...>, N>::stop()
{
    if(base_type::get_is_running())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(m_verbose >= 2, "stopping (index: %zu)", m_idx);
        tracker_type::stop();
        base_type::set_stopped();
        for(auto& itr : m_timers)
            itr->stop();
    }
    auto _alloc_emplace = [&](buffer_t& itr) {
        if(itr.is_initialized() && !itr.is_empty())
        {
            auto _v = buffer_t{};
            std::swap(itr, _v);
            m_alloc.emplace(std::move(_v));
        }
    };
    _alloc_emplace(m_buffer);
    m_exit();
    m_alloc.join();
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename Tp, enable_if_t<Tp::value>>
const typename sampler<CompT<Types...>, N>::bundle_type&
sampler<CompT<Types...>, N>::get(size_t idx) const
{
    return get_data().at(idx % N);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename Tp, enable_if_t<!Tp::value>>
const typename sampler<CompT<Types...>, N>::bundle_type&
sampler<CompT<Types...>, N>::get(size_t idx) const
{
    return get_data().at(idx);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
void
sampler<CompT<Types...>, N>::execute(int signum)
{
    for(auto& itr : get_samplers(threading::get_id()))
    {
        if(!itr)
            continue;

        IF_CONSTEXPR(trait::prevent_reentry<this_type>::value)
        {
            if(itr->m_sig_lock > 0)
                continue;
            itr->m_sig_lock = 1;
        }

        itr->sample(signum);

        IF_CONSTEXPR(trait::prevent_reentry<this_type>::value) { itr->m_sig_lock = 0; }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
void
sampler<CompT<Types...>, N>::execute(int signum, siginfo_t* _info, void* _data)
{
    static thread_local sig_atomic_t _sig_lock = 0;
    IF_CONSTEXPR(trait::prevent_reentry<this_type>::value)
    {
        if(_sig_lock > 0)
            return;
        _sig_lock = 1;
    }

    for(auto& itr : get_samplers(threading::get_id()))
    {
        if(!itr)
            continue;

        IF_CONSTEXPR(trait::prevent_reentry<this_type>::value)
        {
            if(itr->m_sig_lock > 0)
                continue;
            itr->m_sig_lock = 1;
        }

        itr->sample(signum, _info, _data);

        IF_CONSTEXPR(trait::prevent_reentry<this_type>::value) { itr->m_sig_lock = 0; }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
void
sampler<CompT<Types...>, N>::configure(timer&& _v)
{
    auto _verbose = m_verbose;

    TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose >= 3, "configuring sampler (index: %zu)",
                                    m_idx);

    auto _t      = std::make_unique<timer>(std::move(_v));
    auto _signum = _t->signal();

    TIMEMORY_CONDITIONAL_PRINT_HERE(
        _verbose >= 3, "configuring signal handler for %i (index: %zu)", _signum, m_idx);

    auto& _custom_sa = m_custom_sigaction;

    memset(&_custom_sa, 0, sizeof(_custom_sa));

    if((m_flags & (1 << SA_SIGINFO)) != 0)
        _custom_sa.sa_sigaction = &this_type::execute;
    else
        _custom_sa.sa_handler = &this_type::execute;
    _custom_sa.sa_flags = m_flags;

    // provide the signal to the allocator thread so it can block it
    m_alloc.block_signal(_signum);

    TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose >= 3,
                                    "configuring handler for signal %i (index: %zu)",
                                    TIMEMORY_JOIN("", *_t).c_str(), m_idx);

    // configure the sigaction
    if(sigaction(_signum, &_custom_sa, &m_original_sigaction) == 0)
    {
        if(!_t->is_initialized())
            _t->initialize();
        m_timers.emplace_back(std::move(_t));
    }
    else
    {
        TIMEMORY_EXCEPTION(
            TIMEMORY_JOIN(" ", "Error! sigaction could not be set for signal", *_t));
    }

    TIMEMORY_CONDITIONAL_PRINT_HERE(
        _verbose >= 3, "signal handler for %i configuration complete (index: %zu)",
        _signum, m_idx);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
void
sampler<CompT<Types...>, N>::reset(std::vector<timer_pointer_t>&& _timers)
{
    auto _verbose = m_verbose;

    if(_timers.empty())
        _timers = std::move(m_timers);

    auto _signals = std::set<int>{};
    for(auto& itr : _timers)
        _signals.emplace(itr->signal());

    TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose >= 3, "resetting sampler (index: %zu)",
                                    m_idx);

    if(!_timers.empty())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(_verbose >= 3,
                                        "Resetting %zu signal handlers (index: %zu)",
                                        _timers.size(), m_idx);
        // block signals on thread while resetting
        sampling::block_signals(_signals, sigmask_scope::thread);

        // stop the interval timer
        for(auto& itr : _timers)
            itr->stop();

        // unblock signals on thread after resetting
        sampling::unblock_signals(_signals, sigmask_scope::thread);
    }

    TIMEMORY_CONDITIONAL_PRINT_HERE(
        _verbose >= 3, "signal handler configuration complete (index: %zu)", m_idx);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
void
sampler<CompT<Types...>, N>::ignore(std::set<int> _signals)
{
    if(_signals.empty())
    {
        // if specified by set_signals(...)
        for(const auto& itr : m_timers)
            _signals.emplace(itr->signal());
    }

    sampling::block_signals(_signals, sigmask_scope::process);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename Func>
int
sampler<CompT<Types...>, N>::wait(const pid_t wait_pid, int _verbose, bool _debug,
                                  Func&& _callback)
{
    _verbose = std::max<int>(_verbose, m_verbose);
    if(_debug)
        _verbose = 100;

    if(_verbose >= 4)
        TIMEMORY_PRINTF(stderr, "[%i]> waiting for pid %i...\n", process::get_id(),
                        wait_pid);

    int64_t _freq = 10 * units::msec;
    for(auto& itr : m_timers)
        _freq = std::min<int64_t>(_freq, itr->frequency() * units::sec);

    //----------------------------------------------------------------------------------//
    //
    auto print_info = [=](pid_t _pid, int _status, int _errv, int _retv) {
        if(_verbose >= 4)
        {
            TIMEMORY_PRINTF(stderr,
                            "[%i]> return code: %i, error value: %i, status: %i\n", _pid,
                            _retv, _errv, _status);
            fflush(stderr);
        }
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto diagnose_status = [=](pid_t _pid, int status) {
        if(_verbose >= 4)
            TIMEMORY_PRINTF(stderr, "[%i]> diagnosing status %i...\n", _pid, status);

        if(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS)
        {
            if(_verbose >= 4 || (_debug && _verbose >= 2))
            {
                TIMEMORY_PRINTF(stderr,
                                "[%i]> program terminated normally with exit code: %i\n",
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
                TIMEMORY_PRINTF(stderr,
                                "[%i]> program stopped with signal %i. Exit code: %i\n",
                                _pid, sig, ret);
            }
        }
        else if(WCOREDUMP(status))
        {
            if(_verbose >= 5)
            {
                TIMEMORY_PRINTF(stderr,
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
                TIMEMORY_PRINTF(stderr,
                                "[%i]> program terminated because it received a signal "
                                "(%i) that was not handled. Exit code: %i\n",
                                _pid, WTERMSIG(status), ret);
            }
        }
        else if(WIFEXITED(status) && WEXITSTATUS(status))
        {
            if(ret == 127 && (_verbose >= 5))
            {
                TIMEMORY_PRINTF(stderr, "[%i]> execv failed\n", _pid);
            }
            else if(_verbose >= 5)
            {
                TIMEMORY_PRINTF(stderr,
                                "[%i]> program terminated with a non-zero status. Exit "
                                "code: %i\n",
                                _pid, ret);
            }
        }
        else
        {
            if(_verbose >= 5)
                TIMEMORY_PRINTF(stderr, "[%i]> program terminated abnormally.\n", _pid);
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
                std::this_thread::sleep_for(std::chrono::nanoseconds{ _freq });
            } while(true);
        }

        return errval;
    };
    //
    //----------------------------------------------------------------------------------//

    int  status   = 0;
    int  errval   = 0;
    auto _signals = std::set<int>{};
    for(auto& itr : m_timers)
        _signals.emplace(itr->signal());

    // do not wait on self to exit so execute callback until
    if(_signals.empty() && wait_pid == process::get_id())
    {
        do
        {
            std::this_thread::sleep_for(std::chrono::nanoseconds{ _freq });
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
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename Tp, enable_if_t<Tp::value>>
void
sampler<CompT<Types...>, N>::_init_sampler(int64_t _tid)
{
    m_data.fill(bundle_type{ m_label });
    m_last = &m_data.front();
    get_samplers(_tid).emplace_back(this);
    if(settings::debug())
        m_verbose += 16;
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename Tp, enable_if_t<!Tp::value>>
void
sampler<CompT<Types...>, N>::_init_sampler(int64_t _tid)
{
    m_buffer.set_use_mmap(true);
    m_buffer.init(m_buffer_size);
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

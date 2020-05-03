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

#include "timemory/mpl/apply.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/macros.hpp"

// C++ includes
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <type_traits>
#include <vector>

// C includes
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(TIMEMORY_USE_LIBEXPLAIN)
#    include <libexplain/execvp.h>
#endif

#if defined(_UNIX)
#    include <unistd.h>
extern "C"
{
    extern char** environ;
}
#endif

namespace tim
{
namespace sampling
{
//
//--------------------------------------------------------------------------------------//
//
template <typename CompT, size_t N>
struct sampler;
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
struct sampler<CompT<Types...>, N>
{
    using this_type    = sampler<CompT<Types...>, N>;
    using components_t = CompT<Types...>;
    using array_t      = std::array<components_t, N>;
    using signal_set_t = std::set<int>;

    static void  execute(int signum);
    static void  execute(int signum, siginfo_t*, void*);
    static auto& get_samplers() { return get_persistent_data().m_instances; }

    sampler(const std::string& _label, signal_set_t _good,
            signal_set_t _bad = signal_set_t{})
    : m_idx(0)
    , m_last(nullptr)
    , m_good(_good)
    , m_bad(_bad)
    {
        m_data.fill(components_t(_label));
        auto_lock_t lk(type_mutex<this_type>());
        get_samplers().push_back(this);
    }

    ~sampler()
    {
        auto_lock_t lk(type_mutex<this_type>());
        auto&       _samplers = get_samplers();
        auto        itr       = std::find(_samplers.begin(), _samplers.end(), this);
        if(itr != _samplers.end())
            _samplers.erase(itr);
    }

    void sample()
    {
        m_last = &(m_data.at((m_idx++) % N));
        m_last->sample();
    }

    void start()
    {
        for(auto& itr : m_data)
            itr.start();
    }

    void stop()
    {
        for(auto& itr : m_data)
            itr.stop();
    }

    bool is_good(int v) const { return m_good.count(v) > 0; }
    bool is_bad(int v) const { return m_bad.count(v) > 0; }

    components_t*& get_last() { return m_last; }
    components_t*  get_last() const { return m_last; }

    components_t&       get(size_t idx) { return m_data.at(idx % N); }
    const components_t& get(size_t idx) const { return m_data.at(idx % N); }

    array_t&       get_data() { return m_data; }
    const array_t& get_data() const { return m_data; }

public:
    /// \fn configure
    /// \brief Set up the sampler
    static void configure(int _signal = SIGALRM);

    /// \fn ignore
    /// \brief Ignore the sampler
    static void ignore() { signal(get_persistent_data().m_signal, SIG_IGN); }

    /// \fn wait
    /// \brief Wait function with an optional user callback of type:
    ///
    ///         \code bool (*)(int a, int b)
    ///
    /// where 'a' is the status, 'b' is the error value, and returns true if waiting
    /// should continue
    template <typename Func = std::function<bool(int, int)>>
    static int wait(int _verbose = settings::verbose(), bool _debug = settings::debug(),
                    Func&& _callback = [](int, int) { return true; });

    /// \fn set_flags
    /// \brief Set the sigaction flags, e.g. SA_RESTART | SA_SIGINFO
    static void set_flags(int _flags) { get_persistent_data().m_flags = _flags; }

    /// \fn set_delay
    /// \brief Value, expressed in seconds, that sets the length of time the sampler
    /// waits before starting sampling of the relevant measurements
    static void set_delay(const double& fdelay);

    /// \fn set_freq
    /// \brief Value, expressed in 1/seconds, expressed in 1/seconds, that sets the
    /// frequency that the sampler samples the relevant measurements
    static void set_frequency(const double& ffreq);

    /// \fn set_rate
    /// \brief Value, expressed in number of interupts per second, that configures the
    /// frequency that the sampler samples the relevant measurements
    static void set_rate(const double& frate) { set_frequency(1.0 / frate); }

protected:
    size_t        m_idx  = 0;
    components_t* m_last = nullptr;
    signal_set_t  m_good = {};
    signal_set_t  m_bad  = {};
    array_t       m_data = {};

private:
    using sigaction_t = struct sigaction;
    using itimerval_t = struct itimerval;

    struct persistent_data
    {
        int                     m_signal = 0;
        int                     m_flags  = SA_RESTART | SA_SIGINFO;
        double                  m_delay  = 0.001;
        double                  m_freq   = 1.0 / 2.0;
        sigaction_t             m_custom_sigaction;
        itimerval_t             m_custom_itimerval = { { 1, 0 }, { 0, units::msec } };
        sigaction_t             m_original_sigaction;
        itimerval_t             m_original_itimerval;
        std::vector<this_type*> m_instances;
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
void
sampler<CompT<Types...>, N>::execute(int signum)
{
    for(auto& itr : get_samplers())
    {
        if(itr->is_good(signum))
        {
            itr->sample();
        }
        else if(itr->is_bad(signum))
        {
            char msg[1024];
            sprintf(msg, "[timemory]> sampler instance caught bad signal: %i ...",
                    signum);
            perror(msg);
            signal(signum, SIG_DFL);
            raise(signum);
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
void
sampler<CompT<Types...>, N>::execute(int signum, siginfo_t*, void*)
{
    for(auto& itr : get_samplers())
    {
        if(itr->is_good(signum))
        {
            itr->sample();
        }
        else if(itr->is_bad(signum))
        {
            char msg[1024];
            sprintf(msg, "[timemory]> sampler instance caught bad signal: %i ...",
                    signum);
            perror(msg);
            signal(signum, SIG_DFL);
            raise(signum);
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
void
sampler<CompT<Types...>, N>::configure(int _signal)
{
    get_persistent_data().m_signal = _signal;

    int _itimer = 0;
    switch(_signal)
    {
        case SIGALRM: _itimer = ITIMER_REAL; break;
        case SIGVTALRM: _itimer = ITIMER_VIRTUAL; break;
        case SIGPROF: _itimer = ITIMER_PROF; break;
    }

    auto& _custom_sa   = get_persistent_data().m_custom_sigaction;
    auto& _original_sa = get_persistent_data().m_original_sigaction;

    memset(&_custom_sa, 0, sizeof(_custom_sa));
    // sigfillset(&timem_sa.sa_mask);
    // sigdelset(&timem_sa.sa_mask, _signal);

    _custom_sa.sa_handler   = &this_type::execute;
    _custom_sa.sa_sigaction = &this_type::execute;
    _custom_sa.sa_flags     = SA_RESTART | SA_SIGINFO;

    sigaction(_signal, &_custom_sa, &_original_sa);

    for(auto& itr : get_samplers())
        itr->start();

    auto& _custom_it   = get_persistent_data().m_custom_itimerval;
    auto& _original_it = get_persistent_data().m_original_itimerval;

    // start the interval timer
    int _stat = setitimer(_itimer, &_custom_it, &_original_it);
    if(_stat == EFAULT)
    {
        throw std::runtime_error(
            TIMEMORY_JOIN(" ", "Error! setitimer returned EFAULT.",
                          "Either the new itimerval or the old itimerval was invalid"));
    }
    else if(_stat == EINVAL)
    {
        throw std::runtime_error(TIMEMORY_JOIN(
            " ", "Error! setitimer returned EINVAL.",
            "Either the timer was not one of: ['ITIMER_REAL', 'ITIMER_VIRTUAL', "
            "'ITIMER_PROF'] or the old itimerval was invalid"));
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
template <typename Func>
int
sampler<CompT<Types...>, N>::wait(int _verbose, bool _debug, Func&& _callback)
{
    auto diagnose_status = [=](int status) {
        auto _pid = process::get_target_id();

        if(_verbose > 2 || _debug)
            fprintf(stderr, "[%i]> diagnosing status %i...\n", _pid, status);

        if(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS)
        {
            if(_verbose > 2 || (_debug && _verbose > 0))
                fprintf(stderr, "[%i]> program terminated normally with exit code: %i\n",
                        _pid, WEXITSTATUS(status));

            // normal terminatation
            return 0;
        }

        int ret = WEXITSTATUS(status);
        if(WIFSTOPPED(status))
        {
            int sig = WSTOPSIG(status);
            // stopped with signal 'sig'
            if(_debug || _verbose > 3)
                fprintf(stderr, "[%i]> program stopped with signal %i. Exit code: %i\n",
                        _pid, sig, ret);
        }
        else if(WCOREDUMP(status))
        {
            if(_debug || _verbose > 3)
                fprintf(stderr,
                        "[%i]> program terminated and produced a core dump. Exit "
                        "code: %i\n",
                        _pid, ret);
        }
        else if(WIFSIGNALED(status))
        {
            ret = WTERMSIG(status);
            if(_debug || _verbose > 3)
                fprintf(stderr,
                        "[%i]> program terminated because it received a signal "
                        "(%i) that was not handled. Exit code: %i\n",
                        _pid, WTERMSIG(status), ret);
        }
        else if(WIFEXITED(status) && WEXITSTATUS(status))
        {
            if(ret == 127 && (_debug || _verbose > 3))
                fprintf(stderr, "[%i]> execv failed\n", _pid);
            else if(_debug || _verbose > 3)
                fprintf(stderr,
                        "[%i]> program terminated with a non-zero status. Exit "
                        "code: %i\n",
                        _pid, ret);
        }
        else
        {
            if(_debug || _verbose > 3)
                fprintf(stderr, "[%i]> program terminated abnormally.\n", _pid);
            ret = EXIT_FAILURE;
        }

        return ret;
    };

    auto waitpid_eintr = [&](int& status) {
        pid_t pid    = 0;
        int   errval = 0;
        while((pid = waitpid(WAIT_ANY, &status, 0)) == -1)
        {
            errval = errno;
            if(errno != errval)
                perror("Unexpected error in waitpid_eitr");
            int ret = diagnose_status(status);
            if(_debug || _verbose > 2)
                fprintf(stderr, "[%i]> return code: %i\n", pid, ret);
            if(errval == EINTR)
                continue;
            break;
        }
        return errval;
    };

    int _signal = get_persistent_data().m_signal;

    // loop while the errno is not EINTR (interrupt) and status designates
    // it was stopped because of _signal
    int status = 0;
    int errval = 0;
    do
    {
        status = 0;
        errval = waitpid_eintr(status);
    } while((errval == EINTR && diagnose_status(status) == _signal) &&
            (_callback(status, errval)));

    return diagnose_status(status);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
void
sampler<CompT<Types...>, N>::set_delay(const double& fdelay)
{
    get_persistent_data().m_freq = fdelay;
    int delay_sec                = double(fdelay * units::usec) / units::usec;
    int delay_usec               = int(fdelay * units::usec) % units::usec;
    if(settings::debug() || settings::verbose() > 0)
    {
        fprintf(stderr, "sampler delay     : %i sec + %i usec\n", delay_sec, delay_usec);
    }
    // Configure the timer to expire after designated delay...
    get_persistent_data().m_custom_itimerval.it_value.tv_sec  = delay_sec;
    get_persistent_data().m_custom_itimerval.it_value.tv_usec = delay_usec;
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types>
void
sampler<CompT<Types...>, N>::set_frequency(const double& ffreq)
{
    get_persistent_data().m_freq = ffreq;
    int freq_sec                 = double(ffreq * units::usec) / units::usec;
    int freq_usec                = int(ffreq * units::usec) % units::usec;
    if(settings::debug() || settings::verbose() > 0)
    {
        fprintf(stderr, "sampler frequency     : %i sec + %i usec\n", freq_sec,
                freq_usec);
    }
    // Configure the timer to expire after designated delay...
    get_persistent_data().m_custom_itimerval.it_interval.tv_sec  = freq_sec;
    get_persistent_data().m_custom_itimerval.it_interval.tv_usec = freq_usec;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace sampling
}  // namespace tim

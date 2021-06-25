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

#include "timemory/components/base.hpp"
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
#include <utility>
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

#if defined(TIMEMORY_UNIX)
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
template <typename CompT, size_t N, int... SigIds>
struct sampler;
}
//
//--------------------------------------------------------------------------------------//
//
namespace trait
{
template <typename CompT, size_t N>
struct is_component<sampling::sampler<CompT, N>> : true_type
{};
}  // namespace trait
//
//--------------------------------------------------------------------------------------//
//
namespace sampling
{
//
//--------------------------------------------------------------------------------------//
//
/// \value tim::sampling::dynamic
/// \brief A bare enumeration value implicitly convertible to zero.
enum
{
    dynamic = 0
};
//
//--------------------------------------------------------------------------------------//
//
template <size_t N>
struct fixed_size : std::true_type
{};
//
template <>
struct fixed_size<dynamic> : std::false_type
{};
//
template <size_t N>
using fixed_size_t = typename fixed_size<N>::type;
//
//--------------------------------------------------------------------------------------//
//
template <int... Ids>
struct fixed_sig : std::true_type
{};
//
template <>
struct fixed_sig<> : std::false_type
{};
//
template <int... Ids>
using fixed_sig_t = typename fixed_sig<Ids...>::type;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::sampling::sampler
/// \brief The design of the sampler struct is similar to the \ref tim::component::gotcha
/// component: the first template parameter is a specification of a bundle of components
/// which the struct internally takes measurements with and the second template parameter
/// is a size specification. The size specification is to help ensure that the allocation
/// does not grow too significantly, however, by specifying the size as either 0 (zero)
/// or \ref tim::sampling::dynamic, a std::vector is used instead of the fixed-sized
/// std::array.
/// \code{.cpp}
/// // sampling components
/// using sampler_bundle_t = tim::component_tuple<read_char, written_char>;
/// using sample_t         = tim::sampling::sampler<sampler_bundle_t, 1, SIGALRM>;
/// using bundle_t         = tim::component_tuple<wall_clock, sample_t>;
///
/// // create at least one instance before configuring
/// bundle_t sampling_bundle("example");
///
/// sample_t::configure({ SIGALRM });       // configure the sampling
/// sample_t::pause();                      // wait for one signal to be delivered
///
/// sampling_bundle.start();                // start sampling and wall-clock
/// ...
/// sampling_bundle.stop();                 // stop sampling and wall-clock
///
/// sample_t::pause();                      // wait for one signal to be delivered
/// sampler_t::ignore({ SIGALRM });         // ignore future interrupts
/// sampler_t::wait(process::target_pid()); // wait for pid to finish
/// \endcode
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
struct sampler<CompT<Types...>, N, SigIds...>
: component::base<sampler<CompT<Types...>, N, SigIds...>, void>
, private policy::instance_tracker<sampler<CompT<Types...>, N, SigIds...>, false>
{
    using this_type    = sampler<CompT<Types...>, N, SigIds...>;
    using base_type    = component::base<this_type, void>;
    using components_t = CompT<Types...>;
    using signal_set_t = std::set<int>;
    using pid_cb_t     = std::function<bool(pid_t, int, int)>;
    using array_t = conditional_t<fixed_size_t<N>::value, std::array<components_t, N>,
                                  std::vector<components_t>>;

    using array_type   = array_t;
    using tracker_type = policy::instance_tracker<this_type, false>;

    static void  execute(int signum);
    static void  execute(int signum, siginfo_t*, void*);
    static auto& get_samplers() { return get_persistent_data().m_instances; }
    static auto  get_latest_samples();

public:
    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    sampler(const std::string& _label, signal_set_t _good,
            signal_set_t _bad = signal_set_t{});

    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    sampler(const std::string& _label, signal_set_t _good,
            signal_set_t _bad = signal_set_t{});

    ~sampler();

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    void sample();

    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    void sample();

    template <typename Tp = fixed_sig_t<SigIds...>, enable_if_t<Tp::value> = 0>
    void start();
    template <typename Tp = fixed_sig_t<SigIds...>, enable_if_t<Tp::value> = 0>
    void stop();

    template <typename Tp = fixed_sig_t<SigIds...>, enable_if_t<!Tp::value> = 0>
    void start();
    template <typename Tp = fixed_sig_t<SigIds...>, enable_if_t<!Tp::value> = 0>
    void stop();

public:
    TIMEMORY_NODISCARD bool is_good(int v) const { return m_good.count(v) > 0; }
    TIMEMORY_NODISCARD bool is_bad(int v) const { return m_bad.count(v) > 0; }

    auto good_count() const { return m_good.size(); }
    auto bad_count() const { return m_bad.size(); }
    auto count() const { return good_count() + bad_count(); }

    auto& get_good() { return m_good; }
    auto& get_bad() { return m_bad; }

    const auto& get_good() const { return m_good; }
    const auto& get_bad() const { return m_bad; }

    auto backtrace_enabled() const { return m_backtrace; }
    void enable_backtrace(bool val) { m_backtrace = val; }

    components_t*& get_last() { return m_last; }
    components_t*  get_last() const { return m_last; }

    components_t*& get_latest() { return m_last; }
    components_t*  get_latest() const { return m_last; }

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    components_t& get(size_t idx);
    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    components_t& get(size_t idx);

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    const components_t& get(size_t idx) const;
    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    const components_t& get(size_t idx) const;

    array_t&       get_data() { return m_data; }
    const array_t& get_data() const { return m_data; }

public:
    /// \fn void configure(std::set<int> _signals, int _verb)
    /// \param[in] _signals A set of signals to catch
    /// \param[in] _verb Logging Verbosity
    ///
    /// \brief Set up the sampler
    static TIMEMORY_INLINE void configure(std::set<int> _signals, int _verbose = 1);
    static TIMEMORY_INLINE void configure(int _signal = SIGALRM, int _verbose = 1)
    {
        configure({ _signal }, _verbose);
    }

    /// \fn void ignore(const std::set<int>& _signals)
    /// \param[in] _signals Set of signals
    ///
    /// \brief Ignore the signals
    static TIMEMORY_INLINE void ignore(const std::set<int>& _signals);

    /// \fn void clear()
    /// \brief Clear all signals. Recommended to call ignore() prior to clearing all the
    /// signals
    static void clear() { get_persistent_data().m_signals.clear(); }

    /// \fn void pause()
    /// \brief Pause until a signal is delivered
    static TIMEMORY_INLINE void pause()
    {
        if(!get_persistent_data().m_signals.empty())
            ::pause();
    }

    /// \fn int wait(pid_t _pid, int _verb, bool _debug, Func&& _cb)
    /// \param[in] _pid Process id to wait on
    /// \param[in] _verb Logging verbosity
    /// \param[in] _debug Enable debug logging
    /// \param[in] _cb Callback for checking whether to exit
    ///
    /// \brief Wait function with an optional user callback of type:
    ///
    /// \code{.cpp}
    ///     bool (*)(int a, int b)
    /// \endcode
    /// where 'a' is the status, 'b' is the error value, and returns true if waiting
    /// should continue
    template <typename Func = pid_cb_t>
    static int wait(pid_t _pid, int _verbose, bool _debug,
                    Func&& _callback = pid_callback());

    template <typename Func = pid_cb_t>
    static int wait(int _verbose = settings::verbose(), bool _debug = settings::debug(),
                    Func&& _callback = pid_callback())
    {
        return wait(process::get_target_id(), _verbose, _debug,
                    std::forward<Func>(_callback));
    }

    template <typename Func, enable_if_t<std::is_function<Func>::value> = 0>
    static int wait(pid_t _pid, Func&& _callback, int _verbose = settings::verbose(),
                    bool _debug = settings::debug())
    {
        return wait(_pid, _verbose, _debug, std::forward<Func>(_callback));
    }

    template <typename Func, enable_if_t<std::is_function<Func>::value> = 0>
    static int wait(Func&& _callback, int _verbose = settings::verbose(),
                    bool _debug = settings::debug())
    {
        return wait(process::get_target_id(), _verbose, _debug,
                    std::forward<Func>(_callback));
    }

    /// \fn void set_flags(int)
    /// \param flags[in] the sigaction flags to use
    ///
    /// \brief Set the sigaction flags, e.g. SA_RESTART | SA_SIGINFO
    static void set_flags(int _flags) { get_persistent_data().m_flags = _flags; }

    /// \fn void set_delay(double)
    /// \brief Value, expressed in seconds, that sets the length of time the sampler
    /// waits before starting sampling of the relevant measurements
    static void set_delay(double fdelay);

    /// \fn void set_freq(double)
    /// \brief Value, expressed in 1/seconds, expressed in 1/seconds, that sets the
    /// frequency that the sampler samples the relevant measurements
    static void set_frequency(double ffreq);

    /// \fn void set_rate(double)
    /// \brief Value, expressed in number of interupts per second, that configures the
    /// frequency that the sampler samples the relevant measurements
    static void set_rate(double frate) { set_frequency(1.0 / frate); }

    /// \fn int64_t get_delay(int64_t)
    /// \brief Get the delay of the sampler
    static int64_t get_delay(int64_t units = units::usec);

    /// \fn int64_t get_frequency(int64_t)
    /// \brief Get the frequency of the sampler
    static int64_t get_frequency(int64_t units = units::usec);

    /// \fn int get_itimer(int)
    /// \brief Returns the itimer value associated with the given signal
    static int get_itimer(int _signal);

    /// \fn bool check_itimer(int, bool)
    /// \brief Checks to see if there was an error setting or getting itimer val
    static bool check_itimer(int _stat, bool _throw_exception = false);

protected:
    bool          m_backtrace = false;
    size_t        m_idx       = 0;
    components_t* m_last      = nullptr;
    signal_set_t  m_good      = {};
    signal_set_t  m_bad       = {};
    array_t       m_data      = {};

private:
    using sigaction_t = struct sigaction;
    using itimerval_t = struct itimerval;

    struct persistent_data
    {
        bool                    m_active = false;
        int                     m_flags  = SA_RESTART | SA_SIGINFO;
        double                  m_delay  = 0.001;
        double                  m_freq   = 1.0 / 2.0;
        sigaction_t             m_custom_sigaction;
        itimerval_t             m_custom_itimerval = { { 1, 0 }, { 0, units::msec } };
        sigaction_t             m_original_sigaction;
        itimerval_t             m_original_itimerval;
        std::set<int>           m_signals   = {};
        std::vector<this_type*> m_instances = {};
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }

    /// \fn pid_cb_t& pid_callback()
    /// \brief Default callback when configuring sampler
    static pid_cb_t pid_callback()
    {
        return [](pid_t _id, int, int) { return _id != process::get_id(); };
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
inline auto
sampler<CompT<Types...>, N, SigIds...>::get_latest_samples()
{
    std::vector<components_t*> _last{};
    auto_lock_t                lk(type_mutex<this_type>());
    _last.reserve(get_persistent_data().m_instances.size());
    for(auto& itr : get_persistent_data().m_instances)
        _last.emplace_back(itr->get_last());
    return _last;
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<Tp::value>>
sampler<CompT<Types...>, N, SigIds...>::sampler(const std::string& _label,
                                                signal_set_t _good, signal_set_t _bad)
: m_last(nullptr)
, m_good(std::move(_good))
, m_bad(std::move(_bad))
{
    TIMEMORY_FOLD_EXPRESSION(m_good.insert(SigIds));
    m_data.fill(components_t(_label));
    m_last = &m_data.front();
    auto_lock_t lk(type_mutex<this_type>());
    get_samplers().push_back(this);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<!Tp::value>>
sampler<CompT<Types...>, N, SigIds...>::sampler(const std::string& _label,
                                                signal_set_t _good, signal_set_t _bad)
: m_last(nullptr)
, m_good(std::move(_good))
, m_bad(std::move(_bad))
{
    TIMEMORY_FOLD_EXPRESSION(m_good.insert(SigIds));
    m_data.emplace_back(components_t(_label));
    m_last = &m_data.front();
    auto_lock_t lk(type_mutex<this_type>());
    get_samplers().push_back(this);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
sampler<CompT<Types...>, N, SigIds...>::~sampler()
{
    auto_lock_t lk(type_mutex<this_type>());
    auto&       _samplers = get_samplers();
    auto        itr       = std::find(_samplers.begin(), _samplers.end(), this);
    if(itr != _samplers.end())
        _samplers.erase(itr);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::sample()
{
    // if(!base_type::get_is_running())
    //    return;
    m_last = &(m_data.at((m_idx++) % N));
    // get last 4 of 7 backtrace entries (i.e. offset by 3)
    if(m_backtrace)
    {
        m_last->sample(get_backtrace<4, 3>());
    }
    else
    {
        m_last->sample();
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<!Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::sample()
{
    // if(!base_type::get_is_running())
    //    return;
    m_last = &m_data.back();
    m_data.emplace_back(components_t(m_last->hash()));
    // get last 4 of 7 backtrace entries (i.e. offset by 3)
    if(m_backtrace)
    {
        m_last->sample(get_backtrace<4, 3>());
    }
    else
    {
        m_last->sample();
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
    auto cnt = tracker_type::start();
    base_type::set_started();
    for(auto& itr : m_data)
        itr.start();
    if(cnt == 0)
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
    auto cnt = tracker_type::stop();
    base_type::set_stopped();
    for(auto& itr : m_data)
        itr.stop();
    if(cnt == 0)
        ignore({ SigIds... });
}
//
//--------------------------------------------------------------------------------------//
// no signals specified in template parameters
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<!Tp::value>>
void
sampler<CompT<Types...>, N, SigIds...>::start()
{
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
    base_type::set_stopped();
    for(auto& itr : m_data)
        itr.stop();
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<Tp::value>>
typename sampler<CompT<Types...>, N, SigIds...>::components_t&
sampler<CompT<Types...>, N, SigIds...>::get(size_t idx)
{
    return m_data.at(idx % N);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<Tp::value>>
const typename sampler<CompT<Types...>, N, SigIds...>::components_t&
sampler<CompT<Types...>, N, SigIds...>::get(size_t idx) const
{
    return m_data.at(idx % N);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<!Tp::value>>
typename sampler<CompT<Types...>, N, SigIds...>::components_t&
sampler<CompT<Types...>, N, SigIds...>::get(size_t idx)
{
    return m_data.at(idx);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
template <typename Tp, enable_if_t<!Tp::value>>
const typename sampler<CompT<Types...>, N, SigIds...>::components_t&
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
    if(settings::debug())
    {
        printf("[pid=%i][tid=%i][%s]> sampling...\n", (int) process::get_id(),
               (int) threading::get_id(), demangle<this_type>().c_str());
    }

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
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::execute(int signum, siginfo_t*, void*)
{
    if(settings::debug())
    {
        printf("[pid=%i][tid=%i][%s]> sampling...\n", (int) process::get_id(),
               (int) threading::get_id(), demangle<this_type>().c_str());
    }

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
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::configure(std::set<int> _signals, int _verbose)
{
    // already active
    if(get_persistent_data().m_active)
        return;

    size_t wait_count = 0;
    {
        auto_lock_t lk(type_mutex<this_type>());
        for(auto& itr : get_samplers())
            wait_count += itr->count();
    }

    if(wait_count == 0)
    {
        if(_verbose > 0)
        {
            fprintf(
                stderr,
                "[sampler::configure]> No existing sampler has been configured to "
                "sample at a specific signal or fail at a specific signal. itimer "
                "for will not be set. Sampler will only wait for target pid to exit\n");
        }
        _signals.clear();
    }
    else
    {
        TIMEMORY_FOLD_EXPRESSION(_signals.insert(SigIds));
    }

    if(!_signals.empty())
    {
        auto& _custom_sa   = get_persistent_data().m_custom_sigaction;
        auto& _custom_it   = get_persistent_data().m_custom_itimerval;
        auto& _original_sa = get_persistent_data().m_original_sigaction;
        auto& _original_it = get_persistent_data().m_original_itimerval;

        memset(&_custom_sa, 0, sizeof(_custom_sa));

        _custom_sa.sa_handler   = &this_type::execute;
        _custom_sa.sa_sigaction = &this_type::execute;
        _custom_sa.sa_flags     = SA_RESTART | SA_SIGINFO;

        // start the interval timer
        for(const auto& itr : _signals)
        {
            // get the associated itimer type
            auto _itimer = get_itimer(itr);
            if(_itimer < 0)
            {
                TIMEMORY_EXCEPTION(TIMEMORY_JOIN(
                    " ", "Error! Alarm cannot be set for signal", itr,
                    "because the signal does not map to a known itimer value\n"));
            }

            // configure the sigaction
            int _sret = sigaction(itr, &_custom_sa, &_original_sa);
            if(_sret == 0)
            {
                get_persistent_data().m_signals.insert(itr);
            }
            else
            {
                TIMEMORY_EXCEPTION(TIMEMORY_JOIN(
                    " ", "Error! sigaction could not be set for signal", itr));
            }

            // start the alarm (throws if fails)
            check_itimer(setitimer(_itimer, &_custom_it, &_original_it), true);
        }
    }

    // if active field based on whether there are signals
    get_persistent_data().m_active = !get_persistent_data().m_signals.empty();
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
inline void
sampler<CompT<Types...>, N, SigIds...>::ignore(const std::set<int>& _signals)
{
    for(const auto& itr : _signals)
        signal(itr, SIG_IGN);

    auto& _original_it = get_persistent_data().m_original_itimerval;
    for(const auto& itr : _signals)
    {
        itimerval_t _curr;
        auto        _itimer = get_itimer(itr);
        if(_itimer < 0)
            continue;
        check_itimer(getitimer(_itimer, &_curr));
        // stop the alarm
        if(_curr.it_interval.tv_usec > 0 || _curr.it_interval.tv_sec > 0)
            check_itimer(setitimer(_itimer, &_original_it, &_curr));
    }

    // if active field based on whether there are signals
    get_persistent_data().m_active = !get_persistent_data().m_signals.empty();
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
    if(_verbose > 2 || _debug)
        fprintf(stderr, "[%i]> waiting for pid %i...\n", process::get_id(), wait_pid);

    //----------------------------------------------------------------------------------//
    //
    auto print_info = [=](pid_t _pid, int _status, int _errv, int _retv) {
        if(_debug || _verbose > 2)
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
        if(_verbose > 2 || _debug)
            fprintf(stderr, "[%i]> diagnosing status %i...\n", _pid, status);

        if(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS)
        {
            if(_verbose > 2 || (_debug && _verbose > 0))
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
            if(_debug || _verbose > 3)
            {
                fprintf(stderr, "[%i]> program stopped with signal %i. Exit code: %i\n",
                        _pid, sig, ret);
            }
        }
        else if(WCOREDUMP(status))
        {
            if(_debug || _verbose > 3)
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
            if(_debug || _verbose > 3)
            {
                fprintf(stderr,
                        "[%i]> program terminated because it received a signal "
                        "(%i) that was not handled. Exit code: %i\n",
                        _pid, WTERMSIG(status), ret);
            }
        }
        else if(WIFEXITED(status) && WEXITSTATUS(status))
        {
            if(ret == 127 && (_debug || _verbose > 3))
            {
                fprintf(stderr, "[%i]> execv failed\n", _pid);
            }
            else if(_debug || _verbose > 3)
            {
                fprintf(stderr,
                        "[%i]> program terminated with a non-zero status. Exit "
                        "code: %i\n",
                        _pid, ret);
            }
        }
        else
        {
            if(_debug || _verbose > 3)
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
                std::this_thread::sleep_for(
                    std::chrono::microseconds(get_frequency(units::usec)));
            } while(true);
        }

        return errval;
    };
    //
    //----------------------------------------------------------------------------------//

    auto _signals = get_persistent_data().m_signals;
    int  status   = 0;
    int  errval   = 0;

    // do not wait on self to exit so execute callback until
    if(_signals.empty() && wait_pid == process::get_id())
    {
        do
        {
            std::this_thread::sleep_for(
                std::chrono::microseconds(get_frequency(units::usec)));
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
sampler<CompT<Types...>, N, SigIds...>::set_delay(double fdelay)
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
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::set_frequency(double ffreq)
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
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
inline int64_t
sampler<CompT<Types...>, N, SigIds...>::get_delay(int64_t units)
{
    float _us = (get_persistent_data().m_custom_itimerval.it_value.tv_sec * units::usec) +
                get_persistent_data().m_custom_itimerval.it_value.tv_usec;
    _us *= static_cast<float>(units) / units::usec;
    return std::max<int64_t>(_us, 1);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
inline int64_t
sampler<CompT<Types...>, N, SigIds...>::get_frequency(int64_t units)
{
    float _us =
        (get_persistent_data().m_custom_itimerval.it_interval.tv_sec * units::usec) +
        get_persistent_data().m_custom_itimerval.it_interval.tv_usec;
    _us *= static_cast<float>(units) / units::usec;
    return std::max<int64_t>(_us, 1);
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
}  // namespace sampling
}  // namespace tim

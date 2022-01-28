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

template <typename Tp>
struct check_signals : std::true_type
{};

template <typename Tp>
struct prevent_reentry : std::true_type
{};

template <typename Tp>
struct buffer_size : std::integral_constant<size_t, 0>
{};

template <typename Tp>
struct backtrace_depth : std::integral_constant<size_t, TIMEMORY_SAMPLER_DEPTH_DEFAULT>
{};

template <typename Tp>
struct backtrace_offset : std::integral_constant<size_t, TIMEMORY_SAMPLER_OFFSET_DEFAULT>
{};

template <typename Tp>
struct backtrace_use_libunwind
: std::conditional_t<TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT, std::true_type,
                     std::false_type>
{};
}  // namespace trait
//
//--------------------------------------------------------------------------------------//
//
namespace sampling
{
using itimerval_t = struct itimerval;
using sigaction_t = struct sigaction;
//
inline void
set_delay(itimerval_t& _itimer, double fdelay, const std::string& _extra = "")
{
    int delay_sec  = fdelay;
    int delay_usec = static_cast<int>(fdelay * 1000000) % 1000000;
    if(settings::debug() || settings::verbose() > 0)
    {
        fprintf(stderr, "[T%i]%s sampler delay         : %i sec + %i usec\n",
                (int) threading::get_id(), _extra.c_str(), delay_sec, delay_usec);
    }
    // Configure the timer to expire after designated delay...
    _itimer.it_value.tv_sec  = delay_sec;
    _itimer.it_value.tv_usec = delay_usec;
}
//
inline void
set_frequency(itimerval_t& _itimer, double ffreq, const std::string& _extra = "")
{
    int freq_sec  = ffreq;
    int freq_usec = static_cast<int>(ffreq * 1000000) % 1000000;
    if(settings::debug() || settings::verbose() > 0)
    {
        fprintf(stderr, "[T%i]%s sampler frequency     : %i sec + %i usec\n",
                (int) threading::get_id(), _extra.c_str(), freq_sec, freq_usec);
    }
    // Configure the timer to expire after designated delay...
    _itimer.it_interval.tv_sec  = freq_sec;
    _itimer.it_interval.tv_usec = freq_usec;
}
//
inline double
get_delay(const itimerval_t& _itimer, int64_t units = units::sec)
{
    double _ns =
        (_itimer.it_value.tv_sec * units::sec) + (_itimer.it_value.tv_usec * units::usec);
    return _ns / static_cast<double>(units);
}
//
inline double
get_rate(const itimerval_t& _itimer, int64_t units = units::sec)
{
    double _ns = (_itimer.it_interval.tv_sec * units::sec) +
                 (_itimer.it_interval.tv_usec * units::usec);
    return _ns / static_cast<double>(units);
}
//
inline double
get_frequency(const itimerval_t& _itimer, int64_t units = units::sec)
{
    return 1.0 / get_rate(_itimer, units);
}
//
//--------------------------------------------------------------------------------------//
//
/// \value tim::sampling::dynamic
/// \brief A bare enumeration value implicitly convertible to zero.
enum
{
    dynamic = 0,
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
, private policy::instance_tracker<sampler<CompT<Types...>, N, SigIds...>, true>
{
    using this_type    = sampler<CompT<Types...>, N, SigIds...>;
    using base_type    = component::base<this_type, void>;
    using bundle_type  = CompT<Types...>;
    using signal_set_t = std::set<int>;
    using pid_cb_t     = std::function<bool(pid_t, int, int)>;
    using array_t      = conditional_t<fixed_size_t<N>::value, std::array<bundle_type, N>,
                                  std::vector<bundle_type>>;
    using data_type    = array_t;
    using allocator_t =
        conditional_t<fixed_size_t<N>::value, allocator<void>, allocator<this_type>>;

    using array_type   = array_t;
    using tracker_type = policy::instance_tracker<this_type, true>;

    friend struct allocator<this_type>;

    static_assert(!fixed_size_t<N>::value || trait::buffer_size<this_type>::value > 0,
                  "Error! Dynamic sampler has a default buffer size of zero");

    static constexpr bool is_dynamic() { return !fixed_size_t<N>::value; }

    static void execute(int signum);
    static void execute(int signum, siginfo_t*, void*);

    static auto get_latest_samples();
    static auto get_samplers()
    {
        auto_lock_t            _lk{ type_mutex<this_type>() };
        std::deque<this_type*> _v{};
        for(auto itr : get_persistent_data().m_thread_instances)
        {
            for(auto vitr : itr.second)
                _v.emplace_back(vitr);
        }
        return _v;
    }
    static auto& get_samplers(int64_t _v)
    {
        return get_persistent_data().m_thread_instances[_v];
    }

public:
    sampler(std::string _label, signal_set_t _good, signal_set_t _bad = signal_set_t{});

    sampler(std::string _label, int64_t _tid, signal_set_t _good,
            signal_set_t _bad = signal_set_t{});

    sampler(std::string _label, int64_t _tid, signal_set_t _good, int _verbose,
            signal_set_t _bad = signal_set_t{});

    ~sampler();

    template <typename... Args, typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    void sample(Args&&...);

    template <typename... Args, typename Tp = fixed_size_t<N>,
              enable_if_t<!Tp::value> = 0>
    void sample(Args&&...);

    template <typename Tp = fixed_sig_t<SigIds...>, enable_if_t<Tp::value> = 0>
    void start();
    template <typename Tp = fixed_sig_t<SigIds...>, enable_if_t<Tp::value> = 0>
    void stop();

    template <typename Tp = fixed_sig_t<SigIds...>, enable_if_t<!Tp::value> = 0>
    void start();
    template <typename Tp = fixed_sig_t<SigIds...>, enable_if_t<!Tp::value> = 0>
    void stop();

    void stop(std::set<int>);

public:
    bool is_good(int v) const { return m_good.count(v) > 0; }
    bool is_bad(int v) const { return m_bad.count(v) > 0; }

    auto good_count() const { return m_good.size(); }
    auto bad_count() const { return m_bad.size(); }
    auto count() const { return good_count() + bad_count(); }

    auto& get_good() { return m_good; }
    auto& get_bad() { return m_bad; }

    const auto& get_good() const { return m_good; }
    const auto& get_bad() const { return m_bad; }

    auto backtrace_enabled() const { return m_backtrace; }
    void enable_backtrace(bool val) { m_backtrace = val; }

    bundle_type*& get_last() { return m_last; }
    bundle_type*  get_last() const { return m_last; }

    bundle_type*& get_latest() { return m_last; }
    bundle_type*  get_latest() const { return m_last; }

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    bundle_type& get(size_t idx);
    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    bundle_type& get(size_t idx);

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    const bundle_type& get(size_t idx) const;
    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    const bundle_type& get(size_t idx) const;

    array_t&       get_data() { return m_data; }
    const array_t& get_data() const { return m_data; }

public:
    /// \fn void configure(std::set<int> _signals, int _verb)
    /// \param[in] _signals A set of signals to catch
    /// \param[in] _verb Logging Verbosity
    ///
    /// \brief Set up the sampler
    void configure(std::set<int> _signals = {}, int _verbose = 0, bool _unblock = true);

    template <typename Tp>
    void configure(Tp _signal, int _verbose = 0, bool _unblock = true,
                   enable_if_t<!std::is_same<std::decay_t<Tp>, bool>::value &&
                               std::is_integral<Tp>::value> = 0)
    {
        configure({ _signal }, _verbose, _unblock);
    }

    template <typename Tp>
    void configure(Tp _unblock, std::set<int> _signals = {}, int _verbose = 0,
                   enable_if_t<std::is_same<std::decay_t<Tp>, bool>::value> = 0)
    {
        configure(_signals, _verbose, _unblock);
    }

    /// \fn void ignore(std::set<int> _signals)
    /// \param[in] _signals Set of signals
    ///
    /// \brief Ignore the signals (applies to all threads)
    void ignore(std::set<int> _signals);

    /// \fn void clear()
    /// \brief Clear all signals. Recommended to call ignore() prior to clearing all the
    /// signals
    void clear() { m_timer_data.m_signals.clear(); }

    /// \fn void pause()
    /// \brief Pause until a signal is delivered
    TIMEMORY_INLINE void pause()
    {
        if(!m_timer_data.m_signals.empty())
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
    int wait(pid_t _pid, int _verbose, bool _debug, Func&& _callback = pid_callback());

    template <typename Func = pid_cb_t>
    int wait(int _verbose = settings::verbose(), bool _debug = settings::debug(),
             Func&& _callback = pid_callback())
    {
        return wait(process::get_target_id(), _verbose, _debug,
                    std::forward<Func>(_callback));
    }

    template <typename Func, enable_if_t<std::is_function<Func>::value> = 0>
    int wait(pid_t _pid, Func&& _callback, int _verbose = settings::verbose(),
             bool _debug = settings::debug())
    {
        return wait(_pid, _verbose, _debug, std::forward<Func>(_callback));
    }

    template <typename Func, enable_if_t<std::is_function<Func>::value> = 0>
    int wait(Func&& _callback, int _verbose = settings::verbose(),
             bool _debug = settings::debug())
    {
        return wait(process::get_target_id(), _verbose, _debug,
                    std::forward<Func>(_callback));
    }

    /// \fn void set_flags(int)
    /// \param flags[in] the sigaction flags to use
    ///
    /// \brief Set the sigaction flags, e.g. SA_RESTART | SA_SIGINFO
    void set_flags(int _flags) { m_timer_data.m_flags = _flags; }

    /// \fn void set_verbose(int)
    /// \brief Configure the verbosity
    void set_verbose(int _v) { m_verbose = _v; }

    /// \fn void set_signals(std::set<int>)
    /// \brief Signals to catch
    void set_signals(const std::set<int>& _signals) { m_timer_data.m_signals = _signals; }

    /// \fn void add_signal(int)
    /// \brief Append to set of signals
    void add_signal(int _v) { m_timer_data.m_signals.emplace(_v); }

    /// \fn void remove_signal(int)
    /// \brief Remove from set of signals
    void remove_signal(int _v) { m_timer_data.m_signals.erase(_v); }

    /// \fn void set_delay(double, std::set<int>)
    /// \brief Value, expressed in seconds, that sets the length of time the sampler
    /// waits before starting sampling of the relevant measurements
    void set_delay(double fdelay, std::set<int> _signals = {});

    /// \fn void set_frequency(double, std::set<int>)
    /// \brief Value, expressed in 1/seconds, expressed in 1/seconds, that sets the
    /// frequency that the sampler samples the relevant measurements
    void set_frequency(double ffreq, std::set<int> _signals = {});

    /// \fn void set_rate(double, std::set<int>)
    /// \brief Value, expressed in number of interupts per second, that configures the
    /// frequency that the sampler samples the relevant measurements
    void set_rate(double frate, const std::set<int>& _signals = {})
    {
        set_frequency(1.0 / frate, _signals);
    }

    /// \fn size_t get_id() const
    /// \brief Get the unique global identifier
    size_t get_id() const { return m_idx; }

    /// \fn std::set<int> get_signals() const
    /// \brief Get signals handled by the sampler
    std::set<int> get_signals() const { return m_timer_data.m_signals; }

    /// \fn bool get_active(int signum) const
    /// \brief Get whether the specified signal has an active handler
    bool get_active(int signum) const { return m_timer_data.m_active.at(signum); }

    /// \fn int get_verbse() const
    /// \brief Get the verbosity
    int get_verbose() const { return m_verbose; }

    /// \fn double get_delay(int64_t) const
    /// \brief Get the delay of the sampler
    double get_delay(int64_t units = units::sec) const;

    /// \fn double get_rate(int64_t) const
    /// \brief Get the rate (interval) of the sampler
    double get_rate(int64_t units = units::sec) const;

    /// \fn double get_frequency(int64_t) const
    /// \brief Get the frequency of the sampler
    double get_frequency(int64_t units = units::sec) const;

    size_t get_buffer_size() const { return m_buffer_size; }
    void   set_buffer_size(size_t _v) { m_buffer_size = _v; }

    allocator_t&       get_allocator() { return m_alloc; }
    const allocator_t& get_allocator() const { return m_alloc; }

    template <typename FuncT>
    void set_notify(FuncT&& _v)
    {
        m_notify = std::forward<FuncT>(_v);
    }

    template <typename FuncT>
    void set_wait(FuncT&& _v)
    {
        m_wait = std::forward<FuncT>(_v);
    }

    template <typename FuncT>
    void set_exit(FuncT&& _v)
    {
        m_exit = std::forward<FuncT>(_v);
    }

    template <typename FuncT>
    void set_swap_data(FuncT&& _v)
    {
        m_swap_data = std::forward<FuncT>(_v);
    }

    void swap_data() { m_swap_data(m_data); }

protected:
    bool                            m_backtrace   = false;
    int                             m_verbose     = tim::settings::verbose();
    size_t                          m_idx         = get_counter()++;
    size_t                          m_buffer_size = trait::buffer_size<this_type>::value;
    bundle_type*                    m_last        = nullptr;
    signal_set_t                    m_good        = {};
    signal_set_t                    m_bad         = {};
    array_t                         m_data        = {};
    std::function<void()>           m_notify      = []() {};
    std::function<void()>           m_wait        = []() {};
    std::function<void()>           m_exit        = []() {};
    std::function<void(data_type&)> m_swap_data   = [](data_type&) {};
    allocator_t                     m_alloc;
    std::string                     m_label = {};

private:
    struct timer_data
    {
        int                        m_flags = SA_RESTART | SA_SIGINFO;
        double                     m_delay = 0.001;
        double                     m_freq  = 1.0 / 2.0;
        sigaction_t                m_custom_sigaction;
        sigaction_t                m_original_sigaction;
        std::set<int>              m_signals            = {};
        std::map<int, bool>        m_active             = {};
        std::map<int, double>      m_signal_delay       = {};
        std::map<int, double>      m_signal_freq        = {};
        std::map<int, itimerval_t> m_custom_itimerval   = {};
        std::map<int, itimerval_t> m_original_itimerval = {};
    };

    struct persistent_data : timer_data
    {
        std::map<int64_t, std::deque<this_type*>> m_thread_instances = {};
    };

    static std::atomic<int64_t>& get_counter()
    {
        static std::atomic<int64_t> _ntot{ 0 };
        return _ntot;
    }

    static persistent_data& get_persistent_data()
    {
        static std::atomic<int64_t> _ntot{ 0 };
        static auto                 _main = persistent_data{};
        static thread_local auto    _n    = _ntot++;
        if(_n == 0)
            return _main;
        static thread_local auto _instance = _main;
        return _instance;
    }

    /// \fn pid_cb_t& pid_callback()
    /// \brief Default callback when configuring sampler
    static pid_cb_t pid_callback()
    {
        return [](pid_t _id, int, int) { return _id != process::get_id(); };
    }

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    void _init_sampler(int64_t _tid = threading::get_id());

    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    void _init_sampler(int64_t _tid = threading::get_id());

    timer_data m_timer_data = static_cast<persistent_data>(get_persistent_data());

public:
    /// \fn int get_itimer(int)
    /// \brief Returns the itimer value associated with the given signal
    static int get_itimer(int _signal);

    /// \fn bool check_itimer(int, bool)
    /// \brief Checks to see if there was an error setting or getting itimer val
    static bool check_itimer(int _stat, bool _throw_exception = false);

    static timer_data& get_default_config()
    {
        return static_cast<timer_data&>(get_persistent_data());
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
inline auto
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
        else m_last->sample(get_backtrace<_depth, _offset>(),
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
        else m_last->sample(get_backtrace<_depth, _offset>(),
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
    CONDITIONAL_PRINT_HERE(m_verbose > 0, "starting (index: %zu)", m_idx);
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
    CONDITIONAL_PRINT_HERE(m_verbose > 0, "stopping (index: %zu)", m_idx);
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
    CONDITIONAL_PRINT_HERE(m_verbose > 0, "starting (index: %zu)", m_idx);
    auto cnt = tracker_type::start();
    if(cnt.second == 0 && !m_alloc.is_alive())
    {
        CONDITIONAL_PRINT_HERE(m_verbose > 0, "restarting allocator (index: %zu)", m_idx);
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
    CONDITIONAL_PRINT_HERE(m_verbose > 0, "stopping (index: %zu)", m_idx);
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
inline void
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

    CONDITIONAL_PRINT_HERE(m_verbose > 0, "stopping %zu signals (index: %zu)",
                           _signals.size(), m_idx);

    for(auto itr : _signals)
    {
        CONDITIONAL_PRINT_HERE(m_verbose > 0, "stopping signal %i (index: %zu)", itr,
                               m_idx);
        auto&       _original_it = m_timer_data.m_original_itimerval[itr];
        itimerval_t _curr;
        auto        _itimer = get_itimer(itr);
        if(_itimer < 0)
        {
            TIMEMORY_EXCEPTION(
                TIMEMORY_JOIN(" ", "Error! Alarm cannot be set for signal", itr,
                              "because the signal does not map to a known itimer "
                              "value\n"));
        }
        check_itimer(getitimer(_itimer, &_curr));
        // stop the alarm
        if(_curr.it_interval.tv_usec > 0 || _curr.it_interval.tv_sec > 0)
            check_itimer(setitimer(_itimer, &_original_it, &_curr));

        m_timer_data.m_active[itr] = false;
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
        CONDITIONAL_PRINT_HERE(settings::debug(), "initializing %s", "semaphore");
        int _err = 0;
        TIMEMORY_SEMAPHORE_HANDLE_EINTR(sem_init, _err, &_v, 0, 1)
        TIMEMORY_SEMAPHORE_CHECK_MSG(_err, "sem_init(&_v, 0, 1)")
        CONDITIONAL_PRINT_HERE(settings::debug(), "%s initialized", "semaphore");
        return _v;
    }();
    static thread_local auto _sem_dtor = scope::destructor{ []() {
        CONDITIONAL_PRINT_HERE(settings::debug(), "destroying %s", "semaphore");
        int                  _err      = 0;
        TIMEMORY_SEMAPHORE_HANDLE_EINTR(sem_destroy, _err, &_sem)
        TIMEMORY_SEMAPHORE_CHECK_MSG(_err, "sem_destroy(&_sem)")
        CONDITIONAL_PRINT_HERE(settings::debug(), "%s destroyed", "semaphore");
    } };

    IF_CONSTEXPR(trait::prevent_reentry<this_type>::value)
    {
        int _err = 0;
        TIMEMORY_SEMAPHORE_TRYWAIT(_sem, _err);

        if(_err == EAGAIN)
        {
            CONDITIONAL_PRINT_HERE(settings::debug(),
                                   "Ignoring signal %i (raised while sampling)", signum);
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

        CONDITIONAL_PRINT_HERE(itr->m_verbose > 1, "sampling signal %i (index: %zu)",
                               signum, itr->m_idx);

        IF_CONSTEXPR(trait::check_signals<this_type>::value)
        {
            if(itr->is_good(signum))
            {
                itr->sample(signum);
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
        CONDITIONAL_PRINT_HERE(settings::debug(), "initializing %s", "semaphore");
        int _err = 0;
        TIMEMORY_SEMAPHORE_HANDLE_EINTR(sem_init, _err, &_v, 0, 1)
        TIMEMORY_SEMAPHORE_CHECK_MSG(_err, "sem_init(&_v, 0, 1)")
        CONDITIONAL_PRINT_HERE(settings::debug(), "%s initialized", "semaphore");
        return _v;
    }();
    static thread_local auto _sem_dtor = scope::destructor{ []() {
        CONDITIONAL_PRINT_HERE(settings::debug(), "destroying %s", "semaphore");
        int                  _err      = 0;
        TIMEMORY_SEMAPHORE_HANDLE_EINTR(sem_destroy, _err, &_sem)
        TIMEMORY_SEMAPHORE_CHECK_MSG(_err, "sem_destroy(&_sem)")
        CONDITIONAL_PRINT_HERE(settings::debug(), "%s destroyed", "semaphore");
    } };

    IF_CONSTEXPR(trait::prevent_reentry<this_type>::value)
    {
        int _err = 0;
        TIMEMORY_SEMAPHORE_TRYWAIT(_sem, _err);

        if(_err == EAGAIN)
        {
            CONDITIONAL_PRINT_HERE(settings::debug(),
                                   "Ignoring signal %i (raised while sampling)", signum);
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

        CONDITIONAL_PRINT_HERE(itr->m_verbose > 1, "sampling signal %i (index: %zu)",
                               signum, itr->m_idx);

        IF_CONSTEXPR(trait::check_signals<this_type>::value)
        {
            if(itr->is_good(signum))
            {
                itr->sample(signum, _info, _data);
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

    CONDITIONAL_PRINT_HERE(_verbose > 1, "configuring sampler (index: %zu)", m_idx);

    size_t wait_count = 0;
    {
        auto_lock_t _lk{ type_mutex<this_type>() };
        for(auto& itr : get_samplers(threading::get_id()))
            wait_count += itr->count();
    }

    if(wait_count == 0)
    {
        if(_verbose > 0)
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
        CONDITIONAL_PRINT_HERE(_verbose > 1,
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
                CONDITIONAL_PRINT_HERE(
                    _verbose > 1, "handler for signal %i is already active (index: %zu)",
                    itr, m_idx);
                continue;
            }
            CONDITIONAL_PRINT_HERE(_verbose > 1,
                                   "configuring handler for signal %i (index: %zu)", itr,
                                   m_idx);

            // get the associated itimer type
            auto _itimer = get_itimer(itr);
            if(_itimer < 0)
            {
                TIMEMORY_EXCEPTION(
                    TIMEMORY_JOIN(" ", "Error! Alarm cannot be set for signal", itr,
                                  "because the signal does not map to a known itimer "
                                  "value\n"));
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

            // start the alarm (throws if fails)
            check_itimer(setitimer(_itimer, &(m_timer_data.m_custom_itimerval.at(itr)),
                                   &(m_timer_data.m_original_itimerval[itr])),
                         true);

            m_timer_data.m_active[itr] = true;
        }
    }

    CONDITIONAL_PRINT_HERE(_verbose > 1,
                           "signal handler configuration complete (index: %zu)", m_idx);

    // unblock signals on thread after configuring
    if(!_signals.empty() && _unblock)
        sampling::unblock_signals(_signals, sigmask_scope::thread);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
inline void
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

    if(_verbose > 2)
        fprintf(stderr, "[%i]> waiting for pid %i...\n", process::get_id(), wait_pid);

    //----------------------------------------------------------------------------------//
    //
    auto print_info = [=](pid_t _pid, int _status, int _errv, int _retv) {
        if(_verbose > 2)
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
        if(_verbose > 2)
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
            if(_verbose > 3)
            {
                fprintf(stderr, "[%i]> program stopped with signal %i. Exit code: %i\n",
                        _pid, sig, ret);
            }
        }
        else if(WCOREDUMP(status))
        {
            if(_verbose > 3)
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
            if(_verbose > 3)
            {
                fprintf(stderr,
                        "[%i]> program terminated because it received a signal "
                        "(%i) that was not handled. Exit code: %i\n",
                        _pid, WTERMSIG(status), ret);
            }
        }
        else if(WIFEXITED(status) && WEXITSTATUS(status))
        {
            if(ret == 127 && (_verbose > 3))
            {
                fprintf(stderr, "[%i]> execv failed\n", _pid);
            }
            else if(_verbose > 3)
            {
                fprintf(stderr,
                        "[%i]> program terminated with a non-zero status. Exit "
                        "code: %i\n",
                        _pid, ret);
            }
        }
        else
        {
            if(_verbose > 3)
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
sampler<CompT<Types...>, N, SigIds...>::set_delay(double fdelay, std::set<int> _signals)
{
    m_timer_data.m_delay = std::max<double>(fdelay, m_timer_data.m_delay);
    if(_signals.empty())
        _signals = m_timer_data.m_signals;
    for(const auto& itr : _signals)
    {
        sampling::set_delay(m_timer_data.m_custom_itimerval[itr], fdelay,
                            std::string{ "[" } + std::to_string(itr) + "]");
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
void
sampler<CompT<Types...>, N, SigIds...>::set_frequency(double        ffreq,
                                                      std::set<int> _signals)
{
    m_timer_data.m_freq = std::max<double>(ffreq, m_timer_data.m_freq);
    if(_signals.empty())
        _signals = m_timer_data.m_signals;
    for(const auto& itr : _signals)
    {
        sampling::set_frequency(m_timer_data.m_custom_itimerval[itr], ffreq,
                                std::string{ "[" } + std::to_string(itr) + "]");
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
inline double
sampler<CompT<Types...>, N, SigIds...>::get_delay(int64_t units) const
{
    double _ns = m_timer_data.m_delay;
    for(const auto& itr : m_timer_data.m_signals)
    {
        _ns = std::max(
            _ns, sampling::get_delay(m_timer_data.m_custom_itimerval.at(itr), units));
    }
    return _ns / static_cast<double>(units);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
inline double
sampler<CompT<Types...>, N, SigIds...>::get_rate(int64_t units) const
{
    double _ns = 1.0 / m_timer_data.m_freq;
    for(const auto& itr : m_timer_data.m_signals)
    {
        _ns = std::min(
            _ns, sampling::get_rate(m_timer_data.m_custom_itimerval.at(itr), units));
    }
    return _ns / static_cast<double>(units);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class CompT, size_t N, typename... Types, int... SigIds>
inline double
sampler<CompT<Types...>, N, SigIds...>::get_frequency(int64_t units) const
{
    return 1.0 / get_rate(units);
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

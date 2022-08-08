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

#ifndef TIMEMORY_SAMPLING_SAMPLER_HPP_
#    define TIMEMORY_SAMPLING_SAMPLER_HPP_

#    include "timemory/backends/threading.hpp"
#    include "timemory/components/base.hpp"
#    include "timemory/macros/language.hpp"
#    include "timemory/macros/os.hpp"
#    include "timemory/mpl/apply.hpp"
#    include "timemory/operations/types/sample.hpp"
#    include "timemory/sampling/allocator.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/settings/settings.hpp"
#    include "timemory/units.hpp"
#    include "timemory/utility/backtrace.hpp"
#    include "timemory/utility/demangle.hpp"
#    include "timemory/utility/macros.hpp"
#    include "timemory/variadic/macros.hpp"

// C++ includes
#    include <array>
#    include <atomic>
#    include <deque>
#    include <functional>
#    include <map>
#    include <set>
#    include <string>
#    include <type_traits>
#    include <utility>
#    include <vector>

// C includes
#    include <cassert>
#    include <cerrno>
#    include <csignal>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <sys/time.h>
#    include <sys/types.h>
#    include <sys/wait.h>

#    if defined(TIMEMORY_UNIX)
#        include <unistd.h>
#    endif

#    if !defined(TIMEMORY_SAMPLER_DEPTH_DEFAULT)
#        define TIMEMORY_SAMPLER_DEPTH_DEFAULT 64
#    endif

#    if !defined(TIMEMORY_SAMPLER_OFFSET_DEFAULT)
#        define TIMEMORY_SAMPLER_OFFSET_DEFAULT 3
#    endif

#    if !defined(TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT)
#        if defined(TIMEMORY_USE_LIBUNWIND)
#            define TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT true
#        else
#            define TIMEMORY_SAMPLER_USE_LIBUNWIND_DEFAULT false
#        endif
#    endif

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
using itimerval_t  = struct itimerval;
using itimerspec_t = struct itimerspec;
using sigaction_t  = struct sigaction;
//
inline itimerspec_t
get_itimerspec(const itimerval_t& _val)
{
    itimerspec_t _spec;
    _spec.it_interval.tv_sec  = _val.it_interval.tv_sec;
    _spec.it_interval.tv_nsec = 1000 * _val.it_interval.tv_usec;
    _spec.it_value.tv_sec     = _val.it_value.tv_sec;
    _spec.it_value.tv_nsec    = 1000 * _val.it_value.tv_usec;
    return _spec;
}
//
inline itimerval_t
get_itimerval(const itimerspec_t& _spec)
{
    itimerval_t _val;
    _val.it_interval.tv_sec  = _spec.it_interval.tv_sec;
    _val.it_interval.tv_usec = _spec.it_interval.tv_nsec / 1000;
    _val.it_value.tv_sec     = _spec.it_value.tv_sec;
    _val.it_value.tv_usec    = _spec.it_value.tv_nsec / 1000;
    return _val;
}
//
inline void
set_delay(itimerval_t& _itimer, double fdelay, const std::string& _extra = {},
          bool _verbose = false)
{
    int delay_sec  = fdelay;
    int delay_usec = static_cast<int>(fdelay * 1000000) % 1000000;
    if(_verbose)
    {
        fprintf(stderr, "[T%i]%s sampler delay      : %i sec + %i usec\n",
                (int) threading::get_id(), _extra.c_str(), delay_sec, delay_usec);
    }
    // Configure the timer to expire after designated delay...
    _itimer.it_value.tv_sec  = delay_sec;
    _itimer.it_value.tv_usec = delay_usec;
}
//
inline void
set_frequency(itimerval_t& _itimer, double _freq, const std::string& _extra = {},
              bool _verbose = false)
{
    double _period      = 1.0 / _freq;
    int    _period_sec  = _period;
    int    _period_usec = static_cast<int>(_period * 1000000) % 1000000;
    if(_verbose)
    {
        fprintf(stderr, "[T%i]%s sampler period     : %i sec + %i usec\n",
                (int) threading::get_id(), _extra.c_str(), _period_sec, _period_usec);
    }
    // Configure the timer to expire at designated intervals
    _itimer.it_interval.tv_sec  = _period_sec;
    _itimer.it_interval.tv_usec = _period_usec;
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
get_period(const itimerval_t& _itimer, int64_t units = units::sec)
{
    double _ns = (_itimer.it_interval.tv_sec * units::sec) +
                 (_itimer.it_interval.tv_usec * units::usec);
    return _ns / static_cast<double>(units);
}
//
inline double
get_frequency(const itimerval_t& _itimer, int64_t units = units::sec)
{
    return 1.0 / get_period(_itimer, units);
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
    using buffer_t     = data_storage::ring_buffer<bundle_type>;
    using allocator_t =
        conditional_t<fixed_size_t<N>::value, allocator<void>, allocator<this_type>>;

    using array_type   = array_t;
    using tracker_type = policy::instance_tracker<this_type, true>;

    friend struct allocator<this_type>;

    static_assert(fixed_size<N>::value || trait::buffer_size<this_type>::value > 0,
                  "Error! Dynamic sampler has a default buffer size of zero");

    static constexpr bool is_dynamic() { return !fixed_size_t<N>::value; }

    static TIMEMORY_NOINLINE void execute(int signum);
    static TIMEMORY_NOINLINE void execute(int signum, siginfo_t*, void*);

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
    const bundle_type& get(size_t idx) const;
    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    const bundle_type& get(size_t idx) const;

    template <typename Tp = fixed_size_t<N>, enable_if_t<!Tp::value> = 0>
    decltype(auto) get_data() const
    {
        return m_alloc.get_data();
    }

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    array_t& get_data()
    {
        return m_data;
    }

    template <typename Tp = fixed_size_t<N>, enable_if_t<Tp::value> = 0>
    const array_t& get_data() const
    {
        return m_data;
    }

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

    void reset(std::set<int> _signals = {}, int _verbose = 0, bool _unblock = true);

    template <typename Tp>
    void reset(Tp _signal, int _verbose = 0, bool _unblock = true,
               enable_if_t<!std::is_same<std::decay_t<Tp>, bool>::value &&
                           std::is_integral<Tp>::value> = 0)
    {
        reset({ _signal }, _verbose, _unblock);
    }

    template <typename Tp>
    void reset(Tp _unblock, std::set<int> _signals = {}, int _verbose = 0,
               enable_if_t<std::is_same<std::decay_t<Tp>, bool>::value> = 0)
    {
        reset(_signals, _verbose, _unblock);
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
    void set_delay(double fdelay, std::set<int> _signals = {}, bool _verbose = false);

    /// \fn void set_frequency(double, std::set<int>)
    /// \brief Value, expressed in 1/seconds, expressed in 1/seconds, that sets the
    /// frequency that the sampler samples the relevant measurements
    void set_frequency(double ffreq, std::set<int> _signals = {}, bool _verbose = false);

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

    /// \fn double get_delay(int64_t units, int signum) const
    /// \param units the units of the value, e.g. units::sec
    /// \param signum if -1, return the shortest delay
    /// \brief Get the delay of the sampler
    double get_delay(int64_t units = units::sec, int signum = -1) const;

    /// \fn double get_period(int64_t units, int signum) const
    /// \param units the units of the value, e.g. units::sec
    /// \param signum if -1, return the shortest period
    /// \brief Get the rate (interval) of the sampler
    double get_period(int64_t units = units::sec, int signum = -1) const;

    /// \fn double get_frequency(int64_t units, int signum) const
    /// \param units the units of the value, e.g. units::sec
    /// \param signum if -1, return the largest frequency
    /// \brief Get the frequency of the sampler
    double get_frequency(int64_t units = units::sec, int signum = -1) const;

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
    void set_exit(FuncT&& _v)
    {
        m_exit = std::forward<FuncT>(_v);
    }

    auto get_sample_count() const { return m_count; }

protected:
    bool                  m_backtrace   = false;
    int                   m_verbose     = tim::settings::verbose();
    size_t                m_idx         = get_counter()++;
    size_t                m_buffer_size = trait::buffer_size<this_type>::value;
    size_t                m_count       = 0;
    bundle_type*          m_last        = nullptr;
    signal_set_t          m_good        = {};
    signal_set_t          m_bad         = {};
    array_t               m_data        = {};
    std::function<void()> m_notify      = []() {};
    std::function<void()> m_exit        = []() {};
    buffer_t              m_buffer      = {};
    buffer_t              m_filled      = {};
    allocator_t           m_alloc;
    std::string           m_label = {};

private:
    struct timer_data
    {
        int                         m_flags = SA_RESTART | SA_SIGINFO;
        double                      m_delay = 0.001;
        double                      m_freq  = 1.0 / 2.0;
        sigaction_t                 m_custom_sigaction;
        sigaction_t                 m_original_sigaction;
        std::set<int>               m_signals             = {};
        std::map<int, bool>         m_active              = {};
        std::map<int, double>       m_signal_delay        = {};
        std::map<int, double>       m_signal_freq         = {};
        std::map<int, itimerval_t>  m_custom_itimerval    = {};
        std::map<int, itimerval_t>  m_original_itimerval  = {};
        std::map<int, itimerspec_t> m_custom_itimerspec   = {};
        std::map<int, itimerspec_t> m_original_itimerspec = {};
        std::map<int, sigevent>     m_sigevent            = {};
        std::map<int, timer_t>      m_timer               = {};
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
}  // namespace sampling
}  // namespace tim

#    if !defined(TIMEMORY_SAMPLING_SAMPLER_USE_EXTERN)
#        include "timemory/sampling/sampler.cpp"
#    endif
#endif

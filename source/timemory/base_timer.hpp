// MIT License
//
// Copyright (c) 2018, The Regents of the University of California,
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file base_timer.hpp
 * \headerfile base_timer.hpp "timemory/base_timer.hpp"
 * Base class for the timer class
 * Not directly used
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include <atomic>
#include <fstream>
#include <string>

#include "timemory/base_clock.hpp"
#include "timemory/data_types.hpp"
#include "timemory/formatters.hpp"
#include "timemory/macros.hpp"
#include "timemory/serializer.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/string.hpp"
#include "timemory/usage.hpp"
#include "timemory/utility.hpp"

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace internal
{
//======================================================================================//
//
//  Primary base class for handling the timer
//
//======================================================================================//

tim_api class base_timer
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef tim::string                 string_t;
    typedef string_t::size_type         size_type;
    typedef std::mutex                  mutex_t;
    typedef std::recursive_mutex        rmutex_t;
    typedef std::ostream                ostream_t;
    typedef std::ofstream               ofstream_t;
    typedef uomap<ostream_t*, rmutex_t> mutex_map_t;
    typedef std::lock_guard<mutex_t>    auto_lock_t;
    typedef std::lock_guard<rmutex_t>   recursive_lock_t;
    //
    typedef base_delta<timer_data> data_accum_t;
    typedef base_data<timer_data>  data_t;
    //
    typedef tms                                     tms_t;
    typedef std::micro                              ratio_t;
    typedef tim::base_clock<ratio_t>                clock_t;
    typedef clock_t::time_point                     time_point_t;
    typedef std::chrono::duration<clock_t, ratio_t> duration_t;
    typedef base_timer                              this_type;
    typedef format::timer                           format_type;
    typedef std::shared_ptr<format_type>            timer_format_t;

public:
    base_timer(timer_format_t = timer_format_t(), ostream_t* = &std::cout);
    virtual ~base_timer();

    base_timer(const base_timer& rhs);
    base_timer& operator=(const base_timer& rhs);

public:
    // public member functions
    inline void start();
    inline void stop();
    inline bool is_valid() const;
    inline bool is_running() const;
    double      real_elapsed() const;
    double      wall_elapsed() const { return this->real_elapsed(); }
    double      system_elapsed() const;
    double      user_elapsed() const;
    double      cpu_elapsed() const { return user_elapsed() + system_elapsed(); }
    double      cpu_utilization() const { return cpu_elapsed() / real_elapsed() * 100.; }
    inline const char*         clock_time() const;
    inline size_type           laps() const { return m_accum.size(); }
    inline void                reset() { m_accum.reset(); }
    inline data_accum_t&       accum() { return m_accum; }
    inline const data_accum_t& accum() const { return m_accum; }
    inline timer_format_t      format() const { return m_format; }
    inline void                set_format(const format_type& _format);
    inline void                set_format(timer_format_t _format);

public:
    // public member functions
    void report(ostream_t&, bool endline = true, bool ign_cutoff = false) const;
    void report(bool endline = true) const;
    bool above_cutoff(bool ign_cutoff = false) const;
    void sync(const this_type& rhs);
    void thread_timing(bool _val) { m_thread_timing = _val; }
    bool thread_timing() const { return m_thread_timing; }

protected:
    // protected member functions
    data_accum_t&       get_accum() { return m_accum; }
    const data_accum_t& get_accum() const { return m_accum; }
    inline void         start_stop_debug_message(const string_t&, const string_t&);

protected:
    // protected member variables
    bool m_thread_timing;
    // pointers
    ostream_t* m_os;
    // objects
    mutex_t              m_mutex;
    mutable data_t       m_data;
    mutable data_accum_t m_accum;
    timer_format_t       m_format;

private:
    // world mutex map for thread-safe ostreams
    static mutex_map_t f_mutex_map;

public:
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int /*version*/)
    {
        auto _cpu_util = (m_accum.sum().user() + m_accum.sum().sys());
        if(m_accum.sum().real() > 0)
            _cpu_util /= m_accum.sum().real();
        else
            _cpu_util = 0.0;

        if(!tim::isfinite(_cpu_util))
            _cpu_util = 0.0;

        // rss::usage_delta accum_rss;

        ar(serializer::make_nvp("laps", m_accum.size()),
           // user clock elapsed
           serializer::make_nvp("user_elapsed", m_accum.sum().user()),
           // system clock elapsed
           serializer::make_nvp("system_elapsed", m_accum.sum().sys()),
           // wall clock elapsed
           serializer::make_nvp("wall_elapsed", m_accum.sum().real()),
           // cpu elapsed
           serializer::make_nvp("cpu_elapsed",
                                m_accum.sum().user() + m_accum.sum().sys()),
           // cpu utilization
           serializer::make_nvp("cpu_util", _cpu_util),
#if defined(TIMEMORY_STAT_TIMERS)
           serializer::make_nvp("wall_elapsed_sqr", m_accum.get_sqr<2>()),
           serializer::make_nvp("user_elapsed_sqr", m_accum.get_sqr<0>()),
           serializer::make_nvp("system_elapsed_sqr", m_accum.get_sqr<1>()),
#endif
           // conversion to seconds
           serializer::make_nvp("to_seconds_ratio_num", ratio_t::num),
           serializer::make_nvp("to_seconds_ratio_den", ratio_t::den)
           // memory usage
           // serializer::make_nvp("rss_max", accum_rss.total()),
           // serializer::make_nvp("rss_self", accum_rss.self()),
           // memory usage (minimum)
           // serializer::make_nvp("rss_min", accum_rss.total_min()),
           // serializer::make_nvp("rss_self_min", accum_rss.self_min())
        );
    }
};

//--------------------------------------------------------------------------------------//
inline void
base_timer::set_format(const format_type& _format)
{
    m_format = timer_format_t(new format_type(_format));
}
//--------------------------------------------------------------------------------------//
inline void
base_timer::set_format(timer_format_t _format)
{
    m_format = _format;
}
//--------------------------------------------------------------------------------------//
// Print timer status n std::ostream
static inline std::ostream&
operator<<(std::ostream& os, const base_timer& t)
{
    bool restart = !t.is_valid();
    if(restart)
        const_cast<base_timer&>(t).stop();
    t.report(os);
    if(restart)
        const_cast<base_timer&>(t).start();

    return os;
}
//--------------------------------------------------------------------------------------//
// Wall time
inline double
base_timer::real_elapsed() const
{
    if(m_data.running())
        throw std::runtime_error(
            "Error! base_timer::real_elapsed() - "
            "timer not stopped or no times recorded!");
    return m_accum.sum().real() / static_cast<double>(ratio_t::den);
}
//--------------------------------------------------------------------------------------//
// System time
inline double
base_timer::system_elapsed() const
{
    if(m_data.running())
        throw std::runtime_error(
            "Error! base_timer::system_elapsed() - "
            "timer not stopped or no times recorded!");
    return m_accum.sum().sys() / static_cast<double>(ratio_t::den);
}
//--------------------------------------------------------------------------------------//
// CPU time
inline double
base_timer::user_elapsed() const
{
    if(m_data.running())
        throw std::runtime_error(
            "Error! base_timer::user_elapsed() - "
            "timer not stopped or no times recorded!");
    return m_accum.sum().user() / static_cast<double>(ratio_t::den);
}
//--------------------------------------------------------------------------------------//
inline void
base_timer::start_stop_debug_message(const string_t& _func, const string_t& _already)
{
    int32_t _verbose = tim::get_env<int32_t>("TIMEMORY_VERBOSE", 0);
    if(_verbose > 0)
    {
        std::stringstream _msg;
        _msg << "Warning! base_timer::" << _func << " called but already " << _already
             << "..." << std::endl;
        if(_verbose > 1)
            tim::stack_backtrace(_msg);
        std::cerr << _msg.str();
    }
}
//--------------------------------------------------------------------------------------//
inline void
base_timer::start()
{
    if(!m_data.running())
    {
        m_data.resume();
        m_data.start() = std::make_tuple(get_clock_monotonic_now<uintmax_t, ratio_t>(),
                                         get_clock_thread_now<uintmax_t, ratio_t>(),
                                         get_clock_system_now<uintmax_t, ratio_t>());
    }
}
//--------------------------------------------------------------------------------------//
inline void
base_timer::stop()
{
    if(m_data.running())
    {
        m_data.pause();
        m_data.stop() = std::make_tuple(get_clock_monotonic_now<uintmax_t, ratio_t>(),
                                        get_clock_thread_now<uintmax_t, ratio_t>(),
                                        get_clock_system_now<uintmax_t, ratio_t>());
        m_accum += m_data;
    }
}
//--------------------------------------------------------------------------------------//
inline bool
base_timer::is_valid() const
{
    return (m_data.running()) ? false : true;
}
//--------------------------------------------------------------------------------------//
inline bool
base_timer::is_running() const
{
    return m_data.running();
}
//--------------------------------------------------------------------------------------//
inline const char*
base_timer::clock_time() const
{
    time_t     rawtime;
    struct tm* timeinfo;

    std::time(&rawtime);
    timeinfo = localtime(&rawtime);
    return asctime(timeinfo);
}
//--------------------------------------------------------------------------------------//
inline void
base_timer::report(std::ostream& os, bool endline, bool ign_cutoff) const
{
    // stop, if not already stopped
    // if(m_data.running())
    //    const_cast<base_timer*>(this)->stop();

    if(!above_cutoff(ign_cutoff))
        return;

    std::stringstream ss;
    // ss << (*m_format)(this);

    if(endline)
        ss << std::endl;

    // ensure thread-safety
    tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
    recursive_lock_t rlock(f_mutex_map[&os]);
    // output to ostream
    os << ss.str();
}
//--------------------------------------------------------------------------------------//

}  // namespace internal

}  // namespace tim

//--------------------------------------------------------------------------------------//
/*
namespace internal
{
typedef typename tim::internal::base_timer_data::ratio_t       base_ratio_t;
typedef tim::base_clock<base_ratio_t>                          base_clock_t;
typedef tim::base_clock_data<base_ratio_t>                     base_clock_data_t;
typedef std::chrono::duration<base_clock_data_t, base_ratio_t> base_duration_t;
typedef std::chrono::time_point<base_clock_t, base_duration_t> base_time_point_t;
typedef std::tuple<base_time_point_t, base_time_point_t>       base_time_pair_t;
}  // namespace internal
*/
//--------------------------------------------------------------------------------------//

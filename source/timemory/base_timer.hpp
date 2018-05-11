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
//

/** \file base_timer.hpp
 * \headerfile base_timer.hpp "timemory/base_timer.hpp"
 * Base class for the timer class
 * Not directly used
 */

#ifndef base_timer_hpp_
#define base_timer_hpp_

//----------------------------------------------------------------------------//

#include <fstream>
#include <string>
#include <atomic>

#include "timemory/macros.hpp"
#include "timemory/base_clock.hpp"
#include "timemory/rss.hpp"
#include "timemory/utility.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/formatters.hpp"
#include "timemory/serializer.hpp"

//----------------------------------------------------------------------------//

namespace tim
{

namespace internal
{

//============================================================================//
//
//  Class for handling the timing and memory data
//
//============================================================================//

class tim_api base_timer_data
{
public:
    typedef base_timer_data                             this_type;
    typedef std::micro                                  ratio_t;
    typedef tim::base_clock<ratio_t>                    clock_t;
    typedef clock_t::time_point                         time_point_t;
    typedef std::tuple<time_point_t, time_point_t>      data_type;
    typedef tim::rss::usage_delta                       rss_type;
    typedef std::chrono::duration<clock_t, ratio_t>     duration_t;

public:
    base_timer_data() : m_running(false), m_data(data_type()) { }

    time_point_t& start() { m_running = true; return std::get<0>(m_data); }
    time_point_t& stop() { m_running = false; return std::get<1>(m_data); }

    const time_point_t& start() const { m_running = true; return std::get<0>(m_data); }
    const time_point_t& stop() const { m_running = false; return std::get<1>(m_data); }

    const bool& running() const { return m_running; }

    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(serializer::make_nvp("start", std::get<0>(m_data)),
           serializer::make_nvp("stop",  std::get<1>(m_data)));
    }

    inline void rss_init() { m_rss.init(); }
    inline void rss_record() { m_rss.record(); }

    inline rss_type& rss() { return m_rss; }
    inline const rss_type& rss() const { return m_rss; }

    this_type& operator=(const this_type& rhs)
    {
        if(this != &rhs)
        {
            m_running = rhs.m_running;
            m_data = rhs.m_data;
            m_rss = rhs.m_rss;
        }
        return *this;
    }

protected:
    mutable bool    m_running;
    data_type       m_data;
    rss_type        m_rss;

};

//----------------------------------------------------------------------------//

template <int N>
uint64_t get_start(const base_timer_data& data)
{
    return std::get<N>(data.start().time_since_epoch().count().data);
}

//----------------------------------------------------------------------------//

template <int N>
uint64_t get_stop(const base_timer_data& data)
{
    return std::get<N>(data.stop().time_since_epoch().count().data);
}

//============================================================================//
//
//  Class for handling the timing difference
//
//============================================================================//

class tim_api base_timer_delta
{
public:
    typedef base_timer_delta                            this_type;
    typedef uint64_t                                    uint_type;
    typedef std::tuple<uint_type, uint_type, uint_type> data_type;
    typedef std::tuple<uint64_t, uint64_t, uint64_t>    incr_type;
    typedef base_timer_data                             op_type;
    typedef tim::rss::usage_delta                       rss_type;
    typedef rss_type::base_type                         base_rss_type;

public:
    base_timer_delta()
    : m_lap(0),
      m_sum(data_type(0, 0, 0)),
  #if defined(TIMEMORY_STAT_TIMERS)
      m_sqr(data_type(0, 0, 0)),
  #endif
      m_rss(rss_type())
    { }

public:
    const uint64_t& size() const    { return m_lap; }
    uint64_t&       size()          { return m_lap; }

    rss_type&       rss()           { return m_rss; }
    const rss_type& rss() const     { return m_rss; }

    void reset()
    {
        m_lap = 0;
        m_sum = data_type(0, 0, 0);
    #if defined(TIMEMORY_STAT_TIMERS)
        m_sqr = data_type(0, 0, 0);
    #endif
        m_rss = rss_type();
    }

    template <int N> uint64_t get_sum() const { return std::get<N>(m_sum); }

    #if defined(TIMEMORY_STAT_TIMERS)
    template <int N> uint64_t get_sqr() const { return std::get<N>(m_sqr); }
    #else
    template <int N> uint64_t get_sqr() const { return 0; }
    #endif

public:
    //------------------------------------------------------------------------//
    //      operator = this
    //
    this_type& operator=(const this_type& rhs)
    {
        if(this != &rhs)
        {
            m_lap = rhs.m_lap;
            m_sum = rhs.m_sum;
    #if defined(TIMEMORY_STAT_TIMERS)
            m_sqr = rhs.m_sqr;
    #endif
            m_rss = rhs.m_rss;
        }
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator += data
    //
    this_type& operator+=(const op_type& data)
    {
        auto _data = incr_type(compute<0>(data),
                               compute<1>(data),
                               compute<2>(data));
        compute_sum(_data);
        compute_sqr(_data);
        m_lap += 1;
        m_rss.max(data.rss());

        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator += this
    //
    this_type& operator+=(const this_type& rhs)
    {
        m_lap += rhs.m_lap;
        compute_sum(rhs.m_sum);
    #if defined(TIMEMORY_STAT_TIMERS)
        compute_sqr(rhs.m_sqr);
    #endif
        m_rss.max(rhs.m_rss);
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator -= this
    //
    this_type& operator-=(const this_type& rhs)
    {
        subtract_sum(rhs.m_sum);
    #if defined(TIMEMORY_STAT_TIMERS)
        subtract_sqr(rhs.m_sqr);
    #endif
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator += RSS
    //
    this_type& operator+=(const base_rss_type& rhs)
    {
        m_rss += rhs;
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator -= RSS
    //
    this_type& operator-=(const base_rss_type& rhs)
    {
        m_rss -= rhs;
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator /= integer
    //
    this_type& operator/=(const uint64_t& rhs)
    {
        if(rhs > 0)
        {
            divide_sum(rhs);
            divide_sqr(rhs);
        }
        return *this;
    }

protected:
    template <int N> uint64_t compute(const op_type& data)
    {
        auto _ts = get_start<N>(data);
        auto _te = get_stop<N>(data);
        return (_te > _ts) ? (_te - _ts) : uint64_t(0);
    }

    inline void compute_sum(const incr_type& rhs)
    {
        std::get<0>(m_sum) += std::get<0>(rhs);
        std::get<1>(m_sum) += std::get<1>(rhs);
        std::get<2>(m_sum) += std::get<2>(rhs);
    }

    template <int N>
    void subtract_unsigned(data_type& lhs, const data_type& rhs)
    {
        std::get<N>(lhs) = (std::get<N>(lhs) < std::get<N>(rhs))
                           ? (uint_type)(0)
                           : std::get<N>(lhs) - std::get<N>(rhs);
    }

    inline void subtract_sum(const incr_type& rhs)
    {
        subtract_unsigned<0>(m_sum, rhs);
        subtract_unsigned<1>(m_sum, rhs);
        subtract_unsigned<2>(m_sum, rhs);
    }

    #if defined(TIMEMORY_STAT_TIMERS)
    inline void compute_sqr(const incr_type& rhs)
    {
        std::get<0>(m_sqr) += std::pow(std::get<0>(rhs), 2);
        std::get<1>(m_sqr) += std::pow(std::get<1>(rhs), 2);
        std::get<2>(m_sqr) += std::pow(std::get<2>(rhs), 2);
    }
    #else
    inline void compute_sqr(const incr_type&) { }
    #endif

    #if defined(TIMEMORY_STAT_TIMERS)
    inline void subtract_sqr(const incr_type& rhs)
    {
        std::get<0>(m_sqr) -= std::pow(std::get<0>(rhs), 2);
        std::get<1>(m_sqr) -= std::pow(std::get<1>(rhs), 2);
        std::get<2>(m_sqr) -= std::pow(std::get<2>(rhs), 2);
    }
    #else
    inline void subtract_sqr(const incr_type&) { }
    #endif

    inline void divide_sum(const uint64_t& rhs)
    {
        std::get<0>(m_sum) /= rhs;
        std::get<1>(m_sum) /= rhs;
        std::get<2>(m_sum) /= rhs;
    }

    #if defined(TIMEMORY_STAT_TIMERS)
    inline void divide_sqr(const uint64_t& rhs)
    {
        std::get<0>(m_sqr) /= rhs;
        std::get<1>(m_sqr) /= rhs;
        std::get<2>(m_sqr) /= rhs;
    }
    #else
    inline void divide_sqr(const uint64_t&) { }
    #endif

protected:
    uint_type   m_lap;
    data_type   m_sum;
    #if defined(TIMEMORY_STAT_TIMERS)
    data_type   m_sqr;
    #endif
    rss_type    m_rss;

};

//============================================================================//
//
//  Primary base class for handling the timer
//
//============================================================================//


class tim_api base_timer
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef std::string                         string_t;
    typedef string_t::size_type                 size_type;
    typedef std::mutex                          mutex_t;
    typedef std::recursive_mutex                rmutex_t;
    typedef std::ostream                        ostream_t;
    typedef std::ofstream                       ofstream_t;
    typedef uomap<ostream_t*, rmutex_t>         mutex_map_t;
    typedef std::lock_guard<mutex_t>            auto_lock_t;
    typedef std::lock_guard<rmutex_t>           recursive_lock_t;
    typedef tms                                 tms_t;
    typedef base_timer_data                     data_t;
    typedef data_t::ratio_t                     ratio_t;
    typedef data_t::clock_t                     base_clock_t;
    typedef data_t::time_point_t                time_point_t;
    typedef base_timer_delta                    data_accum_t;
    typedef data_t::duration_t                  duration_t;
    typedef base_timer                          this_type;
    typedef tim::rss::usage_delta               rss_type;
    typedef format::timer                       format_type;
    typedef std::shared_ptr<format_type>        timer_format_t;
    typedef std::function<void()>               record_func_t;

public:
    base_timer(timer_format_t = timer_format_t(),
               bool _record_memory = true,
               ostream_t* = &std::cout);
    virtual ~base_timer();

    base_timer(const base_timer& rhs);
    base_timer& operator=(const base_timer& rhs);

public:
    // public member functions
    inline void start() { m_start_func(); }
    inline void stop()  { m_stop_func(); }
    inline bool is_valid() const;
    inline bool is_running() const;
    double real_elapsed() const;
    double wall_elapsed() const { return this->real_elapsed(); }
    double system_elapsed() const;
    double user_elapsed() const;
    double cpu_elapsed() const { return user_elapsed() + system_elapsed(); }
    double cpu_utilization() const { return cpu_elapsed() / real_elapsed() * 100.; }
    inline const char* clock_time() const;
    inline size_type laps() const { return m_accum.size(); }
    inline void rss_init();
    inline void rss_record();
    inline void reset() { m_accum.reset(); }
    inline data_accum_t& accum() { return m_accum; }
    inline const data_accum_t& accum() const { return m_accum; }
    inline timer_format_t format() const { return m_format; }
    inline void set_format(const format_type& _format);
    inline void set_format(timer_format_t _format);

public:
    // public member functions
    void report(ostream_t&, bool endline = true, bool ign_cutoff = false) const;
    void report(bool endline = true) const;
    bool above_cutoff(bool ign_cutoff = false) const;
    void sync(const this_type& rhs);

    inline void configure_record();
    void record_memory(bool _val) { m_record_memory = _val; configure_record(); }
    bool record_memory() const { return m_record_memory; }

protected:
    // protected member functions
    data_t& m_timer() const { return m_data; }
    data_accum_t& get_accum() { return m_accum; }
    const data_accum_t& get_accum() const { return m_accum; }

    inline void start_stop_debug_message(const string_t&, const string_t&);
    inline void start_with_memory();
    inline void stop_with_memory();
    inline void start_without_memory();
    inline void stop_without_memory();

protected:
    // protected member variables
    bool                    m_record_memory;
    // pointers
    ostream_t*              m_os;
    // objects
    mutex_t                 m_mutex;
    record_func_t           m_start_func;
    record_func_t           m_stop_func;
    mutable data_t          m_data;
    mutable data_accum_t    m_accum;
    timer_format_t          m_format;

private:
    // world mutex map for thread-safe ostreams
    static mutex_map_t     f_mutex_map;

public:
    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/)
    {
        auto _cpu_util = (m_accum.get_sum<0>() + m_accum.get_sum<1>())
                         / m_accum.get_sum<2>();
        if(!tim::isfinite(_cpu_util))
            _cpu_util = 0.0;

        ar(serializer::make_nvp("laps", m_accum.size()),
           // user clock elapsed
           serializer::make_nvp("user_elapsed",     m_accum.get_sum<0>()),
           // system clock elapsed
           serializer::make_nvp("system_elapsed",   m_accum.get_sum<1>()),
           // wall clock elapsed
           serializer::make_nvp("wall_elapsed",     m_accum.get_sum<2>()),
           // cpu elapsed
           serializer::make_nvp("cpu_elapsed",
                            m_accum.get_sum<0>() + m_accum.get_sum<1>()),
           // cpu utilization
           serializer::make_nvp("cpu_util",         _cpu_util),
   #if defined(TIMEMORY_STAT_TIMERS)
           serializer::make_nvp("wall_elapsed_sqr", m_accum.get_sqr<2>()),
           serializer::make_nvp("user_elapsed_sqr", m_accum.get_sqr<0>()),
           serializer::make_nvp("system_elapsed_sqr",  m_accum.get_sqr<1>()),
   #endif
           // conversion to seconds
           serializer::make_nvp("to_seconds_ratio_num", ratio_t::num),
           serializer::make_nvp("to_seconds_ratio_den", ratio_t::den),
           // memory usage
           serializer::make_nvp("rss_max",  m_accum.rss().total()),
           serializer::make_nvp("rss_self", m_accum.rss().self()),
           // memory usage (minimum)
           serializer::make_nvp("rss_min",  m_accum.rss().total_min()),
           serializer::make_nvp("rss_self_min", m_accum.rss().self_min()));
    }

};

//----------------------------------------------------------------------------//
inline
void base_timer::set_format(const format_type& _format)
{
    m_format = timer_format_t(new format_type(_format));
}
//----------------------------------------------------------------------------//
inline
void base_timer::set_format(timer_format_t _format)
{
    m_format = _format;
}
//----------------------------------------------------------------------------//
inline void base_timer::rss_init()
{
    m_timer().rss_init();
}
//----------------------------------------------------------------------------//
inline void base_timer::rss_record()
{
    m_timer().rss_record();
}
//----------------------------------------------------------------------------//
// Print timer status n std::ostream
static inline
std::ostream& operator<<(std::ostream& os, const base_timer& t)
{
    bool restart = !t.is_valid();
    if(restart)
        const_cast<base_timer&>(t).stop();
    t.report(os);
    if(restart)
        const_cast<base_timer&>(t).start();

    return os;
}
//----------------------------------------------------------------------------//
inline                                                          // Wall time
double base_timer::real_elapsed() const
{
    if(m_timer().running())
        throw std::runtime_error("Error! base_timer::real_elapsed() - "
                                 "timer not stopped or no times recorded!");
    return m_accum.get_sum<2>() / static_cast<double>(ratio_t::den);
}
//----------------------------------------------------------------------------//
inline                                                          // System time
double base_timer::system_elapsed() const
{
    if(m_timer().running())
        throw std::runtime_error("Error! base_timer::system_elapsed() - "
                                 "timer not stopped or no times recorded!");
    return m_accum.get_sum<1>() / static_cast<double>(ratio_t::den);
}
//----------------------------------------------------------------------------//
inline                                                          // CPU time
double base_timer::user_elapsed() const
{
    if(m_timer().running())
        throw std::runtime_error("Error! base_timer::user_elapsed() - "
                                 "timer not stopped or no times recorded!");
    return m_accum.get_sum<0>() / static_cast<double>(ratio_t::den);
}
//----------------------------------------------------------------------------//
inline void base_timer::start_stop_debug_message(const string_t& _func,
                                                 const string_t& _already)
{
    int32_t _verbose = tim::get_env<int32_t>("TIMEMORY_VERBOSE", 0);
    if(_verbose > 0)
    {
        std::stringstream _msg;
        _msg << "Warning! base_timer::" << _func << " called but already "
             << _already << "..." << std::endl;
        if(_verbose > 1)
            tim::stack_backtrace(_msg);
        std::cerr << _msg.str();
    }
}
//----------------------------------------------------------------------------//
inline
void base_timer::configure_record()
{
    if(m_record_memory)
    {
        auto _start = [=] () { this->start_with_memory(); };
        auto _stop  = [=] () { this->stop_with_memory();  };
        m_start_func = _start;
        m_stop_func = _stop;
    }
    else
    {
        auto _start = [=] () { this->start_without_memory(); };
        auto _stop  = [=] () { this->stop_without_memory();  };
        m_start_func = _start;
        m_stop_func = _stop;
    }
}
//----------------------------------------------------------------------------//
inline
void base_timer::start_with_memory()
{
    if(!m_timer().running())
    {
        m_timer().start() = base_clock_t::now();
        rss_init();
    }
#if !defined(NDEBUG)
    else
        start_stop_debug_message(__FUNCTION__, "running");
#endif
}
//----------------------------------------------------------------------------//
inline
void base_timer::start_without_memory()
{
    if(!m_timer().running())
        m_timer().start() = base_clock_t::now();
#if !defined(NDEBUG)
    else
        start_stop_debug_message(__FUNCTION__, "running");
#endif
}
//----------------------------------------------------------------------------//
inline
void base_timer::stop_with_memory()
{
    if(m_timer().running())
    {
        m_timer().stop() = base_clock_t::now();
        rss_record();
        m_accum += m_timer();
    }
#if !defined(NDEBUG)
    else
        start_stop_debug_message(__FUNCTION__, "stopped");
#endif
}
//----------------------------------------------------------------------------//
inline
void base_timer::stop_without_memory()
{
    if(m_timer().running())
    {
        m_timer().stop() = base_clock_t::now();
        m_accum += m_timer();
    }
#if !defined(NDEBUG)
    else
        start_stop_debug_message(__FUNCTION__, "stopped");
#endif
}
//----------------------------------------------------------------------------//
inline
bool base_timer::is_valid() const
{
    return (m_timer().running()) ? false : true;
}
//----------------------------------------------------------------------------//
inline
bool base_timer::is_running() const
{
    return m_timer().running();
}
//----------------------------------------------------------------------------//
inline const char* base_timer::clock_time() const
{
    time_t rawtime;
    struct tm* timeinfo;

    std::time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    return asctime (timeinfo);
}
//----------------------------------------------------------------------------//

} // namespace internal

} // namespace tim

//----------------------------------------------------------------------------//

namespace internal
{
typedef typename tim::internal::base_timer_data::ratio_t base_ratio_t;
typedef tim::base_clock<base_ratio_t>   base_clock_t;
typedef tim::base_clock_data<base_ratio_t> base_clock_data_t;
typedef std::chrono::duration<base_clock_data_t, base_ratio_t> base_duration_t;
typedef std::chrono::time_point<base_clock_t, base_duration_t>  base_time_point_t;
typedef std::tuple<base_time_point_t, base_time_point_t> base_time_pair_t;
}

//----------------------------------------------------------------------------//

#endif // base_timer_hpp_

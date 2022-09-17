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
#include "timemory/log/logger.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/units.hpp"

#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace tim
{
namespace sampling
{
using itimerval_t  = struct itimerval;
using itimerspec_t = struct itimerspec;
using sigaction_t  = struct sigaction;

inline itimerspec_t
get_itimerspec(const itimerval_t& _val)
{
    itimerspec_t _spec;
    memset(&_spec, 0, sizeof(_spec));
    _spec.it_interval.tv_sec  = _val.it_interval.tv_sec;
    _spec.it_interval.tv_nsec = 1000 * _val.it_interval.tv_usec;
    _spec.it_value.tv_sec     = _val.it_value.tv_sec;
    _spec.it_value.tv_nsec    = 1000 * _val.it_value.tv_usec;
    return _spec;
}

inline itimerval_t
get_itimerval(const itimerspec_t& _spec)
{
    itimerval_t _val;
    memset(&_val, 0, sizeof(_val));
    _val.it_interval.tv_sec  = _spec.it_interval.tv_sec;
    _val.it_interval.tv_usec = _spec.it_interval.tv_nsec / 1000;
    _val.it_value.tv_sec     = _spec.it_value.tv_sec;
    _val.it_value.tv_usec    = _spec.it_value.tv_nsec / 1000;
    return _val;
}

inline void
set_delay(itimerval_t& _itimer, double fdelay, const std::string& _extra = {},
          bool _verbose = false)
{
    int64_t delay_sec  = fdelay;
    int64_t delay_usec = static_cast<int64_t>(fdelay * 1000000) % 1000000;
    if(_verbose)
    {
        fprintf(stderr, "[T%li]%s sampler delay      : %li sec + %li usec\n",
                threading::get_id(), _extra.c_str(), delay_sec, delay_usec);
    }
    // Configure the timer to expire after designated delay...
    _itimer.it_value.tv_sec  = delay_sec;
    _itimer.it_value.tv_usec = delay_usec;
}

inline void
set_frequency(itimerval_t& _itimer, double _freq, const std::string& _extra = {},
              bool _verbose = false)
{
    double  _period      = 1.0 / _freq;
    int64_t _period_sec  = _period;
    int64_t _period_usec = static_cast<int64_t>(_period * 1000000) % 1000000;
    if(_verbose)
    {
        fprintf(stderr, "[T%li]%s sampler period     : %li sec + %li usec\n",
                threading::get_id(), _extra.c_str(), _period_sec, _period_usec);
    }
    // Configure the timer to expire at designated intervals
    _itimer.it_interval.tv_sec  = _period_sec;
    _itimer.it_interval.tv_usec = _period_usec;
}

inline void
set_delay(itimerspec_t& _itimer, double fdelay, const std::string& _extra = {},
          bool _verbose = false)
{
    int64_t delay_sec  = fdelay;
    int64_t delay_nsec = static_cast<int64_t>(fdelay * 1000000000) % 1000000000;
    if(_verbose)
    {
        fprintf(stderr, "[T%li]%s sampler delay      : %li sec + %li nsec\n",
                threading::get_id(), _extra.c_str(), delay_sec, delay_nsec);
    }
    // Configure the timer to expire after designated delay...
    _itimer.it_value.tv_sec  = delay_sec;
    _itimer.it_value.tv_nsec = delay_nsec;
}

inline void
set_frequency(itimerspec_t& _itimer, double _freq, const std::string& _extra = {},
              bool _verbose = false)
{
    double  _period      = 1.0 / _freq;
    int64_t _period_sec  = _period;
    int64_t _period_nsec = static_cast<int64_t>(_period * 1000000000) % 1000000000;
    if(_verbose)
    {
        fprintf(stderr, "[T%li]%s sampler period     : %li sec + %li nsec\n",
                threading::get_id(), _extra.c_str(), _period_sec, _period_nsec);
    }
    // Configure the timer to expire at designated intervals
    _itimer.it_interval.tv_sec  = _period_sec;
    _itimer.it_interval.tv_nsec = _period_nsec;
}

inline double
get_delay(const itimerval_t& _itimer, int64_t units = units::sec)
{
    double _ns =
        (_itimer.it_value.tv_sec * units::sec) + (_itimer.it_value.tv_usec * units::usec);
    return _ns / static_cast<double>(units);
}

inline double
get_period(const itimerval_t& _itimer, int64_t units = units::sec)
{
    double _ns = (_itimer.it_interval.tv_sec * units::sec) +
                 (_itimer.it_interval.tv_usec * units::usec);
    return _ns / static_cast<double>(units);
}

inline double
get_frequency(const itimerval_t& _itimer, int64_t units = units::sec)
{
    return 1.0 / get_period(_itimer, units);
}

inline double
get_delay(const itimerspec_t& _itimer, int64_t units = units::sec)
{
    double _ns =
        (_itimer.it_value.tv_sec * units::sec) + (_itimer.it_value.tv_nsec * units::nsec);
    return _ns / static_cast<double>(units);
}

inline double
get_period(const itimerspec_t& _itimer, int64_t units = units::sec)
{
    double _ns = (_itimer.it_interval.tv_sec * units::sec) +
                 (_itimer.it_interval.tv_nsec * units::nsec);
    return _ns / static_cast<double>(units);
}

inline double
get_frequency(const itimerspec_t& _itimer, int64_t units = units::sec)
{
    return 1.0 / get_period(_itimer, units);
}

struct timer
{
    timer() = default;
    timer(int _signum, int _clock_type, int _notify, double _freq, double _delay,
          int64_t _tim_tid = threading::get_id(), long _sys_tid = -1);

    ~timer() { stop(); }

    timer(const timer&) = delete;
    timer& operator=(const timer&) = delete;

    timer(timer&& rhs) noexcept;
    timer& operator=(timer&& rhs) noexcept;

    bool initialize();
    bool start();
    bool stop();

    bool is_active() const { return m_is_active; }
    bool is_initialized() const { return m_initialized; }

    auto signal() const { return m_signal; }
    auto clock_id() const { return m_clock_id; }
    auto notify_id() const { return m_notify_id; }
    auto frequency() const { return m_freq; }
    auto delay() const { return m_wait; }
    auto get_tid() const { return m_tim_tid; }
    auto get_sys_tid() const { return m_sys_tid; }

    auto get_timerspec() const;
    auto get_frequency(int64_t _units) const;
    auto get_period(int64_t _units) const;
    auto get_delay(int64_t _units) const;

    void set_signal(int);
    void set_clock_id(int);
    void set_notify_id(int);
    void set_tid(int64_t _timemory_tid, long _system_tid);

    friend std::ostream& operator<<(std::ostream& _os, const timer& _v)
    {
        return (_os << _v.as_string());
    }

private:
    static const char* timer_strerror(int _ret)
    {
        switch(_ret)
        {
            case EINVAL: return "Invalid data"; break;
            case ENOMEM: return "Could not allocate memory"; break;
            case ENOTSUP:
                return "Kernel does not support creating a timer against this clock id";
                break;
            case EPERM: return "Caller did not have the CAP_WAKE_ALARM capability"; break;
            case EFAULT: return "Invalid pointer"; break;
            default: break;
        }
        return "Unknown error";
    }

    std::string as_string() const;

private:
    mutable bool m_initialized = false;
    mutable bool m_is_active   = false;
    int          m_signal      = SIGRTMAX - 1;
    int          m_clock_id    = CLOCK_PROCESS_CPUTIME_ID;
    int          m_notify_id   = SIGEV_SIGNAL;
    long         m_sys_tid     = -1;
    int64_t      m_tim_tid     = threading::get_id();
    double       m_freq        = 50.0;
    double       m_wait        = 0.005;
    itimerspec_t m_spec        = {};
    timer_t      m_timer       = {};
};
//
inline timer::timer(int _signum, int _clock_type, int _notify, double _freq,
                    double _delay, int64_t _tim_tid, long _sys_tid)
: m_signal{ _signum }
, m_clock_id{ _clock_type }
, m_notify_id{ _notify }
, m_sys_tid{ _sys_tid }
, m_tim_tid{ _tim_tid }
, m_freq{ _freq }
, m_wait{ _delay }
{
    memset(&m_spec, 0, sizeof(m_spec));
}
/*
inline timer::timer(const timer& rhs)
: m_initialized{ rhs.m_initialized }
, m_is_active{ rhs.m_is_active }
, m_signal{ rhs.m_signal }
, m_clock_id{ rhs.m_clock_id }
, m_notify_id{ rhs.m_notify_id }
, m_sys_tid{ rhs.m_sys_tid }
, m_tim_tid{ rhs.m_tim_tid }
, m_freq{ rhs.m_freq }
, m_wait{ rhs.m_wait }
, m_spec{ rhs.m_spec }
, m_timer{ rhs.m_timer }
{
    rhs.m_initialized = false;
    rhs.m_is_active   = false;
}

inline timer&
timer::operator=(const timer& rhs)
{
    m_initialized     = rhs.m_initialized;
    m_is_active       = rhs.m_is_active;
    m_signal          = rhs.m_signal;
    m_clock_id        = rhs.m_clock_id;
    m_notify_id       = rhs.m_notify_id;
    m_sys_tid         = rhs.m_sys_tid;
    m_tim_tid         = rhs.m_tim_tid;
    m_freq            = rhs.m_freq;
    m_wait            = rhs.m_wait;
    m_spec            = rhs.m_spec;
    m_timer           = rhs.m_timer;
    rhs.m_initialized = false;
    rhs.m_is_active   = false;
    return *this;
}
*/
inline timer::timer(timer&& rhs) noexcept
: m_initialized{ rhs.m_initialized }
, m_is_active{ rhs.m_is_active }
, m_signal{ rhs.m_signal }
, m_clock_id{ rhs.m_clock_id }
, m_notify_id{ rhs.m_notify_id }
, m_sys_tid{ rhs.m_sys_tid }
, m_tim_tid{ rhs.m_tim_tid }
, m_freq{ rhs.m_freq }
, m_wait{ rhs.m_wait }
, m_spec{ rhs.m_spec }
, m_timer{ rhs.m_timer }
{
    rhs.m_initialized = false;
    rhs.m_is_active   = false;
}

inline timer&
timer::operator=(timer&& rhs) noexcept
{
    m_initialized     = rhs.m_initialized;
    m_is_active       = rhs.m_is_active;
    m_signal          = rhs.m_signal;
    m_clock_id        = rhs.m_clock_id;
    m_notify_id       = rhs.m_notify_id;
    m_sys_tid         = rhs.m_sys_tid;
    m_tim_tid         = rhs.m_tim_tid;
    m_freq            = rhs.m_freq;
    m_wait            = rhs.m_wait;
    m_spec            = rhs.m_spec;
    m_timer           = rhs.m_timer;
    rhs.m_initialized = false;
    rhs.m_is_active   = false;
    return *this;
}
//
inline auto
timer::get_timerspec() const
{
    return m_spec;
}

inline auto
timer::get_frequency(int64_t _units) const
{
    return sampling::get_frequency(get_timerspec(), _units);
}

inline auto
timer::get_period(int64_t _units) const
{
    return sampling::get_period(get_timerspec(), _units);
}

inline auto
timer::get_delay(int64_t _units) const
{
    return sampling::get_delay(get_timerspec(), _units);
}
//
inline bool
timer::initialize()
{
    if(m_initialized)
        return false;

    struct sigevent _sigevt;
    memset(&_sigevt, 0, sizeof(_sigevt));
    _sigevt.sigev_notify          = m_notify_id;
    _sigevt.sigev_signo           = m_signal;
    _sigevt.sigev_value.sival_ptr = &m_timer;
    if(m_notify_id == SIGEV_THREAD_ID)
    {
        _sigevt._sigev_un._tid = m_sys_tid;
    }

    memset(&m_spec, 0, sizeof(m_spec));
    set_delay(m_spec, m_wait);
    set_frequency(m_spec, m_freq);

    int _ret = 0;
    TIMEMORY_REQUIRE((_ret = timer_create(m_clock_id, &_sigevt, &m_timer)) == 0)
        << "Failed to create timer! " << timer_strerror(_ret) << " :: " << _ret << ". "
        << *this;

    m_initialized = (_ret == 0);
    return true;
}

inline bool
timer::start()
{
    if(m_is_active)
        return false;

    initialize();

    int _ret = 0;
    TIMEMORY_REQUIRE((_ret = timer_settime(m_timer, 0, &m_spec, nullptr)) == 0)
        << "Failed to start timer " << timer_strerror(_ret) << " :: " << _ret << ". "
        << *this;

    m_is_active = (_ret == 0);

    const double _epsilon = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto         _compute_norm = [](double _lhs, double _rhs) -> double {
        return (std::isfinite(_lhs) && std::isfinite(_rhs))
                   ? std::abs((_lhs / _rhs) - 1.0)
                   : 1.0;
    };

    TIMEMORY_REQUIRE(_compute_norm(get_delay(units::sec), m_wait) < _epsilon)
        << "Wait time is not finite :: computed delay " << get_delay(units::sec)
        << " vs. " << *this << "[norm: " << _compute_norm(get_delay(units::sec), m_wait)
        << ")";
    TIMEMORY_REQUIRE(_compute_norm(get_frequency(units::sec), m_freq) < _epsilon)
        << "Interval time is not finite :: computed frequency "
        << get_frequency(units::sec) << " vs. " << *this
        << "[norm: " << _compute_norm(get_frequency(units::sec), m_freq) << ")";
    TIMEMORY_REQUIRE(_compute_norm(get_period(units::sec), 1.0 / m_freq) < _epsilon)
        << "Period is not finite :: computed period " << get_period(units::sec) << " vs. "
        << *this << "[norm: " << _compute_norm(get_period(units::sec), 1.0 / m_freq)
        << ")";

    return true;
}

inline bool
timer::stop()
{
    if(m_initialized)
    {
        if(m_is_active)
        {
            itimerspec_t _spec;
            memset(&_spec, 0, sizeof(_spec));
            _spec.it_value.tv_sec     = 0;
            _spec.it_value.tv_nsec    = 0;
            _spec.it_interval.tv_sec  = 0;
            _spec.it_interval.tv_nsec = 0;
            int _ret                  = 0;
            TIMEMORY_PREFER((_ret = timer_settime(m_timer, 0, &_spec, nullptr)) == 0)
                << "Failed to stop timer : " << timer_strerror(_ret) << " :: " << _ret
                << ". " << *this;
            m_is_active = false;
        }
        int _ret = 0;
        TIMEMORY_REQUIRE((_ret = timer_delete(m_timer)) == 0)
            << "Failed to delete timer : " << timer_strerror(_ret) << " :: " << _ret
            << ". " << *this;
        m_initialized = false;
        return true;
    }
    return false;
}
//
inline void
timer::set_signal(int _v)
{
    if(!m_is_active)
        m_signal = _v;
}

inline void
timer::set_clock_id(int _v)
{
    if(!m_is_active)
        m_clock_id = _v;
}

inline void
timer::set_notify_id(int _v)
{
    if(!m_is_active)
        m_notify_id = _v;
}

inline void
timer::set_tid(int64_t _tim, long _sys)
{
    if(!m_is_active)
    {
        m_tim_tid = _tim;
        m_sys_tid = _sys;
    }
}
//
inline std::string
timer::as_string() const
{
    std::stringstream _os;
    _os << std::boolalpha;
    _os << "tid=" << m_tim_tid << ", sys_tid=" << m_sys_tid << ", init=" << m_initialized
        << ", is_active=" << m_is_active << ", signal=" << m_signal
        << ", clock_id=" << m_clock_id << ", notify_id=" << m_notify_id
        << ", freq=" << std::fixed << std::setprecision(3) << m_freq
        << " interrupts/sec, period=" << std::scientific << std::setprecision(3)
        << (1.0 / m_freq) << " sec, wait=" << std::setprecision(3) << m_wait << " sec";
    return _os.str();
}
}  // namespace sampling
}  // namespace tim

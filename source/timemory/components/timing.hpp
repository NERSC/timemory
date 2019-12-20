//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file timemory/components/timing.hpp
 * \headerfile timemory/components/timing.hpp "timemory/components/timing.hpp"
 * Provides components for timing-related records
 *
 */

#pragma once

#include "timemory/backends/clocks.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/units.hpp"

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

extern template struct base<wall_clock>;
extern template struct base<system_clock>;
extern template struct base<user_clock>;
extern template struct base<cpu_clock>;
extern template struct base<monotonic_clock>;
extern template struct base<monotonic_raw_clock>;
extern template struct base<thread_cpu_clock>;
extern template struct base<process_cpu_clock>;
extern template struct base<cpu_util, std::pair<int64_t, int64_t>>;
extern template struct base<process_cpu_util, std::pair<int64_t, int64_t>>;
extern template struct base<thread_cpu_util, std::pair<int64_t, int64_t>>;

#endif

//--------------------------------------------------------------------------------------//
//
//          Timing types
//
//--------------------------------------------------------------------------------------//
// the system's real time (i.e. wall time) clock, expressed as the amount of time since
// the epoch.
struct wall_clock : public base<wall_clock, int64_t>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<wall_clock, value_type>;

    static std::string label() { return "wall"; }
    static std::string description() { return "wall time"; }
    static value_type  record() { return tim::get_clock_real_now<int64_t, ratio_t>(); }

    double get_display() const { return get(); }
    double get() const
    {
        auto val = (is_transient) ? static_cast<value_type>(accum) : value;
        return static_cast<double>(val) / ratio_t::den * get_unit();
    }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// alias for "wall_clock"
using real_clock = wall_clock;
// alias for "wall_clock" since time is a construct of our consciousness
using virtual_clock = wall_clock;

//--------------------------------------------------------------------------------------//
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesnâ€™t work that way.
// units are reported in number of clock ticks per second
//
// this struct extracts only the CPU time spent in kernel-mode
struct system_clock : public base<system_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<system_clock, value_type>;

    static std::string label() { return "sys"; }
    static std::string description() { return "system time"; }
    static value_type  record() { return tim::get_clock_system_now<int64_t, ratio_t>(); }
    double             get_display() const
    {
        auto val = (is_transient) ? static_cast<value_type>(accum) : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    double get() const { return get_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesnâ€™t work that way.
// units are reported in number of clock ticks per second
//
// this struct extracts only the CPU time spent in user-mode
struct user_clock : public base<user_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<user_clock, value_type>;

    static std::string label() { return "user"; }
    static std::string description() { return "user time"; }
    static value_type  record() { return tim::get_clock_user_now<int64_t, ratio_t>(); }
    double             get_display() const
    {
        auto val = (is_transient) ? static_cast<value_type>(accum) : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    double get() const { return get_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesnâ€™t work that way.
// units are reported in number of clock ticks per second
//
// this struct extracts only the CPU time spent in both user- and kernel- mode
struct cpu_clock : public base<cpu_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<cpu_clock, value_type>;

    static std::string label() { return "cpu"; }
    static std::string description() { return "cpu time"; }
    static value_type  record() { return tim::get_clock_cpu_now<int64_t, ratio_t>(); }
    double             get_display() const
    {
        auto val = (is_transient) ? static_cast<value_type>(accum) : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    double get() const { return get_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// clock that increments monotonically, tracking the time since an arbitrary point,
// and will continue to increment while the system is asleep.
struct monotonic_clock : public base<monotonic_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<monotonic_clock, value_type>;

    static std::string label() { return "monotonic_clock"; }
    static std::string description() { return "monotonic time"; }
    static value_type  record()
    {
        return tim::get_clock_monotonic_now<int64_t, ratio_t>();
    }
    double get_display() const
    {
        auto val = (is_transient) ? static_cast<value_type>(accum) : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    double get() const { return get_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// clock that increments monotonically, tracking the time since an arbitrary point like
// CLOCK_MONOTONIC.  However, this clock is unaffected by frequency or time adjustments.
// It should not be compared to other system time sources.
struct monotonic_raw_clock : public base<monotonic_raw_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<monotonic_raw_clock, value_type>;

    static std::string label() { return "monotonic_raw_clock"; }
    static std::string description() { return "monotonic raw time"; }
    static value_type  record()
    {
        return tim::get_clock_monotonic_raw_now<int64_t, ratio_t>();
    }
    double get_display() const
    {
        auto val = (is_transient) ? static_cast<value_type>(accum) : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    double get() const { return get_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// this clock measures the CPU time within the current thread (excludes sibling/child
// threads)
// clock that tracks the amount of CPU (in user- or kernel-mode) used by the calling
// thread.
struct thread_cpu_clock : public base<thread_cpu_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<thread_cpu_clock, value_type>;

    static std::string label() { return "thread_cpu"; }
    static std::string description() { return "thread cpu time"; }
    static value_type  record() { return tim::get_clock_thread_now<int64_t, ratio_t>(); }
    double             get_display() const
    {
        auto val = (is_transient) ? static_cast<value_type>(accum) : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    double get() const { return get_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// this clock measures the CPU time within the current process (excludes child processes)
// clock that tracks the amount of CPU (in user- or kernel-mode) used by the calling
// process.
struct process_cpu_clock : public base<process_cpu_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<process_cpu_clock, value_type>;

    static std::string label() { return "process_cpu"; }
    static std::string description() { return "process cpu time"; }
    static value_type  record() { return tim::get_clock_process_now<int64_t, ratio_t>(); }
    double             get_display() const
    {
        auto val = (is_transient) ? static_cast<value_type>(accum) : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    double get() const { return get_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// this computes the CPU utilization percentage for the calling process and child
// processes.
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesnâ€™t work that way.
//
// this struct extracts only the CPU time spent in both user- and kernel- mode
// and divides by wall clock time
struct cpu_util : public base<cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<cpu_util, value_type>;
    using this_type  = cpu_util;

    static std::string label() { return "cpu_util"; }
    static std::string description() { return "cpu utilization"; }
    static value_type  record()
    {
        return value_type(cpu_clock::record(), wall_clock::record());
    }
    double get_display() const
    {
        const auto& _data =
            (is_transient) ? static_cast<const value_type&>(accum) : value;
        double denom = (_data.second > 0) ? _data.second : 1;
        double numer = (_data.second > 0) ? _data.first : 0;
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    double serialization() { return get_display(); }
    double get() const { return get_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        accum += rhs.accum;
        value += rhs.value;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        accum -= rhs.accum;
        value -= rhs.value;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }
};

//--------------------------------------------------------------------------------------//
// this computes the CPU utilization percentage for ONLY the calling process (excludes
// child processes)
//
// this struct extracts only the CPU time spent in both user- and kernel- mode
// and divides by wall clock time
struct process_cpu_util : public base<process_cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<process_cpu_util, value_type>;
    using this_type  = process_cpu_util;

    static std::string label() { return "proc_cpu_util"; }
    static std::string description() { return "process cpu utilization"; }
    static value_type  record()
    {
        return value_type(process_cpu_clock::record(), wall_clock::record());
    }
    double get_display() const
    {
        const auto& _data =
            (is_transient) ? static_cast<const value_type&>(accum) : value;
        double denom = (_data.second > 0) ? _data.second : 1;
        double numer = (_data.second > 0) ? _data.first : 0;
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    double serialization() { return get_display(); }
    double get() const { return get_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        accum += rhs.accum;
        value += rhs.value;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        accum -= rhs.accum;
        value -= rhs.value;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }
};

//--------------------------------------------------------------------------------------//
// this computes the CPU utilization percentage for ONLY the calling thread (excludes
// sibling and child threads)
//
// this struct extracts only the CPU time spent in both user- and kernel- mode
// and divides by wall clock time
struct thread_cpu_util : public base<thread_cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<thread_cpu_util, value_type>;
    using this_type  = thread_cpu_util;

    static std::string label() { return "thread_cpu_util"; }
    static std::string description() { return "thread cpu utilization"; }
    static value_type  record()
    {
        return value_type(thread_cpu_clock::record(), wall_clock::record());
    }
    double get_display() const
    {
        const auto& _data =
            (is_transient) ? static_cast<const value_type&>(accum) : value;
        double denom = (_data.second > 0) ? _data.second : 1;
        double numer = (_data.second > 0) ? _data.first : 0;
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    double serialization() { return get_display(); }
    double get() const { return get_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        accum += rhs.accum;
        value += rhs.value;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        accum -= rhs.accum;
        value -= rhs.value;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }
};

//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim

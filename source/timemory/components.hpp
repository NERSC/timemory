//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California,
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

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <stdio.h>
#include <string>

#include "timemory/clocks.hpp"
#include "timemory/macros.hpp"
#include "timemory/papi.hpp"
#include "timemory/rusage.hpp"
#include "timemory/serializer.hpp"
#include "timemory/units.hpp"

//======================================================================================//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace component
{
//--------------------------------------------------------------------------------------//

template <bool B, typename T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct record_max
{
    // trait that signifies that updating w.r.t. another instance should
    // be a max of the two instances
    static constexpr bool value = false;
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename value_type = intmax_t>
struct base
{
    using Type      = _Tp;
    using this_type = base<_Tp, value_type>;

    bool       is_running   = false;
    bool       is_transient = false;
    value_type value        = value_type();
    value_type accum        = value_type();

    base()                 = default;
    virtual ~base()        = default;
    base(const this_type&) = default;
    base(this_type&&)      = default;
    base& operator=(const this_type&) = default;
    base& operator=(this_type&&) = default;

    //----------------------------------------------------------------------------------//
    // function operator
    //
    value_type& operator()()
    {
        value = Type::record();
        return value;
    }

    //----------------------------------------------------------------------------------//
    // just record a measurment
    //
    void measure() { value = Type::record(); }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start() { static_cast<Type&>(*this).start(); }

    //----------------------------------------------------------------------------------//
    // stop
    //
    void stop() { static_cast<Type&>(*this).stop(); }

    //----------------------------------------------------------------------------------//
    // set the firsts notify that start has been called
    //
    void set_started()
    {
        is_running   = true;
        is_transient = true;
    }

    //----------------------------------------------------------------------------------//
    // set the firsts notify that stop has been called
    //
    void set_stopped() { is_running = false; }

    //----------------------------------------------------------------------------------//
    // conditional start if not running
    //
    bool conditional_start()
    {
        if(!is_running)
        {
            static_cast<Type&>(*this).start();
            return true;
        }
        return false;
    }

    //----------------------------------------------------------------------------------//
    // conditional stop if running
    //
    bool conditional_stop()
    {
        if(is_running)
        {
            static_cast<Type&>(*this).stop();
            return true;
        }
        return false;
    }

    CREATE_STATIC_VARIABLE_ACCESSOR(short, get_precision, precision)
    CREATE_STATIC_VARIABLE_ACCESSOR(short, get_width, width)
    CREATE_STATIC_VARIABLE_ACCESSOR(std::ios_base::fmtflags, get_format_flags,
                                    format_flags)
    CREATE_STATIC_FUNCTION_ACCESSOR(intmax_t, get_unit, unit)
    CREATE_STATIC_FUNCTION_ACCESSOR(std::string, get_label, label)
    CREATE_STATIC_FUNCTION_ACCESSOR(std::string, get_description, descript)
    CREATE_STATIC_FUNCTION_ACCESSOR(std::string, get_display_unit, display_unit)

    //----------------------------------------------------------------------------------//
    // comparison operators
    //
    bool operator==(const base<Type>& rhs) const { return (value == rhs.value); }
    bool operator<(const base<Type>& rhs) const { return (value < rhs.value); }
    bool operator>(const base<Type>& rhs) const { return (value > rhs.value); }
    bool operator!=(const base<Type>& rhs) const { return !(*this == rhs); }
    bool operator<=(const base<Type>& rhs) const { return !(*this > rhs); }
    bool operator>=(const base<Type>& rhs) const { return !(*this < rhs); }

    //----------------------------------------------------------------------------------//
    // this_type operators (plain-old data)
    //
    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    Type& operator+=(const this_type& rhs)
    {
        value += rhs.value;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    Type& operator-=(const this_type& rhs)
    {
        value -= rhs.value;
        return static_cast<Type&>(*this);
    }

    //----------------------------------------------------------------------------------//
    // this_type operators (complex data)
    //
    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value)> = 0>
    Type& operator+=(const this_type& rhs)
    {
        return static_cast<Type&>(*this).operator+=(rhs.value);
    }

    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value)> = 0>
    Type& operator-=(const this_type& rhs)
    {
        return static_cast<Type&>(*this).operator-=(rhs.value);
    }

    //----------------------------------------------------------------------------------//
    // value type operators (plain-old data)
    //
    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    Type& operator+=(const value_type& rhs)
    {
        value += rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    Type& operator-=(const value_type& rhs)
    {
        value -= rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    Type& operator*=(const value_type& rhs)
    {
        value *= rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    Type& operator/=(const value_type& rhs)
    {
        value /= rhs;
        return static_cast<Type&>(*this);
    }

    //----------------------------------------------------------------------------------//
    // value type operators (complex data)
    //
    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value)> = 0>
    Type& operator+=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator+=(rhs);
    }

    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value)> = 0>
    Type& operator-=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator-=(rhs);
    }

    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value)> = 0>
    Type& operator*=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator*=(rhs);
    }

    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value)> = 0>
    Type& operator/=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator/=(rhs);
    }

    //----------------------------------------------------------------------------------//
    // friend operators
    //
    friend Type operator+(const this_type& lhs, const this_type& rhs)
    {
        return base<Type>(lhs) += rhs;
    }

    friend Type operator-(const this_type& lhs, const this_type& rhs)
    {
        return base<Type>(lhs) -= rhs;
    }

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        auto obj_value = static_cast<const Type&>(obj).compute_display();
        auto label     = get_label();
        auto disp      = get_display_unit();
        auto prec      = get_precision();
        auto width     = get_width();
        auto flags     = get_format_flags();

        std::stringstream ss, ssv, ssi;
        ssv.setf(flags);
        ssv << std::setw(width) << std::setprecision(prec) << obj_value;
        if(!disp.empty())
            ssv << " " << disp;
        ssi << " " << label;
        ss << ssv.str() << ssi.str();
        os << ss.str();

        return os;
    }
};

//--------------------------------------------------------------------------------------//
//
//          Timing types
//
//--------------------------------------------------------------------------------------//

struct real_clock : public base<real_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using base_type  = base<real_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "real"; }
    static std::string descript() { return "wall time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record()
    {
        return tim::get_clock_realtime_now<intmax_t, ratio_t>();
    }
    double compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
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

using wall_clock = real_clock;

//--------------------------------------------------------------------------------------//

struct system_clock : public base<system_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using base_type  = base<system_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "sys"; }
    static std::string descript() { return "system time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return tim::get_clock_system_now<intmax_t, ratio_t>(); }
    double             compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
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

struct user_clock : public base<user_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using base_type  = base<user_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "user"; }
    static std::string descript() { return "user time"; }
    static std::string display_unit() { return "sec"; }
    static value_type record() { return tim::get_clock_process_now<intmax_t, ratio_t>(); }
    double            compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
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

struct cpu_clock : public base<cpu_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using base_type  = base<cpu_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "cpu"; }
    static std::string descript() { return "cpu time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return user_clock::record() + system_clock::record(); }
    double             compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
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

struct monotonic_clock : public base<monotonic_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using base_type  = base<monotonic_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "mono"; }
    static std::string descript() { return "monotonic time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record()
    {
        return tim::get_clock_monotonic_now<intmax_t, ratio_t>();
    }
    double compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
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

struct monotonic_raw_clock : public base<monotonic_raw_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using base_type  = base<monotonic_raw_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "raw_mono"; }
    static std::string descript() { return "monotonic raw time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record()
    {
        return tim::get_clock_monotonic_raw_now<intmax_t, ratio_t>();
    }
    double compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
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

struct thread_cpu_clock : public base<thread_cpu_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using base_type  = base<thread_cpu_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "thread_cpu"; }
    static std::string descript() { return "thread cpu time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return tim::get_clock_thread_now<intmax_t, ratio_t>(); }
    double             compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
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

struct process_cpu_clock : public base<process_cpu_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using base_type  = base<process_cpu_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "process_cpu"; }
    static std::string descript() { return "process cpu time"; }
    static std::string display_unit() { return "sec"; }
    static value_type record() { return tim::get_clock_process_now<intmax_t, ratio_t>(); }
    double            compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
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

struct cpu_util : public base<cpu_util, std::pair<intmax_t, intmax_t>>
{
    using ratio_t    = std::micro;
    using value_type = std::pair<intmax_t, intmax_t>;
    using base_type  = base<cpu_util, value_type>;
    using this_type  = cpu_util;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return 1; }
    static std::string label() { return "cpu_util"; }
    static std::string descript() { return "cpu utilization"; }
    static std::string display_unit() { return "%"; }
    static value_type  record()
    {
        return value_type(user_clock::record() + system_clock::record(),
                          real_clock::record());
    }
    double compute_display() const
    {
        double denom =
            (accum.second > 0) ? accum.second : ((value.second > 0) ? value.second : 1);
        double numer =
            (accum.second > 0) ? accum.first : ((value.second > 0) ? value.first : 0);
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    double serial() { return compute_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum.first += (tmp.first - value.first);
        accum.second += (tmp.second - value.second);
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const value_type& rhs)
    {
        this->value.first += rhs.first;
        this->value.second += rhs.second;
        return *this;
    }

    this_type& operator-=(const value_type& rhs)
    {
        this->value.first -= rhs.first;
        this->value.second -= rhs.second;
        return *this;
    }
};

//--------------------------------------------------------------------------------------//

struct process_cpu_util : public base<process_cpu_util, std::pair<intmax_t, intmax_t>>
{
    using ratio_t    = std::micro;
    using value_type = std::pair<intmax_t, intmax_t>;
    using base_type  = base<process_cpu_util, value_type>;
    using this_type  = process_cpu_util;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return 1; }
    static std::string label() { return "process_perc_cpu"; }
    static std::string descript() { return "process cpu utilization"; }
    static std::string display_unit() { return "%"; }
    static value_type  record()
    {
        return value_type(process_cpu_clock::record(), real_clock::record());
    }
    double compute_display() const
    {
        double denom =
            (accum.second > 0) ? accum.second : ((value.second > 0) ? value.second : 1);
        double numer =
            (accum.second > 0) ? accum.first : ((value.second > 0) ? value.first : 0);
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    double serial() { return compute_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum.first += (tmp.first - value.first);
        accum.second += (tmp.second - value.second);
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const value_type& rhs)
    {
        this->value.first += rhs.first;
        this->value.second += rhs.second;
        return *this;
    }

    this_type& operator-=(const value_type& rhs)
    {
        this->value.first -= rhs.first;
        this->value.second -= rhs.second;
        return *this;
    }
};

//--------------------------------------------------------------------------------------//

struct thread_cpu_util : public base<thread_cpu_util, std::pair<intmax_t, intmax_t>>
{
    using ratio_t    = std::micro;
    using value_type = std::pair<intmax_t, intmax_t>;
    using base_type  = base<thread_cpu_util, value_type>;
    using this_type  = thread_cpu_util;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return 1; }
    static std::string label() { return "thread_perc_cpu"; }
    static std::string descript() { return "thread cpu utilization"; }
    static std::string display_unit() { return "%"; }
    static value_type  record()
    {
        return value_type(thread_cpu_clock::record(), real_clock::record());
    }
    double compute_display() const
    {
        double denom =
            (accum.second > 0) ? accum.second : ((value.second > 0) ? value.second : 1);
        double numer =
            (accum.second > 0) ? accum.first : ((value.second > 0) ? value.first : 0);
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    double serial() { return compute_display(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum.first += (tmp.first - value.first);
        accum.second += (tmp.second - value.second);
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const value_type& rhs)
    {
        this->value.first += rhs.first;
        this->value.second += rhs.second;
        return *this;
    }

    this_type& operator-=(const value_type& rhs)
    {
        this->value.first -= rhs.first;
        this->value.second -= rhs.second;
        return *this;
    }
};

//--------------------------------------------------------------------------------------//
//
//          Usage types
//
//--------------------------------------------------------------------------------------//

struct peak_rss : public base<peak_rss>
{
    using value_type = intmax_t;
    using base_type  = base<peak_rss, value_type>;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::megabyte; }
    static std::string label() { return "peak_rss"; }
    static std::string descript() { return "max resident set size"; }
    static std::string display_unit() { return "MB"; }
    static value_type  record() { return get_peak_rss(); }
    double             compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    void start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp   = record();
        auto delta = tmp - value;
        accum      = std::max(accum, delta);
        value      = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
//
template <>
struct record_max<peak_rss>
{
    static constexpr bool value = true;
};

//--------------------------------------------------------------------------------------//

struct current_rss : public base<current_rss>
{
    using value_type = intmax_t;
    using base_type  = base<current_rss, value_type>;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::megabyte; }
    static std::string label() { return "current_rss"; }
    static std::string descript() { return "current resident set size"; }
    static std::string display_unit() { return "MB"; }
    static value_type  record() { return get_current_rss(); }
    double             compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    void start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp   = record();
        auto delta = tmp - value;
        accum      = std::max(accum, delta);
        value      = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
//
template <>
struct record_max<current_rss>
{
    static constexpr bool value = true;
};

//--------------------------------------------------------------------------------------//

struct stack_rss : public base<stack_rss>
{
    using value_type = intmax_t;
    using base_type  = base<stack_rss, value_type>;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::kilobyte; }
    static std::string label() { return "rss_stack"; }
    static std::string descript() { return "integral unshared stack size"; }
    static std::string display_unit() { return "KB"; }
    static value_type  record() { return get_stack_rss(); }
    double             compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    void start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp   = record();
        auto delta = tmp - value;
        accum      = std::max(accum, delta);
        value      = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
//
template <>
struct record_max<stack_rss>
{
    static constexpr bool value = true;
};

//--------------------------------------------------------------------------------------//

struct data_rss : public base<data_rss>
{
    using value_type = intmax_t;
    using base_type  = base<data_rss, value_type>;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::kilobyte; }
    static std::string label() { return "rss_data"; }
    static std::string descript() { return "integral unshared data size"; }
    static std::string display_unit() { return "KB"; }
    static value_type  record() { return get_data_rss(); }
    double             compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    void start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp   = record();
        auto delta = tmp - value;
        accum      = std::max(accum, delta);
        value      = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
//
template <>
struct record_max<data_rss>
{
    static constexpr bool value = true;
};

//--------------------------------------------------------------------------------------//

struct num_swap : public base<num_swap>
{
    using value_type = intmax_t;
    using base_type  = base<num_swap>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "num_swap"; }
    static std::string descript() { return "swaps out of main memory"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return get_num_swap(); }
    value_type         compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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

struct num_io_in : public base<num_io_in>
{
    using value_type = intmax_t;
    using base_type  = base<num_io_in>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "io_in"; }
    static std::string descript() { return "block input operations"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return get_num_io_in(); }
    value_type         compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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

struct num_io_out : public base<num_io_out>
{
    using value_type = intmax_t;
    using base_type  = base<num_io_out>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "io_out"; }
    static std::string descript() { return "block output operations"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return get_num_io_out(); }
    value_type         compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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

struct num_minor_page_faults : public base<num_minor_page_faults>
{
    using value_type = intmax_t;
    using base_type  = base<num_minor_page_faults>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "minor_page_faults"; }
    static std::string descript() { return "page reclaims"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return get_num_minor_page_faults(); }
    value_type         compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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

struct num_major_page_faults : public base<num_major_page_faults>
{
    using value_type = intmax_t;
    using base_type  = base<num_major_page_faults>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "major_page_faults"; }
    static std::string descript() { return "page faults"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return get_num_major_page_faults(); }
    value_type         compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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
//
//          PAPI components
//
//--------------------------------------------------------------------------------------//

template <int EventType, int EventSet>
struct papi_event
: public base<papi_event<EventType, EventSet>, long long>
, public counted_object<papi_event<EventType, EventSet>>
, public counted_object<papi_event<0, EventSet>>
{
    using value_type       = long long;
    using base_type        = base<papi_event<EventType, EventSet>, value_type>;
    using this_type        = papi_event<EventType, EventSet>;
    using event_type_count = counted_object<papi_event<EventType, EventSet>>;
    using event_set_count  = counted_object<papi_event<0, EventSet>>;

    static const short                   precision    = 0;
    static const short                   width        = 6;
    static const std::ios_base::fmtflags format_flags = {};

    using base_type::accum;
    using base_type::is_transient;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    papi_event()
    : read_offset(event_type_count::live() - 1)
    {
        add_event_type();
    }

    ~papi_event() { remove_event_type(); }

    papi_event(const papi_event& rhs) = default;
    this_type& operator=(const this_type& rhs) = default;
    papi_event(papi_event&& rhs)               = default;
    this_type& operator=(this_type&&) = default;

    static PAPI_event_info_t info()
    {
        PAPI_event_info_t evt_info;
#if defined(TIMEMORY_USE_PAPI)
        PAPI_get_event_info(EventType, &evt_info);
#endif
        return evt_info;
    }

    static intmax_t    unit() { return 1; }
    static std::string label() { return info().short_descr; }
    static std::string descript() { return info().long_descr; }
    static std::string display_unit() { return info().units; }
    value_type         record()
    {
        start_event_set();
        std::vector<long long> read_value(event_type_count::live(), 0);
        tim::papi::read(EventSet, read_value.data());
        return read_value[read_offset];
    }
    value_type compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    void start()
    {
        set_started();
        start_event_set();
        value = record();
    }
    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        stop_event_set();
        set_stopped();
    }

private:
    intmax_t read_offset = 0;

    static bool& event_type_added()
    {
        static thread_local bool instance = false;
        return instance;
    }

    static bool& event_set_started()
    {
        static thread_local bool instance = false;
        return instance;
    }

    static void add_event_type()
    {
        if(!event_type_added())
        {
            tim::papi::add_event(EventSet, EventType);
            event_type_added() = true;
        }
    }

    static void remove_event_type()
    {
        if(event_type_added() && event_type_count::live() < 1)
        {
            tim::papi::remove_event(EventSet, EventType);
            event_type_added() = true;
        }
    }

    static void start_event_set()
    {
        if(!event_set_started())
        {
            tim::papi::start(EventSet);
            event_set_started() = true;
        }
    }

    static void stop_event_set()
    {
        if(event_set_started() && event_set_count::live() < 1)
        {
            long long* tmp = new long long(0);
            tim::papi::stop(EventSet, tmp);
            tim::papi::destroy_event_set(EventSet);
            delete tmp;
            event_set_started() = false;
        }
    }
};

}  // namespace component

//--------------------------------------------------------------------------------------//
//
//  component definitions
//
//--------------------------------------------------------------------------------------//

template <typename... Types>
class component_tuple;

//--------------------------------------------------------------------------------------//
//  all configurations
//
using usage_components_t =
    component_tuple<component::peak_rss, component::current_rss, component::stack_rss,
                    component::data_rss, component::num_swap, component::num_io_in,
                    component::num_io_out, component::num_minor_page_faults,
                    component::num_major_page_faults>;

using timing_components_t =
    component_tuple<component::real_clock, component::system_clock, component::user_clock,
                    component::cpu_clock, component::monotonic_clock,
                    component::monotonic_raw_clock, component::thread_cpu_clock,
                    component::process_cpu_clock, component::cpu_util,
                    component::thread_cpu_util, component::process_cpu_util>;

//--------------------------------------------------------------------------------------//
//  standard configurations
//
using standard_usage_components_t =
    component_tuple<component::peak_rss, component::current_rss>;

using standard_timing_components_t =
    component_tuple<component::real_clock, component::thread_cpu_clock,
                    component::thread_cpu_util, component::process_cpu_clock,
                    component::process_cpu_util>;

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

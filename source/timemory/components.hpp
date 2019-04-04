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

enum component_type
{
    REALTIME,
    SYSTEM,
    USER,
    MONOTONIC,
    MONOTONIC_RAW,
    THREAD_CPUTIME,
    PROCESS_CPUTIME,
    CPU_UTIL,
    PROCESS_CPU_UTIL,
    THREAD_CPU_UTIL,
    PEAK_RSS,
    CURRENT_RSS,
    STACK_RSS,
    DATA_RSS,
    NUM_SWAP,
    NUM_IO_IN,
    NUM_IO_OUT,
    NUM_MINOR_PAGE_FAULTS,
    NUM_MAJOR_PAGE_FAULTS
};

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
    // start
    //
    void start()
    {
        is_running   = true;
        is_transient = true;
        Type::start(*this);
    }

    //----------------------------------------------------------------------------------//
    // stop
    //
    void stop()
    {
        Type::stop(*this);
        is_running = false;
    }

    //----------------------------------------------------------------------------------//
    // conditional start if not running
    //
    bool conditional_start()
    {
        if(!is_running)
        {
            start();
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
            stop();
            return true;
        }
        return false;
    }

    CREATE_STATIC_VARIABLE_ACCESSOR(component_type, get_component_type, category)
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
        auto obj_value = Type::compute_display(obj);
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
    using store_type = value_type;
    using base_type  = base<real_clock, value_type>;

    static const component_type          category  = REALTIME;
    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "real"; }
    static std::string descript() { return "wall time"; }
    static std::string display_unit() { return "sec"; }
    static intmax_t record() { return tim::get_clock_realtime_now<intmax_t, ratio_t>(); }
    static double   compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

using wall_clock = real_clock;

//--------------------------------------------------------------------------------------//

struct system_clock : public base<system_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<system_clock, value_type>;

    static const component_type          category  = SYSTEM;
    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "sys"; }
    static std::string descript() { return "system time"; }
    static std::string display_unit() { return "sec"; }
    static intmax_t    record() { return tim::get_clock_system_now<intmax_t, ratio_t>(); }
    static double      compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct user_clock : public base<user_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<user_clock, value_type>;

    static const component_type          category  = USER;
    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "user"; }
    static std::string descript() { return "user time"; }
    static std::string display_unit() { return "sec"; }
    static intmax_t record() { return tim::get_clock_monotonic_now<intmax_t, ratio_t>(); }
    static double   compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct cpu_clock : public base<cpu_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<cpu_clock, value_type>;

    static const component_type          category  = USER;
    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "cpu"; }
    static std::string descript() { return "cpu time"; }
    static std::string display_unit() { return "sec"; }
    static intmax_t    record() { return user_clock::record() + system_clock::record(); }
    static double      compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct monotonic_clock : public base<monotonic_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<monotonic_clock, value_type>;

    static const component_type          category  = MONOTONIC;
    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "mono"; }
    static std::string descript() { return "monotonic time"; }
    static std::string display_unit() { return "sec"; }
    static intmax_t record() { return tim::get_clock_monotonic_now<intmax_t, ratio_t>(); }
    static double   compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct monotonic_raw_clock : public base<monotonic_raw_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<monotonic_raw_clock, value_type>;

    static const component_type          category  = MONOTONIC_RAW;
    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "raw_mono"; }
    static std::string descript() { return "monotonic raw time"; }
    static std::string display_unit() { return "sec"; }
    static intmax_t    record()
    {
        return tim::get_clock_monotonic_raw_now<intmax_t, ratio_t>();
    }
    static double compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct thread_cpu_clock : public base<thread_cpu_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<thread_cpu_clock, value_type>;

    static const component_type          category  = THREAD_CPUTIME;
    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "thread_cpu"; }
    static std::string descript() { return "thread cpu time"; }
    static std::string display_unit() { return "sec"; }
    static intmax_t    record() { return tim::get_clock_thread_now<intmax_t, ratio_t>(); }
    static double      compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct process_cpu_clock : public base<process_cpu_clock>
{
    using ratio_t    = std::micro;
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<process_cpu_clock, value_type>;

    static const component_type          category  = PROCESS_CPUTIME;
    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "process_cpu"; }
    static std::string descript() { return "process cpu time"; }
    static std::string display_unit() { return "sec"; }
    static intmax_t record() { return tim::get_clock_process_now<intmax_t, ratio_t>(); }
    static double   compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return static_cast<double>(val / static_cast<double>(ratio_t::den) *
                                   base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct cpu_util : public base<cpu_util, std::pair<intmax_t, intmax_t>>
{
    using ratio_t    = std::micro;
    using value_type = std::pair<intmax_t, intmax_t>;
    using store_type = double;
    using base_type  = base<cpu_util, value_type>;
    using this_type  = cpu_util;

    static const component_type          category  = CPU_UTIL;
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
    static double compute_display(const base_type& obj)
    {
        double denom = (obj.accum.second > 0)
                           ? obj.accum.second
                           : ((obj.value.second > 0) ? obj.value.second : 1);
        double numer = (obj.accum.second > 0)
                           ? obj.accum.first
                           : ((obj.value.second > 0) ? obj.value.first : 0);
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    static double serial(const base_type& obj) { return compute_display(obj); }
    static void   start(base_type& obj) { obj.value = record(); }
    static void   stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum.first += (tmp.first - obj.value.first);
        obj.accum.second += (tmp.second - obj.value.second);
        obj.value = std::move(tmp);
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
    using store_type = double;
    using base_type  = base<process_cpu_util, value_type>;
    using this_type  = process_cpu_util;

    static const component_type          category  = PROCESS_CPU_UTIL;
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
    static double compute_display(const base_type& obj)
    {
        double denom = (obj.accum.second > 0)
                           ? obj.accum.second
                           : ((obj.value.second > 0) ? obj.value.second : 1);
        double numer = (obj.accum.second > 0)
                           ? obj.accum.first
                           : ((obj.value.second > 0) ? obj.value.first : 0);
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    static double serial(const base_type& obj) { return compute_display(obj); }
    static void   start(base_type& obj) { obj.value = record(); }
    static void   stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum.first += (tmp.first - obj.value.first);
        obj.accum.second += (tmp.second - obj.value.second);
        obj.value = std::move(tmp);
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
    using store_type = double;
    using base_type  = base<thread_cpu_util, value_type>;
    using this_type  = thread_cpu_util;

    static const component_type          category  = THREAD_CPU_UTIL;
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
    static double compute_display(const base_type& obj)
    {
        double denom = (obj.accum.second > 0)
                           ? obj.accum.second
                           : ((obj.value.second > 0) ? obj.value.second : 1);
        double numer = (obj.accum.second > 0)
                           ? obj.accum.first
                           : ((obj.value.second > 0) ? obj.value.first : 0);
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    static double serial(const base_type& obj) { return compute_display(obj); }
    static void   start(base_type& obj) { obj.value = record(); }
    static void   stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum.first += (tmp.first - obj.value.first);
        obj.accum.second += (tmp.second - obj.value.second);
        obj.value = std::move(tmp);
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
    using store_type = value_type;
    using base_type  = base<peak_rss, value_type>;

    static const component_type          category  = PEAK_RSS;
    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::megabyte; }
    static std::string label() { return "rss_peak"; }
    static std::string descript() { return "max resident set size"; }
    static std::string display_unit() { return "MB"; }
    static intmax_t    record() { return get_peak_rss(); }
    static double      compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return val / static_cast<double>(base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp  = record();
        obj.accum = std::max(obj.accum, tmp);
        obj.value = std::move(tmp);
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
    using store_type = value_type;
    using base_type  = base<current_rss, value_type>;

    static const component_type          category  = CURRENT_RSS;
    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::megabyte; }
    static std::string label() { return "rss_curr"; }
    static std::string descript() { return "current resident set size"; }
    static std::string display_unit() { return "MB"; }
    static intmax_t    record() { return get_current_rss(); }
    static double      compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return val / static_cast<double>(base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp  = record();
        obj.accum = std::max(obj.accum, tmp);
        obj.value = std::move(tmp);
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
    using store_type = value_type;
    using base_type  = base<stack_rss, value_type>;

    static const component_type          category  = STACK_RSS;
    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::kilobyte; }
    static std::string label() { return "rss_stack"; }
    static std::string descript() { return "integral unshared stack size"; }
    static std::string display_unit() { return "KB"; }
    static intmax_t    record() { return get_stack_rss(); }
    static double      compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return val / static_cast<double>(base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp  = record();
        obj.accum = std::max(obj.accum, tmp);
        obj.value = std::move(tmp);
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
    using store_type = value_type;
    using base_type  = base<data_rss, value_type>;

    static const component_type          category  = DATA_RSS;
    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::kilobyte; }
    static std::string label() { return "rss_data"; }
    static std::string descript() { return "integral unshared data size"; }
    static std::string display_unit() { return "KB"; }
    static intmax_t    record() { return get_data_rss(); }
    static double      compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return val / static_cast<double>(base_type::get_unit());
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp  = record();
        obj.accum = std::max(obj.accum, tmp);
        obj.value = std::move(tmp);
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
    using store_type = value_type;
    using base_type  = base<num_swap>;

    static const component_type          category     = NUM_SWAP;
    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "num_swap"; }
    static std::string descript() { return "swaps out of main memory"; }
    static std::string display_unit() { return ""; }
    static intmax_t    record() { return get_num_swap(); }
    static intmax_t    compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return val;
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct num_io_in : public base<num_io_in>
{
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<num_io_in>;

    static const component_type          category     = NUM_IO_IN;
    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "io_in"; }
    static std::string descript() { return "block input operations"; }
    static std::string display_unit() { return ""; }
    static intmax_t    record() { return get_num_io_in(); }
    static intmax_t    compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return val;
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct num_io_out : public base<num_io_out>
{
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<num_io_out>;

    static const component_type          category     = NUM_IO_OUT;
    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "io_out"; }
    static std::string descript() { return "block output operations"; }
    static std::string display_unit() { return ""; }
    static intmax_t    record() { return get_num_io_out(); }
    static intmax_t    compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return val;
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct num_minor_page_faults : public base<num_minor_page_faults>
{
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<num_minor_page_faults>;

    static const component_type          category     = NUM_MINOR_PAGE_FAULTS;
    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "minor_page_faults"; }
    static std::string descript() { return "page reclaims"; }
    static std::string display_unit() { return ""; }
    static intmax_t    record() { return get_num_minor_page_faults(); }
    static intmax_t    compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return val;
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//

struct num_major_page_faults : public base<num_major_page_faults>
{
    using value_type = intmax_t;
    using store_type = value_type;
    using base_type  = base<num_major_page_faults>;

    static const component_type          category     = NUM_MAJOR_PAGE_FAULTS;
    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "major_page_faults"; }
    static std::string descript() { return "page faults"; }
    static std::string display_unit() { return ""; }
    static intmax_t    record() { return get_num_major_page_faults(); }
    static intmax_t    compute_display(const base_type& obj)
    {
        auto val = (obj.is_transient) ? obj.accum : obj.value;
        return val;
    }
    static void start(base_type& obj) { obj.value = record(); }
    static void stop(base_type& obj)
    {
        auto tmp = record();
        obj.accum += (tmp - obj.value);
        obj.value = std::move(tmp);
    }
};

//--------------------------------------------------------------------------------------//
//
//          PAPI components
//
//--------------------------------------------------------------------------------------//

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
    component_tuple<component::real_clock, component::system_clock, component::user_clock,
                    component::cpu_clock, component::cpu_util, component::thread_cpu_util,
                    component::process_cpu_util>;

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

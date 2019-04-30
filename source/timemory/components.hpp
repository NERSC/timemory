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

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#include "timemory/clocks.hpp"
#include "timemory/graph.hpp"
#include "timemory/macros.hpp"
#include "timemory/papi.hpp"
#include "timemory/rusage.hpp"
#include "timemory/serializer.hpp"
#include "timemory/singleton.hpp"
#include "timemory/storage.hpp"
#include "timemory/units.hpp"
#include "timemory/utility.hpp"

#if defined(TIMEMORY_USE_CUDA) && defined(__NVCC__)
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#endif

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

// trait that signifies that updating w.r.t. another instance should
// be a max of the two instances
template <typename _Tp>
struct record_max : std::false_type
{
};

//--------------------------------------------------------------------------------------//

// trait that signifies that an implementation (e.g. PAPI) is available
template <typename _Tp>
struct impl_available : std::true_type
{
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename value_type = intmax_t>
struct base
{
    using Type         = _Tp;
    using this_type    = base<_Tp, value_type>;
    using storage_type = graph_storage<Type>;

    bool                            is_running   = false;
    bool                            is_transient = false;
    value_type                      value        = value_type();
    value_type                      accum        = value_type();
    intmax_t                        hashid       = 0;
    intmax_t                        laps         = 0;
    typename storage_type::iterator itr;

    base()                          = default;
    virtual ~base()                 = default;
    explicit base(const this_type&) = default;
    explicit base(this_type&&)      = default;
    base& operator=(const this_type&) = default;
    base& operator=(this_type&&) = default;

    //----------------------------------------------------------------------------------//
    //
    value_type get_measurement() const { return Type::record(); }

    //----------------------------------------------------------------------------------//
    // function operator
    //
    value_type& operator()()
    {
        value = Type::record();
        return value;
    }

    //----------------------------------------------------------------------------------//
    // set the graph node prefix
    //
    void set_prefix(const string_t& _prefix)
    {
        storage_type::instance()->set_prefix(_prefix);
    }

    //----------------------------------------------------------------------------------//
    // insert the node into the graph
    //
    void insert_node(bool& exists, const intmax_t& _hashid)
    {
        hashid    = _hashid;
        Type& obj = static_cast<Type&>(*this);
        itr       = storage_type::instance()->insert(hashid, obj, exists);
    }

    //----------------------------------------------------------------------------------//
    // pop the node off the graph
    //
    template <typename U = value_type, enable_if_t<(!std::is_class<U>::value)> = 0>
    void pop_node()
    {
        Type& obj = itr->obj();
        obj.accum += accum;
        obj.value += value;
        obj.is_transient = is_transient;
        obj.is_running   = false;
        obj.laps += laps;
        itr = storage_type::instance()->pop();
    }

    //----------------------------------------------------------------------------------//
    // pop the node off the graph
    //
    template <typename U = value_type, enable_if_t<(std::is_class<U>::value)> = 0>
    void pop_node()
    {
        Type& obj = itr->obj();
        Type& rhs = static_cast<Type&>(*this);
        obj += rhs;
        obj.laps += rhs.laps;
        storage_type::instance()->pop();
    }

    //----------------------------------------------------------------------------------//
    // reset the values
    //
    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    void reset()
    {
        is_running   = false;
        is_transient = false;
        laps         = 0;
        value        = value_type();
        accum        = value_type();
    }

    //----------------------------------------------------------------------------------//
    // reset the values
    //
    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value)> = 0>
    void reset()
    {
        is_running   = false;
        is_transient = false;
        laps         = 0;
        static_cast<Type&>(*this).reset();
    }

    //----------------------------------------------------------------------------------//
    // just record a measurment
    //
    void measure()
    {
        is_running   = false;
        is_transient = false;
        value        = Type::record();
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        ++laps;
        set_started();
        static_cast<Type&>(*this).start();
    }

    //----------------------------------------------------------------------------------//
    // stop
    //
    void stop()
    {
        static_cast<Type&>(*this).stop();
        set_stopped();
    }

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
    void set_stopped()
    {
        is_running   = false;
        is_transient = true;
    }

    //----------------------------------------------------------------------------------//
    // conditional start if not running
    //
    bool conditional_start()
    {
        if(!is_running)
        {
            set_started();
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
            set_stopped();
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
    template <typename U = value_type, enable_if_t<(!std::is_class<U>::value)> = 0>
    Type& operator+=(const this_type& rhs)
    {
        value += rhs.value;
        accum += rhs.accum;
        laps += rhs.laps;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(!std::is_class<U>::value)> = 0>
    Type& operator-=(const this_type& rhs)
    {
        value -= rhs.value;
        accum -= rhs.accum;
        laps -= rhs.laps;
        return static_cast<Type&>(*this);
    }

    //----------------------------------------------------------------------------------//
    // this_type operators (complex data)
    //
    template <typename U = value_type, enable_if_t<(std::is_class<U>::value)> = 0>
    Type& operator+=(const this_type& rhs)
    {
        laps += rhs.laps;
        return static_cast<Type&>(*this).operator+=(static_cast<const Type&>(rhs));
    }

    template <typename U = value_type, enable_if_t<(std::is_class<U>::value)> = 0>
    Type& operator-=(const this_type& rhs)
    {
        laps -= rhs.laps;
        return static_cast<Type&>(*this).operator-=(static_cast<const Type&>(rhs));
    }

    //----------------------------------------------------------------------------------//
    // value type operators (plain-old data)
    //
    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    Type& operator+=(const value_type& rhs)
    {
        value += rhs;
        accum += rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    Type& operator-=(const value_type& rhs)
    {
        value -= rhs;
        accum -= rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    Type& operator*=(const value_type& rhs)
    {
        value *= rhs;
        accum *= rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    Type& operator/=(const value_type& rhs)
    {
        value /= rhs;
        accum /= rhs;
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

    template <typename U = Type, enable_if_t<(impl_available<U>::value)> = 0>
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        auto obj_value = static_cast<const Type&>(obj).compute_display();
        auto label     = get_label();
        auto disp      = get_display_unit();
        auto prec      = get_precision();
        auto width     = get_width();
        auto flags     = get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(flags);
        ss_value << std::setw(width) << std::setprecision(prec) << obj_value;
        if(!disp.empty())
            ss_extra << " " << disp;
        if(!label.empty())
            ss_extra << " " << label;
        os << ss_value.str() << ss_extra.str();

        return os;
    }

    template <typename U = Type, enable_if_t<(!impl_available<U>::value)> = 0>
    friend std::ostream& operator<<(std::ostream& os, const this_type&)
    {
        os << " ?";
        return os;
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        auto _disp = static_cast<const Type&>(*this).compute_display();
        ar(serializer::make_nvp(Type::label() + ".is_transient", is_transient),
           serializer::make_nvp(Type::label() + ".laps", laps),
           serializer::make_nvp(Type::label() + ".value", value),
           serializer::make_nvp(Type::label() + ".accum", accum),
           serializer::make_nvp(Type::label() + ".display", _disp),
           serializer::make_nvp(Type::label() + ".unit.value", Type::unit()),
           serializer::make_nvp(Type::label() + ".unit.repr", Type::display_unit()));
    }

    /*
    template <typename Archive, typename U = value_type,
              enable_if_t<(std::is_class<U>::value)> = 0>
    void serialize(Archive& ar, const unsigned int)
    {
        auto obj_value = static_cast<Type&>(*this).serial();
        ar(serializer::make_nvp(Type::label() + ".value", value),
           serializer::make_nvp(Type::label() + ".unit.value", Type::unit()),
           serializer::make_nvp(Type::label() + ".unit.repr", Type::display_unit()));
    }*/
};

//--------------------------------------------------------------------------------------//
//
//          Timing types
//
//--------------------------------------------------------------------------------------//
// the system's real time (i.e. wall time) clock, expressed as the amount of time since
// the epoch.
struct real_clock : public base<real_clock>
{
    using ratio_t    = std::nano;
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
    static value_type  record() { return tim::get_clock_real_now<intmax_t, ratio_t>(); }

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
// alias for "real" clock
using wall_clock = real_clock;

//--------------------------------------------------------------------------------------//
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesn’t work that way.
// units are reported in number of clock ticks per second
//
// this struct extracts only the CPU time spent in kernel-mode
struct system_clock : public base<system_clock>
{
    using ratio_t    = std::nano;
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
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesn’t work that way.
// units are reported in number of clock ticks per second
//
// this struct extracts only the CPU time spent in user-mode
struct user_clock : public base<user_clock>
{
    using ratio_t    = std::nano;
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
    static value_type  record() { return tim::get_clock_user_now<intmax_t, ratio_t>(); }
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
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesn’t work that way.
// units are reported in number of clock ticks per second
//
// this struct extracts only the CPU time spent in both user- and kernel- mode
struct cpu_clock : public base<cpu_clock>
{
    using ratio_t    = std::nano;
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
    static value_type  record() { return tim::get_clock_cpu_now<intmax_t, ratio_t>(); }
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
// clock that increments monotonically, tracking the time since an arbitrary point,
// and will continue to increment while the system is asleep.
struct monotonic_clock : public base<monotonic_clock>
{
    using ratio_t    = std::nano;
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
// clock that increments monotonically, tracking the time since an arbitrary point like
// CLOCK_MONOTONIC.  However, this clock is unaffected by frequency or time adjustments.
// It should not be compared to other system time sources.
struct monotonic_raw_clock : public base<monotonic_raw_clock>
{
    using ratio_t    = std::nano;
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
// this clock measures the CPU time within the current thread (excludes sibling/child
// threads)
// clock that tracks the amount of CPU (in user- or kernel-mode) used by the calling
// thread.
struct thread_cpu_clock : public base<thread_cpu_clock>
{
    using ratio_t    = std::nano;
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
// this clock measures the CPU time within the current process (excludes child processes)
// clock that tracks the amount of CPU (in user- or kernel-mode) used by the calling
// process.
struct process_cpu_clock : public base<process_cpu_clock>
{
    using ratio_t    = std::nano;
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
// this computes the CPU utilization percentage for the calling process and child
// processes.
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesn’t work that way.
//
// this struct extracts only the CPU time spent in both user- and kernel- mode
// and divides by wall clock time
struct cpu_util : public base<cpu_util, std::pair<intmax_t, intmax_t>>
{
    using ratio_t    = std::nano;
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
        return value_type(cpu_clock::record(), real_clock::record());
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
        if(is_transient)
        {
            accum.first += rhs.first;
            accum.second += rhs.second;
        }
        else
        {
            value.first += rhs.first;
            value.second += rhs.second;
        }
        return *this;
    }

    this_type& operator-=(const value_type& rhs)
    {
        if(is_transient)
        {
            accum.first -= rhs.first;
            accum.second -= rhs.second;
        }
        else
        {
            value.first -= rhs.first;
            value.second -= rhs.second;
        }
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        accum.first += rhs.accum.first;
        accum.second += rhs.accum.second;
        value.first += rhs.value.first;
        value.second += rhs.value.second;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        accum.first -= rhs.accum.first;
        accum.second -= rhs.accum.second;
        value.first -= rhs.value.first;
        value.second -= rhs.value.second;
        return *this;
    }
};

//--------------------------------------------------------------------------------------//
// this computes the CPU utilization percentage for ONLY the calling process (excludes
// child processes)
//
// this struct extracts only the CPU time spent in both user- and kernel- mode
// and divides by wall clock time
struct process_cpu_util : public base<process_cpu_util, std::pair<intmax_t, intmax_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<intmax_t, intmax_t>;
    using base_type  = base<process_cpu_util, value_type>;
    using this_type  = process_cpu_util;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return 1; }
    static std::string label() { return "proc_cpu_util"; }
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
        if(is_transient)
        {
            accum.first += rhs.first;
            accum.second += rhs.second;
        }
        else
        {
            value.first += rhs.first;
            value.second += rhs.second;
        }
        return *this;
    }

    this_type& operator-=(const value_type& rhs)
    {
        if(is_transient)
        {
            accum.first -= rhs.first;
            accum.second -= rhs.second;
        }
        else
        {
            value.first -= rhs.first;
            value.second -= rhs.second;
        }
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        accum.first += rhs.accum.first;
        accum.second += rhs.accum.second;
        value.first += rhs.value.first;
        value.second += rhs.value.second;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        accum.first -= rhs.accum.first;
        accum.second -= rhs.accum.second;
        value.first -= rhs.value.first;
        value.second -= rhs.value.second;
        return *this;
    }
};

//--------------------------------------------------------------------------------------//
// this computes the CPU utilization percentage for ONLY the calling thread (excludes
// sibling and child threads)
//
// this struct extracts only the CPU time spent in both user- and kernel- mode
// and divides by wall clock time
struct thread_cpu_util : public base<thread_cpu_util, std::pair<intmax_t, intmax_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<intmax_t, intmax_t>;
    using base_type  = base<thread_cpu_util, value_type>;
    using this_type  = thread_cpu_util;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return 1; }
    static std::string label() { return "thread_cpu_util"; }
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
        if(is_transient)
        {
            accum.first += rhs.first;
            accum.second += rhs.second;
        }
        else
        {
            value.first += rhs.first;
            value.second += rhs.second;
        }
        return *this;
    }

    this_type& operator-=(const value_type& rhs)
    {
        if(is_transient)
        {
            accum.first -= rhs.first;
            accum.second -= rhs.second;
        }
        else
        {
            value.first -= rhs.first;
            value.second -= rhs.second;
        }
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        accum.first += rhs.accum.first;
        accum.second += rhs.accum.second;
        value.first += rhs.value.first;
        value.second += rhs.value.second;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        accum.first -= rhs.accum.first;
        accum.second -= rhs.accum.second;
        value.first -= rhs.value.first;
        value.second -= rhs.value.second;
        return *this;
    }
};

//--------------------------------------------------------------------------------------//
//
//          Usage types
//
//--------------------------------------------------------------------------------------//
// this struct extracts the high-water mark (or a change in the high-water mark) of
// the resident set size (RSS). Which is current amount of memory in RAM
//
// when used on a system with swap enabled, this value may fluctuate but should not
// on an HPC system.
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
struct record_max<peak_rss> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
// this struct measures the resident set size (RSS) currently allocated in pages of
// memory. Unlike the peak_rss, this value will fluctuate as memory gets freed and
// allocated
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
struct record_max<current_rss> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
// an integral value indicating the amount of memory used by the text segment that was
// also shared among other processes.
// an integral value of the amount of unshared memory residing in the stack segment
// of a process
struct stack_rss : public base<stack_rss>
{
    using value_type = intmax_t;
    using base_type  = base<stack_rss, value_type>;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::kilobyte; }
    static std::string label() { return "stack_rss"; }
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
struct record_max<stack_rss> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
// an integral value of the amount of unshared memory residing in the data segment of
// a process
struct data_rss : public base<data_rss>
{
    using value_type = intmax_t;
    using base_type  = base<data_rss, value_type>;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::kilobyte; }
    static std::string label() { return "data_rss"; }
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
struct record_max<data_rss> : std::true_type
{
};

//--------------------------------------------------------------------------------------//
// the number of times a process was swapped out of main memory.
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
// the number of times the file system had to perform input.
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
// the number of times the file system had to perform output.
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
// the number of page faults serviced without any I/O activity; here I/O activity is
// avoided by reclaiming a page frame from the list of pages awaiting reallocation.
struct num_minor_page_faults : public base<num_minor_page_faults>
{
    using value_type = intmax_t;
    using base_type  = base<num_minor_page_faults>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "minor_page_flts"; }
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
// the number of page faults serviced that required I/O activity.
struct num_major_page_faults : public base<num_major_page_faults>
{
    using value_type = intmax_t;
    using base_type  = base<num_major_page_faults>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "major_page_flts"; }
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
// the number of IPC messages sent.
struct num_msg_sent : public base<num_msg_sent>
{
    using value_type = intmax_t;
    using base_type  = base<num_msg_sent>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "num_msg_sent"; }
    static std::string descript() { return "messages sent"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return get_num_messages_sent(); }
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
// the number of IPC messages received.
struct num_msg_recv : public base<num_msg_recv>
{
    using value_type = intmax_t;
    using base_type  = base<num_msg_recv>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "num_msg_recv"; }
    static std::string descript() { return "messages received"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return get_num_messages_received(); }
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
// the number of signals delivered
struct num_signals : public base<num_signals>
{
    using value_type = intmax_t;
    using base_type  = base<num_signals>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "num_signals"; }
    static std::string descript() { return "signals delievered"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return get_num_signals(); }
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
// the number of times a context switch resulted due to a process voluntarily giving up
// the processor before its time slice was completed (usually to await availability of a
// resource).
struct voluntary_context_switch : public base<voluntary_context_switch>
{
    using value_type = intmax_t;
    using base_type  = base<voluntary_context_switch>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "vol_cxt_swch"; }
    static std::string descript() { return "voluntary context switches"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return get_num_voluntary_context_switch(); }
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
// the number of times a context switch resulted due to a process voluntarily giving up
// the processor before its time slice was completed (usually to await availability of a
// resource).
struct priority_context_switch : public base<priority_context_switch>
{
    using value_type = intmax_t;
    using base_type  = base<priority_context_switch>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static intmax_t    unit() { return 1; }
    static std::string label() { return "prio_cxt_swch"; }
    static std::string descript() { return "priority context switches"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return get_num_priority_context_switch(); }
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

template <int EventSet, int... EventTypes>
struct papi_event
: public base<papi_event<EventSet, EventTypes...>,
              std::array<long long, sizeof...(EventTypes)>>
, public counted_object<papi_event<EventSet>>
{
    using size_type   = std::size_t;
    using value_type  = std::array<long long, sizeof...(EventTypes)>;
    using base_type   = base<papi_event<EventSet, EventTypes...>, value_type>;
    using this_type   = papi_event<EventSet, EventTypes...>;
    using event_count = counted_object<papi_event<EventSet>>;

    static const size_type               num_events = sizeof...(EventTypes);
    static const short                   precision  = 6;
    static const short                   width      = 8;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::scientific | std::ios_base::dec;

    using base_type::accum;
    using base_type::is_transient;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;
    using event_count::m_count;

    papi_event()
    {
        if(m_count == 0 && event_count::is_master())
        {
            add_event_types();
            start_event_set();
        }
    }
    ~papi_event()
    {
        if(m_count == 0 && event_count::is_master())
        {
            // stop_event_set();
            // remove_event_types();
        }
    }

    papi_event(const papi_event& rhs) = default;
    this_type& operator=(const this_type& rhs) = default;
    papi_event(papi_event&& rhs)               = default;
    this_type& operator=(this_type&&) = default;

    static PAPI_event_info_t info(int evt_type)
    {
        PAPI_event_info_t evt_info;
#if defined(TIMEMORY_USE_PAPI)
        PAPI_get_event_info(evt_type, &evt_info);
#else
        consume_parameters(std::move(evt_type));
#endif
        return evt_info;
    }

    static intmax_t unit() { return 1; }
    // leave these empty
    static std::string label() { return "papi" + std::to_string(EventSet); }
    static std::string descript() { return ""; }
    static std::string display_unit() { return ""; }
    // use these instead
    static std::string label(int evt_type) { return info(evt_type).short_descr; }
    static std::string descript(int evt_type) { return info(evt_type).long_descr; }
    static std::string display_unit(int evt_type) { return info(evt_type).units; }

    static value_type record()
    {
        value_type read_value;
        apply<void>::set_value(read_value, 0);
        tim::papi::read(EventSet, read_value.data());
        return read_value;
    }

    string_t compute_display() const
    {
        auto val              = (is_transient) ? accum : value;
        int  evt_types[]      = { EventTypes... };
        auto _compute_display = [&](std::ostream& os, size_type idx) {
            double _obj_value = val[idx];
            auto   _evt_type  = evt_types[idx];
            auto   _label     = label(_evt_type);
            auto   _disp      = display_unit(_evt_type);
            auto   _prec      = base_type::get_precision();
            auto   _width     = base_type::get_width();
            auto   _flags     = base_type::get_format_flags();

            std::stringstream ss, ssv, ssi;
            ssv.setf(_flags);
            ssv << std::setw(_width) << std::setprecision(_prec) << _obj_value;
            if(!_disp.empty())
                ssv << " " << _disp;
            if(!_label.empty())
                ssi << " " << _label;
            ss << ssv.str() << ssi.str();
            os << ss.str();
        };
        std::stringstream ss;
        for(size_type i = 0; i < num_events; ++i)
        {
            _compute_display(ss, i);
            if(i + 1 < num_events)
                ss << ", ";
        }
        return ss.str();
    }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        auto tmp = record();
        for(size_type i = 0; i < num_events; ++i)
            accum[i] += (tmp[i] - value[i]);
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
            accum[i] += rhs.accum[i];
        for(size_type i = 0; i < num_events; ++i)
            value[i] += rhs.value[i];
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
            accum[i] -= rhs.accum[i];
        for(size_type i = 0; i < num_events; ++i)
            value[i] -= rhs.value[i];
        return *this;
    }

    value_type serial() { return accum; }

private:
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

    void add_event_types()
    {
        if(!event_type_added() && m_count == 0 && event_count::is_master())
        {
            // DEBUG_PRINT_HERE(std::to_string(event_count::live()).c_str());
            int evt_types[] = { EventTypes... };
            tim::papi::add_events(EventSet, evt_types, num_events);
            event_type_added() = true;
        }
    }

    void remove_event_types()
    {
        if(event_type_added() && m_count == 0 && event_count::is_master())
        {
            // DEBUG_PRINT_HERE(std::to_string(event_count::live()).c_str());
            int evt_types[] = { EventTypes... };
            for(size_type i = 0; i < num_events; ++i)
                tim::papi::remove_event(EventSet, evt_types[i]);
            event_type_added() = false;
        }
    }

    void start_event_set()
    {
        if(!event_set_started() && m_count == 0 && event_count::is_master())
        {
            // DEBUG_PRINT_HERE(std::to_string(event_count::live()).c_str());
            tim::papi::start(EventSet);
            event_set_started() = true;
        }
    }

    void stop_event_set()
    {
        if(event_set_started() && m_count == 0 && event_count::is_master())
        {
            // DEBUG_PRINT_HERE(std::to_string(event_count::live()).c_str());
            long long* tmp = new long long(0);
            tim::papi::stop(EventSet, tmp);
            tim::papi::destroy_event_set(EventSet);
            delete tmp;
            event_set_started() = false;
        }
    }
};

#if !defined(TIMEMORY_USE_PAPI)
//--------------------------------------------------------------------------------------//
//
template <int EventSet, int... EventTypes>
struct impl_available<papi_event<EventSet, EventTypes...>> : std::false_type
{
};

#endif
//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_CUDA) && defined(__NVCC__)
//--------------------------------------------------------------------------------------//
//
//
// this struct extracts only the CPU time spent in kernel-mode
struct cuda_event : public base<cuda_event, float>
{
    using ratio_t    = std::milli;
    using value_type = float;
    using base_type  = base<cuda_event, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static intmax_t    unit() { return units::sec; }
    static std::string label() { return "evt"; }
    static std::string descript() { return "event time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return 0.0f; }

    cuda_event(cudaStream_t _stream = 0)
    : m_stream(_stream)
    {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }

    ~cuda_event()
    {
        sync();
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }

    float compute_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<float>(val / static_cast<float>(ratio_t::den) *
                                  base_type::get_unit());
    }

    void start()
    {
        set_started();
        cudaStreamAddCallback(m_stream, &cuda_event::callback, static_cast<void*>(this),
                              0);
        cudaEventRecord(m_start, m_stream);
    }

    void stop()
    {
        cudaEventRecord(m_stop, m_stream);
        set_stopped();
    }

    void set_stream(cudaStream_t _stream = 0) { m_stream = _stream; }

    static void callback(cudaStream_t _stream, cudaError_t _status, void* user_data)
    {
        cuda_event* _this = static_cast<cuda_event*>(user_data);
        float       tmp   = 0.0f;
        cudaEventElapsedTime(&tmp, _this->m_start, _this->m_stop);
        _this->accum += tmp;
        _this->value = std::move(tmp);
    }

    void sync() { cudaEventSynchronize(m_stop); }

private:
    cudaStream_t m_stream = 0;
    cudaEvent_t  m_start;
    cudaEvent_t  m_stop;
};

#endif

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
using usage_components_t = component_tuple<
    component::current_rss, component::peak_rss, component::stack_rss,
    component::data_rss, component::num_swap, component::num_io_in, component::num_io_out,
    component::num_minor_page_faults, component::num_major_page_faults,
    component::num_msg_sent, component::num_msg_recv, component::num_signals,
    component::voluntary_context_switch, component::priority_context_switch>;

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
    component_tuple<component::current_rss, component::peak_rss, component::num_io_in,
                    component::num_io_out, component::num_minor_page_faults,
                    component::num_major_page_faults, component::priority_context_switch>;

using standard_timing_components_t =
    component_tuple<component::real_clock, component::thread_cpu_clock,
                    component::thread_cpu_util, component::process_cpu_clock,
                    component::process_cpu_util>;

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

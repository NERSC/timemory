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

/** \file components.hpp
 * \headerfile components.hpp "timemory/components.hpp"
 * These are core tools provided by TiMemory. These tools can be used individually
 * or bundled together in a component_tuple (C++) or component_list (C, Python)
 *
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <numeric>
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

#if defined(TIMEMORY_USE_CUDA)
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
//
template <typename _Tp>
struct record_max : std::false_type
{
};

//--------------------------------------------------------------------------------------//
// trait that signifies that data is an array type
//
template <typename _Tp>
struct array_serialization
{
    using type = std::false_type;
};

//--------------------------------------------------------------------------------------//
// trait that signifies that an implementation (e.g. PAPI) is available
//
template <typename _Tp>
struct impl_available : std::true_type
{
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename value_type = int64_t>
struct base : public tim::counted_object<_Tp>
{
    using Type           = _Tp;
    using this_type      = base<_Tp, value_type>;
    using storage_type   = graph_storage<Type>;
    using graph_iterator = typename storage_type::iterator;

    bool           is_running   = false;
    bool           is_transient = false;
    value_type     value        = value_type();
    value_type     accum        = value_type();
    int64_t        hashid       = 0;
    int64_t        laps         = 0;
    graph_iterator itr;

    base()                          = default;
    ~base()                         = default;
    explicit base(const this_type&) = default;
    explicit base(this_type&&)      = default;
    base& operator=(const this_type&) = default;
    base& operator=(this_type&&) = default;

    //----------------------------------------------------------------------------------//
    // function operator
    //
    value_type operator()() { return Type::record(); }

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
    void insert_node(bool& exists, const int64_t& _hashid)
    {
        hashid    = _hashid;
        Type& obj = static_cast<Type&>(*this);
        itr       = storage_type::instance()->insert(hashid, obj, exists);
    }

    void insert_node(const string_t& _prefix, const int64_t& _hashid)
    {
        hashid    = _hashid;
        Type& obj = static_cast<Type&>(*this);
        itr       = storage_type::instance()->insert(hashid, obj, _prefix);
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
        static_cast<Type&>(*this).start();
        set_started();
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
    CREATE_STATIC_FUNCTION_ACCESSOR(int64_t, get_unit, unit)
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
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(!std::is_class<U>::value)> = 0>
    Type& operator-=(const this_type& rhs)
    {
        value -= rhs.value;
        accum -= rhs.accum;
        laps -= rhs.laps;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
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

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        auto _disp = static_cast<const Type&>(*this).compute_display();
        ar(serializer::make_nvp("is_transient", is_transient),
           serializer::make_nvp("laps", laps), serializer::make_nvp("value", value),
           serializer::make_nvp("accum", accum), serializer::make_nvp("display", _disp));
    }
};

//======================================================================================//
// component initialization
//
/*
class init
{
public:
    using string_t  = std::string;
    bool     store  = false;
    int32_t  ncount = 0;
    int32_t  nhash  = 0;
    string_t key    = "";
    string_t tag    = "";
};
*/
//======================================================================================//
// construction tuple for a component
//
template <typename Type, typename... Args>
class constructor : public std::tuple<Args...>
{
public:
    using base_type                    = std::tuple<Args...>;
    static constexpr std::size_t nargs = std::tuple_size<decay_t<base_type>>::value;

    explicit constructor(Args&&... _args)
    : base_type(std::forward<Args>(_args)...)
    {
    }

    template <typename _Tuple, size_t... _Idx>
    Type operator()(_Tuple&& __t, index_sequence<_Idx...>)
    {
        return Type(std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }

    Type operator()()
    {
        return (*this)(static_cast<base_type>(*this), make_index_sequence<nargs>{});
    }
};

//--------------------------------------------------------------------------------------//
//  component_tuple initialization
//
using init = constructor<void, std::string, std::string, int32_t, int32_t, bool>;

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
    using value_type = int64_t;
    using base_type  = base<real_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "real"; }
    static std::string descript() { return "wall time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return tim::get_clock_real_now<int64_t, ratio_t>(); }

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
    using value_type = int64_t;
    using base_type  = base<system_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "sys"; }
    static std::string descript() { return "system time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return tim::get_clock_system_now<int64_t, ratio_t>(); }
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
    using value_type = int64_t;
    using base_type  = base<user_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "user"; }
    static std::string descript() { return "user time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return tim::get_clock_user_now<int64_t, ratio_t>(); }
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
    using value_type = int64_t;
    using base_type  = base<cpu_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "cpu"; }
    static std::string descript() { return "cpu time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return tim::get_clock_cpu_now<int64_t, ratio_t>(); }
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
    using value_type = int64_t;
    using base_type  = base<monotonic_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "mono"; }
    static std::string descript() { return "monotonic time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record()
    {
        return tim::get_clock_monotonic_now<int64_t, ratio_t>();
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
    using value_type = int64_t;
    using base_type  = base<monotonic_raw_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "raw_mono"; }
    static std::string descript() { return "monotonic raw time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record()
    {
        return tim::get_clock_monotonic_raw_now<int64_t, ratio_t>();
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
    using value_type = int64_t;
    using base_type  = base<thread_cpu_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "thread_cpu"; }
    static std::string descript() { return "thread cpu time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return tim::get_clock_thread_now<int64_t, ratio_t>(); }
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
    using value_type = int64_t;
    using base_type  = base<process_cpu_clock, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "process_cpu"; }
    static std::string descript() { return "process cpu time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return tim::get_clock_process_now<int64_t, ratio_t>(); }
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
// this computes the CPU utilization percentage for the calling process and child
// processes.
// uses clock() -- only relevant as a time when a different is computed
// Do not use a single CPU time as an amount of time; it doesn’t work that way.
//
// this struct extracts only the CPU time spent in both user- and kernel- mode
// and divides by wall clock time
struct cpu_util : public base<cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<cpu_util, value_type>;
    using this_type  = cpu_util;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return 1; }
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

    this_type& operator+=(const this_type& rhs)
    {
        accum.first += rhs.accum.first;
        accum.second += rhs.accum.second;
        value.first += rhs.value.first;
        value.second += rhs.value.second;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        accum.first -= rhs.accum.first;
        accum.second -= rhs.accum.second;
        value.first -= rhs.value.first;
        value.second -= rhs.value.second;
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

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return 1; }
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

    this_type& operator+=(const this_type& rhs)
    {
        accum.first += rhs.accum.first;
        accum.second += rhs.accum.second;
        value.first += rhs.value.first;
        value.second += rhs.value.second;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        accum.first -= rhs.accum.first;
        accum.second -= rhs.accum.second;
        value.first -= rhs.value.first;
        value.second -= rhs.value.second;
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

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return 1; }
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

    this_type& operator+=(const this_type& rhs)
    {
        accum.first += rhs.accum.first;
        accum.second += rhs.accum.second;
        value.first += rhs.value.first;
        value.second += rhs.value.second;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        accum.first -= rhs.accum.first;
        accum.second -= rhs.accum.second;
        value.first -= rhs.value.first;
        value.second -= rhs.value.second;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
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
    using value_type = int64_t;
    using base_type  = base<peak_rss, value_type>;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::megabyte; }
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
    using value_type = int64_t;
    using base_type  = base<current_rss, value_type>;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::megabyte; }
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
    using value_type = int64_t;
    using base_type  = base<stack_rss, value_type>;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::kilobyte; }
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
    using value_type = int64_t;
    using base_type  = base<data_rss, value_type>;

    static const short                   precision = 1;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::kilobyte; }
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
    using value_type = int64_t;
    using base_type  = base<num_swap>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static int64_t     unit() { return 1; }
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
    using value_type = int64_t;
    using base_type  = base<num_io_in>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static int64_t     unit() { return 1; }
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
    using value_type = int64_t;
    using base_type  = base<num_io_out>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static int64_t     unit() { return 1; }
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
    using value_type = int64_t;
    using base_type  = base<num_minor_page_faults>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static int64_t     unit() { return 1; }
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
    using value_type = int64_t;
    using base_type  = base<num_major_page_faults>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static int64_t     unit() { return 1; }
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
    using value_type = int64_t;
    using base_type  = base<num_msg_sent>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static int64_t     unit() { return 1; }
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
    using value_type = int64_t;
    using base_type  = base<num_msg_recv>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static int64_t     unit() { return 1; }
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
    using value_type = int64_t;
    using base_type  = base<num_signals>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static int64_t     unit() { return 1; }
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
    using value_type = int64_t;
    using base_type  = base<voluntary_context_switch>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static int64_t     unit() { return 1; }
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
    using value_type = int64_t;
    using base_type  = base<priority_context_switch>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static int64_t     unit() { return 1; }
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
, public static_counted_object<papi_event<EventSet>>
{
    using size_type   = std::size_t;
    using value_type  = std::array<long long, sizeof...(EventTypes)>;
    using entry_type  = typename value_type::value_type;
    using base_type   = base<papi_event<EventSet, EventTypes...>, value_type>;
    using this_type   = papi_event<EventSet, EventTypes...>;
    using event_count = static_counted_object<papi_event<EventSet>>;

    static const size_type               num_events = sizeof...(EventTypes);
    static const short                   precision  = 6;
    static const short                   width      = 8;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::scientific | std::ios_base::dec;

    using base_type::accum;
    using base_type::is_running;
    using base_type::is_transient;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;
    using event_count::m_count;

    template <typename _Tp>
    using array_t = std::array<_Tp, num_events>;

    papi_event()
    {
        if(event_count::is_master())
        {
            // add_event_types();
            start_event_set();
        }
        apply<void>::set_value(value, 0);
        apply<void>::set_value(accum, 0);
    }

    ~papi_event()
    {
        if(event_count::live() < 1 && event_count::is_master())
        {
            stop_event_set();
            // remove_event_types();
        }
    }

    papi_event(const papi_event& rhs) = default;
    this_type& operator=(const this_type& rhs) = default;
    papi_event(papi_event&& rhs)               = default;
    this_type& operator=(this_type&&) = default;

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        array_t<double> _disp;
        array_t<double> _value;
        array_t<double> _accum;
        for(int i = 0; i < num_events; ++i)
        {
            _disp[i]  = compute_display(i);
            _value[i] = value[i];
            _accum[i] = accum[i];
        }
        ar(serializer::make_nvp("is_transient", is_transient),
           serializer::make_nvp("laps", laps), serializer::make_nvp("value", _value),
           serializer::make_nvp("accum", _accum), serializer::make_nvp("display", _disp));
    }

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

    static int64_t unit() { return 1; }
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
        if(event_count::is_master())
            tim::papi::read(EventSet, read_value.data());
        return read_value;
    }

    entry_type compute_display(int evt_type) const
    {
        auto val = (is_transient) ? accum[evt_type] : value[evt_type];
        return val;
    }

    string_t compute_display() const
    {
        auto val              = (is_transient) ? accum : value;
        int  evt_types[]      = { EventTypes... };
        auto _compute_display = [&](std::ostream& os, size_type idx) {
            auto _obj_value = val[idx];
            auto _evt_type  = evt_types[idx];
            auto _label     = label(_evt_type);
            auto _disp      = display_unit(_evt_type);
            auto _prec      = base_type::get_precision();
            auto _width     = base_type::get_width();
            auto _flags     = base_type::get_format_flags();

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

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    static array_t<std::string> label_array()
    {
        array_t<std::string> arr;
        int                  evt_types[] = { EventTypes... };
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = label(evt_types[i]);
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    static array_t<std::string> descript_array()
    {
        array_t<std::string> arr;
        int                  evt_types[] = { EventTypes... };
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = descript(evt_types[i]);
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    static array_t<std::string> display_unit_array()
    {
        array_t<std::string> arr;
        int                  evt_types[] = { EventTypes... };
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = display_unit(evt_types[i]);
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    static array_t<int64_t> unit_array()
    {
        array_t<int64_t> arr;
        for(size_type i = 0; i < num_events; ++i)
            arr[i] = 1;
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        auto tmp = record();
        for(size_type i = 0; i < num_events; ++i)
        {
            accum[i] += (tmp[i] - value[i]);
            // auto diff = (tmp[i] - value[i]);
            // accum[i] += (diff > 0) ? diff : 0;
        }
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
            accum[i] += rhs.accum[i];
        for(size_type i = 0; i < num_events; ++i)
            value[i] += rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
            accum[i] -= rhs.accum[i];
        for(size_type i = 0; i < num_events; ++i)
            value[i] -= rhs.value[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    value_type serial() { return accum; }

private:
    inline bool acquire_claim(std::atomic<bool>& m_check)
    {
        bool is_set = m_check.load(std::memory_order_relaxed);
        if(is_set)
            return false;
        return m_check.compare_exchange_strong(is_set, true, std::memory_order_relaxed);
    }

    inline bool release_claim(std::atomic<bool>& m_check)
    {
        bool is_set = m_check.load(std::memory_order_relaxed);
        if(!is_set)
            return false;
        return m_check.compare_exchange_strong(is_set, false, std::memory_order_relaxed);
    }

    static std::atomic<bool>& event_type_added()
    {
        static std::atomic<bool> instance(false);
        return instance;
    }

    static std::atomic<bool>& event_set_started()
    {
        static std::atomic<bool> instance(false);
        return instance;
    }

    void add_event_types()
    {
        if(acquire_claim(event_type_added()))
        {
            // PRINT_HERE("");
            int evt_types[] = { EventTypes... };
            tim::papi::add_events(EventSet, evt_types, num_events);
            // PRINT_HERE("");
        }
    }

    void remove_event_types()
    {
        if(release_claim(event_type_added()))
        {
            // PRINT_HERE("");
            int evt_types[] = { EventTypes... };
            tim::papi::remove_events(EventSet, evt_types, num_events);
            // PRINT_HERE("");
        }
    }

    void start_event_set()
    {
        if(acquire_claim(event_set_started()))
        {
            // PRINT_HERE("");
            // tim::papi::start(EventSet);
            int events[] = { EventTypes... };
            tim::papi::start_counters(events, num_events);
            // PRINT_HERE("");
        }
    }

    void stop_event_set()
    {
        if(release_claim(event_set_started()))
        {
            // PRINT_HERE("");
            value_type events;
#if defined(_WINDOWS)
            for(std::size_t i = 0; i < num_events; ++i)
                events[i] = 0;
#else
            apply<void>::set_value(events, 0);
#endif
            tim::papi::stop_counters(events.data(), num_events);
            // tim::papi::stop(EventSet, read_value.data());
            // tim::papi::destroy_event_set(EventSet);
            // PRINT_HERE("");
        }
    }
};

//--------------------------------------------------------------------------------------//
//
template <int EventSet, int... EventTypes>
struct array_serialization<papi_event<EventSet, EventTypes...>>
{
    using type = std::true_type;
};

//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_PAPI)
template <int EventSet, int... EventTypes>
struct impl_available<papi_event<EventSet, EventTypes...>> : std::false_type
{
};
#endif

//--------------------------------------------------------------------------------------//
// this computes the numerator of the roofline for a given set of PAPI counters.
// e.g. for FLOPS roofline (floating point operations / second:
//
//  single precision:
//              cpu_roofline<PAPI_SP_OPS>
//
//  double precision:
//              cpu_roofline<PAPI_DP_OPS>
//
//  generic:
//              cpu_roofline<PAPI_FP_OPS>
//              cpu_roofline<PAPI_SP_OPS, PAPI_DP_OPS>
//
// NOTE: in order to do a roofline, the peak must be calculated with ERT
//      (eventually will be integrated)
//
template <int... EventTypes>
struct cpu_roofline
: public base<cpu_roofline<EventTypes...>,
              std::pair<std::array<long long, sizeof...(EventTypes)>, int64_t>>
{
    using size_type  = std::size_t;
    using array_type = std::array<long long, sizeof...(EventTypes)>;
    using papi_type  = papi_event<0, EventTypes...>;
    using ratio_t    = typename real_clock::ratio_t;
    using value_type = std::pair<array_type, int64_t>;
    using base_type  = base<cpu_roofline, value_type>;
    using this_type  = cpu_roofline<EventTypes...>;

    using base_type::accum;
    using base_type::is_running;
    using base_type::is_transient;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    static const size_type               num_events = sizeof...(EventTypes);
    static const short                   precision  = 3;
    static const short                   width      = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return 1; }
    static std::string label() { return "cpu_roofline"; }
    static std::string descript() { return "cpu roofline"; }
    static std::string display_unit() { return "OPS/s"; }
    static value_type  record()
    {
        return value_type(papi_type::record(), real_clock::record());
    }

    double compute_display() const
    {
        auto& obj = (accum.second > 0) ? accum : value;
        if(obj.second == 0)
            return 0.0;
        return std::accumulate(obj.begin(), obj.end(), 0) /
               static_cast<double>(obj.second);
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
        for(size_type i = 0; i < num_events; ++i)
            accum.first[i] += (tmp.first[i] - value.first[i]);
        accum.second += (tmp.second - value.second);
        value = std::move(tmp);
        set_stopped();
    }

    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
        {
            accum.first[i] += rhs.accum.first[i];
            value.first[i] += rhs.value.first[i];
        }
        accum.second += rhs.accum.second;
        value.second += rhs.value.second;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < num_events; ++i)
        {
            accum.first[i] -= rhs.accum.first[i];
            value.first[i] -= rhs.value.first[i];
        }
        accum.second -= rhs.accum.second;
        value.second -= rhs.value.second;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }
};

//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_PAPI)
template <int... EventTypes>
struct impl_available<cpu_roofline<EventTypes...>> : std::false_type
{
};
#endif

//--------------------------------------------------------------------------------------//
// Shorthand aliases for common roofline types
//
using cpu_roofline_sflops = cpu_roofline<PAPI_SP_OPS>;
using cpu_roofline_dflops = cpu_roofline<PAPI_DP_OPS>;
using cpu_roofline_flops  = cpu_roofline<PAPI_FP_OPS>;
// TODO: check if L1 roofline wants L1 total cache hits (below) or L1 composite of
// accesses/reads/writes/etc.
using cpu_roofline_l1 = cpu_roofline<PAPI_L1_TCH>;
// TODO: check if L2 roofline wants L2 total cache hits (below) or L2 composite of
// accesses/reads/writes/etc.
using cpu_roofline_l2 = cpu_roofline<PAPI_L2_TCH>;

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_CUDA)
//--------------------------------------------------------------------------------------//
//
//
// this struct extracts only the CPU time spent in kernel-mode
struct cuda_event : public base<cuda_event, float>
{
    using ratio_t    = std::micro;
    using value_type = float;
    using base_type  = base<cuda_event, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "cuda_event"; }
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
        destroy();
    }

    float compute_display() const
    {
        const_cast<cuda_event&>(*this).sync();
        auto val = (is_transient) ? accum : value;
        return static_cast<float>(val / static_cast<float>(ratio_t::den) *
                                  base_type::get_unit());
    }

    void start()
    {
        set_started();
        m_is_synced = false;
        // cuda_event* _this = static_cast<cuda_event*>(this);
        // cudaStreamAddCallback(m_stream, &cuda_event::callback, _this, 0);
        cudaEventRecord(m_start, m_stream);
    }

    void stop()
    {
        cudaEventRecord(m_stop, m_stream);
        sync();
        set_stopped();
    }

    void set_stream(cudaStream_t _stream = 0) { m_stream = _stream; }

    void sync()
    {
        if(!m_is_synced)
        {
            cudaEventSynchronize(m_stop);
            float tmp = 0.0f;
            cudaEventElapsedTime(&tmp, m_start, m_stop);
            accum += tmp;
            value       = std::move(tmp);
            m_is_synced = true;
        }
    }

    void destroy()
    {
        if(is_valid())
        {
            cudaEventDestroy(m_start);
            cudaEventDestroy(m_stop);
        }
    }

    bool is_valid() const
    {
        auto ret = cudaEventQuery(m_stop);
        return (ret == cudaSuccess && ret == cudaErrorNotReady);
    }

protected:
    static void callback(cudaStream_t /*_stream*/, cudaError_t /*_status*/,
                         void* user_data)
    {
        cuda_event* _this = static_cast<cuda_event*>(user_data);
        if(!_this->m_is_synced && _this->is_valid())
        {
            cudaEventSynchronize(_this->m_stop);
            float tmp = 0.0f;
            cudaEventElapsedTime(&tmp, _this->m_start, _this->m_stop);
            _this->accum += tmp;
            _this->value       = std::move(tmp);
            _this->m_is_synced = true;
        }
    }

private:
    bool         m_is_synced = false;
    cudaStream_t m_stream    = 0;
    cudaEvent_t  m_start;
    cudaEvent_t  m_stop;
};

#else
//--------------------------------------------------------------------------------------//
// dummy for cuda_event when CUDA is not available
//
using cudaStream_t = int;
using cudaError_t = int;
// this struct extracts only the CPU time spent in kernel-mode
struct cuda_event : public base<cuda_event, float>
{
    using ratio_t = std::micro;
    using value_type = float;
    using base_type = base<cuda_event, value_type>;

    static const short precision = 3;
    static const short width = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t unit() { return units::sec; }
    static std::string label() { return "cuda_event"; }
    static std::string descript() { return "event time"; }
    static std::string display_unit() { return "sec"; }
    static value_type record() { return 0.0f; }

    cuda_event(cudaStream_t _stream = 0)
    : m_stream(_stream)
    {
    }

    ~cuda_event() {}

    float compute_display() const { return 0.0f; }

    void start() {}

    void stop() {}

    void set_stream(cudaStream_t _stream = 0) { m_stream = _stream; }

    static void callback(cudaStream_t, cudaError_t, void*) {}

    void sync() {}

private:
    cudaStream_t m_stream = 0;
};

#endif

#if !defined(TIMEMORY_USE_CUDA)
//--------------------------------------------------------------------------------------//
//  disable cuda_event if not enabled via preprocessor
//
template <>
struct impl_available<cuda_event> : std::false_type
{
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
using rusage_components_t = component_tuple<
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
    component_tuple<component::real_clock, component::cpu_clock, component::cpu_util,
                    component::thread_cpu_clock, component::thread_cpu_util,
                    component::process_cpu_clock, component::process_cpu_util>;

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

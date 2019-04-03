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
#include "timemory/rusage.hpp"
#include "timemory/serializer.hpp"
#include "timemory/units.hpp"

#if defined(_UNIX)
#    include <sys/resource.h>
#    include <unistd.h>
#    if defined(_MACOS)
#        include <mach/mach.h>
#    endif
#elif defined(_WINDOWS)
#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
#    include <psapi.h>
#    include <stdio.h>
#    include <windows.h>
#else
#    error "Cannot define get_peak_rss() or get_current_rss() for an unknown OS."
#endif

//============================================================================//

namespace tim
{
//----------------------------------------------------------------------------//

namespace component
{
//----------------------------------------------------------------------------//

enum component_type
{
    REALTIME,
    SYSTEM,
    USER,
    MONOTONIC,
    MONOTONIC_RAW,
    THREAD_CPUTIME,
    PROCESS_CPUTIME,
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

//----------------------------------------------------------------------------//

template <typename _Tp, typename value_type = intmax_t>
struct base
{
    using Type       = _Tp;
    value_type value = value_type(0);

    base(value_type _value = value_type(0))
    : value(_value)
    {
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar(serializer::make_nvp(Type::label(), value),
           serializer::make_nvp("units", Type::unit));
    }

    value_type operator()() { return (value = Type::record()); }
    value_type start() { return (*this)(); }
    value_type stop() { return (*this)(); }

    _Tp& max(const base<_Tp>& rhs) { return (value = std::max(value, rhs.value)); }
    _Tp  max(const base<_Tp>& rhs) const { return std::max(value, rhs.value); }
    _Tp& min(const base<_Tp>& rhs) { return (value = std::min(value, rhs.value)); }
    _Tp  min(const base<_Tp>& rhs) const { return std::min(value, rhs.value); }

    _Tp& operator+=(const base<_Tp>& rhs)
    {
        value += rhs.value;
        return static_cast<_Tp&>(*this);
    }

    _Tp& operator-=(const base<_Tp>& rhs)
    {
        value -= rhs.value;
        return static_cast<_Tp&>(*this);
    }

    friend _Tp operator+(const base<_Tp>& lhs, const base<_Tp>& rhs)
    {
        return base<_Tp>(lhs) += rhs;
    }

    friend _Tp operator-(const base<_Tp>& lhs, const base<_Tp>& rhs)
    {
        return base<_Tp>(lhs) -= rhs;
    }

    _Tp& operator+=(const value_type& rhs)
    {
        value += rhs;
        return static_cast<_Tp&>(*this);
    }

    _Tp& operator-=(const value_type& rhs)
    {
        value -= rhs;
        return static_cast<_Tp&>(*this);
    }

    _Tp& operator*=(const value_type& rhs)
    {
        value *= rhs;
        return static_cast<_Tp&>(*this);
    }

    _Tp& operator/=(const value_type& rhs)
    {
        value /= rhs;
        return static_cast<_Tp&>(*this);
    }

    friend std::ostream& operator<<(std::ostream& os, const base<_Tp>& obj)
    {
        auto tmp   = Type::compute_display(obj);
        auto label = Type::label();
        auto disp  = Type::display_unit();

        std::stringstream ss, ssv, ssi;
        ssv << std::setprecision(3) << std::setw(8) << std::fixed << tmp;
        ssi << " " << std::setw(8) << std::left << label;
        ss << ssv.str() << ssi.str() << " [" << disp << "]";
        os << ss.str();

        return os;
    }
};

//----------------------------------------------------------------------------//

struct realtime_clock : public base<realtime_clock>
{
    using ratio_t   = std::micro;
    using this_type = base<realtime_clock>;

    static const component_type category = REALTIME;
    static const intmax_t       unit     = units::usec;
    static std::string          label() { return "real"; }
    static std::string          descript() { return "wall time"; }
    static std::string          display_unit() { return "sec"; }
    static double record() { return tim::get_clock_realtime_now<intmax_t, ratio_t>(); }
    static double compute_display(const this_type& obj)
    {
        return static_cast<double>(obj.value / static_cast<double>(ratio_t::den));
    }
};

//----------------------------------------------------------------------------//

struct system_clock : public base<system_clock>
{
    using ratio_t   = std::micro;
    using this_type = base<system_clock>;

    static const component_type category = SYSTEM;
    static const intmax_t       unit     = units::usec;
    static std::string          label() { return "sys"; }
    static std::string          descript() { return "system time"; }
    static std::string          display_unit() { return "sec"; }
    static double record() { return tim::get_clock_system_now<intmax_t, ratio_t>(); }
    static double compute_display(const this_type& obj)
    {
        return static_cast<double>(obj.value / static_cast<double>(ratio_t::den));
    }
};

//----------------------------------------------------------------------------//

struct user_clock : public base<user_clock>
{
    using ratio_t   = std::micro;
    using this_type = base<user_clock>;

    static const component_type category = USER;
    static const intmax_t       unit     = units::usec;
    static std::string          label() { return "user"; }
    static std::string          descript() { return "user time"; }
    static std::string          display_unit() { return "sec"; }
    static double record() { return tim::get_clock_monotonic_now<intmax_t, ratio_t>(); }
    static double compute_display(const this_type& obj)
    {
        return static_cast<double>(obj.value / static_cast<double>(ratio_t::den));
    }
};

//----------------------------------------------------------------------------//

struct monotonic_clock : public base<monotonic_clock>
{
    using ratio_t   = std::micro;
    using this_type = base<monotonic_clock>;

    static const component_type category = MONOTONIC;
    static const intmax_t       unit     = units::usec;
    static std::string          label() { return "mono"; }
    static std::string          descript() { return "monotonic time"; }
    static std::string          display_unit() { return "sec"; }
    static double record() { return tim::get_clock_monotonic_now<intmax_t, ratio_t>(); }
    static double compute_display(const this_type& obj)
    {
        return static_cast<double>(obj.value / static_cast<double>(ratio_t::den));
    }
};

//----------------------------------------------------------------------------//

struct monotonic_raw_clock : public base<monotonic_raw_clock>
{
    using ratio_t   = std::micro;
    using this_type = base<monotonic_raw_clock>;

    static const component_type category = MONOTONIC_RAW;
    static const intmax_t       unit     = units::usec;
    static std::string          label() { return "raw_mono"; }
    static std::string          descript() { return "monotonic raw time"; }
    static std::string          display_unit() { return "sec"; }
    static double               record()
    {
        return tim::get_clock_monotonic_raw_now<intmax_t, ratio_t>();
    }
    static double compute_display(const this_type& obj)
    {
        return static_cast<double>(obj.value / static_cast<double>(ratio_t::den));
    }
};

//----------------------------------------------------------------------------//

struct thread_cpu_clock : public base<thread_cpu_clock>
{
    using ratio_t   = std::micro;
    using this_type = base<thread_cpu_clock>;

    static const component_type category = THREAD_CPUTIME;
    static const intmax_t       unit     = units::usec;
    static std::string          label() { return "thr_cpu"; }
    static std::string          descript() { return "thread cpu time"; }
    static std::string          display_unit() { return "sec"; }
    static double record() { return tim::get_clock_thread_now<intmax_t, ratio_t>(); }
    static double compute_display(const this_type& obj)
    {
        return static_cast<double>(obj.value / static_cast<double>(ratio_t::den));
    }
};

//----------------------------------------------------------------------------//

struct process_cpu_clock : public base<process_cpu_clock>
{
    using ratio_t   = std::micro;
    using this_type = base<process_cpu_clock>;

    static const component_type category = PROCESS_CPUTIME;
    static const intmax_t       unit     = units::usec;
    static std::string          label() { return "proc_cpu"; }
    static std::string          descript() { return "process cpu time"; }
    static std::string          display_unit() { return "sec"; }
    static double record() { return tim::get_clock_process_now<intmax_t, ratio_t>(); }
    static double compute_display(const this_type& obj)
    {
        return static_cast<double>(obj.value / static_cast<double>(ratio_t::den));
    }
};

//----------------------------------------------------------------------------//

typedef std::tuple<realtime_clock, system_clock, user_clock, monotonic_clock,
                   monotonic_raw_clock, thread_cpu_clock, process_cpu_clock>
    timing_types_t;

//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//

struct peak_rss : public base<peak_rss>
{
    using this_type = base<peak_rss>;

    static const component_type category = PEAK_RSS;
    static const intmax_t       units    = units::kilobyte;
    static std::string          label() { return "rss_peak"; }
    static std::string          descript() { return "max resident set size"; }
    static std::string          display_unit() { return "sec"; }
    static intmax_t             record() { return get_peak_rss(); }
    static intmax_t compute_display(const this_type& obj) { return obj.value; }
};

//----------------------------------------------------------------------------//

struct current_rss : public base<current_rss>
{
    using this_type = base<current_rss>;

    static const component_type category = CURRENT_RSS;
    static const intmax_t       units    = units::kilobyte;
    static std::string          label() { return "rss_curr"; }
    static std::string          descript() { return "current resident set size"; }
    static intmax_t             record() { return get_current_rss(); }
    static intmax_t compute_display(const this_type& obj) { return obj.value; }
};

//----------------------------------------------------------------------------//

struct stack_rss : public base<stack_rss>
{
    using this_type = base<stack_rss>;

    static const component_type category = STACK_RSS;
    static const intmax_t       units    = units::kilobyte;
    static std::string          label() { return "rss_stack"; }
    static std::string          descript() { return "integral unshared stack size"; }
    static intmax_t             record() { return get_stack_rss(); }
    static intmax_t compute_display(const this_type& obj) { return obj.value; }
};

//----------------------------------------------------------------------------//

struct data_rss : public base<data_rss>
{
    using this_type = base<data_rss>;

    static const component_type category = DATA_RSS;
    static const intmax_t       units    = units::kilobyte;
    static std::string          label() { return "rss_data"; }
    static std::string          descript() { return "integral unshared data size"; }
    static intmax_t             record() { return get_data_rss(); }
    static intmax_t compute_display(const this_type& obj) { return obj.value; }
};

//----------------------------------------------------------------------------//

struct num_swap : public base<num_swap>
{
    using this_type = base<num_swap>;

    static const component_type category = NUM_SWAP;
    static const intmax_t       units    = 1;
    static std::string          label() { return "num_swap"; }
    static std::string          descript() { return "swaps out of main memory"; }
    static intmax_t             record() { return get_num_swap(); }
    static intmax_t compute_display(const this_type& obj) { return obj.value; }
};

//----------------------------------------------------------------------------//

struct num_io_in : public base<num_io_in>
{
    using this_type = base<num_io_in>;

    static const component_type category = NUM_IO_IN;
    static const intmax_t       units    = 1;
    static std::string          label() { return "io_in"; }
    static std::string          descript() { return "block input operations"; }
    static intmax_t             record() { return get_num_io_in(); }
    static intmax_t compute_display(const this_type& obj) { return obj.value; }
};

//----------------------------------------------------------------------------//

struct num_io_out : public base<num_io_out>
{
    using this_type = base<num_io_out>;

    static const component_type category = NUM_IO_OUT;
    static const intmax_t       units    = 1;
    static std::string          label() { return "io_out"; }
    static std::string          descript() { return "block output operations"; }
    static intmax_t             record() { return get_num_io_out(); }
    static intmax_t compute_display(const this_type& obj) { return obj.value; }
};

//----------------------------------------------------------------------------//

struct num_minor_page_faults : public base<num_minor_page_faults>
{
    using this_type = base<num_minor_page_faults>;

    static const component_type category = NUM_MINOR_PAGE_FAULTS;
    static const intmax_t       units    = 1;
    static std::string          label() { return "minor_page_faults"; }
    static std::string          descript() { return "page reclaims"; }
    static intmax_t             record() { return get_num_minor_page_faults(); }
    static intmax_t compute_display(const this_type& obj) { return obj.value; }
};

//----------------------------------------------------------------------------//

struct num_major_page_faults : public base<num_major_page_faults>
{
    using this_type = base<num_major_page_faults>;

    static const component_type category = NUM_MAJOR_PAGE_FAULTS;
    static const intmax_t       units    = 1;
    static std::string          label() { return "major_page_faults"; }
    static std::string          descript() { return "page faults"; }
    static intmax_t             record() { return get_num_major_page_faults(); }
    static intmax_t compute_display(const this_type& obj) { return obj.value; }
};

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

template <typename _Tp>
struct max : public base<_Tp>
{
    max(base<_Tp>& obj) { obj.value = std::max(obj.value, _Tp::record()); }
};

//----------------------------------------------------------------------------//

template <typename _Tp>
struct record : public base<_Tp>
{
    record(base<_Tp>& obj) { obj(); }
    record(base<_Tp>& obj, const base<_Tp>& rhs) { obj += rhs.value; }
};

//----------------------------------------------------------------------------//

template <typename _Tp>
struct reset : public base<_Tp>
{
    reset(base<_Tp>& obj) { obj.value = 0; }
};

//----------------------------------------------------------------------------//

template <typename _Tp>
struct print : public base<_Tp>
{
    print(base<_Tp>& obj, std::ostream& os) { os << obj << std::endl; }
};

//----------------------------------------------------------------------------//

template <typename _Tp>
struct minus : public base<_Tp>
{
    minus(base<_Tp>& obj, const base<_Tp>& rhs) { obj.value -= rhs.value; }
    minus(base<_Tp>& obj, const intmax_t& rhs) { obj.value -= rhs; }
};

//----------------------------------------------------------------------------//

template <typename _Tp>
struct plus : public base<_Tp>
{
    plus(base<_Tp>& obj, const base<_Tp>& rhs) { obj.value += rhs.value; }
    plus(base<_Tp>& obj, const intmax_t& rhs) { obj.value += rhs; }
};

//----------------------------------------------------------------------------//

template <typename _Tp>
struct multiply : public base<_Tp>
{
    multiply(base<_Tp>& obj, const base<_Tp>& rhs) { obj.value *= rhs.value; }
    multiply(base<_Tp>& obj, const intmax_t& rhs) { obj.value *= rhs; }
};

//----------------------------------------------------------------------------//

template <typename _Tp>
struct divide : public base<_Tp>
{
    divide(base<_Tp>& obj, const base<_Tp>& rhs) { obj.value /= rhs.value; }
    divide(base<_Tp>& obj, const intmax_t& rhs) { obj.value /= rhs; }
};

//----------------------------------------------------------------------------//

template <typename _Tp, typename Archive>
struct serial : public base<_Tp>
{
    serial(base<_Tp>& obj, Archive& ar, const unsigned int)
    {
        ar(serializer::make_nvp(_Tp::label(), obj.value));
    }
};

//----------------------------------------------------------------------------//

}  // namespace component

//----------------------------------------------------------------------------//

}  // namespace tim

//----------------------------------------------------------------------------//

#include "timemory/component.icpp"

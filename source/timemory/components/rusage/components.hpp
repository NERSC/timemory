//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

/**
 * \file timemory/components/rusage/components.hpp
 * \brief Implementation of the rusage component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/data/statistics.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include "timemory/components/rusage/backends.hpp"
#include "timemory/components/rusage/types.hpp"
#include "timemory/components/timing/backends.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//          Resource Usage types
//
//--------------------------------------------------------------------------------------//
/// \class peak_rss
/// \brief
/// this struct extracts the high-water mark (or a change in the high-water mark) of
/// the resident set size (RSS). Which is current amount of memory in RAM.
/// When used on a system with swap enabled, this value may fluctuate but should not
/// on an HPC system.
//
struct peak_rss : public base<peak_rss>
{
    static std::string label() { return "peak_rss"; }
    static std::string description() { return "max resident set size"; }
    static value_type  record() { return get_peak_rss(); }
    double             get() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    double get_display() const { return get(); }
    void   start()
    {
        set_started();
        value = record();
    }
    void stop()
    {
        auto tmp   = record();
        auto delta = tmp - value;
        accum      = std::max(static_cast<const value_type&>(accum), delta);
        value      = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
/// \class page_rss
/// \brief
/// this struct measures the resident set size (RSS) currently allocated in pages of
/// memory. Unlike the peak_rss, this value will fluctuate as memory gets freed and
/// allocated
//
struct page_rss : public base<page_rss, int64_t>
{
    using value_type  = int64_t;
    using result_type = double;
    using this_type   = page_rss;
    using base_type   = base<this_type, value_type>;

    static std::string label() { return "page_rss"; }
    static std::string description() { return "resident set size of memory pages"; }
    static value_type  record() { return get_page_rss(); }
    double             get() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    double get_display() const { return get(); }
    void   start()
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

// using current_peak_rss = page_rss;

//--------------------------------------------------------------------------------------//
/// \class stack_rss
/// \brief
/// an integral value of the amount of unshared memory residing in the stack segment
/// of a process
/// Unmaintained by Unix kernel. Globally disabled
/// unless TIMEMORY_USE_UNMAINTAINED_RUSAGE is defined
//
struct stack_rss : public base<stack_rss>
{
    using value_type = int64_t;
    using base_type  = base<stack_rss, value_type>;

    static int64_t     units() { return units::kilobyte; }
    static std::string label() { return "stack_rss"; }
    static std::string description() { return "integral unshared stack size"; }
    static value_type  record() { return get_stack_rss(); }
    double             get() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    double get_display() const { return get(); }
    void   start()
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
/// \class data_rss
/// \brief
/// an integral value of the amount of unshared memory residing in the data segment of
/// a process
/// Unmaintained by Unix kernel. Globally disabled
/// unless TIMEMORY_USE_UNMAINTAINED_RUSAGE is defined
//
struct data_rss : public base<data_rss>
{
    using value_type = int64_t;
    using base_type  = base<data_rss, value_type>;

    static int64_t     units() { return units::kilobyte; }
    static std::string label() { return "data_rss"; }
    static std::string description() { return "integral unshared data size"; }
    static value_type  record() { return get_data_rss(); }
    double             get() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    double get_display() const { return get(); }
    void   start()
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
/// \class num_swap
/// \brief
/// the number of times a process was swapped out of main memory.
/// Unmaintained by Unix kernel. Globally disabled
/// unless TIMEMORY_USE_UNMAINTAINED_RUSAGE is defined
//
struct num_swap : public base<num_swap>
{
    using value_type = int64_t;
    using base_type  = base<num_swap>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "num_swap"; }
    static std::string description() { return "swaps out of main memory"; }
    static value_type  record() { return get_num_swap(); }
    value_type         get() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    value_type get_display() const { return get(); }
    void       start()
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
/// \class num_io_in
/// \brief
/// the number of times the file system had to perform input.
//
struct num_io_in : public base<num_io_in>
{
    using value_type = int64_t;
    using base_type  = base<num_io_in>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "io_in"; }
    static std::string description() { return "block input operations"; }
    static value_type  record() { return get_num_io_in(); }
    value_type         get() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    value_type get_display() const { return get(); }
    void       start()
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
/// \class num_io_out
/// \brief
/// the number of times the file system had to perform output.
//
struct num_io_out : public base<num_io_out>
{
    using value_type = int64_t;
    using base_type  = base<num_io_out>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "io_out"; }
    static std::string description() { return "block output operations"; }
    static value_type  record() { return get_num_io_out(); }
    value_type         get() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    value_type get_display() const { return get(); }
    void       start()
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
/// \class num_minor_page_faults
/// \brief
/// the number of page faults serviced without any I/O activity; here I/O activity is
/// avoided by reclaiming a page frame from the list of pages awaiting reallocation.
//
struct num_minor_page_faults : public base<num_minor_page_faults>
{
    using value_type = int64_t;
    using base_type  = base<num_minor_page_faults>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "minor_page_flts"; }
    static std::string description() { return "page reclaims"; }
    static value_type  record() { return get_num_minor_page_faults(); }
    value_type         get() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    value_type get_display() const { return get(); }
    void       start()
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
/// \class num_major_page_faults
/// \brief
/// the number of page faults serviced that required I/O activity.
//
struct num_major_page_faults : public base<num_major_page_faults>
{
    using value_type = int64_t;
    using base_type  = base<num_major_page_faults>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "major_page_flts"; }
    static std::string description() { return "page faults"; }
    static value_type  record() { return get_num_major_page_faults(); }
    value_type         get() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    value_type get_display() const { return get(); }
    void       start()
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
/// \class num_msg_sent
/// \brief
/// the number of IPC messages sent.
/// Unmaintained by Unix kernel. Globally disabled
/// unless TIMEMORY_USE_UNMAINTAINED_RUSAGE is defined
//
struct num_msg_sent : public base<num_msg_sent>
{
    using value_type = int64_t;
    using base_type  = base<num_msg_sent>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "num_msg_sent"; }
    static std::string description() { return "messages sent"; }
    static value_type  record() { return get_num_messages_sent(); }
    value_type         get() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    value_type get_display() const { return get(); }
    void       start()
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
/// \class num_msg_recv
/// \brief
/// the number of IPC messages received.
/// Unmaintained by Unix kernel. Globally disabled
/// unless TIMEMORY_USE_UNMAINTAINED_RUSAGE is defined
//
struct num_msg_recv : public base<num_msg_recv>
{
    using value_type = int64_t;
    using base_type  = base<num_msg_recv>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "num_msg_recv"; }
    static std::string description() { return "messages received"; }
    static value_type  record() { return get_num_messages_received(); }
    value_type         get() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    value_type get_display() const { return get(); }
    void       start()
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
/// \class num_signals
/// \brief the number of signals delivered.
/// Unmaintained by Unix kernel. Globally disabled
/// unless TIMEMORY_USE_UNMAINTAINED_RUSAGE is defined
//
struct num_signals : public base<num_signals>
{
    using value_type = int64_t;
    using base_type  = base<num_signals>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "num_signals"; }
    static std::string description() { return "signals delievered"; }
    static value_type  record() { return get_num_signals(); }
    value_type         get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    value_type get() const { return get_display(); }
    void       start()
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
/// \class voluntary_context_switch
/// \brief
/// the number of times a context switch resulted due to a process voluntarily giving up
/// the processor before its time slice was completed (usually to await availability of a
/// resource).
//
struct voluntary_context_switch : public base<voluntary_context_switch>
{
    using value_type = int64_t;
    using base_type  = base<voluntary_context_switch>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "vol_cxt_swch"; }
    static std::string description() { return "voluntary context switches"; }
    static value_type  record() { return get_num_voluntary_context_switch(); }
    value_type         get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    value_type get() const { return get_display(); }
    void       start()
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

using vol_cxt_switch = voluntary_context_switch;

//--------------------------------------------------------------------------------------//
/// \class priority_context_switch
/// \brief
/// the number of times a context switch resulted due to a process voluntarily giving up
/// the processor before its time slice was completed (usually to await availability of a
/// resource).
//
struct priority_context_switch : public base<priority_context_switch>
{
    using value_type = int64_t;
    using base_type  = base<priority_context_switch>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "prio_cxt_swch"; }
    static std::string description() { return "priority context switches"; }
    static value_type  record() { return get_num_priority_context_switch(); }
    value_type         get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
    }
    value_type get() const { return get_display(); }
    void       start()
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

using prio_cxt_switch = priority_context_switch;

//--------------------------------------------------------------------------------------//
/// \class read_bytes
/// \brief I/O counter: bytes read Attempt to count the number of bytes which this process
/// really did cause to be fetched from the storage layer. Done at the submit_bio() level,
/// so it is accurate for block-backed filesystems.
//
struct read_bytes : public base<read_bytes, std::pair<int64_t, int64_t>>
{
    using this_type   = read_bytes;
    using value_type  = std::pair<int64_t, int64_t>;
    using base_type   = base<this_type, value_type>;
    using result_type = std::pair<double, double>;

    static std::string label() { return "read_bytes"; }
    static std::string description() { return "physical I/O reads"; }

    static std::pair<double, double> unit()
    {
        return std::pair<double, double>{
            units::kilobyte, static_cast<double>(units::kilobyte) / units::sec
        };
    }

    static std::vector<std::string> display_unit_array()
    {
        return std::vector<std::string>{ std::get<0>(get_display_unit()),
                                         std::get<1>(get_display_unit()) };
    }

    static std::vector<std::string> label_array()
    {
        return std::vector<std::string>{ label(), "read_rate" };
    }

    static display_unit_type display_unit()
    {
        return display_unit_type{ "KB", "KB/sec" };
    }

    static std::pair<double, double> unit_array() { return unit(); }

    static std::vector<std::string> description_array()
    {
        return std::vector<std::string>{ "Number of bytes read", "Rate of bytes read" };
    }

    static value_type record()
    {
        return value_type(get_bytes_read(),
                          tim::get_clock_real_now<int64_t, std::nano>());
    }

    static auto get_timing_unit()
    {
        static auto _value = units::sec;
        if(settings::timing_units().length() > 0)
            _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
        return _value;
    }

    std::string get_display() const
    {
        std::stringstream ss, ssv, ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = get_display_unit();

        auto _val = get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!std::get<1>(_disp).empty())
            ssr << " " << std::get<1>(_disp);

        ss << ssv.str() << ", " << ssr.str();
        ss << " read";
        return ss.str();
    }

    result_type get() const
    {
        auto val = (is_transient) ? accum : value;

        double data  = std::get<0>(val);
        double delta = std::get<1>(val);

        if(!is_transient)
            delta = tim::get_clock_real_now<int64_t, std::nano>() - delta;

        delta /= static_cast<double>(std::nano::den);
        delta *= get_timing_unit();

        double rate = 0.0;
        if(delta != 0.0)
            rate = data / delta;

        if(laps > 0)
            rate *= laps;

        data /= std::get<0>(get_unit());
        rate /= std::get<0>(get_unit());

        if(!std::isfinite(rate))
            rate = 0.0;

        return result_type(data, rate);
    }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        auto tmp          = record();
        auto diff         = (tmp - value);
        std::get<0>(diff) = std::abs(std::get<0>(diff));
        accum += diff;
        value = std::move(tmp);
        set_stopped();
    }

    static unit_type get_unit()
    {
        static auto  _instance = this_type::unit();
        static auto& _mem      = std::get<0>(_instance);
        static auto& _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<1>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _timing_val =
                std::get<1>(units::get_timing_unit(settings::timing_units()));
            _rate = _mem / (_timing_val);
        }

        static const auto factor = static_cast<double>(std::nano::den);
        unit_type         _tmp   = _instance;
        std::get<1>(_tmp) *= factor;

        return _tmp;
    }

    static display_unit_type get_display_unit()
    {
        static display_unit_type _instance = this_type::display_unit();
        static auto&             _mem      = std::get<0>(_instance);
        static auto&             _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<0>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _tval = std::get<0>(units::get_timing_unit(settings::timing_units()));
            _rate      = apply<std::string>::join("/", _mem, _tval);
        }
        else if(settings::memory_units().length() > 0)
        {
            _rate = apply<std::string>::join("/", _mem, "sec");
        }

        return _instance;
    }

    //----------------------------------------------------------------------------------//
    // record a measurment (for file sampling)
    //
    void measure() { std::get<0>(value) = get_bytes_read(); }
};

//--------------------------------------------------------------------------------------//
/// \class written_bytes
/// \brief I/O counter: Attempt to count the number of bytes which this process caused to
/// be sent to the storage layer. This is done at page-dirtying time.
//
struct written_bytes : public base<written_bytes, std::array<int64_t, 2>>
{
    using this_type   = written_bytes;
    using value_type  = std::array<int64_t, 2>;
    using base_type   = base<this_type, value_type>;
    using result_type = std::array<double, 2>;

    static std::string label() { return "written_bytes"; }
    static std::string description() { return "physical I/O writes"; }

    static result_type unit()
    {
        return result_type{ { units::kilobyte,
                              static_cast<double>(units::kilobyte) / units::sec } };
    }

    static std::vector<std::string> display_unit_array()
    {
        return std::vector<std::string>{ std::get<0>(get_display_unit()),
                                         std::get<1>(get_display_unit()) };
    }

    static std::vector<std::string> label_array()
    {
        return std::vector<std::string>{ label(), "written_rate" };
    }

    static display_unit_type display_unit()
    {
        return display_unit_type{ { "KB", "KB/sec" } };
    }

    static std::array<double, 2> unit_array() { return unit(); }

    static std::vector<std::string> description_array()
    {
        return std::vector<std::string>{ "Number of bytes written",
                                         "Rate of bytes written" };
    }

    static value_type record()
    {
        return value_type{ { get_bytes_written(),
                             tim::get_clock_real_now<int64_t, std::nano>() } };
    }

    static auto get_timing_unit()
    {
        static auto _value = units::sec;
        if(settings::timing_units().length() > 0)
            _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
        return _value;
    }

    std::string get_display() const
    {
        std::stringstream ss, ssv, ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = get_display_unit();

        auto _val = get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!std::get<1>(_disp).empty())
            ssr << " " << std::get<1>(_disp);

        ss << ssv.str() << ", " << ssr.str();
        ss << " written";
        return ss.str();
    }

    result_type get() const
    {
        auto val = (is_transient) ? accum : value;

        double data  = std::get<0>(val);
        double delta = std::get<1>(val);

        if(!is_transient)
            delta = tim::get_clock_real_now<int64_t, std::nano>() - delta;

        delta /= static_cast<double>(std::nano::den);
        delta *= get_timing_unit();

        double rate = 0.0;
        if(delta != 0.0)
            rate = data / delta;

        if(laps > 0)
            rate *= laps;

        data /= std::get<0>(get_unit());
        rate /= std::get<0>(get_unit());

        if(!std::isfinite(rate))
            rate = 0.0;

        return result_type{ { data, rate } };
    }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        auto tmp  = record();
        auto diff = tmp;
        diff[0] -= value[0];
        diff[1] -= value[1];
        diff[0] = std::abs(diff[0]);
        accum[0] += diff[0];
        accum[1] += diff[1];
        value = tmp;
        set_stopped();
    }

    static unit_type get_unit()
    {
        static auto  _instance = this_type::unit();
        static auto& _mem      = std::get<0>(_instance);
        static auto& _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<1>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _timing_val =
                std::get<1>(units::get_timing_unit(settings::timing_units()));
            _rate = _mem / (_timing_val);
        }

        static const auto factor = static_cast<double>(std::nano::den);
        unit_type         _tmp   = _instance;
        std::get<1>(_tmp) *= factor;

        return _tmp;
    }

    static display_unit_type get_display_unit()
    {
        static display_unit_type _instance = this_type::display_unit();
        static auto&             _mem      = std::get<0>(_instance);
        static auto&             _rate     = std::get<1>(_instance);

        if(settings::memory_units().length() > 0)
            _mem = std::get<0>(units::get_memory_unit(settings::memory_units()));

        if(settings::timing_units().length() > 0)
        {
            auto _tval = std::get<0>(units::get_timing_unit(settings::timing_units()));
            _rate      = apply<std::string>::join("/", _mem, _tval);
        }
        else if(settings::memory_units().length() > 0)
        {
            _rate = apply<std::string>::join("/", _mem, "sec");
        }

        return _instance;
    }

    //----------------------------------------------------------------------------------//
    // record a measurment (for file sampling)
    //
    void measure() { std::get<0>(value) = get_bytes_written(); }
};

//--------------------------------------------------------------------------------------//
/// \class virtual_memory
/// \brief
/// this struct extracts the virtual memory usage
//
struct virtual_memory : public base<virtual_memory>
{
    using value_type = int64_t;
    using base_type  = base<virtual_memory, value_type>;

    static std::string label() { return "virtual_memory"; }
    static std::string description() { return "virtual memory usage"; }
    static value_type  record() { return get_virt_mem(); }
    double             get() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    double get_display() const { return get(); }
    void   start()
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
/// \class user_mode_time
/// \brief This is the total amount of time spent executing in user mode
//
struct user_mode_time : public base<user_mode_time, int64_t>
{
    using ratio_t    = std::micro;
    using value_type = int64_t;
    using this_type  = user_mode_time;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "user_mode"; }
    static std::string description() { return "Time spent executing in user mode"; }
    static value_type  record() { return get_user_mode_time(); }

    double get_display() const { return get(); }
    double get() const
    {
        auto val = (is_transient) ? accum : value;
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
        if(tmp > value)
        {
            accum += (tmp - value);
            value = std::move(tmp);
        }
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
/// \class kernel_mode_time
/// \brief This is the total amount of time spent executing in kernel mode
//
struct kernel_mode_time : public base<kernel_mode_time, int64_t>
{
    using ratio_t    = std::micro;
    using value_type = int64_t;
    using this_type  = kernel_mode_time;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "kernel_mode"; }
    static std::string description() { return "Time spent executing in kernel mode"; }
    static value_type  record() { return get_kernel_mode_time(); }

    double get_display() const { return get(); }
    double get() const
    {
        auto val = (is_transient) ? accum : value;
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
        if(tmp > value)
        {
            accum += (tmp - value);
            value = std::move(tmp);
        }
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
/// \class current_peak_rss
/// \brief
/// this struct extracts the high-water mark of the resident set size (RSS) at start
/// and stop. RSS is current amount of memory in RAM.
//
struct current_peak_rss : public base<current_peak_rss, std::pair<int64_t, int64_t>>
{
    using unit_type         = std::pair<int64_t, int64_t>;
    using display_unit_type = std::pair<std::string, std::string>;
    using result_type       = std::pair<double, double>;
    using this_type         = current_peak_rss;

    static std::string label() { return "current_peak_rss"; }
    static std::string description() { return "current resident set size"; }
    static value_type  record() { return value_type{ get_peak_rss(), 0 }; }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        value = value_type{ value.first, record().first };
        accum = std::max(accum, value);
        set_stopped();
    }

    std::string get_display() const
    {
        std::stringstream ss, ssv, ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = get_display_unit();

        auto _val = get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!std::get<1>(_disp).empty())
            ssr << " " << std::get<1>(_disp);

        ss << ssv.str() << ", " << ssr.str();
        return ss.str();
    }

    result_type get() const
    {
        result_type data = (is_transient) ? accum : value;
        data.first /= get_unit().first;
        data.second /= get_unit().second;
        return data;
    }

    static std::pair<double, double> unit()
    {
        return std::pair<double, double>{ units::megabyte, units::megabyte };
    }

    static std::vector<std::string> display_unit_array()
    {
        return std::vector<std::string>{ get_display_unit().first,
                                         get_display_unit().second };
    }

    static std::vector<std::string> label_array()
    {
        return std::vector<std::string>{ "start peak rss", " stop peak rss" };
    }

    static display_unit_type display_unit() { return display_unit_type{ "MB", "MB" }; }

    static std::pair<double, double> unit_array() { return unit(); }

    static std::vector<std::string> description_array()
    {
        return std::vector<std::string>{ "Resident set size at start",
                                         "Resident set size at stop" };
    }

    static unit_type get_unit()
    {
        static auto  _instance = this_type::unit();
        static auto& _mem      = _instance;

        if(settings::memory_units().length() > 0)
        {
            _mem.first  = std::get<1>(units::get_memory_unit(settings::memory_units()));
            _mem.second = std::get<1>(units::get_memory_unit(settings::memory_units()));
        }

        return _mem;
    }

    static display_unit_type get_display_unit()
    {
        static display_unit_type _instance = this_type::display_unit();
        static auto&             _mem      = _instance;

        if(settings::memory_units().length() > 0)
        {
            _mem.first  = std::get<0>(units::get_memory_unit(settings::memory_units()));
            _mem.second = std::get<0>(units::get_memory_unit(settings::memory_units()));
        }

        return _mem;
    }
};

//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
//
//======================================================================================//

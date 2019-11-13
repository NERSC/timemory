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

#include "timemory/backends/rusage.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/storage.hpp"

//======================================================================================//

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

extern template struct base<peak_rss>;
extern template struct base<page_rss>;
extern template struct base<stack_rss>;
extern template struct base<data_rss>;
extern template struct base<num_swap>;
extern template struct base<num_io_in>;
extern template struct base<num_io_out>;
extern template struct base<num_minor_page_faults>;
extern template struct base<num_major_page_faults>;
extern template struct base<num_msg_sent>;
extern template struct base<num_msg_recv>;
extern template struct base<num_signals>;
extern template struct base<voluntary_context_switch>;
extern template struct base<priority_context_switch>;
extern template struct base<read_bytes, std::tuple<int64_t, int64_t>>;
extern template struct base<written_bytes, std::tuple<int64_t, int64_t>>;
extern template struct base<virtual_memory>;

#endif

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
    using value_type = int64_t;
    using base_type  = base<peak_rss, value_type>;

    static std::string label() { return "peak_rss"; }
    static std::string description() { return "max resident set size"; }
    static value_type  record() { return get_peak_rss(); }
    double             get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    double get() const { return get_display(); }
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
/// \class page_rss
/// \brief
/// this struct measures the resident set size (RSS) currently allocated in pages of
/// memory. Unlike the peak_rss, this value will fluctuate as memory gets freed and
/// allocated
//
struct page_rss : public base<page_rss>
{
    using value_type = int64_t;
    using base_type  = base<page_rss, value_type>;

    static std::string label() { return "page_rss"; }
    static std::string description() { return "resident set size of memory pages"; }
    static value_type  record() { return get_page_rss(); }
    double             get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    double get() const { return get_display(); }
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

using current_rss = page_rss;

//--------------------------------------------------------------------------------------//
/// \class stack_rss
/// \brief
/// an integral value indicating the amount of memory used by the text segment that was
/// also shared among other processes.
/// an integral value of the amount of unshared memory residing in the stack segment
/// of a process
//
struct stack_rss : public base<stack_rss>
{
    using value_type = int64_t;
    using base_type  = base<stack_rss, value_type>;

    static int64_t     units() { return units::kilobyte; }
    static std::string label() { return "stack_rss"; }
    static std::string description() { return "integral unshared stack size"; }
    static value_type  record() { return get_stack_rss(); }
    double             get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    double get() const { return get_display(); }
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
//
struct data_rss : public base<data_rss>
{
    using value_type = int64_t;
    using base_type  = base<data_rss, value_type>;

    static int64_t     units() { return units::kilobyte; }
    static std::string label() { return "data_rss"; }
    static std::string description() { return "integral unshared data size"; }
    static value_type  record() { return get_data_rss(); }
    double             get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    double get() const { return get_display(); }
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
    value_type         get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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
    value_type         get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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
    value_type         get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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
    value_type         get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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
    value_type         get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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
/// \class num_msg_sent
/// \brief
/// the number of IPC messages sent.
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
    value_type         get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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
/// \class num_msg_recv
/// \brief
/// the number of IPC messages received.
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
    value_type         get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val;
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
/// \class num_signals
/// \brief
/// the number of signals delivered
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

using prio_cxt_switch = priority_context_switch;

//--------------------------------------------------------------------------------------//
/// \class read_bytes
/// \brief I/O counter: bytes read Attempt to count the number of bytes which this process
/// really did cause to be fetched from the storage layer. Done at the submit_bio() level,
/// so it is accurate for block-backed filesystems.
//
struct read_bytes : public base<read_bytes, std::tuple<int64_t, int64_t>>
{
    using value_type  = std::tuple<int64_t, int64_t>;
    using base_type   = base<read_bytes, value_type>;
    using timer_type  = real_clock;
    using result_type = std::tuple<int64_t, double>;

    static int64_t     unit() { return units::kilobyte; }
    static std::string label() { return "read_bytes"; }
    static std::string description() { return "physical I/O reads"; }

    static value_type record()
    {
        return value_type(get_bytes_read(), timer_type::record());
    }

    std::string get_display() const
    {
        std::stringstream ss, ssv, ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = base_type::get_display_unit();

        auto _val = get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!_disp.empty())
            ssv << " " << _disp;

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!_disp.empty())
            ssr << " " << _disp << "/" << timer_type::get_display_unit();

        ss << ssv.str() << ", " << ssr.str();
        ss << " read";
        return ss.str();
    }

    result_type get() const
    {
        auto val = (is_transient) ? accum : value;

        auto data  = std::get<0>(val) / base_type::get_unit();
        auto delta = static_cast<double>(std::get<1>(val) /
                                         static_cast<double>(timer_type::ratio_t::den) *
                                         timer_type::get_unit());
        auto rate  = data / delta;
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
        auto tmp = record();
        std::get<0>(accum) += (std::get<0>(tmp) - std::get<0>(value));
        std::get<1>(accum) += (std::get<1>(tmp) - std::get<1>(value));
        value = std::move(tmp);
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
/// \class written_bytes
/// \brief I/O counter: Attempt to count the number of bytes which this process caused to
/// be sent to the storage layer. This is done at page-dirtying time.
//
struct written_bytes : public base<written_bytes, std::tuple<int64_t, int64_t>>
{
    using value_type  = std::tuple<int64_t, int64_t>;
    using base_type   = base<written_bytes, value_type>;
    using timer_type  = real_clock;
    using result_type = std::tuple<int64_t, double>;

    static int64_t     unit() { return units::kilobyte; }
    static std::string label() { return "written_bytes"; }
    static std::string description() { return "physical I/O writes"; }

    static value_type record()
    {
        return value_type(get_bytes_written(), timer_type::record());
    }

    std::string get_display() const
    {
        std::stringstream ss, ssv, ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = base_type::get_display_unit();

        auto _val = get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!_disp.empty())
            ssv << " " << _disp;

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!_disp.empty())
            ssr << " " << _disp << "/" << timer_type::get_display_unit();

        ss << ssv.str() << ", " << ssr.str();
        ss << " written";
        return ss.str();
    }

    result_type get() const
    {
        auto val = (is_transient) ? accum : value;

        auto data  = std::get<0>(val) / base_type::get_unit();
        auto delta = static_cast<double>(std::get<1>(val) /
                                         static_cast<double>(timer_type::ratio_t::den) *
                                         timer_type::get_unit());
        auto rate  = data / delta;
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
        auto tmp = record();
        std::get<0>(accum) += (std::get<0>(tmp) - std::get<0>(value));
        std::get<1>(accum) += (std::get<1>(tmp) - std::get<1>(value));
        value = std::move(tmp);
        set_stopped();
    }
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
    double             get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return val / static_cast<double>(base_type::get_unit());
    }
    double get() const { return get_display(); }
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
}  // namespace component
}  // namespace tim

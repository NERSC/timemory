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

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/macros.hpp"
#include "timemory/rusage.hpp"
#include "timemory/storage.hpp"
#include "timemory/units.hpp"

//======================================================================================//

namespace tim
{
namespace component
{
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
}  // namespace component
}  // namespace tim

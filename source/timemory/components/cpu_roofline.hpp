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
#include "timemory/components/timing.hpp"
#include "timemory/components/types.hpp"
#include "timemory/macros.hpp"
#include "timemory/papi.hpp"
#include "timemory/storage.hpp"
#include "timemory/units.hpp"

//======================================================================================//

namespace tim
{
namespace component
{
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
              std::pair<std::array<long long, sizeof...(EventTypes)>, int64_t>,
              policy::initialization, policy::finalization>
{
    friend struct policy::wrapper<policy::initialization, policy::finalization>;

    using size_type  = std::size_t;
    using array_type = std::array<long long, sizeof...(EventTypes)>;
    using value_type = std::pair<array_type, int64_t>;
    using this_type  = cpu_roofline<EventTypes...>;
    using base_type =
        base<this_type, value_type, policy::initialization, policy::finalization>;

    using papi_type = papi_tuple<0, EventTypes...>;
    using ratio_t   = typename real_clock::ratio_t;

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

    static void invoke_initialize()
    {
        int events[] = { EventTypes... };
        tim::papi::start_counters(events, num_events);
    }
    static void invoke_finalize()
    {
        array_type events = {};
        tim::papi::stop_counters(events.data(), num_events);
    }

    static int64_t     unit() { return 1; }
    static std::string label() { return "cpu_roofline"; }
    static std::string descript() { return "cpu roofline"; }
    static std::string display_unit()
    {
        std::stringstream ss;
        ss << "(";
        auto labels = papi_type::label_array();
        for(size_type i = 0; i < labels.size(); ++i)
        {
            ss << labels[i];
            if(i + 1 < labels.size())
            {
                ss << " + ";
            }
        }
        ss << ") / " << real_clock::display_unit();
        return ss.str();
    }

    static value_type record()
    {
        return value_type(papi_type::record(), real_clock::record());
    }

    double compute_display() const
    {
        auto& obj = (accum.second > 0) ? accum : value;
        if(obj.second == 0)
            return 0.0;
        return std::accumulate(obj.first.begin(), obj.first.end(), 0) /
               (static_cast<double>(obj.second / static_cast<double>(ratio_t::den) *
                                    real_clock::get_unit()));
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

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        auto _value = obj.compute_display();
        auto _label = this_type::get_label();
        auto _disp  = this_type::display_unit();
        auto _prec  = real_clock::get_precision();
        auto _width = this_type::get_width();
        auto _flags = real_clock::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);
        ss_value << std::setw(_width) << std::setprecision(_prec) << _value;
        if(!_disp.empty())
            ss_extra << " " << _disp;
        else if(!_label.empty())
            ss_extra << " " << _label;
        os << ss_value.str() << ss_extra.str();

        return os;
    }

private:
    // create and destroy a papi_tuple<...> so that the event set gets registered
    // papi_type _impl;
};

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
}  // namespace component
}  // namespace tim

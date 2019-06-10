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
              std::pair<std::array<long long, sizeof...(EventTypes)>, int64_t>>
{
    using size_type  = std::size_t;
    using array_type = std::array<long long, sizeof...(EventTypes)>;
    using papi_type  = papi_tuple<0, EventTypes...>;
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
        base_type::get_precision()    = real_clock::get_precision();
        base_type::get_format_flags() = real_clock::get_format_flags();
        base_type::get_width()        = real_clock::get_width();
        auto& obj                     = (accum.second > 0) ? accum : value;
        if(obj.second == 0)
            return 0.0;
        return std::accumulate(obj.first.begin(), obj.first.end(), 0) /
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

private:
    // create and destroy a papi_tuple<...> so that the event set gets registered
    papi_type _impl;
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

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

/** \file timemory/components/timing/components.hpp
 * \brief Provides components for timing-related components
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include "timemory/components/timing/backends.hpp"
#include "timemory/components/timing/types.hpp"

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//          Timing types
//
//--------------------------------------------------------------------------------------//
// the system's real time (i.e. wall time) clock, expressed as the amount of time since
// the epoch.
struct wall_clock : public base<wall_clock, int64_t>
{
    using ratio_t           = std::nano;
    using value_type        = int64_t;
    using base_type         = base<wall_clock, value_type>;
    using statistics_policy = policy::record_statistics<wall_clock>;

    static std::string label() { return "wall"; }
    static std::string description() { return "wall time"; }
    static value_type  record() { return tim::get_clock_real_now<int64_t, ratio_t>(); }

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
        value = (record() - value);
        accum += value;
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// alias for "wall_clock"
using real_clock = wall_clock;
// alias for "wall_clock" since time is a construct of our consciousness
using virtual_clock = wall_clock;
//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim

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

/** \file cupti.hpp
 * \headerfile cupti_activity.hpp "timemory/cupti_activity.hpp"
 * Provides implementation of CUPTI routines.
 *
 */

#pragma once

#include "timemory/backends/cuda.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/details/cupti.hpp"
#include "timemory/units.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#endif

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//          CUPTI component
//
//--------------------------------------------------------------------------------------//

//#if defined(TIMEMORY_USE_CUPTI)

struct cupti_activity : public base<cupti_activity, int64_t>
{
    using ratio_t       = std::nano;
    using size_type     = std::size_t;
    using string_t      = std::string;
    using receiver_type = cupti::activity::receiver<cupti_activity>;
    using value_type    = int64_t;
    using base_type     = base<cupti_activity, value_type>;
    using this_type     = cupti_activity;

    static const short                   precision = 3;
    static const short                   width     = 8;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::dec | std::ios_base::showpoint;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "real"; }
    static std::string descript() { return "wall time"; }
    static std::string display_unit() { return "sec"; }

    static value_type record()
    {
        return cupti::activity::get_receiver<this_type>().get();
    }

    // make sure it is removed
    ~cupti_activity() { cupti::activity::get_receiver<this_type>().remove(this); }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        set_started();
        cupti::activity::start_trace(this);
        value = record();
    }

    void stop()
    {
        cupti::activity::stop_trace(this);
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }

    float get_display() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<float>(val / static_cast<float>(ratio_t::den) *
                                  base_type::get_unit());
    }

    float get() const
    {
        auto val = (is_transient) ? accum : value;
        return static_cast<float>(val / static_cast<float>(ratio_t::den) *
                                  base_type::get_unit());
    }
};

}  // namespace component
}  // namespace tim

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
 * \file timemory/components/caliper/components.hpp
 * \brief Implementation of the caliper component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#if defined(TIMEMORY_USE_CALIPER)
//
//  Temporarily locating this inside the timemory package, in the future,
//  the plan is for it to live inside the Caliper repository so that Caliper can
//  can arbitrarily extend/modify/etc.
//
#    include "timemory/components/caliper/timemory.hpp"
#else
#    include "timemory/components/caliper/backends.hpp"
#    include "timemory/components/caliper/types.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
struct caliper_marker : public base<caliper_marker, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = caliper_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "caliper"; }
    static std::string description()
    {
        return "Forwards markers to Caliper instrumentation";
    }
};
//
//======================================================================================//
//
}  // namespace component
}  // namespace tim
#endif

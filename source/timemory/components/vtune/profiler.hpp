// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/** \file components/vtune/profiler.hpp
 * \headerfile components/vtune/profiler.hpp "timemory/components/vtune/profiler.hpp"
 * This component provides VTune profiler start/stop
 *
 */

#pragma once

#include "timemory/backends/ittnotify.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

extern template struct base<vtune_profiler, void>;

#endif

//--------------------------------------------------------------------------------------//
// control VTune profiler
//
struct vtune_profiler
: public base<vtune_profiler, void>
, public policy::instance_tracker<vtune_profiler, false>
{
    using value_type   = void;
    using this_type    = vtune_profiler;
    using base_type    = base<this_type, value_type>;
    using tracker_type = policy::instance_tracker<vtune_profiler, false>;

    static std::string label() { return "vtune_profiler"; }
    static std::string description() { return "Start/stop Intel profiling"; }
    static value_type  record() {}

    static void global_init(storage_type*) { ittnotify::pause(); }
    static void global_finalize(storage_type*) { ittnotify::pause(); }

    using tracker_type::m_tot;

    void start()
    {
        tracker_type::start();
        set_started();
        if(m_tot == 0)
            ittnotify::resume();
    }

    void stop()
    {
        tracker_type::stop();
        set_stopped();
        if(m_tot == 0)
            ittnotify::pause();
    }
};
}  // namespace component
}  // namespace tim

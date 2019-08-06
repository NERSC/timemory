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
//
// Author: Jonathan R. Madsen
//

#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <vector>

#include <timemory/timemory.hpp>

using string_t     = std::string;
using string_set_t = std::set<string_t>;

//--------------------------------------------------------------------------------------//
// placeholder for hardware counter extension in development
//
#if !defined(CPU_HW_COUNTERS)
#    define CPU_HW_COUNTERS 32
#endif

// using papi_array_t = tim::component::papi_array<CPU_HW_COUNTERS>;
// using papi_tuple_t = tim::component_tuple<papi_array_t>;

#if !defined(GPU_HW_COUNTERS)
#    define GPU_HW_COUNTERS 8
#endif
//
// EVENTUALLY, will have something like:
//
//      using papi_array_t  = tim::papi_array < CPU_HW_COUNTERS >;
//      using cupti_array_t = tim::cupti_array< GPU_HW_COUNTERS >;
//
// that will let the user define a string and the array will be populated similar to
// the component_list

//--------------------------------------------------------------------------------------//
//  declare the allowed component types
//  components are in tim::component:: namespace but thats a lot to type...
//
using namespace tim::component;
using auto_list_t = tim::auto_list<
    real_clock, system_clock, user_clock, cpu_clock, monotonic_clock, monotonic_raw_clock,
    thread_cpu_clock, process_cpu_clock, cpu_util, thread_cpu_util, process_cpu_util,
    current_rss, peak_rss, stack_rss, data_rss, num_swap, num_io_in, num_io_out,
    num_minor_page_faults, num_major_page_faults, num_msg_sent, num_msg_recv, num_signals,
    voluntary_context_switch, priority_context_switch, cuda_event,
    papi_array<CPU_HW_COUNTERS>, cpu_roofline_sp_flops, cpu_roofline_dp_flops>;

//--------------------------------------------------------------------------------------//
// use this type derived from the auto type because the "auto" types do template
// filtering if a component is marked as unavailable, e.g.:
//
//      namespace tim
//      {
//      namespace component
//      {
//
//      template <>
//      struct impl_available<data_rss> : std::false_type;
//
//      } // namespace component
//      } // namespace tim
//
using comp_list_t = typename auto_list_t::component_type;

//--------------------------------------------------------------------------------------//
// default components to record -- maybe should be empty?
//
inline string_t
get_default_components()
{
    return "real_clock, user_clock, system_clock, cpu_util, current_rss, peak_rss, "
           "cuda_event";
}

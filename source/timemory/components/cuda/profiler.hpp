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

/** \file components/cuda/profiler.hpp
 * \headerfile components/cuda/profiler.hpp "timemory/components/cuda/profiler.hpp"
 * This component provides start and stop for the profiler
 *
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/policy.hpp"

#if defined(TIMEMORY_USE_CUDA)
#    include <cuda_profiler_api.h>
#endif

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)
extern template struct base<cuda_profiler, void>;
#endif

//--------------------------------------------------------------------------------------//
// controls the CUDA profiler
//
struct cuda_profiler
: public base<cuda_profiler, void>
, private policy::instance_tracker<cuda_profiler>
{
    using value_type   = void;
    using this_type    = cuda_profiler;
    using base_type    = base<this_type, value_type>;
    using tracker_type = policy::instance_tracker<cuda_profiler>;

    static std::string label() { return "cuda_profiler"; }
    static std::string description() { return "CUDA profiler controller"; }

    enum class mode : short
    {
        nvp,
        csv
    };

    static void configure(const std::string& _infile, const std::string& _outfile,
                          mode _mode)
    {
#if defined(TIMEMORY_USE_CUDA)
        cudaProfilerInitialize(_infile.c_str(), _outfile.c_str(),
                               (_mode == mode::nvp) ? cudaKeyValuePair : cudaCSV);
#else
        consume_parameters(_infile, _outfile, _mode);
#endif
    }

    void start()
    {
#if defined(TIMEMORY_USE_CUDA)
        tracker_type::start();
        if(m_tot == 0)
            cudaProfilerStart();
#endif
    }

    void stop()
    {
        tracker_type::stop();
#if defined(TIMEMORY_USE_CUDA)
        if(m_tot == 0)
            cudaProfilerStop();
#endif
    }
};

}  // namespace component
}  // namespace tim

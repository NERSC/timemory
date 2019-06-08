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

/** \file components.hpp
 * \headerfile components.hpp "timemory/components.hpp"
 * These are core tools provided by TiMemory. These tools can be used individually
 * or bundled together in a component_tuple (C++) or component_list (C, Python)
 *
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <numeric>
#include <string>

#include "timemory/macros.hpp"
#include "timemory/papi.hpp"
#include "timemory/storage.hpp"
#include "timemory/units.hpp"
#include "timemory/utility.hpp"

#include "timemory/components/base.hpp"
#include "timemory/components/cpu_roofline.hpp"
#include "timemory/components/cupti_array.hpp"
#include "timemory/components/cupti_tuple.hpp"
#include "timemory/components/gpu_roofline.hpp"
#include "timemory/components/papi_array.hpp"
#include "timemory/components/papi_tuple.hpp"
#include "timemory/components/resource_usage.hpp"
#include "timemory/components/timing.hpp"
#include "timemory/components/type_traits.hpp"
#include "timemory/components/types.hpp"

#if defined(TIMEMORY_USE_CUDA)
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#endif

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/cupti.hpp"
#endif

//======================================================================================//

namespace tim
{
namespace component
{
//======================================================================================//
// component initialization
//
/*
class init
{
public:
    using string_t  = std::string;
    bool     store  = false;
    int32_t  ncount = 0;
    int32_t  nhash  = 0;
    string_t key    = "";
    string_t tag    = "";
};
*/

//======================================================================================//
// construction tuple for a component
//
template <typename Type, typename... Args>
class constructor : public std::tuple<Args...>
{
public:
    using base_type                    = std::tuple<Args...>;
    static constexpr std::size_t nargs = std::tuple_size<decay_t<base_type>>::value;

    explicit constructor(Args&&... _args)
    : base_type(std::forward<Args>(_args)...)
    {
    }

    template <typename _Tuple, size_t... _Idx>
    Type operator()(_Tuple&& __t, index_sequence<_Idx...>)
    {
        return Type(std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }

    Type operator()()
    {
        return (*this)(static_cast<base_type>(*this), make_index_sequence<nargs>{});
    }
};

//--------------------------------------------------------------------------------------//
//  component_tuple initialization
//
using init = constructor<void, std::string, std::string, int32_t, int32_t, bool>;

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_CUDA)

//--------------------------------------------------------------------------------------//
// this component extracts the time spent in GPU kernels
//
struct cuda_event : public base<cuda_event, float>
{
    using ratio_t    = std::milli;
    using value_type = float;
    using base_type  = base<cuda_event, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "cuda_event"; }
    static std::string descript() { return "event time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return 0.0f; }

    cuda_event(cudaStream_t _stream = 0)
    : m_stream(_stream)
    {
        m_is_valid = check(cudaEventCreate(&m_start));
        if(m_is_valid)
            m_is_valid = check(cudaEventCreate(&m_stop));
    }

    ~cuda_event()
    {
        /*if(m_is_valid && is_valid())
        {
            sync();
            destroy();
        }*/
    }

    float compute_display() const
    {
        const_cast<cuda_event&>(*this).sync();
        auto val = (is_transient) ? accum : value;
        return static_cast<float>(val / static_cast<float>(ratio_t::den) *
                                  base_type::get_unit());
    }

    void start()
    {
        set_started();
        if(m_is_valid)
        {
            m_is_synced = false;
            // cuda_event* _this = static_cast<cuda_event*>(this);
            // cudaStreamAddCallback(m_stream, &cuda_event::callback, _this, 0);
            cudaEventRecord(m_start, m_stream);
        }
    }

    void stop()
    {
        if(m_is_valid)
        {
            cudaEventRecord(m_stop, m_stream);
            sync();
        }
        set_stopped();
    }

    void set_stream(cudaStream_t _stream = 0) { m_stream = _stream; }

    void sync()
    {
        if(m_is_valid && !m_is_synced)
        {
            cudaEventSynchronize(m_stop);
            float tmp = 0.0f;
            cudaEventElapsedTime(&tmp, m_start, m_stop);
            accum += tmp;
            value       = std::move(tmp);
            m_is_synced = true;
        }
    }

    void destroy()
    {
        if(m_is_valid && is_valid())
        {
            cudaEventDestroy(m_start);
            cudaEventDestroy(m_stop);
        }
    }

    bool is_valid() const
    {
        // get last error but don't reset last error to cudaSuccess
        auto ret = cudaPeekAtLastError();
        // if failure previously, return false
        if(ret != cudaSuccess)
            return false;
        // query
        ret = cudaEventQuery(m_stop);
        // if all good, return valid
        if(ret == cudaSuccess)
            return true;
        // if not all good, clear the last error bc if was from failed query
        ret = cudaGetLastError();
        // return if not ready (OK) or something else
        return (ret == cudaErrorNotReady);
    }

    bool check(cudaError_t err) const { return (err == cudaSuccess); }

protected:
    static void callback(cudaStream_t /*_stream*/, cudaError_t /*_status*/,
                         void* user_data)
    {
        cuda_event* _this = static_cast<cuda_event*>(user_data);
        if(!_this->m_is_synced && _this->is_valid())
        {
            cudaEventSynchronize(_this->m_stop);
            float tmp = 0.0f;
            cudaEventElapsedTime(&tmp, _this->m_start, _this->m_stop);
            _this->accum += tmp;
            _this->value       = std::move(tmp);
            _this->m_is_synced = true;
        }
    }

private:
    bool         m_is_synced = false;
    bool         m_is_valid  = true;
    cudaStream_t m_stream    = 0;
    cudaEvent_t  m_start;
    cudaEvent_t  m_stop;
};

#else
//--------------------------------------------------------------------------------------//
// dummy for cuda_event when CUDA is not available
//
using cudaStream_t = int;
using cudaError_t  = int;
// this struct extracts only the CPU time spent in kernel-mode
struct cuda_event : public base<cuda_event, float>
{
    using ratio_t    = std::micro;
    using value_type = float;
    using base_type  = base<cuda_event, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "cuda_event"; }
    static std::string descript() { return "event time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return 0.0f; }

    cuda_event(cudaStream_t _stream = 0)
    : m_stream(_stream)
    {
    }

    ~cuda_event() {}

    float compute_display() const { return 0.0f; }

    void start() {}

    void stop() {}

    void set_stream(cudaStream_t _stream = 0) { m_stream = _stream; }

    static void callback(cudaStream_t, cudaError_t, void*) {}

    void sync() {}

private:
    cudaStream_t m_stream = 0;
};

#endif

}  // namespace component

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/cupti_event.hpp"
#endif

//--------------------------------------------------------------------------------------//

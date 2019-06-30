// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

/** \file cuda_event.hpp
 * \headerfile cuda_event.hpp "timemory/cuda_event.hpp"
 * This component provides kernel timer
 *
 */

#pragma once

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
// this struct extracts only the CPU time spent in kernel-mode
struct cuda_event : public base<cuda_event, float>
{
    using cudaStream_t = int;
    using cudaError_t  = int;
    using ratio_t      = std::milli;
    using value_type   = float;
    using base_type    = base<cuda_event, value_type>;

    static const short                   precision = 3;
    static const short                   width     = 6;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec;

    static int64_t     unit() { return units::sec; }
    static std::string label() { return "cuda_event"; }
    static std::string descript() { return "event time"; }
    static std::string display_unit() { return "sec"; }
    static value_type  record() { return 0.0f; }

    cuda_event() {}
    explicit cuda_event(cudaStream_t) {}
    ~cuda_event() {}

    float       compute_display() const { return 0.0f; }
    void        start() {}
    void        stop() {}
    void        set_stream(cudaStream_t = 0) {}
    static void callback(cudaStream_t, cudaError_t, void*) {}
    void        sync() {}
};

#endif

}  // namespace component

//--------------------------------------------------------------------------------------//

}  // namespace tim

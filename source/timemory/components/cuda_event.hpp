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

#include "timemory/backends/cuda.hpp"
#include "timemory/bits/settings.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/units.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#endif

#include <vector>

//======================================================================================//

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

#endif

//--------------------------------------------------------------------------------------//
// this component extracts the time spent in GPU kernels
//
struct cuda_event : public base<cuda_event, float>
{
    struct marker
    {
        bool          valid   = true;
        bool          synced  = false;
        bool          running = false;
        cuda::event_t first   = cuda::event_t{};
        cuda::event_t second  = cuda::event_t{};

        marker() { valid = (cuda::event_create(first) && cuda::event_create(second)); }

        ~marker()
        {
            // cuda::event_destroy(first);
            // cuda::event_destroy(second);
        }

        void start(cuda::stream_t& stream)
        {
            if(!valid)
                return;
            synced  = false;
            running = true;
            cuda::event_record(first, stream);
        }

        void stop(cuda::stream_t& stream)
        {
            if(!valid || !running)
                return;
            cuda::event_record(second, stream);
            running = false;
        }

        float sync()
        {
            if(!valid)
                return 0.0;
            if(!synced)
                cuda::event_sync(second);
            synced = true;
            return cuda::event_elapsed_time(first, second);
        }
    };

    using ratio_t       = std::milli;
    using value_type    = float;
    using base_type     = base<cuda_event, value_type>;
    using marker_list_t = std::vector<marker>;

    static std::string label() { return "cuda_event"; }
    static std::string description() { return "event time"; }
    static value_type  record() { return 0.0f; }

    static uint64_t& get_batched_marker_size()
    {
        static uint64_t _instance = settings::cuda_event_batch_size();
        return _instance;
    }

public:
    explicit cuda_event(cuda::stream_t _stream = 0)
    : m_stream(_stream)
    , m_global(marker())
    {}

    ~cuda_event() {}
    cuda_event(const cuda_event&) = default;
    cuda_event(cuda_event&&)      = default;
    cuda_event& operator=(const cuda_event&) = default;
    cuda_event& operator=(cuda_event&&) = default;

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

    void start()
    {
        set_started();
        m_global_synced = false;
        m_global.start(m_stream);
        // cuda_event* _this = static_cast<cuda_event*>(this);
        // cudaStreamAddCallback(m_stream, &cuda_event::callback, _this, 0);
    }

    void stop()
    {
        for(uint64_t i = 0; i < m_num_markers; ++i)
            m_markers[i].stop(m_stream);
        if(m_current_marker == 0 && m_num_markers == 0)
            m_global.stop(m_stream);
        sync();
        set_stopped();
    }

    void sync()
    {
        if(m_current_marker == 0 && m_num_markers == 0)
        {
            if(!m_global_synced)
            {
                float tmp       = m_global.sync();
                m_global_synced = true;
                accum += tmp;
                value = std::move(tmp);
            }
        }
        else if(m_current_marker > m_synced_markers)
        {
            float tmp = 0.0;
            for(uint64_t i = m_synced_markers; i < m_num_markers; ++i, ++m_synced_markers)
                tmp += m_markers[i].sync();
            m_markers_synced = true;
            accum += tmp;
            value = std::move(tmp);
        }
    }

    void set_stream(cuda::stream_t _stream = 0) { m_stream = _stream; }

    void mark_begin()
    {
        m_markers_synced = false;
        m_current_marker = m_num_markers++;
        if(m_current_marker >= m_markers.size())
            append_marker_list(std::max<uint64_t>(m_marker_batch_size, 1));
        m_markers[m_current_marker].start(m_stream);
    }

    void mark_end() { m_markers[m_current_marker].stop(m_stream); }

    void mark_begin(cuda::stream_t _stream)
    {
        m_markers_synced = false;
        m_current_marker = m_num_markers++;
        if(m_current_marker >= m_markers.size())
            append_marker_list(std::max<uint64_t>(m_marker_batch_size, 1));
        m_markers[m_current_marker].start(_stream);
    }

    void mark_end(cuda::stream_t _stream) { m_markers[m_current_marker].stop(_stream); }

protected:
    void append_marker_list(const uint64_t nsize)
    {
        for(uint64_t i = 0; i < nsize; ++i)
            m_markers.emplace_back(marker());
    }

private:
    bool           m_global_synced     = false;
    bool           m_markers_synced    = false;
    uint64_t       m_synced_markers    = 0;
    uint64_t       m_current_marker    = 0;
    uint64_t       m_num_markers       = 0;
    uint64_t       m_marker_batch_size = get_batched_marker_size();
    cuda::stream_t m_stream            = 0;
    marker         m_global;
    marker_list_t  m_markers;

    /*
    static void callback(cuda::stream_t stream, cuda::error_t status, void* user_data)
    {
        cuda_event* _this = static_cast<cuda_event*>(user_data);
        if(!_this->m_is_synced && _this->is_valid())
        {
            cuda::event_sync(_this->m_stop);
            float tmp = cuda::event_elapsed_time(_this->m_start, _this->m_stop);
            _this->accum += tmp;
            _this->value       = std::move(tmp);
            _this->m_is_synced = true;
        }
    }
    */

    /*
    bool is_valid(const marker& m) const
    {
        // get last error but don't reset last error to cudaSuccess
        auto ret = cuda::peek_at_last_error();
        // if failure previously, return false
        if(ret != cuda::success_v)
            return false;
        // query
        ret = cuda::event_query(m.second);
        // if all good, return valid
        if(ret == cuda::success_v)
            return true;
        // if not all good, clear the last error bc if was from failed query
        ret = cuda::get_last_error();
        // return if not ready (OK) or something else
        return (ret == cuda::err_not_ready_v);
    }
    */
};

}  // namespace component

//--------------------------------------------------------------------------------------//

namespace trait
{
template <>
struct supports_args<component::cuda_event, std::tuple<>> : std::true_type
{};

template <>
struct supports_args<component::cuda_event, std::tuple<cuda::stream_t>> : std::true_type
{};
}  // namespace trait

//--------------------------------------------------------------------------------------//

}  // namespace tim

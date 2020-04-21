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
 * \file timemory/components/cuda/components.hpp
 * \brief Implementation of the cuda component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"

#include "timemory/components/cuda/backends.hpp"
#include "timemory/components/cuda/types.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//
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
};
//
//======================================================================================//
//
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

    using config_type      = std::tuple<std::string, std::string, mode>;
    using initializer_type = std::function<config_type()>;

    static initializer_type& get_initializer()
    {
        static initializer_type _instance = []() {
            return config_type("cuda_profiler.inp", "cuda_profiler.out", mode::nvp);
        };
        return _instance;
    }

    static void global_init(storage_type*)
    {
#if defined(TIMEMORY_USE_CUDA)
        cudaProfilerStop();
#endif
    }

    static void global_finalize(storage_type*)
    {
#if defined(TIMEMORY_USE_CUDA)
        cudaProfilerStop();
#endif
    }

    static void configure()
    {
        auto _config = get_initializer()();
        configure(std::get<0>(_config), std::get<1>(_config), std::get<2>(_config));
    }

    static void configure(const std::string& _infile, const std::string& _outfile,
                          mode _mode)
    {
        static std::atomic<int32_t> _once;
        if(_once++ > 0)
            return;
#if defined(TIMEMORY_USE_CUDA)
        cudaProfilerInitialize(_infile.c_str(), _outfile.c_str(),
                               (_mode == mode::nvp) ? cudaKeyValuePair : cudaCSV);
#else
        consume_parameters(_infile, _outfile, _mode);
#endif
    }

    cuda_profiler() { configure(); }

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
#if defined(TIMEMORY_USE_CUDA)
        tracker_type::stop();
        if(m_tot == 0)
            cudaProfilerStop();
#endif
    }
};
//
//======================================================================================//
// adds NVTX markers
//
struct nvtx_marker : public base<nvtx_marker, void>
{
    using value_type = void;
    using this_type  = nvtx_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "nvtx_marker"; }
    static std::string description() { return "NVTX markers"; }
    static value_type  record() {}

    static bool& use_device_sync()
    {
        static bool _instance = settings::nvtx_marker_device_sync();
        return _instance;
    }

    static const int32_t& get_thread_id()
    {
        static std::atomic<int32_t> _thread_counter(0);
        static thread_local int32_t _thread_id = _thread_counter++;
        return _thread_id;
    }

    static const int32_t& get_stream_id(cuda::stream_t _stream)
    {
        using pair_t    = std::pair<cuda::stream_t, int32_t>;
        using map_t     = std::map<cuda::stream_t, int32_t>;
        using map_ptr_t = std::unique_ptr<map_t>;

        static thread_local map_ptr_t _instance = map_ptr_t(new map_t);
        if(_instance->find(_stream) == _instance->end())
            _instance->insert(pair_t(_stream, _instance->size()));
        return _instance->find(_stream)->second;
    }

    static void thread_init(storage_type*) { nvtx::name_thread(get_thread_id()); }

    nvtx_marker(const nvtx::color::color_t& _color = 0, const std::string& _prefix = "",
                cuda::stream_t _stream = 0)
    : color(_color)
    , prefix(_prefix)
    , stream(_stream)
    {}

    void start() { range_id = nvtx::range_start(get_attribute()); }
    void stop()
    {
        if(use_device_sync())
            cuda::device_sync();
        else
            cuda::stream_sync(stream);
        nvtx::range_stop(range_id);
    }

    void mark_begin()
    {
        nvtx::mark(prefix + "_begin_t" + std::to_string(get_thread_id()));
    }

    void mark_end() { nvtx::mark(prefix + "_end_t" + std::to_string(get_thread_id())); }

    void mark_begin(cuda::stream_t _stream)
    {
        nvtx::mark(prefix + "_begin_t" + std::to_string(get_thread_id()) + "_s" +
                   std::to_string(get_stream_id(_stream)));
    }

    void mark_end(cuda::stream_t _stream)
    {
        nvtx::mark(prefix + "_end_t" + std::to_string(get_thread_id()) + "_s" +
                   std::to_string(get_stream_id(_stream)));
    }

    void set_stream(cuda::stream_t _stream) { stream = _stream; }

    void set_prefix(const std::string& _prefix) { prefix = _prefix; }

private:
    nvtx::color::color_t     color = 0;
    nvtx::event_attributes_t attribute;
    nvtx::range_id_t         range_id = 0;
    std::string              prefix   = "";
    cuda::stream_t           stream   = 0;

private:
    nvtx::event_attributes_t& get_attribute()
    {
        if(!has_attribute)
        {
            if(settings::debug())
            {
                std::stringstream ss;
                ss << "[nvtx_marker]> Creating NVTX marker with label: \"" << prefix
                   << "\" and color " << std::hex << color << "...";
                std::cout << ss.str() << std::endl;
            }
            attribute     = nvtx::init_marker(prefix, color);
            has_attribute = true;
        }
        return attribute;
    }

    bool has_attribute = false;
};
//
//======================================================================================//
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//

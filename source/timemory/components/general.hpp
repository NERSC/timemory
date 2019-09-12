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
//

#pragma once

#include "timemory/backends/cuda.hpp"
#include "timemory/backends/gperf.hpp"
#include "timemory/backends/nvtx.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/details/settings.hpp"

#include <cstdint>

//======================================================================================//

namespace tim
{
template <typename... _Types>
class component_list;
template <typename... _Types>
class component_tuple;

namespace component
{
//--------------------------------------------------------------------------------------//
//
//          General Components with no specific category
//
//--------------------------------------------------------------------------------------//
// returns the trip count
//
struct trip_count : public base<trip_count>
{
    using value_type = int64_t;
    using this_type  = trip_count;
    using base_type  = base<this_type, value_type>;

    static const short                   precision = 0;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint;

    static int64_t     unit() { return 1; }
    static std::string label() { return "trip_count"; }
    static std::string description() { return "trip counts"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return 1; }

    value_type get_display() const { return accum; }
    value_type get() const { return accum; }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        accum += value;
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// start/stop gperftools cpu profiler
//
struct gperf_cpu_profiler
: public base<gperf_cpu_profiler, void, policy::thread_init, policy::global_finalize>
{
    using value_type = void;
    using this_type  = gperf_cpu_profiler;
    using base_type =
        base<this_type, value_type, policy::thread_init, policy::global_finalize>;

    static std::string label() { return "gperf_cpu_profiler"; }
    static std::string description() { return "gperftools cpu profiler"; }
    static value_type  record() {}

    static void invoke_thread_init() { gperf::cpu::register_thread(); }

    static void invoke_global_finalize()
    {
        if(gperf::cpu::is_running())
        {
            gperf::cpu::profiler_flush();
            gperf::cpu::profiler_stop();
        }
    }

    void start()
    {
        set_started();
        if(!gperf::cpu::is_running())
        {
            index      = this_type::get_index()++;
            auto fname = settings::compose_output_filename(
                label() + "_" + std::to_string(index), ".dat");
            auto ret = gperf::cpu::profiler_start(fname);
            if(ret == 0)
                fprintf(stderr, "[gperf_cpu_profiler]> Error starting %s...",
                        fname.c_str());
        }
    }

    void stop()
    {
        if(index >= 0)
        {
            gperf::cpu::profiler_flush();
            gperf::cpu::profiler_stop();
        }
        set_stopped();
    }

protected:
    int32_t index = -1;  // if this is >= zero, then we flush and stop

    template <typename... _Types>
    friend class component_tuple;

    template <typename... _Types>
    friend class component_list;

private:
    static std::atomic<int64_t>& get_index()
    {
        static std::atomic<int64_t> _instance;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
// start/stop gperftools cpu profiler
//
struct gperf_heap_profiler
: public base<gperf_heap_profiler, void, policy::global_finalize>
{
    using value_type = void;
    using this_type  = gperf_heap_profiler;
    using base_type  = base<this_type, value_type, policy::global_finalize>;

    static std::string label() { return "gperf_heap_profiler"; }
    static std::string description() { return "gperftools heap profiler"; }
    static value_type  record() {}

    static void invoke_global_finalize()
    {
        if(gperf::heap::is_running())
        {
            gperf::heap::profiler_flush("global_finalize");
            gperf::heap::profiler_stop();
        }
    }

    void start()
    {
        set_started();
        if(!gperf::heap::is_running())
        {
            index      = this_type::get_index()++;
            auto fname = settings::compose_output_filename(label(), ".dat");
            auto ret   = gperf::heap::profiler_start(fname);
            if(ret > 0)
                fprintf(stderr, "[gperf_heap_profiler]> Error starting %s...",
                        prefix.c_str());
        }
    }

    void stop()
    {
        if(index >= 0)
        {
            gperf::heap::profiler_flush(prefix);
            gperf::heap::profiler_stop();
        }
        set_stopped();
    }

protected:
    std::string prefix;
    int32_t     index = -1;  // if this is >= zero, then we flush and stop

    template <typename... _Types>
    friend class component_tuple;

    template <typename... _Types>
    friend class component_list;

private:
    static std::atomic<int64_t>& get_index()
    {
        static std::atomic<int64_t> _instance;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
// adds NVTX markers
//
struct nvtx_marker : public base<nvtx_marker, void, policy::thread_init>
{
    using value_type = void;
    using this_type  = nvtx_marker;
    using base_type  = base<this_type, value_type, policy::thread_init>;

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
        static std::atomic<int32_t> _thread_counter;
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

    static void invoke_thread_init() { nvtx::name_thread(get_thread_id()); }

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

    nvtx::color::color_t     color = 0;
    nvtx::event_attributes_t attribute;
    nvtx::range_id_t         range_id;
    std::string              prefix = "";
    cuda::stream_t           stream = 0;

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

//--------------------------------------------------------------------------------------//

}  // namespace component

//--------------------------------------------------------------------------------------//

namespace trait
{
template <>
struct supports_args<component::nvtx_marker, std::tuple<>> : std::true_type
{};

template <>
struct supports_args<component::nvtx_marker, std::tuple<cuda::stream_t>> : std::true_type
{};
}  // namespace trait

//--------------------------------------------------------------------------------------//
}  // namespace tim

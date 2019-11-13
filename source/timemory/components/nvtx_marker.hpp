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

#pragma once

#include "timemory/backends/cuda.hpp"
#include "timemory/backends/nvtx.hpp"
#include "timemory/bits/settings.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"

//======================================================================================//

namespace tim
{
template <typename... _Types>
class component_list;
template <typename... _Types>
class component_tuple;

namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

#endif

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

    static void invoke_thread_init(storage_type*) { nvtx::name_thread(get_thread_id()); }

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

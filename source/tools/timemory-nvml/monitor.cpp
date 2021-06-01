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

#include "nvml.hpp"
#include "nvml_memory_info.hpp"
#include "nvml_processes.hpp"
#include "nvml_temperature.hpp"
#include "nvml_utilization_rate.hpp"

#include <timemory/storage/ring_buffer.hpp>

using bundle_t = tim::lightweight_tuple<nvml_processes, nvml_memory_info,
                                        nvml_temperature, nvml_utilization_rate>;

using ring_buffer_t       = tim::data_storage::ring_buffer<bundle_t>;
using device_bundle_map_t = std::map<int, ring_buffer_t>;

void
dump(const device_bundle_map_t& _data)
{
    using map_type = std::map<int, std::vector<bundle_t>>;
    map_type _mdata{};

    for(auto& itr : _data)
    {
        std::vector<bundle_t> _vdata{};
        size_t                n = itr.second.count();
        _vdata.resize(n);
        for(size_t i = 0; i < n; ++i)
        {
            itr.second.read(&_vdata[i]);
        }
        _mdata.emplace(itr.first, _vdata);
    }
    using json_type = tim::cereal::PrettyJSONOutputArchive;
    // extra function
    auto _cmdline = [](json_type& ar) {
        ar(tim::cereal::make_nvp("config", get_config()));
    };

    auto fname = get_config().get_output_filename();
    fname += ".json";
    fprintf(stderr, "%s>>> Outputting '%s'...\n", (verbose() < 0) ? "" : "\n",
            fname.c_str());
    tim::generic_serialization<json_type>(fname, _mdata, "timemory", "nvml", _cmdline);
}

void
monitor(const std::vector<nvml_device_info>& _device_info)
{
    auto get_record_time = []() {
        char _buffer[128];
        auto _now = std::time_t{ std::time(nullptr) };

        if(std::strftime(_buffer, sizeof(_buffer), time_format().c_str(),
                         std::localtime(&_now)))
            return std::string{ _buffer };
        return std::string{};
    };

    device_bundle_map_t device_bundle_map{};

    // create a ring buffer for each device
    for(const auto& itr : _device_info)
    {
        device_bundle_map.emplace(itr.index, buffer_count());
    }

    while(true)
    {
        size_t ncount = 0;
        for(const auto& itr : _device_info)
        {
            bundle_t _bundle{ get_record_time() };
            _bundle.sample(itr.device);
            device_bundle_map[itr.index].write(&_bundle);
        }
        auto nidx = ++ncount;
        if(nidx >= max_samples() && max_samples() > 0)
        {
            break;
        }
        else if(nidx >= dump_interval())
        {
            ncount = 0;
            dump(device_bundle_map);
        }
    }

    dump(device_bundle_map);
}

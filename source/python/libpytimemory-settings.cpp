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

#if !defined(TIMEMORY_PYSETTINGS_SOURCE)
#    define TIMEMORY_PYSETTINGS_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/settings/extern.hpp"

using string_t = std::string;

#define SETTING_PROPERTY(TYPE, FUNC)                                                     \
    settings.def_property_static(                                                        \
        TIMEMORY_STRINGIZE(FUNC), [](py::object) { return tim::settings::FUNC(); },      \
        [](py::object, TYPE v) { tim::settings::FUNC() = v; },                           \
        "Binds to 'tim::settings::" TIMEMORY_STRINGIZE(FUNC) "()'")

//======================================================================================//
//
namespace pysettings
{
//
//--------------------------------------------------------------------------------------//
//
py::class_<pysettings::settings>
generate(py::module& _pymod)
{
    py::class_<pysettings::settings> settings(
        _pymod, "settings", "Global configuration settings for timemory");

    settings.def(py::init<>(), "Dummy");
    // to parse changes in env vars
    settings.def("parse", &tim::settings::parse);

    using strvector_t = std::vector<std::string>;

    SETTING_PROPERTY(bool, suppress_parsing);
    SETTING_PROPERTY(bool, enabled);
    SETTING_PROPERTY(bool, auto_output);
    SETTING_PROPERTY(bool, cout_output);
    SETTING_PROPERTY(bool, file_output);
    SETTING_PROPERTY(bool, text_output);
    SETTING_PROPERTY(bool, json_output);
    SETTING_PROPERTY(bool, dart_output);
    SETTING_PROPERTY(bool, time_output);
    SETTING_PROPERTY(bool, plot_output);
    SETTING_PROPERTY(bool, diff_output);
    SETTING_PROPERTY(bool, flamegraph_output);
    SETTING_PROPERTY(int, verbose);
    SETTING_PROPERTY(bool, debug);
    SETTING_PROPERTY(bool, banner);
    SETTING_PROPERTY(bool, flat_profile);
    SETTING_PROPERTY(bool, timeline_profile);
    SETTING_PROPERTY(bool, collapse_threads);
    SETTING_PROPERTY(bool, collapse_processes);
    SETTING_PROPERTY(bool, destructor_report);
    SETTING_PROPERTY(uint16_t, max_depth);
    SETTING_PROPERTY(string_t, time_format);
    SETTING_PROPERTY(string_t, python_exe);
    SETTING_PROPERTY(strvector_t, command_line);
    SETTING_PROPERTY(size_t, throttle_count);
    SETTING_PROPERTY(size_t, throttle_value);
    // width/precision
    SETTING_PROPERTY(int16_t, precision);
    SETTING_PROPERTY(int16_t, width);
    SETTING_PROPERTY(bool, scientific);
    SETTING_PROPERTY(int16_t, timing_precision);
    SETTING_PROPERTY(int16_t, timing_width);
    SETTING_PROPERTY(string_t, timing_units);
    SETTING_PROPERTY(bool, timing_scientific);
    SETTING_PROPERTY(int16_t, memory_precision);
    SETTING_PROPERTY(int16_t, memory_width);
    SETTING_PROPERTY(string_t, memory_units);
    SETTING_PROPERTY(bool, memory_scientific);
    // output
    SETTING_PROPERTY(string_t, output_path);
    SETTING_PROPERTY(string_t, output_prefix);
    // dart
    SETTING_PROPERTY(string_t, dart_type);
    SETTING_PROPERTY(uint64_t, dart_count);
    SETTING_PROPERTY(bool, dart_label);
    // parallelism
    SETTING_PROPERTY(size_t, max_thread_bookmarks);
    SETTING_PROPERTY(bool, cpu_affinity);
    SETTING_PROPERTY(bool, mpi_init);
    SETTING_PROPERTY(bool, mpi_finalize);
    SETTING_PROPERTY(bool, mpi_thread);
    SETTING_PROPERTY(string_t, mpi_thread_type);
    SETTING_PROPERTY(bool, upcxx_init);
    SETTING_PROPERTY(bool, upcxx_finalize);
    SETTING_PROPERTY(int32_t, node_count);
    // misc
    SETTING_PROPERTY(bool, stack_clearing);
    SETTING_PROPERTY(bool, add_secondary);
    SETTING_PROPERTY(tim::process::id_t, target_pid);
    // components
    SETTING_PROPERTY(string_t, global_components);
    SETTING_PROPERTY(string_t, tuple_components);
    SETTING_PROPERTY(string_t, list_components);
    SETTING_PROPERTY(string_t, ompt_components);
    SETTING_PROPERTY(string_t, mpip_components);
    SETTING_PROPERTY(string_t, trace_components);
    SETTING_PROPERTY(string_t, profiler_components);
    SETTING_PROPERTY(string_t, components);
    // papi
    SETTING_PROPERTY(bool, papi_multiplexing);
    SETTING_PROPERTY(bool, papi_fail_on_error);
    SETTING_PROPERTY(bool, papi_quiet);
    SETTING_PROPERTY(string_t, papi_events);
    SETTING_PROPERTY(bool, papi_attach);
    SETTING_PROPERTY(int, papi_overflow);
    // cuda/nvtx/cupti
    SETTING_PROPERTY(uint64_t, cuda_event_batch_size);
    SETTING_PROPERTY(bool, nvtx_marker_device_sync);
    SETTING_PROPERTY(int32_t, cupti_activity_level);
    SETTING_PROPERTY(string_t, cupti_activity_kinds);
    SETTING_PROPERTY(string_t, cupti_events);
    SETTING_PROPERTY(string_t, cupti_metrics);
    SETTING_PROPERTY(int, cupti_device);
    // roofline
    SETTING_PROPERTY(string_t, roofline_mode);
    SETTING_PROPERTY(string_t, cpu_roofline_mode);
    SETTING_PROPERTY(string_t, gpu_roofline_mode);
    SETTING_PROPERTY(string_t, cpu_roofline_events);
    SETTING_PROPERTY(string_t, gpu_roofline_events);
    SETTING_PROPERTY(bool, roofline_type_labels);
    SETTING_PROPERTY(bool, roofline_type_labels_cpu);
    SETTING_PROPERTY(bool, roofline_type_labels_gpu);
    SETTING_PROPERTY(bool, instruction_roofline);
    // ert
    SETTING_PROPERTY(uint64_t, ert_num_threads);
    SETTING_PROPERTY(uint64_t, ert_num_threads_cpu);
    SETTING_PROPERTY(uint64_t, ert_num_threads_gpu);
    SETTING_PROPERTY(uint64_t, ert_num_streams);
    SETTING_PROPERTY(uint64_t, ert_grid_size);
    SETTING_PROPERTY(uint64_t, ert_block_size);
    SETTING_PROPERTY(uint64_t, ert_alignment);
    SETTING_PROPERTY(uint64_t, ert_min_working_size);
    SETTING_PROPERTY(uint64_t, ert_min_working_size_cpu);
    SETTING_PROPERTY(uint64_t, ert_min_working_size_gpu);
    SETTING_PROPERTY(uint64_t, ert_max_data_size);
    SETTING_PROPERTY(uint64_t, ert_max_data_size_cpu);
    SETTING_PROPERTY(uint64_t, ert_max_data_size_gpu);
    SETTING_PROPERTY(string_t, ert_skip_ops);
    // signals
    SETTING_PROPERTY(bool, allow_signal_handler);
    SETTING_PROPERTY(bool, enable_signal_handler);
    SETTING_PROPERTY(bool, enable_all_signals);
    SETTING_PROPERTY(bool, disable_all_signals);

    return settings;
}
}  // namespace pysettings
//
//======================================================================================//
